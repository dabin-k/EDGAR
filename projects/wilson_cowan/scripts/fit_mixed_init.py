#!/usr/bin/env python
"""Hybrid fit: per-parameter mixed init, then gradient-descend + evaluate on the h10 data.

The init takes the "dynamics/scale" params (tau_E, tau_I, E_max, I_max, C_E, C_I, tau_S, XE,
XI, s0_S) from the h40 fit and the connection weights (W_EE, W_IE, W_EI, W_II, W_ES, W_IS) from
the h5 fit, then refines everything by GD on the h5 data (loss also evaluated on h5).

Motivation: the earlier all-h40-init hybrid landed in the h40 param basin (large tau_E etc.) yet
generalised better on h10 than the param_est2-init h10 fit. This isolates which half of the
h40 solution carries that benefit — keep h40's slow time-constants/scales but let the couplings
start from the native h10 fit.

Reuses the fit machinery in ``fit_smoothing_sweep.py`` (same objective A / GD config / data
build). Writes ``mixed_params.npz`` + ``mixed_fit.png`` and regenerates
``param_vs_smoothing.png`` with the hybrid fit drawn as a single red X at width=10 across
every panel (params + train/test loss), alongside the standard per-width sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]
REPO = WC.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))
sys.path.insert(0, str(WC / "scripts"))

import jax.numpy as jnp                                       # noqa: E402

import fit_smoothing_sweep as F                               # noqa: E402 (sets Agg backend)
import matplotlib.pyplot as plt                               # noqa: E402
from neural_data import build_cv_samples                      # noqa: E402
from load_data import apply_model, loss_fn                    # noqa: E402
from edgar.scoring.scoring import _optimize, _eval_loss       # noqa: E402
import model2                                                 # noqa: E402

INIT_WIDTH = 40      # source of the "dynamics/scale" params in the mixed init
REST_WIDTH = 5       # source of the remaining params (the connection weights) in the init
FIT_WIDTH = 5        # data GD refines on AND evaluates on

# Mixed init: these keys come from h40; every other param comes from REST_WIDTH (the connection
# weights W_EE/W_IE/W_EI/W_II/W_ES/W_IS).
FROM_INIT_KEYS = ["tau_E", "tau_I", "E_max", "I_max", "C_E", "C_I",
                  "tau_S", "XE", "XI", "s0_S"]

OBJECTIVES = ["A", "B", "C", "D"]   # A one-step | B rollout | C +latent | D +signatures
GD = {**F.GD, "max_iter": 1000}     # more iters than the sweep default (200) for the rollout objectives


def _load_saved(w: int) -> tuple[dict, float, float]:
    """Return (param dict, train_loss, test_loss) from ``h{w}_params.npz``."""
    d = np.load(F.PARAMS_DIR / f"h{w}_params.npz", allow_pickle=True)
    params = {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}
    return params, float(d["train_loss"]), float(d["test_loss"])


def _plot_param_vs_lossfn(results: dict) -> None:
    """One panel per parameter: fitted value vs training objective; plus a loss panel.

    ``results`` maps objective label -> (param dict, train_loss, test_loss). The loss panel is
    log-scaled and labelled non-comparable: each objective minimises a different quantity, so the
    loss *values* are not comparable across objectives — only the per-objective params/trends are.
    """
    objs = list(results.keys())
    x = np.arange(len(objs))
    n = len(F.PARAM_KEYS) + 1
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow))
    axes = axes.ravel()
    for i, k in enumerate(F.PARAM_KEYS):
        vals = [results[o][0][k] for o in objs]
        ax = axes[i]
        ax.plot(x, vals, "o-", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels(objs)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("objective")
        ax.grid(alpha=0.3)
    ax = axes[len(F.PARAM_KEYS)]
    ax.plot(x, [results[o][1] for o in objs], "o-", label="train", color="tab:blue")
    ax.plot(x, [results[o][2] for o in objs], "s-", label="test", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(objs)
    ax.set_title("loss (log; NOT comparable across obj)", fontsize=9)
    ax.set_xlabel("objective")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    for j in range(len(F.PARAM_KEYS) + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"WCS hybrid-init fitted params vs training objective  "
                 f"(init h{INIT_WIDTH} dyn/scale + h{REST_WIDTH} weights; GD+eval on h{FIT_WIDTH})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(F.PARAMS_DIR / "param_vs_loss_fn.png", dpi=110)
    plt.close(fig)
    print(f"[saved] {F.PARAMS_DIR / 'param_vs_loss_fn.png'}")


def _config_rollout() -> tuple[int, int]:
    """Read (rollout_k, anchor_stride) from the project config.yaml (drives objectives B/C/D)."""
    import yaml
    with open(F.WC / "config.yaml") as fh:
        pp = yaml.safe_load(fh).get("project_params", {})
    return int(pp["rollout_k"]), int(pp["anchor_stride"])


def _load_mixed(obj: str) -> tuple[dict, float, float]:
    """Read a saved ``mixed_params_{obj}.npz`` back into (param dict, train_loss, test_loss)."""
    d = np.load(F.PARAMS_DIR / f"mixed_params_{obj}.npz", allow_pickle=True)
    params = {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}
    return params, float(d["train_loss"]), float(d["test_loss"])


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default=",".join(OBJECTIVES),
                    help="comma-separated objectives to (re)fit; the rest are loaded from disk "
                         "for the combined plot (default: all)")
    ap.add_argument("--max-iter", type=int, default=GD["max_iter"],
                    help=f"GD steps for the fitted objectives (default {GD['max_iter']})")
    ap.add_argument("--warm-start", action="store_true",
                    help="seed each fitted objective from its existing mixed_params_{obj}.npz "
                         "(continue training) instead of the hybrid init. NOTE: Adam moment "
                         "state is reset, so this is a continuation, not a bit-identical resume.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, e.g. --tag nsteps_10000 writes "
                         "mixed_params_{obj}_{tag}.npz / mixed_fit_{obj}_{tag}.png and does NOT "
                         "touch the untagged files or param_vs_loss_fn.png (a separate run).")
    ap.add_argument("--lambda-dyn", type=float, default=1.0,
                    help="objective-D feature-term weight (EDGAR_WC_LAMBDA_DYN); ignored by A/B/C.")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    os.environ["EDGAR_WC_LAMBDA_DYN"] = str(args.lambda_dyn)   # only affects objective D
    run_objs = [o.strip().upper() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in run_objs if o not in OBJECTIVES]
    if bad:
        raise SystemExit(f"unknown objective(s) {bad}; valid: {OBJECTIVES}")
    gd = {**GD, "max_iter": args.max_iter}

    files = F._discover_widths()
    for w in (INIT_WIDTH, REST_WIDTH, FIT_WIDTH):
        if w not in files:
            raise SystemExit(f"width {w} not found; have {sorted(files)}")

    # Publish config's rollout settings to env so _rollout_anchors / apply_model pick them up
    # (my scripts don't go through load_data, which is what normally republishes these).
    rollout_k, anchor_stride = _config_rollout()
    os.environ["EDGAR_WC_ROLLOUT_K"] = str(rollout_k)
    os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(anchor_stride)

    init_scalar_src, _, _ = _load_saved(INIT_WIDTH)
    rest_scalar, _, _ = _load_saved(REST_WIDTH)
    # Per-parameter mixed init: FROM_INIT_KEYS from h40, the rest from REST_WIDTH.
    init_scalar = {k: (init_scalar_src[k] if k in FROM_INIT_KEYS else rest_scalar[k])
                   for k in F.PARAM_KEYS}
    print(f"mixed init: {FROM_INIT_KEYS} from h{INIT_WIDTH}, rest from h{REST_WIDTH}")
    print(f"GD + eval on h{FIT_WIDTH} data; fitting {run_objs} (max_iter={args.max_iter}); "
          f"rollout_k={rollout_k}, anchor_stride={anchor_stride}\n")

    # Build the h5 data once (objective-independent: anchors/target_y_future depend only on
    # rollout_k/anchor_stride/warmup, not the objective).
    cv = build_cv_samples(F._canonical_path(files[FIT_WIDTH], FIT_WIDTH),
                          cv_type="k_fold", held_out_fold=0, chop=F.CHOP)
    time_axis = np.asarray(cv.time)
    T = time_axis.shape[0]
    dt_s = float(time_axis[1] - time_axis[0]) if T > 1 else 0.001
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(int(round((F.WARMUP_MS / 1000.0) / dt_s)))
    os.environ["EDGAR_WC_DT"] = str(dt_s)

    X_train = F._build_data(cv.train, time_axis)
    X_test = F._build_data(cv.test, time_axis)
    C = int(cv.train.n)
    hybrid_init = {k: jnp.asarray(init_scalar[k])[None] for k in F.PARAM_KEYS}  # n=1

    results: dict = {}
    for obj in run_objs:
        os.environ["EDGAR_WC_OBJECTIVE"] = obj   # apply_model / loss_fn read this lazily
        if args.warm_start:
            prev, _, _ = _load_mixed(obj)        # continue from the saved fit for this objective
            params_init = {k: jnp.asarray(prev[k])[None] for k in F.PARAM_KEYS}
            print(f"  [warm-start] obj {obj} seeded from mixed_params_{obj}.npz")
        else:
            params_init = hybrid_init
        fit = _optimize(model2.model_jax, loss_fn, params_init, X_train, gd, apply_model)
        train_loss = float(_eval_loss(model2.model_jax, loss_fn, fit, X_train, apply_model))
        test_loss = float(_eval_loss(model2.model_jax, loss_fn, fit, X_test, apply_model))

        fit_scalar = {k: float(np.asarray(fit[k]).reshape(-1)[0]) for k in F.PARAM_KEYS}
        p_scalar = {k: jnp.asarray(v) for k, v in fit_scalar.items()}

        np.savez(
            F.PARAMS_DIR / f"mixed_params_{obj}{suffix}.npz",
            param_keys=np.array(F.PARAM_KEYS),
            param_values=np.array([fit_scalar[k] for k in F.PARAM_KEYS], dtype=np.float64),
            train_loss=np.float64(train_loss),
            test_loss=np.float64(test_loss),
            init_from=np.str_(f"h{INIT_WIDTH}:{','.join(FROM_INIT_KEYS)} + h{REST_WIDTH}:rest"),
            init_from_h40_keys=np.array(FROM_INIT_KEYS),
            fit_to=np.str_(f"h{FIT_WIDTH}"),
            smoothing_ms=np.int64(FIT_WIDTH),
            chop=np.array(F.CHOP, dtype=np.float64),
            objective=np.str_(obj),
            lambda_dyn=np.float64(args.lambda_dyn),   # only meaningful for objective D
            n_conditions=np.int64(C),
        )
        lam_str = f", lambda_dyn={args.lambda_dyn:g}" if obj == "D" else ""
        F._plot_fit(
            fit, p_scalar, cv, time_axis,
            title=(f"objective {obj} — mixed init (h{INIT_WIDTH} dyn/scale + "
                   f"h{REST_WIDTH} weights) -> h{FIT_WIDTH} data  "
                   f"(gd_steps={args.max_iter}{lam_str}, "
                   f"train={train_loss:.4g}, test={test_loss:.4g})"),
            out_png=F.PARAMS_DIR / f"mixed_fit_{obj}{suffix}.png",
        )
        results[obj] = (fit_scalar, train_loss, test_loss)
        print(f"  obj {obj}: C={C}  train={train_loss:.5g}  test={test_loss:.5g}  "
              f"-> mixed_params_{obj}{suffix}.npz, mixed_fit_{obj}{suffix}.png")

    # A tagged run is a standalone experiment: leave the canonical files / combined plot alone.
    if suffix:
        print(f"[done] tagged run '{args.tag}' -> per-objective files in {F.PARAMS_DIR} "
              f"(param_vs_loss_fn.png left untouched)")
        return

    # Fill the combined plot with the objectives we didn't refit this run (from disk).
    for obj in OBJECTIVES:
        if obj not in results:
            results[obj] = _load_mixed(obj)
            print(f"  obj {obj}: loaded from mixed_params_{obj}.npz (not refit)")

    _plot_param_vs_lossfn({o: results[o] for o in OBJECTIVES})
    print(f"[done] per-objective params + figures in {F.PARAMS_DIR}")


if __name__ == "__main__":
    main()
