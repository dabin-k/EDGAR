#!/usr/bin/env python
"""Fit the WCS model (seed_programs/model2.py) to one session preprocessed at several
Hamming-smoothing widths, and see how the fitted parameters / fit quality move with width.

Motivation (journal 2026-08-24): Lin et al. smooth population PSTHs with a 40 ms Hamming
window, which induces acausality (rates move before the stimulus); no smoothing leaves the
t-1 signal too noisy for a one-step autoregressive model. We fit the SAME session at 5
widths (0/2/5/20/40 ms) and compare.

Reuses the real scoring path: the project's ``apply_model`` / ``loss_fn`` (objective A,
one-step teacher-forced MSE) and ``edgar.scoring.scoring._get_params`` /
``_optimize`` / ``_eval_loss``. Each smoothing file is fit independently as a single
(n=1) sample built directly from ``neural_data.build_cv_samples`` (the mouse-split
``load_data`` real path does not apply — it needs >=2 sessions and a different glob).

Data lives in a SEPARATE tree from config's data_path:
    /home/dabin/code/ichun_opto/results/smoothing_windows/h{w}_population_rates_M150605_ICTP1_s1.npz

Usage:
    python fit_smoothing_sweep.py                 # all widths
    python fit_smoothing_sweep.py --only 40       # one width (smoke)
    python fit_smoothing_sweep.py --max-iter 50   # faster GD
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]                             # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))

# ── The smoothing-sweep data tree (NOT config's data_path). ──
DATA_DIR = Path("/home/dabin/code/ichun_opto/results/smoothing_windows")
PARAMS_DIR = DATA_DIR / "params"
FNAME_RE = re.compile(r"^h(\d+)_population_rates_.*\.npz$")

# Peri-stimulus chop / warmup / objective (mirror config.yaml project_params).
CHOP = (-0.150, 0.400)          # seconds kept around onset (chop_pre_ms/post_ms)
WARMUP_MS = 100.0               # burn-in excluded from the loss
OBJECTIVE = "A"                 # one-step teacher-forced MSE

# Set objective env BEFORE importing the project entry points read it lazily anyway,
# but be explicit so apply_model / loss_fn / _rollout_anchors see the right values.
os.environ["EDGAR_WC_OBJECTIVE"] = OBJECTIVE

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

from load_data import apply_model, loss_fn, _rollout_anchors   # noqa: E402
from neural_data import build_cv_samples                       # noqa: E402
import model2                                                  # noqa: E402
import param_est2                                              # noqa: E402
from edgar.scoring.scoring import _get_params, _optimize, _eval_loss  # noqa: E402

PARAM_KEYS = list(model2.model.DEFAULT_PARAMS.keys())          # 16 incl. s0_S
GD = {"learning_rate": 0.001, "max_iter": 200, "gradient_clip_norm": 5.0}


_LINKDIR = tempfile.TemporaryDirectory(prefix="wc_smoothing_")


def _canonical_path(path: Path, w: int) -> str:
    """build_cv_samples' `_animal_id` regex needs a `population_rates_<id>_s1_trimmed.npz`
    basename; our files are `h{w}_population_rates_<id>_s1.npz`. Return a symlink with a
    canonical name (animal_id carries the width so meta stays distinct)."""
    m = re.match(r"^h\d+_population_rates_(.+)_s1\.npz$", path.name)
    animal = m.group(1) if m else path.stem
    link = Path(_LINKDIR.name) / f"population_rates_{animal}w{w}_s1_trimmed.npz"
    if not link.exists():
        link.symlink_to(path.resolve())
    return str(link)


def _build_data(split, time_axis: np.ndarray) -> dict:
    """One CVSplit (target_y (C,T,2), stim (C,T,2)) -> the n=1 apply_model data dict."""
    T = time_axis.shape[0]
    anchor_starts, K = _rollout_anchors(T)
    target_y = jnp.asarray(split.target_y)[None]              # (1, C, T, 2)
    stim = np.asarray(split.stim)                             # (C, T, 2)
    sE = jnp.asarray(stim[..., 0])[None]                     # (1, C, T)
    sI = jnp.asarray(stim[..., 1])[None]
    target_y_future = jnp.stack(
        [target_y[:, :, a + 1: a + 1 + K, :] for a in anchor_starts], axis=2
    )                                                         # (1, C, A, K, 2)
    time = jnp.broadcast_to(jnp.asarray(time_axis)[None, :], (1, T))
    return {"target_y": target_y, "stim_E": sE, "stim_I": sI,
            "target_y_future": target_y_future, "time": time}


def _simulate_free(p: dict, sE: np.ndarray, sI: np.ndarray, E0, I0, S0):
    """Free-running WCS rollout for one condition: feed the model's own E/I back in.

    Carries the slow-inhibition state S (seeded from the fitted s0_S). Uses the
    stim-at-(t-1) convention (matching apply_model). Returns E, I of length T.
    """
    xs = {"sE": jnp.asarray(sE[:-1]), "sI": jnp.asarray(sI[:-1])}

    def step(carry, s):
        state, E_prev, I_prev = carry
        y_prev = {"E_prev": E_prev, "I_prev": I_prev,
                  "stim_E_prev": s["sE"], "stim_I_prev": s["sI"]}
        new_state, (E, I) = model2.model_jax(state, y_prev, p)
        return (new_state, E, I), (E, I)

    carry0 = ({"S": jnp.asarray(S0)}, jnp.asarray(E0), jnp.asarray(I0))
    _, (Es, Is) = jax.lax.scan(step, carry0, xs)
    Es = np.concatenate([[E0], np.asarray(Es)])
    Is = np.concatenate([[I0], np.asarray(Is)])
    return Es, Is


def _stim_spans(stim_vec):
    """Contiguous (start, end) bins where the stimulus is on (for axvspan markers)."""
    on = np.abs(np.asarray(stim_vec)) > 0
    if not on.any():
        return []
    edges = np.diff(on.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on)]
    return list(zip(starts, ends))


def _fit_one(path: Path, w: int, gd: dict):
    """Fit WCS to one smoothing file; save params npz + a per-file fit figure.

    Returns (fitted_scalar_params_dict, train_loss, test_loss, n_conditions).
    """
    cv = build_cv_samples(_canonical_path(path, w), cv_type="k_fold",
                          held_out_fold=0, chop=CHOP)
    time_axis = np.asarray(cv.time)
    T = time_axis.shape[0]
    dt_s = float(time_axis[1] - time_axis[0]) if T > 1 else 0.001
    warmup_bins = int(round((WARMUP_MS / 1000.0) / dt_s))
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(warmup_bins)
    os.environ["EDGAR_WC_DT"] = str(dt_s)

    X_train = _build_data(cv.train, time_axis)
    X_test = _build_data(cv.test, time_axis)
    C = int(cv.train.n)

    params_init = _get_params(param_est2.parameter_estimator,
                              model2.model.DEFAULT_PARAMS, X_train)
    fit = _optimize(model2.model_jax, loss_fn, params_init, X_train, gd, apply_model)
    train_loss = _eval_loss(model2.model_jax, loss_fn, fit, X_train, apply_model)
    test_loss = _eval_loss(model2.model_jax, loss_fn, fit, X_test, apply_model)

    fit_scalar = {k: float(np.asarray(fit[k]).reshape(-1)[0]) for k in PARAM_KEYS}
    p_scalar = {k: jnp.asarray(v) for k, v in fit_scalar.items()}

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PARAMS_DIR / f"h{w}_params.npz",
        param_keys=np.array(PARAM_KEYS),
        param_values=np.array([fit_scalar[k] for k in PARAM_KEYS], dtype=np.float64),
        train_loss=np.float64(train_loss),
        test_loss=np.float64(test_loss),
        smoothing_ms=np.int64(w),
        chop=np.array(CHOP, dtype=np.float64),
        objective=np.str_(OBJECTIVE),
        n_conditions=np.int64(C),
    )

    _plot_fit(fit, p_scalar, cv, time_axis,
              title=f"h{w} smoothing — WCS fit  (train={train_loss:.4f}, test={test_loss:.4f})",
              out_png=PARAMS_DIR / f"h{w}_fit.png")
    print(f"  h{w:>2}: C={C}  train={train_loss:.5f}  test={test_loss:.5f}")
    return fit_scalar, float(train_loss), float(test_loss), C


def _plot_fit(fit, p_scalar, cv, time_axis, title, out_png):
    """Overlay teacher-forced one-step + free-run rollout on the real E/I traces."""
    X_train = _build_data(cv.train, time_axis)
    out = apply_model(model2.model_jax, X_train, fit)
    pred_1step = np.asarray(out["pred_y_1step"])[0]           # (C, T-1, 2)

    target = np.asarray(cv.train.target_y)                    # (C, T, 2)
    stim = np.asarray(cv.train.stim)                          # (C, T, 2)
    C = target.shape[0]
    t_ms = time_axis * 1000.0
    n_show = min(C, 4)
    idx = np.linspace(0, C - 1, n_show).round().astype(int)
    s0_S = p_scalar["s0_S"]

    fig, axes = plt.subplots(2, n_show, figsize=(4.2 * n_show, 6.4), sharex=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=11)
    for col, c in enumerate(idx):
        E_real, I_real = target[c, :, 0], target[c, :, 1]
        sE, sI = stim[c, :, 0], stim[c, :, 1]
        Ef, If = _simulate_free(p_scalar, sE, sI, E_real[0], I_real[0], s0_S)
        for row, (real, free, one_step, chan, s) in enumerate([
            (E_real, Ef, pred_1step[c, :, 0], "E", sE),
            (I_real, If, pred_1step[c, :, 1], "I", sI),
        ]):
            ax = axes[row, col]
            for a, b in _stim_spans(s):
                ax.axvspan(t_ms[a], t_ms[min(b, len(t_ms) - 1)], color="0.85", zorder=0)
            ax.scatter(t_ms, real, color="k", s=3, alpha=0.25, label="real")
            ax.plot(t_ms[1:], one_step, color="tab:red", lw=0.8, alpha=0.8,
                    label="one-step (TF)")
            ax.plot(t_ms, free, color="tab:blue", lw=1.1, ls="--", label="free-run")
            if row == 0:
                ax.set_title(f"cond {c}", fontsize=9)
            ax.set_ylabel(f"{chan} rate")
            if row == 1:
                ax.set_xlabel("time (ms)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PARAMS_DIR / f"h{w}_fit.png", dpi=110)
    plt.close(fig)


def _plot_sweep(widths, fits, train_losses, test_losses):
    """One panel per parameter: fitted value vs smoothing width; plus a loss panel."""
    order = np.argsort(widths)
    ws = np.asarray(widths)[order]
    n = len(PARAM_KEYS) + 1
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow))
    axes = axes.ravel()
    for i, k in enumerate(PARAM_KEYS):
        vals = np.array([fits[w][k] for w in ws])
        ax = axes[i]
        ax.plot(ws, vals, "o-", color="tab:purple")
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("Hamming width (ms)")
        ax.grid(alpha=0.3)
    ax = axes[len(PARAM_KEYS)]
    ax.plot(ws, [train_losses[w] for w in ws], "o-", label="train", color="tab:blue")
    ax.plot(ws, [test_losses[w] for w in ws], "s-", label="test", color="tab:orange")
    ax.set_title("loss", fontsize=9)
    ax.set_xlabel("Hamming width (ms)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    for j in range(len(PARAM_KEYS) + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("WCS fitted parameters vs Hamming-smoothing width", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(PARAMS_DIR / "param_vs_smoothing.png", dpi=110)
    plt.close(fig)
    print(f"[saved] {PARAMS_DIR / 'param_vs_smoothing.png'}")


def _discover_widths() -> dict[int, Path]:
    out = {}
    for p in sorted(DATA_DIR.glob("h*_population_rates_*.npz")):
        m = FNAME_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="fit a single width (ms)")
    ap.add_argument("--max-iter", type=int, default=GD["max_iter"])
    ap.add_argument("--lr", type=float, default=GD["learning_rate"])
    ap.add_argument("--clip", type=float, default=GD["gradient_clip_norm"])
    args = ap.parse_args()

    gd = {"learning_rate": args.lr, "max_iter": args.max_iter,
          "gradient_clip_norm": args.clip}

    files = _discover_widths()
    if not files:
        raise SystemExit(f"no h*_population_rates_*.npz files under {DATA_DIR}")
    if args.only is not None:
        if args.only not in files:
            raise SystemExit(f"width {args.only} not found; have {sorted(files)}")
        files = {args.only: files[args.only]}

    print(f"data dir: {DATA_DIR}")
    print(f"widths:   {sorted(files)}  objective={OBJECTIVE}  chop={CHOP}  "
          f"gd={{lr:{gd['learning_rate']}, max_iter:{gd['max_iter']}, "
          f"clip:{gd['gradient_clip_norm']}}}\n")

    fits, train_losses, test_losses = {}, {}, {}
    for w in sorted(files):
        fit_scalar, tr, te, _C = _fit_one(files[w], w, gd)
        fits[w], train_losses[w], test_losses[w] = fit_scalar, tr, te

    if len(fits) > 1:
        _plot_sweep(sorted(fits), fits, train_losses, test_losses)
    print(f"\n[done] params + figures in {PARAMS_DIR}")


if __name__ == "__main__":
    main()
