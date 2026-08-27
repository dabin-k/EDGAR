#!/usr/bin/env python
"""Plot one-step + free-run predictions for the check_objective_setup fits (objectives A, B, E, F).

For each objective this loads the fitted params saved by ``check_objective_setup.py``
(``/home/dabin/data/wc_simulations/check_fit_{obj}.npz``), rebuilds the SAME M150605 CV sample the fit used
(``data_file`` / ``chop`` / ``held_out_fold`` / rollout config are all read back from the npz),
picks a few random conditions, and overlays on the real E/I traces:

  * one-step  — the next-step forecast. For A/B/F this is the teacher-forced one-step
    (``pred_y_1step``, real E/I fed in each step); for E it is the EKF one-step ``H·m_t^-``,
    which honestly uses only PAST data through the filter (matching how objective E scores it).
  * free-run  — the autonomous rollout: the model's own predicted (E, I, S) fed back in each step
    (deterministic WCS via ``fit_smoothing_sweep._simulate_free``). Identical machinery for all
    four objectives — the objective-E ``kf_*`` noise params do NOT enter a free run (process noise
    sits at its mean 0; obs noise never touches the latent), so E's free run is the same
    deterministic WCS rollout as A/B/F.

Reuses ``fit_smoothing_sweep.py`` (``_simulate_free`` / ``_stim_spans`` / ``_build_data``).

Usage:
    python plot_objective_fits.py                    # A,B,E,F, 4 random conditions, seed 0
    python plot_objective_fits.py --objectives A,E    # subset
    python plot_objective_fits.py --n-show 6 --seed 3 # more conditions / different draw
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]                             # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))
sys.path.insert(0, str(WC / "scripts"))

import jax.numpy as jnp                          # noqa: E402
import fit_smoothing_sweep as F                  # noqa: E402 (sets Agg backend + adds paths)
import matplotlib.pyplot as plt                  # noqa: E402
from neural_data import build_cv_samples         # noqa: E402
from load_data import apply_model                # noqa: E402
import model2                                    # noqa: E402
import model2_kalman                             # noqa: E402

PARAMS_DIR = "/home/dabin/data/wc_simulations"
MODEL_BY_OBJECTIVE = {"A": model2, "B": model2, "E": model2_kalman, "F": model2}
OBJECTIVES = ["A", "B", "E", "F"]


def _load_fit(obj: str, suffix: str = "") -> tuple[dict, dict]:
    """Load ``check_fit_{obj}{suffix}.npz`` -> (scalar param dict, metadata dict)."""
    path = PARAMS_DIR / f"check_fit_{obj}{suffix}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} — run check_objective_setup.py first")
    d = np.load(path, allow_pickle=True)
    params = {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}
    init = {k: float(v) for k, v in zip(d["param_keys"], d["init_values"])}
    meta = {
        "init": init,
        "data_file": str(d["data_file"]),
        "chop": tuple(float(x) for x in d["chop"]),
        "held_out_fold": int(d["held_out_fold"]),
        "warmup_bins": int(d["warmup_bins"]),
        "rollout_k": int(d["rollout_k"]),
        "anchor_stride": int(d["anchor_stride"]),
        "dt_s": float(d["dt_s"]),
        "train_loss": float(d["train_loss"]),
        "test_loss": float(d["test_loss"]),
        "model": str(d["model"]),
        "cv_type": str(d["cv_type"]) if "cv_type" in d.files else "k_fold",
        "sample_idx": int(d["sample_idx"]) if "sample_idx" in d.files else None,
    }
    return params, meta


def _rebuild_cv(meta: dict, sample_fallback: int = 0):
    """Rebuild the exact CV sample the fit used and republish its rollout/warmup env settings.

    Synthetic fits (``cv_type == 'synthetic_clean'``) can't go through ``build_cv_samples`` — the
    stimulus is rebuilt straight from the design arrays via ``check_objective_setup._build_synthetic_cv``.
    ``sample_idx`` is read from the npz; files saved before it was recorded fall back to
    ``sample_fallback`` (the ``--sample`` CLI arg). The resolved index is written back into ``meta``.
    """
    os.environ["EDGAR_WC_ROLLOUT_K"] = str(meta["rollout_k"])
    os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(meta["anchor_stride"])
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(meta["warmup_bins"])
    os.environ["EDGAR_WC_DT"] = str(meta["dt_s"])
    if meta.get("cv_type") == "synthetic_clean":
        from check_objective_setup import _build_synthetic_cv
        s = meta.get("sample_idx")
        s = sample_fallback if s is None or s < 0 else s
        meta["sample_idx"] = s
        cv = _build_synthetic_cv(meta["data_file"], s, meta["chop"])
    else:
        cv = build_cv_samples(meta["data_file"], cv_type="k_fold",
                              held_out_fold=meta["held_out_fold"], chop=meta["chop"])
    return cv, np.asarray(cv.time)


def _one_step(obj: str, model_mod, cv, time_axis, fit_scalar: dict) -> np.ndarray:
    """One-step prediction ``(C, T-1, 2)`` under objective ``obj`` (routes E through the EKF)."""
    os.environ["EDGAR_WC_OBJECTIVE"] = obj          # apply_model reads this: E -> Kalman path
    X_train = F._build_data(cv.train, time_axis)
    fit_n1 = {k: jnp.asarray(v)[None] for k, v in fit_scalar.items()}   # add n=1 sample axis
    out = apply_model(model_mod.model_jax, X_train, fit_n1)
    return np.asarray(out["pred_y_1step"])[0]


def _plot_objective(obj: str, cv, time_axis, fit_scalar: dict, meta: dict,
                    idx: np.ndarray, out_png: Path, show_one_step: bool = False,
                    truth_path: str | None = None, truth_sample: int | None = None) -> None:
    """Overlay real / free-run (+ optional one-step / clean-truth) for ``obj`` on the conditions."""
    model_mod = MODEL_BY_OBJECTIVE[obj]
    # One-step teacher-forced prediction is off by default (it flatters the fit — real E/I is fed
    # in each step); only computed/plotted with --plot-one-step-teacher-forced.
    pred_1step = _one_step(obj, model_mod, cv, time_axis, fit_scalar) if show_one_step else None

    target = np.asarray(cv.train.target_y)          # (C, T, 2)
    stim = np.asarray(cv.train.stim)                # (C, T, 2)

    # Noiseless reference: the paired CLEAN synthetic trajectory (--truth-traj). Rebuilt via the
    # same _build_synthetic_cv so conditions/sample/chop line up exactly with the (noisy) fit data.
    truth_traj = None
    if truth_path is not None:
        from check_objective_setup import _build_synthetic_cv
        ts = truth_sample if truth_sample is not None else meta.get("sample_idx", 0)
        ts = 0 if ts is None or ts < 0 else ts
        tcv = _build_synthetic_cv(truth_path, ts, meta["chop"])
        truth_traj = np.asarray(tcv.train.target_y)          # (C, T, 2)
        if truth_traj.shape[0] != target.shape[0] or truth_traj.shape[1] != len(time_axis):
            raise SystemExit(f"--truth-traj shape {truth_traj.shape} does not align with the fit "
                             f"data (C={target.shape[0]}, T={len(time_axis)}) — wrong file/sample?")
    t_ms = time_axis * 1000.0
    p_scalar = {k: jnp.asarray(v) for k, v in fit_scalar.items()}
    p_init = {k: jnp.asarray(v) for k, v in meta["init"].items()}   # param_est2 init (pre-GD)
    s0_S = p_scalar["s0_S"]
    s0_S_init = p_init["s0_S"]
    one_step_label = "EKF one-step (H·mₜ⁻)" if obj == "E" else "one-step (teacher-forced)"

    n_show = len(idx)
    fig, axes = plt.subplots(2, n_show, figsize=(4.2 * n_show, 6.4), sharex=True)
    axes = np.atleast_2d(axes)
    src = Path(meta["data_file"]).stem
    if meta.get("cv_type") == "synthetic_clean" and meta.get("sample_idx") not in (None, -1):
        src += f" (sample {meta['sample_idx']})"
    kind = "one-step + free-run" if show_one_step else "free-run"
    fig.suptitle(
        f"objective {obj} ({meta['model']}) — {kind} on {src}  "
        f"(train={meta['train_loss']:.4g}, test={meta['test_loss']:.4g})",
        fontsize=11,
    )
    for col, c in enumerate(idx):
        E_real, I_real = target[c, :, 0], target[c, :, 1]
        sE, sI = stim[c, :, 0], stim[c, :, 1]
        Ef, If = F._simulate_free(p_scalar, sE, sI, E_real[0], I_real[0], s0_S)
        Ef0, If0 = F._simulate_free(p_init, sE, sI, E_real[0], I_real[0], s0_S_init)
        for row, (real, free, free_init, one_step, truth, chan, s) in enumerate([
            (E_real, Ef, Ef0, None if pred_1step is None else pred_1step[c, :, 0],
             None if truth_traj is None else truth_traj[c, :, 0], "E", sE),
            (I_real, If, If0, None if pred_1step is None else pred_1step[c, :, 1],
             None if truth_traj is None else truth_traj[c, :, 1], "I", sI),
        ]):
            ax = axes[row, col]
            for a, b in F._stim_spans(s):
                ax.axvspan(t_ms[a], t_ms[min(b, len(t_ms) - 1)], color="0.85", zorder=0)
            ax.scatter(t_ms, real, color="k", s=3, alpha=0.25, label="real")
            if truth is not None:
                ax.plot(t_ms, truth, color="tab:blue", lw=1.6, label="truth (clean)")
            if one_step is not None:
                ax.plot(t_ms[1:], one_step, color="tab:purple", lw=0.8, alpha=0.85,
                        label=one_step_label)
            ax.plot(t_ms, free_init, color="tab:green", lw=1.1, alpha=0.85, label="free-run (init)")
            ax.plot(t_ms, free, color="tab:red", lw=1.1, label="free-run (fitted)")
            if row == 0:
                ax.set_title(f"cond {c}", fontsize=9)
            ax.set_ylabel(f"{chan} rate")
            if row == 1:
                ax.set_xlabel("time (ms)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default=",".join(OBJECTIVES),
                    help="comma-separated subset of A,B,E,F (default: all)")
    ap.add_argument("--n-show", type=int, default=4, help="number of random conditions to plot")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random condition draw")
    ap.add_argument("--sample", type=int, default=0,
                    help="synthetic sample index to rebuild when the fit npz predates sample_idx "
                         "being saved (ignored for real-data fits and fits that recorded it)")
    ap.add_argument("--plot-one-step-teacher-forced", action="store_true",
                    help="also overlay the one-step teacher-forced prediction (real E/I fed in "
                         "each step; EKF one-step for E). Off by default — free-run only.")
    ap.add_argument("--truth-traj", default=None,
                    help="path to the paired CLEAN synthetic npz; its noiseless E/I is overlaid as "
                         "a solid blue 'truth (clean)' line so you can judge the fitted free-run "
                         "against the true trajectory (synthetic fits only)")
    ap.add_argument("--truth-traj-sample", type=int, default=None,
                    help="sample index within --truth-traj (default: the fit's own sample)")
    ap.add_argument("--tag", default="",
                    help="read check_fit_{obj}_{tag}.npz and write fit_predictions_{obj}_{tag}.png "
                         "(e.g. --tag rollout40_stride1), so a re-fit's figures don't overwrite the "
                         "existing untagged PNGs.")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    run_objs = [o.strip().upper() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in run_objs if o not in OBJECTIVES]
    if bad:
        raise SystemExit(f"unknown objective(s) {bad}; valid: {OBJECTIVES}")

    # Same random conditions across every objective (they share the sample) for a fair comparison.
    cv_ref, _ = _rebuild_cv(_load_fit(run_objs[0], suffix)[1], sample_fallback=args.sample)
    C = int(cv_ref.train.n)
    n_show = int(min(max(1, args.n_show), C))
    idx = np.sort(np.random.default_rng(args.seed).choice(C, n_show, replace=False))
    print(f"conditions (seed {args.seed}): {idx.tolist()}  of {C}\n")

    for obj in run_objs:
        fit_scalar, meta = _load_fit(obj, suffix)
        cv, time_axis = _rebuild_cv(meta, sample_fallback=args.sample)
        _plot_objective(obj, cv, time_axis, fit_scalar, meta, idx,
                        PARAMS_DIR / f"fit_predictions_{obj}{suffix}.png",
                        show_one_step=args.plot_one_step_teacher_forced,
                        truth_path=args.truth_traj, truth_sample=args.truth_traj_sample)


if __name__ == "__main__":
    main()
