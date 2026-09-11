#!/usr/bin/env python
"""Aim of this script : evaluate the trained parameters on the held out experiment condition dataset (i.e. the test data)

General guideline
- Always use sample 0 for now.
- Always use noise_level =0.3 for now.

Some things to note
- Ground truth params files are found in : /home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic/parameters.json.
    This file is identical to /home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic/test_parameters.json
- Noiseless data and stim conditions are found in /home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic/
- Noisy data simulations are found in /home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic/noise_0.30/. This data is used for training the model parameters
- Parameters are optimised using 4 different objectives, labelled A, B, E, F. Each optimised set of parameters is found in /home/dabin/data/wc_simulations/check_fit_{obj}_syn_noise_0.3.npz where obj is one of A, B, E, or F.
- Held out test data is found in /home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic/noise_0.30/synthetic_test_data_noisy_wcs.npz

Things for this script to do
1. Sanity check that the train parameters.json file is identical to the test_parameters.json
2. Simulate the test stimulus conditions using the params trained on the noisy data.
3. Simulate the test stimulus condition using the ground truth parameters
4. Calculate the squared error between the simulated and ground truth test data
5. Visualise the two fits

Implementation notes
- The "ground truth test data" the squared error is measured against is the NOISELESS clean test
  trajectory (``synthetic_test_data_clean_wcs.npz``) — the traces the generator actually produced for
  the held-out stimulus conditions. The noisy test file supplies the observed points shown as scatter.
- Both simulations are free-runs (the model's own E/I fed back each step) through the SAME
  ``fit_smoothing_sweep._simulate_free`` machinery. The generator's WCS update (simulate_data.py) is the
  identical dt=1 Euler step as ``model2.model_jax``, so the ground-truth-param free-run reproduces the
  clean trajectory (its MSE vs clean is the pipeline's numerical-consistency check, ~0).
- The ground-truth WCS params (parameters.json -> sample_0 -> "wcs") cover every model2 param except the
  learned ``s0_S``; the generator seeds S0 = I0, so the GT free-run is seeded from I0. Trained free-runs
  use their own fitted ``s0_S``.

Usage:
    python plot_synthetic_data.py                 # A,B,E,F; 6 conditions; SE table + figures
    python plot_synthetic_data.py --objectives A,F
    python plot_synthetic_data.py --all           # show all 12 test conditions
    python plot_synthetic_data.py --n-show 4 --seed 3
"""
from __future__ import annotations

import argparse
import filecmp
import json
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
import model2                                    # noqa: E402
from check_objective_setup import _build_synthetic_cv  # noqa: E402

# ---- fixed config (guideline: always sample 0, always noise level 0.3) --------------------------
SAMPLE = 0
SYN = Path("/home/dabin/code/EDGAR-gamma/projects/wilson_cowan/data_loader/synthetic")
PARAMS_JSON = SYN / "parameters.json"
TEST_PARAMS_JSON = SYN / "test_parameters.json"
TEST_NOISY = SYN / "noise_0.30" / "synthetic_test_data_noisy_wcs.npz"
TEST_CLEAN = SYN / "synthetic_test_data_clean_wcs.npz"
PARAMS_DIR = Path("/home/dabin/data/wc_simulations")
FIT_TEMPLATE = "check_fit_{obj}_syn_noise_0.3.npz"
OBJECTIVES = ["A", "B", "E", "F"]
CHOP = (-0.15, 0.4)                              # same window the fits used (keeps all 450 test bins)
MODEL_KEYS = list(model2.model_jax.DEFAULT_PARAMS)   # 16 keys incl. s0_S


def _load_trained(obj: str) -> dict:
    """Fitted params for one objective -> {name: float} over model2's param keys (incl. s0_S)."""
    path = PARAMS_DIR / FIT_TEMPLATE.format(obj=obj)
    if not path.exists():
        raise SystemExit(f"missing {path} — run check_objective_setup.py for objective {obj} first")
    d = np.load(path, allow_pickle=True)
    return {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}


def _load_gt_params() -> dict:
    """Ground-truth WCS params for sample 0 -> model2 param dict (s0_S is seeded per-condition)."""
    j = json.loads(PARAMS_JSON.read_text())
    wcs = j[f"sample_{SAMPLE}"]["wcs"]
    missing = [k for k in MODEL_KEYS if k not in wcs and k != "s0_S"]
    if missing:
        raise SystemExit(f"ground-truth wcs params missing model2 keys {missing}")
    return {k: float(wcs[k]) for k in MODEL_KEYS if k in wcs}


def _free_run(params: dict, sE, sI, E0, I0, S0) -> np.ndarray:
    """Free-running WCS rollout for one condition -> (T, 2) = (E, I)."""
    p = {k: jnp.asarray(v) for k, v in params.items()}
    Ec, Ic = F._simulate_free(p, sE, sI, E0, I0, S0)
    return np.stack([np.asarray(Ec), np.asarray(Ic)], axis=-1)


def _simulate(params: dict, stim: np.ndarray, clean: np.ndarray, s0_from_I0: bool) -> np.ndarray:
    """Free-run every condition. Seeds E0/I0 from the clean trace; S0 from I0 (GT) or fitted s0_S."""
    C, T, _ = clean.shape
    out = np.zeros((C, T, 2))
    for c in range(C):
        E0, I0 = float(clean[c, 0, 0]), float(clean[c, 0, 1])
        S0 = I0 if s0_from_I0 else float(params["s0_S"])
        out[c] = _free_run(params, stim[c, :, 0], stim[c, :, 1], E0, I0, S0)
    return out


def _load_test():
    """Return (clean (C,T,2), noisy (C,T,2), stim (C,T,2), time (T,) ms, pulse_type (C,))."""
    cv_clean = _build_synthetic_cv(str(TEST_CLEAN), SAMPLE, CHOP)
    cv_noisy = _build_synthetic_cv(str(TEST_NOISY), SAMPLE, CHOP)
    clean = np.asarray(cv_clean.train.target_y)
    noisy = np.asarray(cv_noisy.train.target_y)
    stim = np.asarray(cv_clean.train.stim)
    time_ms = np.asarray(cv_clean.time) * 1000.0
    pulse_type = np.asarray(np.load(TEST_CLEAN, allow_pickle=True)["pulse_type"])
    return clean, noisy, stim, time_ms, pulse_type


def _plot_objective(obj, sim_trained, sim_gt, clean, noisy, stim, time_ms, pulse_type,
                    idx, mse_cond, out_png):
    """Overlay the trained-param and ground-truth-param free-runs on the held-out conditions."""
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), sharex=True)
    axes = np.atleast_2d(axes)
    mse_all = float(np.mean((sim_trained - clean) ** 2))
    fig.suptitle(
        f"objective {obj} on held-out test stimuli (sample {SAMPLE}, noise 0.3) — "
        f"free-run vs ground-truth params   [MSE(trained vs clean, all 12 cond) = {mse_all:.4g}]",
        fontsize=11,
    )
    for col, c in enumerate(idx):
        for row, ch in enumerate(("E", "I")):
            ax = axes[row, col]
            for a, b in F._stim_spans(stim[c, :, row]):
                ax.axvspan(time_ms[a], time_ms[min(b, len(time_ms) - 1)], color="0.85", zorder=0)
            ax.scatter(time_ms, noisy[c, :, row], color="k", s=3, alpha=0.25, label="noisy observed")
            ax.plot(time_ms, clean[c, :, row], color="0.4", lw=1.0, ls="--", label="ground truth (clean)")
            ax.plot(time_ms, sim_gt[c, :, row], color="tab:blue", lw=1.4, label="GT-param free-run")
            ax.plot(time_ms, sim_trained[c, :, row], color="tab:red", lw=1.4, label="trained free-run")
            if row == 0:
                ax.set_title(f"cond {c}: {pulse_type[c]}\nMSE={mse_cond[c]:.3g}", fontsize=8)
            ax.set_ylabel(f"{ch} rate")
            if row == 1:
                ax.set_xlabel("time (ms)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[saved] {out_png}")


def _plot_mse_summary(mse_by_obj, gt_mse, out_png):
    """Bar chart of per-objective MSE (trained free-run vs clean ground truth)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    objs = list(mse_by_obj)
    ax.bar(objs, [mse_by_obj[o] for o in objs], color="tab:red", alpha=0.85)
    ax.axhline(gt_mse, color="tab:blue", ls="--", lw=1.2,
               label=f"GT-param free-run vs clean ({gt_mse:.2g})")
    ax.set_ylabel("MSE vs clean ground truth (all 12 conditions)")
    ax.set_xlabel("training objective")
    ax.set_title(f"Held-out test error by objective (sample {SAMPLE}, noise 0.3)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default=",".join(OBJECTIVES),
                    help="comma-separated subset of A,B,E,F (default: all)")
    ap.add_argument("--n-show", type=int, default=6, help="number of conditions to plot per objective")
    ap.add_argument("--all", action="store_true", help="plot all 12 test conditions (overrides --n-show)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random condition draw")
    args = ap.parse_args()

    run_objs = [o.strip().upper() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in run_objs if o not in OBJECTIVES]
    if bad:
        raise SystemExit(f"unknown objective(s) {bad}; valid: {OBJECTIVES}")

    # 1. sanity check: the train and test ground-truth param files must be identical.
    same = filecmp.cmp(PARAMS_JSON, TEST_PARAMS_JSON, shallow=False)
    print(f"[1] parameters.json == test_parameters.json : {same}")
    if not same:
        raise SystemExit("parameters.json and test_parameters.json differ — aborting")

    clean, noisy, stim, time_ms, pulse_type = _load_test()
    C = clean.shape[0]
    n_show = C if args.all else int(min(max(1, args.n_show), C))
    idx = np.sort(np.random.default_rng(args.seed).choice(C, n_show, replace=False))

    # 3. ground-truth-param free-run (seeded S0 = I0, matching the generator).
    gt_params = _load_gt_params()
    sim_gt = _simulate(gt_params, stim, clean, s0_from_I0=True)
    gt_mse = float(np.mean((sim_gt - clean) ** 2))
    print(f"[3] GT-param free-run vs clean ground truth: MSE = {gt_mse:.4g} "
          f"(pipeline consistency check, expect ~0)\n")

    # 2 + 4. trained free-runs and their squared error against the clean ground truth.
    print(f"[4] squared error vs clean ground truth (sample {SAMPLE}, noise 0.3, all {C} conditions)")
    print(f"    {'obj':>4} {'MSE':>12} {'SSE':>14}")
    mse_by_obj = {}
    sims_trained = {}
    for obj in run_objs:
        params = _load_trained(obj)
        sim = _simulate(params, stim, clean, s0_from_I0=False)
        sims_trained[obj] = sim
        se = (sim - clean) ** 2
        mse_by_obj[obj] = float(se.mean())
        print(f"    {obj:>4} {se.mean():>12.6g} {se.sum():>14.6g}")
    print()

    # 5. visualise the two fits (trained vs ground-truth params) + an MSE-by-objective summary.
    for obj in run_objs:
        mse_cond = np.mean((sims_trained[obj] - clean) ** 2, axis=(1, 2))   # (C,)
        _plot_objective(obj, sims_trained[obj], sim_gt, clean, noisy, stim, time_ms, pulse_type,
                        idx, mse_cond, PARAMS_DIR / f"synthetic_test_fit_{obj}.png")
    _plot_mse_summary(mse_by_obj, gt_mse, PARAMS_DIR / "synthetic_test_mse_by_objective.png")


if __name__ == "__main__":
    main()
