#!/usr/bin/env python
"""Plot the FULL state-space (Kalman / objective-E) model prediction, not the one-step fit.

Loads saved params (a ``mixed_params_{tag}.npz`` from ``fit_mixed_init.py``), optionally refines
them under objective E (the EKF marginal-NLL loss) for ``--fit-iters`` GD steps, then draws the
state-space view via ``fit_smoothing_sweep._plot_fit_statespace``:
  * free-run (predict-only) trajectory — the model's prediction, no data after the seed,
  * filtered (denoised) trajectory — EKF posterior mean,
  * real data, with the fitted +-2*sigma observation-noise band.

The KF noise/init params (``kf_log_*``, log-variances) are merged in from the model's
``KF_DEFAULT_PARAMS`` when absent, so a fit produced under objectives A-D can still be viewed
(fixed default noise) or refined (``--fit-iters > 0`` makes the noise learnable).

Usage:
    python plot_statespace_fit.py                                 # tag B_nsteps_10000, view only
    python plot_statespace_fit.py --tag A_nsteps_10000
    python plot_statespace_fit.py --fit-iters 500                 # refine under objective E first
    python plot_statespace_fit.py --out /tmp/ss.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]
REPO = WC.parents[1]
for _p in (REPO, WC / "data_loader", WC / "seed_programs", WC / "scripts"):
    sys.path.insert(0, str(_p))

import jax.numpy as jnp                          # noqa: E402
import fit_smoothing_sweep as F                  # noqa: E402 (sets Agg backend)
from neural_data import build_cv_samples         # noqa: E402
from load_data import apply_model, loss_fn      # noqa: E402
import model2_kalman as model2                   # noqa: E402 (objective-E variant: kf_* in DEFAULT_PARAMS)
from edgar.scoring.scoring import _optimize, _eval_loss        # noqa: E402


def _load_params(tag: str) -> dict:
    d = np.load(F.PARAMS_DIR / f"mixed_params_{tag}.npz", allow_pickle=True)
    return {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="B_nsteps_10000",
                    help="mixed_params_{tag}.npz to load (default: B_nsteps_10000)")
    ap.add_argument("--width", type=int, default=5, help="smoothing width h{width} (default: 5)")
    ap.add_argument("--fit-iters", type=int, default=0,
                    help="GD steps to refine params under objective E before plotting (default: 0 "
                         "= view the loaded params as-is with default fixed noise)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate for --fit-iters")
    ap.add_argument("--out", default=None,
                    help="output png (default: <params>/statespace_fit_{tag}.png)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (F.PARAMS_DIR / f"statespace_fit_{args.tag}.png")

    # Merge KF noise/init params so the state-space model is fully specified; they are learnable
    # only if --fit-iters > 0 (otherwise fixed at these defaults).
    params = _load_params(args.tag)
    for k, v in model2.model.DEFAULT_PARAMS.items():
        if k.startswith("kf_"):
            params.setdefault(k, v)
    fit = {k: jnp.asarray(v)[None] for k, v in params.items()}

    files = F._discover_widths()
    if args.width not in files:
        raise SystemExit(f"width {args.width} not found; have {sorted(files)}")
    cv = build_cv_samples(F._canonical_path(files[args.width], args.width),
                          cv_type="k_fold", held_out_fold=0, chop=F.CHOP)
    time_axis = np.asarray(cv.time)
    dt_s = float(time_axis[1] - time_axis[0])
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(int(round((F.WARMUP_MS / 1000.0) / dt_s)))
    os.environ["EDGAR_WC_DT"] = str(dt_s)
    os.environ["EDGAR_WC_OBJECTIVE"] = "E"

    X_train = F._build_data(cv.train, time_axis)
    X_test = F._build_data(cv.test, time_axis)

    if args.fit_iters > 0:
        gd = {"learning_rate": args.lr, "max_iter": args.fit_iters, "gradient_clip_norm": 5.0}
        l0 = float(_eval_loss(model2.model_jax, loss_fn, fit, X_train, apply_model))
        fit = _optimize(model2.model_jax, loss_fn, fit, X_train, gd, apply_model)
        l1 = float(_eval_loss(model2.model_jax, loss_fn, fit, X_train, apply_model))
        print(f"objective-E refine ({args.fit_iters} steps): NLL {l0:.5f} -> {l1:.5f}")

    train_nll = float(_eval_loss(model2.model_jax, loss_fn, fit, X_train, apply_model))
    test_nll = float(_eval_loss(model2.model_jax, loss_fn, fit, X_test, apply_model))
    p_scalar = {k: jnp.asarray(np.asarray(v).reshape(-1)[0]) for k, v in fit.items()}

    F._plot_fit_statespace(
        fit, p_scalar, cv, time_axis,
        title=(f"state-space (objective E) — {args.tag}"
               + (f" (+{args.fit_iters} GD)" if args.fit_iters else " (loaded, fixed noise)")
               + f"  train_NLL={train_nll:.4g}, test_NLL={test_nll:.4g}"),
        out_png=out,
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
