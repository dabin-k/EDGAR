#!/usr/bin/env python
"""Zoomed single-trial overlay: real data vs model one-step (TF) vs persistence y[t-1].

Diagnostic for the "the WCS one-step fit looks too good" observation (journal 2026-08-25):
the teacher-forced one-step prediction is fed the true y[t-1] at every step, so on smoothed
(autocorrelated) rates it can track every wriggle just by re-emitting the previous value.
This plot makes that visible — if the model is ~persistence, the red (model) curve sits on
top of the green (persistence = data shifted one bin right) and both LAG the black data by
~one bin. Full trace on top; a small x/y zoom window below.

Reads a saved ``mixed_params_{tag}.npz`` (from ``fit_mixed_init.py``) and rebuilds the same
h{width} smoothing-sweep data it was fit on (via ``fit_smoothing_sweep``).

Usage:
    python plot_onestep_vs_persistence.py                          # defaults: tag B_nsteps_10000, cond 0, E, 150-260 ms
    python plot_onestep_vs_persistence.py --tag A_nsteps_10000 --chan I
    python plot_onestep_vs_persistence.py --cond 17 --zoom 0 120   # post-stim window of another condition
    python plot_onestep_vs_persistence.py --out /tmp/foo.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]                             # repo root
for _p in (REPO, WC / "data_loader", WC / "seed_programs", WC / "scripts"):
    sys.path.insert(0, str(_p))

import jax.numpy as jnp                          # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

import fit_smoothing_sweep as F                  # noqa: E402
from neural_data import build_cv_samples         # noqa: E402
from load_data import apply_model                # noqa: E402
import model2                                    # noqa: E402


def _load_params(tag: str) -> dict:
    d = np.load(F.PARAMS_DIR / f"mixed_params_{tag}.npz", allow_pickle=True)
    return {k: float(v) for k, v in zip(d["param_keys"], d["param_values"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="B_nsteps_10000",
                    help="mixed_params_{tag}.npz to load (default: B_nsteps_10000)")
    ap.add_argument("--width", type=int, default=5,
                    help="smoothing width h{width} the params were fit on (default: 5)")
    ap.add_argument("--cond", type=int, default=0, help="condition/trial index to plot (default: 0)")
    ap.add_argument("--chan", choices=["E", "I"], default="E", help="channel to plot (default: E)")
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("LO_MS", "HI_MS"),
                    default=[150.0, 260.0], help="zoom x-window in ms (default: 150 260)")
    ap.add_argument("--out", default=None,
                    help="output png (default: <params>/onestep_vs_persistence_{tag}_c{cond}_{chan}.png)")
    args = ap.parse_args()

    chan = 0 if args.chan == "E" else 1
    zoom_lo, zoom_hi = args.zoom
    out = Path(args.out) if args.out else (
        F.PARAMS_DIR / f"onestep_vs_persistence_{args.tag}_c{args.cond}_{args.chan}.png")

    params = _load_params(args.tag)
    fit = {k: jnp.asarray(params[k])[None] for k in F.PARAM_KEYS}

    files = F._discover_widths()
    if args.width not in files:
        raise SystemExit(f"width {args.width} not found; have {sorted(files)}")
    cv = build_cv_samples(F._canonical_path(files[args.width], args.width),
                          cv_type="k_fold", held_out_fold=0, chop=F.CHOP)
    time_axis = np.asarray(cv.time)
    dt_s = float(time_axis[1] - time_axis[0])
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(int(round((F.WARMUP_MS / 1000.0) / dt_s)))
    os.environ["EDGAR_WC_DT"] = str(dt_s)
    os.environ["EDGAR_WC_OBJECTIVE"] = "A"       # one-step is objective-independent; A builds pred_y_1step

    C = int(cv.train.n)
    if not 0 <= args.cond < C:
        raise SystemExit(f"cond {args.cond} out of range; this session has {C} conditions (0..{C-1})")

    X = F._build_data(cv.train, time_axis)
    out_model = apply_model(model2.model_jax, X, fit)
    pred_1step = np.asarray(out_model["pred_y_1step"])[0]     # (C, T-1, 2)
    target = np.asarray(cv.train.target_y)                    # (C, T, 2)

    t_ms = time_axis * 1000.0
    real = target[args.cond, :, chan]                         # (T,)
    model_1s = pred_1step[args.cond, :, chan]                 # (T-1,) prediction for t = 1..T-1
    persist = real[:-1]                                       # (T-1,) predict y[t] = y[t-1]
    t_pred = t_ms[1:]                                         # times the predictions correspond to

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 8))

    def draw(ax):
        ax.plot(t_ms, real, color="k", lw=0.8, alpha=0.5, zorder=1)
        ax.scatter(t_ms, real, color="k", s=14, alpha=0.7, label="real  y[t]", zorder=2)
        ax.plot(t_pred, model_1s, color="tab:red", lw=1.4, marker="o", ms=4,
                label="model one-step (TF)", zorder=4)
        ax.plot(t_pred, persist, color="tab:green", lw=1.2, ls="--", marker="s", ms=3,
                label="persistence  y[t-1]", zorder=3)

    tau = params["tau_" + args.chan]
    draw(ax_full)
    ax_full.set_title(f"{args.tag}  cond {args.cond}  {args.chan}-rate  (tau_{args.chan}={tau:.1f})  "
                      f"— full trace", fontsize=11)
    ax_full.set_ylabel(f"{args.chan} rate")
    ax_full.legend(fontsize=9, loc="upper left")
    ax_full.axvspan(zoom_lo, zoom_hi, color="tab:orange", alpha=0.12, zorder=0)

    draw(ax_zoom)
    m = (t_ms >= zoom_lo) & (t_ms <= zoom_hi)
    if not m.any():
        raise SystemExit(f"zoom window {zoom_lo}-{zoom_hi} ms is outside the data range "
                         f"[{t_ms[0]:.0f}, {t_ms[-1]:.0f}] ms")
    yv = real[m]
    pad = 0.08 * (yv.max() - yv.min() + 1e-6)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_ylim(yv.min() - pad, yv.max() + pad)
    ax_zoom.set_title(f"zoom {zoom_lo:.0f}-{zoom_hi:.0f} ms  — if model ≈ persistence, red sits on "
                      f"green, both lag black by ~1 bin (dt={dt_s * 1000:.1f} ms)", fontsize=10)
    ax_zoom.set_xlabel("time (ms)")
    ax_zoom.set_ylabel(f"{args.chan} rate")
    ax_zoom.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[saved] {out}")

    # Quick numeric check on the zoom window: model's departure from persistence vs a data step.
    mp = (t_pred >= zoom_lo) & (t_pred <= zoom_hi)
    d_mp = float(np.mean(np.abs(model_1s[mp] - persist[mp])))
    d_data = float(np.mean(np.abs(np.diff(real[m]))))
    print(f"zoom: mean|model-persistence|={d_mp:.4f}   mean|y[t]-y[t-1]|(data step)={d_data:.4f}   "
          f"ratio={d_mp / d_data:.3f}  (small ratio => model is essentially persistence)")


if __name__ == "__main__":
    main()
