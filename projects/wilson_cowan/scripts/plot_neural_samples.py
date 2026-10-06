#!/usr/bin/env python
"""Plot real single-trial E/I responses with their stimulus channels, as EDGAR sees them.

Eyeball check of ``neural_data.build_cv_samples``: for one condition of each chosen experiment
type, overlays a few individual train trials (thin) and the mean of all that condition's train
trials (bold) for E and I, and shades the ``u_E``/``u_I`` stimulus bins. Confirms the train
baseline ~ 1, pulse onsets at 0 and ``onset_ipi_ms``, and the single-trial noise level.

Usage:
  python plot_neural_samples.py [--animal M150605_ICTP1] [--types single_E,single_I,paired_EI,paired_IE]
                                [--fold 0] [--bin-ms 1] [--xlim -0.1,0.4]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]           # projects/wilson_cowan/
sys.path.insert(0, str(WC / "data_loader"))

from neural_data import DEFAULT_RESULTS_DIR, build_cv_samples, verify_cv_samples   # noqa: E402

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

N_TRIALS_SHOWN = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="M150605_ICTP1")
    ap.add_argument("--types", default="single_E,single_I,paired_EI,paired_IE")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bin-ms", type=float, default=1.0)
    ap.add_argument("--xlim", default="-0.1,0.4", help="seconds")
    ap.add_argument("--outdir", default=str(WC / "scripts"))
    args = ap.parse_args()

    path = os.path.join(DEFAULT_RESULTS_DIR, f"trial_counts_b30_{args.animal}_s1.npz")
    cv = build_cv_samples(path, "k_fold", held_out_fold=args.fold, chop=(-0.5, 1.5),
                          bin_ms=args.bin_ms)
    verify_cv_samples(cv)
    sp, time = cv.train, np.asarray(cv.time)
    types = [t for t in args.types.split(",") if t]

    fig, axes = plt.subplots(len(types), 1, figsize=(9, 2.4 * len(types)), sharex=True,
                             squeeze=False)
    for ax, etype in zip(axes[:, 0], types):
        rows = [i for i, m in enumerate(sp.meta) if m["experiment_type"] == etype]
        if not rows:
            ax.set_title(f"{etype}: no trials")
            continue
        k = sp.cond_idx[rows[0]]                     # first condition of this type
        same = np.flatnonzero(sp.cond_idx == k)
        m = sp.meta[same[0]]
        mean = sp.target_y[same].mean(axis=0)        # (T, 2)

        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        for ch, col in ((0, "tab:red"), (1, "tab:blue")):
            for i in same[:N_TRIALS_SHOWN]:
                ax.plot(time, sp.target_y[i, :, ch], color=col, lw=0.4, alpha=0.3)
            ax.plot(time, mean[:, ch], color=col, lw=1.3, label=f"{'EI'[ch]} mean")
        ymax = float(mean.max()) * 1.3
        for ch, col in ((0, "tab:red"), (1, "tab:blue")):
            on = sp.stim[same[0], :, ch] > 0
            if on.any():
                ax.fill_between(time, 0, ymax, where=on, color=col, alpha=0.15, step="mid")
        ax.set_ylim(0, ymax)
        ax.set_title(
            f"{m['animal_id']} · {etype} · onset ipi={m['onset_ipi_ms']:g} ms · "
            f"dur={m['dur1_ms']:g}/{m['dur2_ms']:g} ms · fold≠{args.fold} · n={same.size} trials",
            fontsize=9)
        ax.set_ylabel("rate / train baseline")
        ax.set_xlim(*(float(x) for x in args.xlim.split(",")))
        ax.legend(fontsize=7, loc="upper right")

    axes[-1, 0].set_xlabel("time (s, rel. first-pulse onset)")
    fig.tight_layout()
    out = Path(args.outdir) / f"neural_samples_{args.animal}_fold{args.fold}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
