#!/usr/bin/env python
"""Plot a few real E/I population-rate trajectories with their stimulus channels.

Phase 0 eyeball check (plan §17 step 7): confirms baseline ~ 1, pulse onsets line up with
``ipi_ms``, and paired-pulse timing is right, before any model is trained. For one sample of
each of a chosen set of experiment types, overlays E and I traces vs time and shades the
``u_E``/``u_I`` stimulus boxcars.

Usage:
  python plot_neural_samples.py [--animal M150605_ICTP1] [--types single_E,single_I,paired_EI,paired_IE]
                                [--fold 0] [--xlim -0.1,0.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]           # projects/wilson_cowan/
sys.path.insert(0, str(WC / "data_loader"))

from neural_data import load_neural_dataset          # noqa: E402

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402


def _pick(ds, animal: str | None, etype: str, fold: int) -> int | None:
    """First sample index matching (animal, type, fold); animal=None = any."""
    for i, m in enumerate(ds.meta):
        if m["experiment_type"] != etype or m["fold"] != fold:
            continue
        if animal is not None and m["animal_id"] != animal:
            continue
        return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, help="animal_id filter (default: any)")
    ap.add_argument("--types", default="single_E,single_I,paired_EI,paired_IE")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--xlim", default=None, help="e.g. -0.1,0.5 (seconds)")
    ap.add_argument("--outdir", default=str(WC / "scripts"))
    args = ap.parse_args()

    ds = load_neural_dataset()
    time = np.asarray(ds.time)
    y = np.asarray(ds.target_y)
    u = np.asarray(ds.u)

    types = [t for t in args.types.split(",") if t]
    xlim = tuple(float(x) for x in args.xlim.split(",")) if args.xlim else None

    fig, axes = plt.subplots(len(types), 1, figsize=(9, 2.4 * len(types)), sharex=True)
    if len(types) == 1:
        axes = [axes]

    for ax, etype in zip(axes, types):
        idx = _pick(ds, args.animal, etype, args.fold)
        if idx is None:
            ax.set_title(f"{etype}: no matching sample")
            continue
        m = ds.meta[idx]
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        ax.plot(time, y[idx, :, 0], color="tab:red", lw=1.0, label="E")
        ax.plot(time, y[idx, :, 1], color="tab:blue", lw=1.0, label="I")

        ymax = float(np.nanmax(y[idx])) * 1.05
        for ch, col in ((0, "tab:red"), (1, "tab:blue")):
            on = u[idx, :, ch] > 0
            if on.any():
                ax.fill_between(time, 0, ymax, where=on, color=col, alpha=0.12, step="mid")

        ax.set_title(
            f"{m['animal_id']} · {etype} · ipi={m['ipi_ms']}ms · dur={m['dur_ms']}ms "
            f"· fold={m['fold']} · n={m['n_trials']}", fontsize=9)
        ax.set_ylabel("norm. rate")
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("time (s, rel. first-pulse onset)")
    fig.tight_layout()
    tag = args.animal or "any"
    out = Path(args.outdir) / f"neural_samples_{tag}_fold{args.fold}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
