#!/usr/bin/env python
"""Plot how each parameter (and the loss) moves from the param_est2 init to the GD-optimised fit,
for every objective (A, B, E, F), using the npz files saved by ``check_objective_setup.py``.

Two kinds of panel, all in one figure (``fitted_params/param_change.png``):

  * one panel PER PARAMETER — a 2-tick scatter ``['init', 'optimised']`` with the parameter value
    on the y-axis; each objective draws a line from its init value to its optimised value, coloured
    by objective. Dynamics params (16) get a line from all four objectives; the objective-E EKF
    ``kf_*`` params only exist for E, so those panels show a single (green) line.
  * one panel PER OBJECTIVE for the LOSS — the training loss from init to optimised. Losses are on
    different scales across objectives (MSE vs NLL), so each objective gets its own panel rather
    than being overlaid; the held-out test loss is marked at 'optimised' for reference.

Init vs optimised are read from each ``check_fit_{obj}.npz`` (``init_values`` / ``param_values``),
so the init shown is exactly the one that seeded that fit.

Usage:
    python plot_param_change.py                 # A,B,E,F
    python plot_param_change.py --objectives A,B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WC / "seed_programs"))

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

PARAMS_DIR = WC / "scripts" / "fitted_params"
OBJECTIVES = ["A", "B", "E", "F"]
COLORS = {"A": "tab:blue", "B": "tab:orange", "E": "tab:green", "F": "tab:red"}


def _load(obj: str) -> dict:
    path = PARAMS_DIR / f"check_fit_{obj}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} — run check_objective_setup.py first")
    d = np.load(path, allow_pickle=True)
    keys = [str(k) for k in d["param_keys"]]
    return {
        "keys": keys,
        "init": dict(zip(keys, d["init_values"].astype(float))),
        "opt": dict(zip(keys, d["param_values"].astype(float))),
        "init_train": float(d["init_train_loss"]),
        "train": float(d["train_loss"]),
        "test": float(d["test_loss"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default=",".join(OBJECTIVES),
                    help="comma-separated subset of A,B,E,F (default: all)")
    args = ap.parse_args()
    run_objs = [o.strip().upper() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in run_objs if o not in OBJECTIVES]
    if bad:
        raise SystemExit(f"unknown objective(s) {bad}; valid: {OBJECTIVES}")

    data = {o: _load(o) for o in run_objs}

    # Ordered union of all parameter keys across the objectives (dynamics first, kf_* last — the
    # order model2_kalman declares them). Preserves first-seen order.
    param_keys: list[str] = []
    for o in run_objs:
        for k in data[o]["keys"]:
            if k not in param_keys:
                param_keys.append(k)

    n_panels = len(param_keys) + len(run_objs)     # params + one loss panel per objective
    ncol = 4
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    x = [0, 1]
    # ── one panel per parameter ──
    for i, k in enumerate(param_keys):
        ax = axes[i]
        for o in run_objs:
            if k not in data[o]["init"]:
                continue
            ax.plot(x, [data[o]["init"][k], data[o]["opt"][k]], "o-",
                    color=COLORS[o], lw=1.4, ms=5, label=o)
        ax.set_xticks(x)
        ax.set_xticklabels(["init", "optimised"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(k, fontsize=9)
        ax.grid(alpha=0.3)

    # ── one loss panel per objective (separate scales) ──
    for j, o in enumerate(run_objs):
        ax = axes[len(param_keys) + j]
        d = data[o]
        ax.plot(x, [d["init_train"], d["train"]], "o-", color=COLORS[o], lw=1.6, ms=6,
                label="train")
        ax.plot(1, d["test"], "*", color=COLORS[o], ms=12, mec="k", mew=0.5, label="test (opt)")
        ax.set_xticks(x)
        ax.set_xticklabels(["init", "optimised"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"obj {o} loss", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # One shared objective legend (colour = loss function) on the first param panel.
    handles = [plt.Line2D([0], [0], color=COLORS[o], marker="o", lw=1.4, label=f"obj {o}")
               for o in run_objs]
    axes[0].legend(handles=handles, fontsize=7, loc="best")

    fig.suptitle("Parameter & loss change: param_est2 init -> GD-optimised (colour = loss function)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = PARAMS_DIR / "param_change.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
