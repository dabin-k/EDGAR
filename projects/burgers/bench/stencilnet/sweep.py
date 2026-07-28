"""Noise x rollout-horizon sweep for STENCIL-NET on Burgers (Step 2 deliverable).

Trains one STENCIL-NET per (noise level, rollout_steps) pair and collects the
shared forecast-MSE benchmark curve. Training and scoring share one rollout
horizon (see runner.py), so each rollout_steps value is a *separately trained*
net, not a rescore of one net.

Run on the GPU box:
    python sweep.py --epochs 30000
    python sweep.py --epochs 30000 --levels 0.0 --rollout-steps 1 2 4 8
Writes results/stencilnet_sweep.json (one dict per run, each tagged with
noise_level + rollout_steps); per-run JSON/npz artifacts go to results/rollout<r>/.
"""

from __future__ import annotations

import argparse
import json
import os

import runner

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(levels, rollout_steps, epochs, seed, device):
    out_root = os.path.join(_HERE, "results")
    os.makedirs(out_root, exist_ok=True)
    all_results = []
    n = len(levels) * len(rollout_steps)
    print(f"STENCIL-NET sweep: {len(rollout_steps)} rollout horizon(s) x "
          f"{len(levels)} noise level(s) = {n} training run(s), epochs={epochs}", flush=True)
    for r in rollout_steps:
        out_dir = os.path.join(out_root, f"rollout{r}")
        os.makedirs(out_dir, exist_ok=True)
        for nl in levels:
            print(f"=== STENCIL-NET  rollout_steps={r}  noise={nl}  epochs={epochs} ===", flush=True)
            res = runner.run(noise_level=nl, epochs=epochs, seed=seed, device=device,
                             out_dir=out_dir, rollout_steps=r)
            print(json.dumps({k: v for k, v in res.items() if k != "loss_hist"}), flush=True)
            all_results.append(res)
    with open(os.path.join(out_root, "stencilnet_sweep.json"), "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"wrote {out_root}/stencilnet_sweep.json  ({len(all_results)} runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--rollout-steps", type=int, nargs="+", default=[1, 2, 4, 8],
                    dest="rollout_steps")
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu; default auto-detect")
    args = ap.parse_args()
    main(args.levels, args.rollout_steps, args.epochs, args.seed, args.device)
