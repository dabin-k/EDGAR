"""Noise sweep for STENCIL-NET on Burgers (Step 2 deliverable).

Trains one STENCIL-NET per noise level and collects forecast-MSE-vs-noise, the
shared benchmark curve. Clean levels use the forcing-aware RK3 loss; noisy levels
use the TVD-RK3 + latent-noise path (noise-as-latent), exactly as the two
reference notebooks split.

Run on the GPU box:
    python sweep.py --epochs 30000
    python sweep.py --epochs 30000 --levels 0.0 0.05 0.1 0.3
Writes results/stencilnet_sweep.json (list of per-level result dicts).
"""

from __future__ import annotations

import argparse
import json
import os

import runner

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(levels, epochs, seed, device):
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    for nl in levels:
        print(f"=== STENCIL-NET  noise={nl}  epochs={epochs} ===", flush=True)
        r = runner.run(noise_level=nl, epochs=epochs, seed=seed, device=device, out_dir=out_dir)
        summary = {k: v for k, v in r.items() if k != "loss_hist"}
        print(json.dumps(summary), flush=True)
        all_results.append(r)
    with open(os.path.join(out_dir, "stencilnet_sweep.json"), "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"wrote {out_dir}/stencilnet_sweep.json  ({len(all_results)} levels)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu; default auto-detect")
    args = ap.parse_args()
    main(args.levels, args.epochs, args.seed, args.device)
