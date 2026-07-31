"""Noise x rollout-horizon x sample sweep for STENCIL-NET on Burgers (Step 2 deliverable).

Trains one STENCIL-NET per (noise level, rollout_steps, sample) triple over the
shared unforced benchmark datasets and collects the forecast-MSE curve. Two
reasons this is one net per cell rather than one per file:

  * training and scoring share a rollout horizon (see runner.py), so each
    rollout_steps value is a *separately trained* net, not a rescore of one net;
  * the samples within a file differ only in viscosity D, and MLPConv takes no D
    input, so one net cannot serve several viscosities.

Run on the GPU box:
    python sweep.py --epochs 30000 --device cuda
    python sweep.py --dry-run
Writes results/stencilnet_sweep_ic<seed>.json (every run tagged with noise_level,
rollout_steps, sample_idx, D, plus per-(noise, rollout) discover/validate means);
per-run JSON/npz artifacts go to results/rollout<r>/, weights to --weights-dir.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import runner
from runner import ld  # data_loader.load_data, already on sys.path via runner

_HERE = os.path.dirname(os.path.abspath(__file__))


def _dataset_path(data_dir, ic_seed, noise_level):
    return os.path.join(data_dir, f"ic_seed_{ic_seed}_nl_{noise_level}.npz")


def _existing(out_root, rollout, ic_seed, noise_level, sample_idx, epochs):
    """A completed run of at least `epochs` for this cell, or None.

    48 runs is long enough that an interrupted sweep should resume rather than
    start over, but a shorter smoke run must not be mistaken for a full one.
    """
    path = os.path.join(
        out_root,
        f"rollout{rollout}",
        f"stencilnet_ic{ic_seed}_nl{noise_level}_s{sample_idx}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        res = json.load(fh)
    if res.get("epochs", 0) < epochs or res.get("forecast_mse_test") is None:
        return None
    return res


def _summarise(results, n_samples):
    """Mean test scores over the discover and validate samples, per (noise, rollout).

    The two groups come from `load_data.discover_validate_samples`, the same
    function EDGAR's `load_data` splits on, so these means aggregate exactly the
    samples EDGAR reports its discover / validate losses over.

    `skill_test` (1 - MSE/persistence) is the comparable number across cells: the
    persistence floor varies ~50x between the low-D and high-D samples, so raw MSE
    means are dominated by whichever sample decays slowest.
    """
    disc_idx, val_idx = ld.discover_validate_samples(n_samples)
    out = []
    keys = sorted({(r["noise_level"], r["rollout_steps"]) for r in results})
    for nl, roll in keys:
        cell = {
            r["sample_idx"]: r
            for r in results
            if r["noise_level"] == nl and r["rollout_steps"] == roll
        }

        def mean_over(idxs, field):
            vals = [cell[i][field] for i in idxs if i in cell]
            return float(np.mean(vals)) if vals else None

        out.append(
            {
                "noise_level": nl,
                "rollout_steps": roll,
                "n_samples": len(cell),
                "discover_samples": disc_idx.tolist(),
                "validate_samples": val_idx.tolist(),
                "mse_test_discover_mean": mean_over(disc_idx, "forecast_mse_test"),
                "mse_test_validate_mean": mean_over(val_idx, "forecast_mse_test"),
                "persistence_test_discover_mean": mean_over(
                    disc_idx, "persistence_mse_test"
                ),
                "persistence_test_validate_mean": mean_over(
                    val_idx, "persistence_mse_test"
                ),
                "skill_test_discover_mean": mean_over(disc_idx, "skill_test"),
                "skill_test_validate_mean": mean_over(val_idx, "skill_test"),
            }
        )
    return out


def main(
    data_dir,
    ic_seed,
    levels,
    rollout_steps,
    samples,
    epochs,
    seed,
    device,
    weights_dir,
    overwrite,
    dry_run,
):
    out_root = os.path.join(_HERE, "results")
    os.makedirs(out_root, exist_ok=True)
    # How many samples the files hold is a property of the data, not a flag.
    n_samples = int(
        ld.load_dataset(_dataset_path(data_dir, ic_seed, levels[0]))["D"].shape[0]
    )
    samples = list(range(n_samples)) if samples is None else list(samples)
    n = len(levels) * len(rollout_steps) * len(samples)
    print(
        f"STENCIL-NET sweep: {len(rollout_steps)} rollout horizon(s) x {len(levels)} "
        f"noise level(s) x {len(samples)} sample(s) = {n} training run(s), "
        f"ic_seed={ic_seed}, epochs={epochs}",
        flush=True,
    )

    if dry_run:
        for r in rollout_steps:
            for nl in levels:
                for j in samples:
                    done = _existing(out_root, r, ic_seed, nl, j, epochs) is not None
                    print(
                        f"  rollout={r} noise={nl} sample={j} "
                        f"{'SKIP (done)' if done and not overwrite else 'RUN'}  "
                        f"{_dataset_path(data_dir, ic_seed, nl)}"
                    )
        return []

    all_results = []
    for r in rollout_steps:
        out_dir = os.path.join(out_root, f"rollout{r}")
        os.makedirs(out_dir, exist_ok=True)
        for nl in levels:
            data_path = _dataset_path(data_dir, ic_seed, nl)
            for j in samples:
                cached = (
                    None
                    if overwrite
                    else _existing(out_root, r, ic_seed, nl, j, epochs)
                )
                if cached is not None:
                    print(
                        f"=== skip (already done) rollout={r} noise={nl} sample={j} ===",
                        flush=True,
                    )
                    all_results.append(cached)
                    continue
                print(
                    f"=== STENCIL-NET  rollout_steps={r}  noise={nl}  sample={j}  "
                    f"epochs={epochs} ===",
                    flush=True,
                )
                res = runner.run(
                    data_path=data_path,
                    sample_idx=j,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                    out_dir=out_dir,
                    rollout_steps=r,
                    weights_dir=weights_dir,
                )
                print(
                    json.dumps({k: v for k, v in res.items() if k != "loss_hist"}),
                    flush=True,
                )
                all_results.append(res)

    payload = {"runs": all_results, "summary": _summarise(all_results, n_samples)}
    sweep_path = os.path.join(out_root, f"stencilnet_sweep_ic{ic_seed}.json")
    with open(sweep_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {sweep_path}  ({len(all_results)} runs)")
    return all_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir", default="/home/dabin/data/burgers_simulated", dest="data_dir"
    )
    ap.add_argument("--ic-seed", type=int, default=0, dest="ic_seed")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    ap.add_argument(
        "--rollout-steps", type=int, nargs="+", default=[1, 2, 4], dest="rollout_steps"
    )
    ap.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="sample indices to train; default every sample in the file",
    )
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu; default auto-detect")
    ap.add_argument("--weights-dir", default=None, dest="weights_dir")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="retrain cells that already have a completed run",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="print the run matrix and exit",
    )
    args = ap.parse_args()
    main(
        args.data_dir,
        args.ic_seed,
        args.levels,
        args.rollout_steps,
        args.samples,
        args.epochs,
        args.seed,
        args.device,
        args.weights_dir,
        args.overwrite,
        args.dry_run,
    )
