"""Noise x sample x {strong, weak} sweep for SINDy on the shared unforced datasets.

Mirrors bench/stencilnet/sweep.py so the two reference methods are produced the same
way, with one structural difference: **SINDy's fit does not depend on the rollout
horizon.** It recovers a continuous operator, and the horizon only changes how far
that operator is rolled at score time, so each fit is scored at every horizon rather
than refitted per horizon. That makes this 3 x 4 x 2 = 24 fits (seconds each), not
72 — against STENCIL-NET's 36 separately-trained nets.

Run:
    python sweep.py
    python sweep.py --levels 0.0 --samples 1 --dry-run
Writes results/sindy_sweep.json (every fit tagged with noise_level, sample_idx, D,
variant, plus its per-horizon scores) and per-fit JSON to results/.
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


def _summarise(results, n_samples):
    """Per (variant, noise, horizon): coefficient recovery + discover/validate means.

    The two sample groups come from `load_data.discover_validate_samples`, the same
    function EDGAR's `load_data` splits on, so these aggregate exactly the samples
    EDGAR reports discover / validate losses over. As with STENCIL-NET, SINDy refits
    per sample, so its "validate" mean is a second set of in-sample scores, NOT a
    generalisation test — only EDGAR's validate split is one.
    """
    disc_idx, val_idx = ld.discover_validate_samples(n_samples)
    out = []
    keys = sorted({(r["method"], r["noise_level"]) for r in results})
    for method, nl in keys:
        cell = {
            r["sample_idx"]: r
            for r in results
            if r["method"] == method and r["noise_level"] == nl
        }
        horizons = sorted(
            {h["rollout_steps"] for r in cell.values() for h in r["horizons"]}
        )

        def mean_over(idxs, get):
            vals = [get(cell[i]) for i in idxs if i in cell]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        row = {
            "method": method,
            "noise_level": nl,
            "n_samples": len(cell),
            "discover_samples": disc_idx.tolist(),
            "validate_samples": val_idx.tolist(),
            # |recovered - true| / true, averaged; truth for u_xx is the sample's own D
            "abs_rel_err_u_xx_mean": mean_over(
                range(n_samples), lambda r: abs(r["rel_err_u_xx"])
            ),
            "abs_rel_err_uu_x_mean": mean_over(
                range(n_samples), lambda r: abs(r["rel_err_uu_x"])
            ),
            "n_terms_mean": mean_over(range(n_samples), lambda r: r["n_terms"]),
            "by_horizon": [],
        }
        for h in horizons:

            def at(r, field, _h=h):
                for hh in r["horizons"]:
                    if hh["rollout_steps"] == _h:
                        return hh[field]
                return None

            row["by_horizon"].append(
                {
                    "rollout_steps": h,
                    "n_unstable": sum(
                        1 for r in cell.values() if at(r, "stable") is False
                    ),
                    "mse_test_discover_mean": mean_over(
                        disc_idx, lambda r: at(r, "forecast_mse_test")
                    ),
                    "mse_test_validate_mean": mean_over(
                        val_idx, lambda r: at(r, "forecast_mse_test")
                    ),
                    "skill_test_discover_mean": mean_over(
                        disc_idx, lambda r: at(r, "skill_test")
                    ),
                    "skill_test_validate_mean": mean_over(
                        val_idx, lambda r: at(r, "skill_test")
                    ),
                }
            )
        out.append(row)
    return out


def main(
    data_dir,
    ic_seed,
    levels,
    samples,
    rollout_steps,
    threshold,
    variants,
    block_len,
    dry_run,
):
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    n_samples = int(
        ld.load_dataset(_dataset_path(data_dir, ic_seed, levels[0]))["D"].shape[0]
    )
    samples = list(range(n_samples)) if samples is None else list(samples)
    n = len(levels) * len(samples) * len(variants)
    print(
        f"SINDy sweep: {len(levels)} noise level(s) x {len(samples)} sample(s) x "
        f"{len(variants)} variant(s) = {n} fit(s), each scored at "
        f"h={list(rollout_steps)}; ic_seed={ic_seed}, threshold={threshold}",
        flush=True,
    )

    if dry_run:
        for weak in variants:
            for nl in levels:
                for j in samples:
                    print(
                        f"  {'weak' if weak else 'strong':>6} noise={nl} sample={j}  "
                        f"{_dataset_path(data_dir, ic_seed, nl)}"
                    )
        return []

    all_results = []
    for weak in variants:
        for nl in levels:
            data_path = _dataset_path(data_dir, ic_seed, nl)
            for j in samples:
                res = runner.run(
                    data_path=data_path,
                    sample_idx=j,
                    weak=weak,
                    threshold=threshold,
                    rollout_steps=rollout_steps,
                    block_len=block_len,
                    out_dir=out_dir,
                )
                h1 = res["horizons"][0]
                print(
                    f"{res['method']:>12} nl={nl:<5} s={j} D={res['D']:.3f} | "
                    f"u_xx {res['coef_u_xx']:8.5f} (true {res['D']:.3f}) "
                    f"uu_x {res['coef_uu_x']:7.3f} | n_terms {res['n_terms']:2d} | "
                    f"h={h1['rollout_steps']} mse_test {h1['forecast_mse_test']:.3e} "
                    f"skill {h1['skill_test']:.4f}"
                    f"{'' if h1['stable'] else '  UNSTABLE'}",
                    flush=True,
                )
                all_results.append(res)

    payload = {"runs": all_results, "summary": _summarise(all_results, n_samples)}
    with open(os.path.join(out_dir, "sindy_sweep.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_dir}/sindy_sweep.json  ({len(all_results)} fits)")
    return all_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=runner._DEFAULT_DATA_DIR, dest="data_dir")
    ap.add_argument("--ic-seed", type=int, default=0, dest="ic_seed")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    ap.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="sample indices to fit; default every sample in the file",
    )
    ap.add_argument(
        "--rollout-steps", type=int, nargs="+", default=[1, 2, 4], dest="rollout_steps"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="STLSQ threshold; must sit below the smallest true D (0.005)",
    )
    ap.add_argument("--block-len", type=int, default=200, dest="block_len")
    ap.add_argument("--strong-only", action="store_true", dest="strong_only")
    ap.add_argument("--weak-only", action="store_true", dest="weak_only")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()
    variants = [False, True]
    if args.strong_only:
        variants = [False]
    elif args.weak_only:
        variants = [True]
    main(
        args.data_dir,
        args.ic_seed,
        args.levels,
        args.samples,
        args.rollout_steps,
        args.threshold,
        variants,
        args.block_len,
        args.dry_run,
    )
