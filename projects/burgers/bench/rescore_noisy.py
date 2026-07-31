"""Re-score the trained SINDy and STENCIL-NET fits against the OBSERVED field.

Both methods were originally scored with `teacher_forced_forecast(rhs, u_clean, ...)`,
which injects the clean field twice: once as the restart state seeding every rollout,
and once as the target. EDGAR gets neither — `load_data` feeds it `u_noisy` for both.
So the published numbers were not comparable across methods, and the noise axis could
not move them except through training quality.

This re-scores on `u_noisy` for both roles. It is a pure re-score, NOT a retrain:
neither method ever saw `u_clean` during fitting (STENCIL-NET's loss residual is
`target - (rollout + latent noise)` with `target = u_noisy`, see
stencilnet_src/timestepping.py; SINDy fits `u_obs` via fit_pdefind), so the 12.7 h of
GPU training and the 24 SINDy fits stand. SINDy's RHS is rebuilt from the saved
coefficients, STENCIL-NET's from the 36 saved checkpoints — forward passes only.

Writes results/rescore_noisy.json with both scorings side by side per cell.

Run:
    python rescore_noisy.py
    python rescore_noisy.py --levels 0.0 --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "stencilnet", "stencilnet_src"))
sys.path.insert(0, os.path.join(_HERE, "..", "data_loader"))
sys.path.insert(0, os.path.join(_HERE, "sindy"))

from network import MLPConv  # noqa: E402  (vendored)
import load_data as ld  # noqa: E402
from runner import make_rhs, MSE_CAP  # noqa: E402  (bench/sindy/runner.py)

_DEFAULT_DATA_DIR = "/home/dabin/data/burgers_simulated"
_DEFAULT_WEIGHTS_DIR = os.path.join(_DEFAULT_DATA_DIR, "stencilnet_weights")


def _score(rhs, field, dtc, train_cols, test_cols, rollout_steps):
    """Forecast MSE + persistence floor, seeded from and scored against `field`."""
    T = field.shape[1]
    preds, targets = ld.teacher_forced_forecast(rhs, field, dtc, rollout_steps)
    stable = bool(np.all(np.isfinite(preds)))
    preds = np.nan_to_num(preds, nan=1e30, posinf=1e30, neginf=-1e30)
    p_preds, p_targets = ld.teacher_forced_forecast(
        lambda state, t_arr: np.zeros_like(state), field, dtc, rollout_steps
    )
    tr, te = ld.split_start_masks(train_cols, test_cols, T, rollout_steps)
    mse_te = min(ld.forecast_mse(preds[te], targets[te]), MSE_CAP)
    pers_te = ld.forecast_mse(p_preds[te], p_targets[te])
    return {
        "rollout_steps": int(rollout_steps),
        "stable": stable,
        "forecast_mse_train": min(ld.forecast_mse(preds[tr], targets[tr]), MSE_CAP),
        "forecast_mse_test": mse_te,
        "persistence_mse_train": ld.forecast_mse(p_preds[tr], p_targets[tr]),
        "persistence_mse_test": pers_te,
        "skill_test": 1.0 - mse_te / pers_te,
    }


def _dataset(data_dir, ic_seed, noise_level, _cache={}):
    key = (ic_seed, noise_level)
    if key not in _cache:
        path = os.path.join(data_dir, f"ic_seed_{ic_seed}_nl_{noise_level}.npz")
        _cache[key] = ld.load_dataset(path)
    return _cache[key]


def rescore_sindy(data_dir, levels, block_len, dry_run):
    """Rebuild each saved SINDy operator from its coefficients and re-score it."""
    out = []
    pattern = os.path.join(_HERE, "sindy", "results", "sindy_*_nl*_s*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            old = json.load(fh)
        if old["noise_level"] not in levels:
            continue
        if dry_run:
            print(f"  {old['method']:>12} nl={old['noise_level']:<5} s={old['sample_idx']}")
            out.append(old)
            continue

        ds = _dataset(data_dir, old["ic_seed"], old["noise_level"])
        j = old["sample_idx"]
        u_obs = ds["u_noisy"][j]
        dx = float(ds["x_coarse"][1] - ds["x_coarse"][0])
        train_cols, test_cols = ld.block_split(u_obs.shape[1], block_len=block_len)

        N = make_rhs(old["feature_names"], old["coefficients"], dx)
        rhs = lambda state, t_arr: N(state)  # noqa: E731  (unforced)
        horizons = [
            _score(rhs, u_obs, ds["dtc"], train_cols, test_cols, h["rollout_steps"])
            for h in old["horizons"]
        ]
        out.append(_pair(old, horizons))
        _report(out[-1])
    return out


def rescore_stencilnet(data_dir, levels, weights_dir, device, dry_run):
    """Rebuild each trained net from its checkpoint and re-score it (forward only)."""
    out = []
    for path in sorted(glob.glob(os.path.join(weights_dir, "ic*_nl*_r*_s*.pt"))):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["noise_level"] not in levels:
            continue
        r, j = ckpt["rollout_steps"], ckpt["sample_idx"]
        res_path = os.path.join(
            _HERE, "stencilnet", "results", f"rollout{r}",
            f"stencilnet_nl{ckpt['noise_level']}_s{j}.json",
        )
        with open(res_path) as fh:
            old = json.load(fh)
        if dry_run:
            print(f"  stencilnet nl={ckpt['noise_level']:<5} r={r} s={j}  {os.path.basename(path)}")
            out.append(old)
            continue

        ds = _dataset(data_dir, ckpt["ic_seed"], ckpt["noise_level"])
        u_obs = ds["u_noisy"][j]
        train_cols, test_cols = ld.block_split(u_obs.shape[1], block_len=ckpt["block_len"])

        # MLPConv prints its layer sizes on construction; 36 nets x 4 lines of it
        # would bury the actual results.
        with contextlib.redirect_stdout(io.StringIO()):
            net = MLPConv(
                ckpt["sizes"], seed=ckpt["seed"], fs=ckpt["fs"], activation=nn.ELU()
            ).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.noise = None  # inference: no latent-noise field
        net.eval()

        def rhs(state, t_arr, _net=net):
            with torch.no_grad():
                tens = torch.tensor(state, dtype=torch.float, device=device)
                return _net(tens).cpu().numpy()

        horizons = [_score(rhs, u_obs, ds["dtc"], train_cols, test_cols, r)]
        out.append(_pair(old, horizons))
        _report(out[-1])
    return out


def _pair(old, horizons):
    """One cell: its identity, the new noisy-scored horizons, and the old clean ones.

    The two horizon lists are named for the field they were scored on rather than
    carrying a separate `scored_on` tag, so neither can be read under the other's
    convention. At nl=0.0 they are identical by construction (u_noisy IS u_clean
    there), which is the control that the re-score changed nothing else.
    """
    old_h = old["horizons"] if "horizons" in old else [
        {
            "rollout_steps": old["rollout_steps"],
            "forecast_mse_test": old["forecast_mse_test"],
            "persistence_mse_test": old["persistence_mse_test"],
            "skill_test": old["skill_test"],
        }
    ]
    return {
        "method": old["method"],
        "ic_seed": old["ic_seed"],
        "sample_idx": old["sample_idx"],
        "D": old["D"],
        "noise_level": old["noise_level"],
        "horizons_noisy_scored": horizons,
        "horizons_clean_scored": old_h,
    }


def _report(cell):
    for h, old in zip(cell["horizons_noisy_scored"], _aligned(cell)):
        flag = "" if h["stable"] else "  UNSTABLE"
        print(
            f"{cell['method']:>13} nl={cell['noise_level']:<5} s={cell['sample_idx']} "
            f"D={cell['D']:.3f} h={h['rollout_steps']} | "
            f"mse_test {h['forecast_mse_test']:.3e} (was {old['forecast_mse_test']:.3e}) "
            f"| skill {h['skill_test']:7.4f} (was {old['skill_test']:7.4f}){flag}",
            flush=True,
        )


def _aligned(cell):
    by_h = {h["rollout_steps"]: h for h in cell["horizons_clean_scored"]}
    return [by_h[h["rollout_steps"]] for h in cell["horizons_noisy_scored"]]


def _summarise(cells, n_samples):
    """Per (method, noise, horizon): discover/validate means, on EDGAR's sample split.

    Summarises the noisy-scored horizons only — the clean-scored ones are kept
    per-cell for provenance, not to be aggregated into a headline number.
    """
    disc_idx, val_idx = ld.discover_validate_samples(n_samples)
    disc, val = set(disc_idx.tolist()), set(val_idx.tolist())
    rows = []
    for method, nl in sorted({(c["method"], c["noise_level"]) for c in cells}):
        group = [c for c in cells if c["method"] == method and c["noise_level"] == nl]
        horizons = sorted(
            {h["rollout_steps"] for c in group for h in c["horizons_noisy_scored"]}
        )
        row = {"method": method, "noise_level": nl, "by_horizon": []}
        for h in horizons:
            def mean_over(idxs, key, _h=h):
                vals = [
                    hh[key]
                    for c in group
                    if c["sample_idx"] in idxs
                    for hh in c["horizons_noisy_scored"]
                    if hh["rollout_steps"] == _h
                ]
                return float(np.mean(vals)) if vals else None

            row["by_horizon"].append(
                {
                    "rollout_steps": h,
                    "mse_test_discover_mean": mean_over(disc, "forecast_mse_test"),
                    "mse_test_validate_mean": mean_over(val, "forecast_mse_test"),
                    "skill_test_discover_mean": mean_over(disc, "skill_test"),
                    "skill_test_validate_mean": mean_over(val, "skill_test"),
                    "n_unstable": sum(
                        1
                        for c in group
                        for hh in c["horizons_noisy_scored"]
                        if hh["rollout_steps"] == h and not hh["stable"]
                    ),
                }
            )
        rows.append(row)
    return rows


def main(data_dir, levels, weights_dir, block_len, device, methods, dry_run):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    cells = []
    if "sindy" in methods:
        print("── SINDy (rebuilt from saved coefficients) ─ scoring on u_noisy")
        cells += rescore_sindy(data_dir, levels, block_len, dry_run)
    if "stencilnet" in methods:
        print(f"── STENCIL-NET (forward passes from saved weights, device={device}) ─")
        cells += rescore_stencilnet(data_dir, levels, weights_dir, device, dry_run)
    if dry_run:
        print(f"\n{len(cells)} cell(s) would be re-scored.")
        return cells

    n_samples = int(_dataset(data_dir, 0, levels[0])["D"].shape[0])
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"runs": cells, "summary": _summarise(cells, n_samples)}
    out_path = os.path.join(out_dir, "rescore_noisy.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}  ({len(cells)} cells)")
    return cells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, dest="data_dir")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    ap.add_argument("--weights-dir", default=_DEFAULT_WEIGHTS_DIR, dest="weights_dir")
    ap.add_argument("--block-len", type=int, default=200, dest="block_len")
    ap.add_argument("--device", default=None, help="cuda / cpu; default auto-detect")
    ap.add_argument(
        "--methods", nargs="+", default=["sindy", "stencilnet"],
        choices=["sindy", "stencilnet"],
    )
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()
    main(
        args.data_dir, args.levels, args.weights_dir, args.block_len,
        args.device, args.methods, args.dry_run,
    )
