"""Compute the ORACLE Gaussian NLL floor for the van der Pol testbed.

The oracle knows the true dynamics (mu) and — for each trajectory — the true
(x_raw, u_raw) sequence and the affine (shift, scale) that took raw x to
normalised y-space. Its one-step prediction of y[t+1] is the deterministic
Euler step of the true VdP position update in raw scale, then affine-mapped
back to y-scale:

    E[x_raw[t+1]] = x_raw[t] + dt · u_raw[t]
    E[y[t+1]]     = (E[x_raw[t+1]] - y_shift) / y_scale

σ is fit per-trajectory by MLE on the residuals. Any evolved model's post-
optimisation NLL is bounded below by this value.

Note vs FHN: the position update dx/dt = u has NO process noise (all process
noise sits on the velocity update). The oracle's residual is therefore driven
purely by observation noise, so the floor is set by obs_noise_std / y_scale.

Run:
    python projects/vdp_relaxation/scripts/vdp_oracle_nll.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.vdp_relaxation.data_loader.load_data import (   # noqa: E402
    load_data, TEST_WARMUP_STEPS,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))
    (Xd_tr, Xd_te), _, _ = load_data(**pp)

    y = np.asarray(Xd_te["y"], dtype=np.float64)
    x_true = np.asarray(Xd_te["_x_true"], dtype=np.float64)
    u_true = np.asarray(Xd_te["_u_true"], dtype=np.float64)
    y_shift = np.asarray(Xd_te["_y_shift"], dtype=np.float64)[:, None]     # (n, 1)
    y_scale = np.asarray(Xd_te["_y_scale"], dtype=np.float64)[:, None]
    dt = pp["dt"]

    # One-step VdP position Euler in raw scale, using true x and true u.
    # Position update has no process noise (noise sits on u only).
    x_next_raw = x_true[:, :-1] + dt * u_true[:, :-1]
    x_next_y = (x_next_raw - y_shift) / y_scale                             # (n, T-1)

    resid = y[:, 1:] - x_next_y
    resid_pw = resid[:, TEST_WARMUP_STEPS:]
    sigma_mle = np.maximum(resid_pw.std(axis=1), 1e-6)
    nll_per_traj = np.log(sigma_mle) + 0.5
    nll_oracle = float(nll_per_traj.mean())

    pers = float(np.asarray(Xd_te["_persistence_nll"]).mean())

    print(f"\n=== VdP oracle NLL (discovery test window) ===")
    print(f"  post-warmup residual std      : min={sigma_mle.min():.4f}  "
          f"mean={sigma_mle.mean():.4f}  max={sigma_mle.max():.4f}")
    print(f"  ORACLE NLL floor              : {nll_oracle:+.4f} nat / bin")
    print(f"  PERSISTENCE baseline NLL      : {pers:+.4f} nat / bin")
    print(f"  Discovery budget for evolution: {pers - nll_oracle:+.4f} nat / bin")

    if nll_oracle > pers:
        print("\n  ⚠ WARNING: oracle NLL > persistence NLL. Either the oracle "
              "computation is wrong, or noise is small enough that persistence "
              "is already close to the theoretical floor.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
