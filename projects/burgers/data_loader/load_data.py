"""Shared Burgers data-access layer for the benchmark (Step 1).

This is the single source of truth for the field that STENCIL-NET, SINDy, and
both EDGAR variants all consume, so their numbers are comparable. It loads the
bundle produced by `burgers_sim` (`burgers_clean.npz`) and exposes:

  * the clean coarse field and every noisy variant,
  * the grid metadata (dxc, dtc, D, L, forcing coefficients),
  * a leak-free train/test split by contiguous time blocks.

It deliberately does NOT reshape the field into an EDGAR sample tensor. The
sample layout differs between the two EDGAR runs — Step 5 (temporal control:
each grid point is an independent lag series) and Step 6 (spatial-stencil
propagator: one global field, window = spatial neighbourhood) — so each
project's own `evaluate/` builds its samples on top of this loader. Keeping the
split here means both variants (and the two reference methods) share exactly the
same held-out data.

Regeneration: if `burgers_clean.npz` is absent, run
`python -m projects.burgers.data_loader.regenerate` (see `regenerate.py`), or
call `burgers_sim.simulate()/coarsen()/add_noise()` directly.
"""

from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NPZ = os.path.join(_HERE, "burgers_clean.npz")


def load_bundle(path: str | None = None) -> dict:
    """Load the saved Burgers bundle into a plain dict of arrays/scalars."""
    path = path or _DEFAULT_NPZ
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Regenerate with burgers_sim.simulate()/coarsen()/add_noise() "
            "or projects/burgers/data_loader/regenerate.py."
        )
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    # de-array 0-d scalars for ergonomics
    for k in ("dxc", "dtc", "D", "L", "dt", "Lx", "Tsim", "N", "s_factor", "t_factor", "forcing_seed"):
        if k in out:
            out[k] = out[k].item()
    return out


def noise_field(bundle: dict, noise_level: float) -> np.ndarray:
    """Return the observed field at a given noise level (0.0 => clean)."""
    if noise_level in (0, 0.0):
        return bundle["u_coarse"]
    key = f"u_noisy_{noise_level}"
    if key not in bundle:
        raise KeyError(
            f"{key} not in bundle; available levels: {list(bundle.get('noise_levels', []))}. "
            "Add it with burgers_sim.add_noise(bundle['u_coarse'], level)."
        )
    return bundle[key]


def block_split(n_times: int, block_len: int = 200, train_frac: float = 0.6, seed: int = 0):
    """Leak-free train/test split by contiguous time blocks.

    Chops the time axis into `block_len`-column blocks and assigns whole blocks
    to train or test (never splitting a block), so no training window can peek
    across the boundary into a test window. Mirrors the alternating-block scheme
    used by synthetic_data_v2/v3. Returns (train_cols, test_cols) as int arrays
    of column indices into the (space, time) field.
    """
    n_blocks = n_times // block_len
    rng = np.random.RandomState(seed)
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)
    n_train = max(1, int(round(train_frac * n_blocks)))
    train_blocks = set(block_ids[:n_train].tolist())
    train_cols, test_cols = [], []
    for b in range(n_blocks):
        cols = np.arange(b * block_len, (b + 1) * block_len)
        (train_cols if b in train_blocks else test_cols).append(cols)
    train_cols = np.concatenate(train_cols) if train_cols else np.array([], int)
    test_cols = np.concatenate(test_cols) if test_cols else np.array([], int)
    return train_cols, test_cols


def forecast_mse(u_pred: np.ndarray, u_clean: np.ndarray) -> float:
    """Shared benchmark metric: MSE of a prediction against the CLEAN field.

    All methods are scored on this identical function so numbers are comparable:
    the target is always the noise-free ground truth, never the noisy observation.
    """
    u_pred = np.asarray(u_pred); u_clean = np.asarray(u_clean)
    assert u_pred.shape == u_clean.shape, (u_pred.shape, u_clean.shape)
    return float(np.mean((u_pred - u_clean) ** 2))
