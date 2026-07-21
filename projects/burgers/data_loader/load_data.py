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


# ══════════════════════════════════════════════════════════════════════════════
# EDGAR entry points
# ------------------------------------------------------------------------------
# `load_data()` and `loss_fn()` below are the interface EDGAR's TaskSpec loads
# (see edgar/io/task_spec.py). They are separate from the benchmark helpers above
# so the reference methods (SINDy / STENCIL-NET) and EDGAR share one field
# definition but lay it out differently.
#
# EDGAR control task (plan Step 5): fit a discrete autoregressive map
#
#     x(t) = f( x(t-1), ..., x(t-m) )                         [m = evaluate.MAX_LENGTH]
#
# to the UNFORCED (autonomous) field. There is no exogenous input, so the map is
# well-posed in the observed variable alone. Note there is NO exact closed-form
# ground-truth map: the field is a continuous PDE integrated at fine dt then
# coarsened x20 in time and x4 in space, so (a) the exact one-step map is the
# time-dt flow map, which has no finite closed form, and (b) space-coarsening
# makes the coarse field non-Markovian in the observed variable — a second lag
# carries information about the unresolved sub-grid state. Scoring is therefore
# forecast MSE only, graded against three reference posts (see README):
#     do-nothing (persistence)          ~1.5e-4
#     naive one-step surrogate          ~6.8e-6
#     best-achievable AR map (LS floor) ~2.7e-6
#
# The model works in sensor-index space (neighbours via np.roll); physical grid
# spacing is absorbed into its free parameters, so nothing about the grid or the
# generating physics leaks into the model interface.
# ══════════════════════════════════════════════════════════════════════════════

import jax.numpy as jnp

_UNFORCED_CACHE = os.path.join(_HERE, "burgers_unforced_coarse.npz")


def _unforced_coarse_field(s_factor: int, t_factor: int, seed_note: str = "forcing_seed=None"):
    """Simulate the autonomous (unforced) Burgers field and coarsen it.

    Cached to `burgers_unforced_coarse.npz` next to this file, because the fine
    simulation takes ~70 s. Delete that file to force regeneration (e.g. after
    changing the simulator).
    """
    from . import burgers_sim as bs  # local import; heavy (jax-free) numpy sim

    if os.path.exists(_UNFORCED_CACHE):
        with np.load(_UNFORCED_CACHE) as z:
            if int(z["s_factor"]) == s_factor and int(z["t_factor"]) == t_factor:
                return z["u_coarse"], float(z["dxc"]), float(z["dtc"])

    sim = bs.simulate(forcing_seed=None)  # autonomous field
    u_coarse = np.asarray(bs.coarsen(sim, s_factor=s_factor, t_factor=t_factor)["u_coarse"])
    dxc = sim["L"] / u_coarse.shape[0]
    dtc = sim["dt"] * t_factor
    np.savez(
        _UNFORCED_CACHE,
        u_coarse=u_coarse, dxc=dxc, dtc=dtc,
        s_factor=s_factor, t_factor=t_factor,
    )
    return u_coarse, dxc, dtc


def load_data(
    data_path: str = "",
    s_factor: int = 4,
    t_factor: int = 20,
    block_len: int = 100,
    train_frac: float = 0.5,   # kept for API symmetry; split is round-robin below
    n_eval_samples: int = 4,
    random_seed: int = 0,
):
    """Build EDGAR's sample tensors from the unforced coarse field.

    A *sample* is the whole field; parameters are fitted once per field (one
    global map). The time axis is cut into non-overlapping `block_len` blocks and
    the blocks are dealt round-robin into four bins:

        block i -> [disc_train, disc_test, val_train, val_test][i % 4]

    so every split spans the whole trajectory (no split is only-transient or
    only-steady-state), and no block is shared between splits. Autoregressive
    windows never cross a block boundary, so no test step is reachable from a
    train window.

    `data_path` is unused: the field is generated deterministically by the
    simulator (autonomous, so no seed variation is needed).

    Returns:
        (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval
        each dict has key 'x' of shape (n_samples=1, n_blocks, n_sensors, block_len).
    """
    u_coarse, _dxc, _dtc = _unforced_coarse_field(s_factor, t_factor)
    n_sensors, n_times = u_coarse.shape

    n_blocks = n_times // block_len
    field = u_coarse[:, : n_blocks * block_len]
    # (n_sensors, n_blocks, block_len) -> (n_blocks, n_sensors, block_len)
    blocks = field.reshape(n_sensors, n_blocks, block_len).transpose(1, 0, 2)

    bins = [np.arange(i, n_blocks, 4) for i in range(4)]  # disc_tr, disc_te, val_tr, val_te

    def pack(idx):
        # leading n_samples axis of 1: one global field / one param set
        return {"x": blocks[idx][None, ...]}

    X_disc_train = pack(bins[0])
    X_disc_test = pack(bins[1])
    X_val_train = pack(bins[2])
    X_val_test = pack(bins[3])

    # X_eval: a few discover-train blocks for image feedback (as its own samples)
    rng = np.random.default_rng(random_seed)
    n_dt_blocks = X_disc_train["x"].shape[1]
    k = min(n_eval_samples, n_dt_blocks)
    eval_blocks = np.sort(rng.choice(n_dt_blocks, k, replace=False))
    X_eval = {
        "x": X_disc_train["x"][:, eval_blocks],
        "_sample_indices": np.zeros(1, dtype=int),
    }

    def to_jax(d):
        return {k: (jnp.array(v) if k != "_sample_indices" else v) for k, v in d.items()}

    return (
        (to_jax(X_disc_train), to_jax(X_disc_test)),
        (to_jax(X_val_train), to_jax(X_val_test)),
        to_jax(X_eval),
    )


def loss_fn(preds, targets):
    """Mean squared error per sample.

    `evaluate` has already aligned predictions with their targets, so this does
    no indexing of its own. Both arrays are
    (n_samples, n_blocks, n_starts, ROLLOUT_STEPS, n_sensors); reduce everything
    but the sample axis.
    """
    return jnp.mean((preds - targets) ** 2, axis=(1, 2, 3, 4))
