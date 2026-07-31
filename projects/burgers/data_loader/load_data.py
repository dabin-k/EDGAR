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

Two dataset families live here:

  * the original single-field *forced* bundle `burgers_clean.npz` (`load_bundle`,
    `noise_field`), still consumed by the SINDy / STENCIL-NET benchmark scripts.
    Regenerate with `python projects/burgers/data_loader/regenerate.py`.
  * the shared *unforced* comparison datasets, one file per (initial condition,
    noise level), each holding several samples that differ only in viscosity `D`
    (`load_dataset`, and `load_data` on top of it). Written by
    `python projects/burgers/data_loader/generate_datasets.py` to
    `/home/dabin/data/burgers_simulated/ic_seed_{ic}_nl_{nl}.npz`. Noise is baked
    into the file so EDGAR, SINDy and STENCIL-NET see the identical observations.
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
    for k in (
        "dxc",
        "dtc",
        "D",
        "L",
        "dt",
        "Lx",
        "Tsim",
        "N",
        "s_factor",
        "t_factor",
        "forcing_seed",
    ):
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


def train_test_blocks(n_times: int, block_len: int = 200):
    """Which time blocks are dealt to train / test.

    THE definition of the leak-free temporal split: the time axis is chopped into
    `block_len`-column blocks and whole blocks are dealt to train / test in strict
    alternation starting from train (block 0 -> train, block 1 -> test, ...), never
    splitting a block, so no training window can peek across a boundary into a test
    window. With an odd block count train keeps the extra block; any tail columns
    beyond `n_blocks * block_len` are dropped.

    Every consumer derives from this one function -- `block_split` for the
    reference methods (which want columns) and `load_data` for EDGAR (which wants
    blocks) -- so changing the split policy here changes it for all methods at once.

    Args:
        n_times: Length of the time axis.
        block_len: Time-block length.

    Returns:
        (train_blocks, test_blocks), each an int array of block indices.
    """
    blocks = np.arange(n_times // block_len)
    return blocks[blocks % 2 == 0], blocks[blocks % 2 == 1]


def block_split(n_times: int, block_len: int = 200):
    """`train_test_blocks` expressed as column indices into the (space, time) field.

    Returns (train_cols, test_cols) as int arrays.
    """

    def to_cols(blocks):
        if blocks.size == 0:
            return np.array([], int)
        return np.concatenate(
            [np.arange(b * block_len, (b + 1) * block_len) for b in blocks]
        )

    train_blocks, test_blocks = train_test_blocks(n_times, block_len)
    return to_cols(train_blocks), to_cols(test_blocks)


def discover_validate_samples(n_samples: int):
    """Which samples of a dataset file are discover / validate.

    THE definition of the sample-axis split: the first half of a file's samples are
    discover, the rest validate. Since the samples differ only in viscosity `D`,
    validate is a generalisation test over *unseen D*. EDGAR's `load_data` builds
    its tensors from this, and the reference-method sweeps report their per-sample
    scores aggregated over the same two groups, so one change here moves every
    method's discover / validate partition together.

    Args:
        n_samples: Number of samples in the dataset file.

    Returns:
        (discover_idx, validate_idx), each an int array of sample indices.
    """
    n_disc = n_samples // 2
    return np.arange(n_disc), np.arange(n_disc, n_samples)


def contiguous_blocks(field: np.ndarray, cols: np.ndarray) -> list[np.ndarray]:
    """Split `field[:, cols]` back into its maximal runs of consecutive columns.

    `block_split` returns the train (or test) columns as one concatenated index
    array; this recovers the individual contiguous time-blocks, so a per-block
    consumer (SINDy's joint fit, STENCIL-NET's rollout loss) never forms a
    finite-difference / rollout window that straddles a block boundary.
    """
    cols = np.asarray(cols, int)
    if cols.size == 0:
        return []
    breaks = np.where(np.diff(cols) != 1)[0] + 1
    return [field[:, run] for run in np.split(cols, breaks)]


def split_start_masks(train_cols, test_cols, n_times: int, rollout_steps: int):
    """Leak-free train/test masks over the teacher-forced restarts.

    `teacher_forced_forecast` produces `n_starts = n_times - rollout_steps` restarts;
    restart `i` seeds at column `i` and is scored against columns `i+1 … i+rollout_steps`,
    so its whole window is `{i, …, i+rollout_steps}`. A restart is a *train* (resp.
    *test*) restart only if that entire window lies in `train_cols` (resp. `test_cols`);
    windows straddling a block boundary belong to neither, so no scored forecast crosses
    the split. Both SINDy and STENCIL-NET score with these identical masks.

    Returns:
        (train_mask, test_mask), each a boolean array of length `n_starts`.
    """
    h = int(rollout_steps)
    n_starts = n_times - h
    is_tr = np.zeros(n_times, bool)
    is_tr[np.asarray(train_cols, int)] = True
    is_te = np.zeros(n_times, bool)
    is_te[np.asarray(test_cols, int)] = True

    def window_all(flag):
        m = np.ones(n_starts, bool)
        for k in range(h + 1):
            m &= flag[k : k + n_starts]
        return m

    return window_all(is_tr), window_all(is_te)


def forecast_mse(u_pred: np.ndarray, u_target: np.ndarray) -> float:
    """Shared benchmark metric: MSE of a prediction against a target field.

    All methods are scored on this identical function so numbers are comparable.
    The caller chooses the target field
    """
    u_pred = np.asarray(u_pred)
    u_target = np.asarray(u_target)
    assert u_pred.shape == u_target.shape, (u_pred.shape, u_target.shape)
    return float(np.mean((u_pred - u_target) ** 2))


def benchmark_rollout_steps() -> int:
    """`rollout_steps` used to score the reference methods (STENCIL-NET, SINDy).

    Read from this project's own `evaluate` config so the reference-method
    forecasts and EDGAR's teacher-forced rollout (see
    `projects/burgers/evaluate/evaluate.py`) are graded on the SAME horizon.
    """
    import yaml

    cfg = os.path.join(_HERE, "..", "config.yaml")
    with open(cfg) as fh:
        return int(yaml.safe_load(fh)["evaluate"]["rollout_steps"])


def teacher_forced_forecast(rhs, field: np.ndarray, dtc: float, rollout_steps: int):
    """Teacher-forced restart forecast for the continuous-operator reference methods.

    Mirrors EDGAR's scoring protocol (`evaluate.py`): from every time column that
    has `rollout_steps` steps ahead, seed a classic-RK3 rollout at the state
    `field[:, s]` and integrate `rollout_steps` steps on the model's own
    predictions (closed-loop within the window), then score every step against
    `field`. Re-anchoring at each start — rather than a single full-horizon
    free-run — is exactly how EDGAR models are graded, so the reference methods and
    EDGAR share one protocol; it also stops a single diverging window from
    cascading into a global NaN.

    `field` supplies BOTH the restart states and the targets, so passing the clean
    field would hand the method a noise-free initial condition that EDGAR never
    gets (`load_data` feeds it `u_noisy`). Benchmark call sites therefore pass the
    OBSERVED field: seeded from noisy states, scored against noisy targets, on
    both counts matching EDGAR.

    Both STENCIL-NET and SINDy learn a continuous-time RHS operator that is stepped
    externally by RK3, so this takes the RHS as a callable and owns the integrator,
    guaranteeing the two methods are stepped identically.

    Args:
        rhs: callable mapping a batched state `(n_starts, Lx)` and per-start
            physical times `(n_starts,)` to du/dt `(n_starts, Lx)`. Any known
            forcing must already be folded in by the caller.
        field: observed field, shape `(Lx, T)`; used for restart states AND targets.
        dtc: coarse timestep.
        rollout_steps: steps rolled per restart before scoring.

    Returns:
        (preds, targets), both `(n_starts, rollout_steps, Lx)` with
        `n_starts = T - rollout_steps`. The start of restart `i` is time column `i`.
    """
    field = np.asarray(field, dtype=float)
    Lx, T = field.shape
    h = int(rollout_steps)
    n_starts = T - h
    starts = np.arange(n_starts)

    state = field[:, starts].T.copy()  # (n_starts, Lx) at t = starts*dtc
    t = starts * dtc
    preds = np.empty((n_starts, h, Lx))
    for j in range(h):
        # One step of Kutta's 3rd-order RK: sample the slope k = dtc*rhs at three
        # stage points (start, midpoint, end) and combine with Simpson's-rule
        # weights (1,4,1)/6. The k3 stage state is the method's prescribed
        # end-of-interval extrapolation (y - k1 + 2*k2), not a typo. This is the
        # identical integrator the reference free-runs used, so switching to
        # teacher-forced restarts changes only the anchoring, not the stepping.
        k1 = dtc * rhs(state, t)
        k2 = dtc * rhs(state + 0.5 * k1, t + 0.5 * dtc)
        k3 = dtc * rhs(state - k1 + 2.0 * k2, t + dtc)
        state = state + (1.0 / 6.0) * (k1 + 4.0 * k2 + k3)
        preds[:, j] = state
        t = t + dtc

    idx = starts[:, None] + 1 + np.arange(h)[None, :]  # (n_starts, h)
    targets = field.T[idx]  # (n_starts, h, Lx)
    return preds, targets


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
#     x(t) = f( x(t-1), ..., x(t-m) )                  [m = input_sequence_length]
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


def load_dataset(path: str) -> dict:
    """Load one benchmark dataset written by `generate_datasets.py`.

    These are the shared unforced-Burgers files
    (`/home/dabin/data/burgers_simulated/ic_seed_{ic}_nl_{nl}.npz`): one initial
    condition, one noise level, and `n_samples` samples that differ only in their
    viscosity `D`. Noise is baked in, so every method reads the identical
    observations — that is the whole point of the file, and why nothing here
    re-simulates or re-noises.

    Args:
        path: Path to an `ic_seed_*_nl_*.npz` file.

    Returns:
        Dict of the npz contents with the 0-d metadata arrays unwrapped to
        Python scalars. Key arrays: `u_noisy` and `u_clean`, both
        `(n_samples, n_sensors, n_times)`, and `D`, `(n_samples,)`.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate the benchmark datasets with "
            "`uv run python projects/burgers/data_loader/generate_datasets.py`."
        )
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    for k in (
        "dxc",
        "dtc",
        "noise_level",
        "ic_seed",
        "L",
        "Lx",
        "dt",
        "Tsim",
        "s_factor",
        "t_factor",
        "N",
        "forced",
    ):
        if k in out:
            out[k] = out[k].item()
    return out


def load_data(
    data_path: str = "",
    block_len: int = 200,
    train_frac: float = 0.5,  # kept for API symmetry; split is deterministic below
    n_eval_samples: int = 4,
    random_seed: int = 0,
):
    """Build EDGAR's sample tensors from one shared benchmark dataset file.

    `data_path` points at an `ic_seed_*_nl_*.npz` written by
    `generate_datasets.py`. Every sample in the file shares one initial condition
    and one noise realisation but has its own viscosity `D`, so each becomes one
    EDGAR *sample* with independently-fitted parameters while all samples share
    the evolved model. The first half of the samples form the discover split and
    the rest the validate split, so validate is a genuine generalisation test:
    unseen viscosities, not just other time-slices of a field discover has seen.

    Within each sample the time axis is cut into non-overlapping `block_len`
    blocks and alternate blocks are dealt to train / test (leak-free:
    autoregressive windows never cross a block boundary, so no test step is
    reachable from a train window, and no sample is shared across splits). Both
    train and test span the whole trajectory, so neither is only-transient or
    only-steady-state.

    The observations are the file's `u_noisy` field — noise is baked into the
    dataset, applied to the whole field, train / test / eval blocks alike.
    Targets are therefore noisy too, so the achievable loss floor rises with the
    file's noise level: absolute losses are NOT comparable across noise levels,
    nor to the zero-noise scoreboard posts in the README.

    Args:
        data_path: Path to an `ic_seed_*_nl_*.npz` dataset file.
        block_len: Time-block length; alternate blocks go to train / test.
        train_frac: Unused; kept for API symmetry (the split is deterministic).
        n_eval_samples: Number of discover-train blocks in the eval tensor.
        random_seed: Seed for choosing those eval blocks.

    Returns:
        (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval.
        Each dict has key 'x' of shape (n_samples, n_blocks, n_sensors, block_len),
        with n_samples = (samples in the file) // 2.

    Raises:
        ValueError: If `data_path` is empty.
    """
    if not data_path:
        raise ValueError(
            "burgers load_data requires io.data_path to point at a benchmark dataset, "
            "e.g. /home/dabin/data/burgers_simulated/ic_seed_0_nl_0.0.npz "
            "(write them with data_loader/generate_datasets.py)."
        )
    fields = load_dataset(data_path)["u_noisy"]

    def to_blocks(u):
        n_sensors, n_times = u.shape
        n_blocks = n_times // block_len
        field = u[:, : n_blocks * block_len]
        # (n_sensors, n_blocks, block_len) -> (n_blocks, n_sensors, block_len)
        return field.reshape(n_sensors, n_blocks, block_len).transpose(1, 0, 2)

    recordings = [to_blocks(u) for u in fields]

    # Both splits come from the shared definitions above, so EDGAR and the
    # reference-method sweeps partition the same data the same way.
    disc_idx, val_idx = discover_validate_samples(len(recordings))
    n_disc = len(disc_idx)
    disc_recs = [recordings[i] for i in disc_idx]
    val_recs = [recordings[i] for i in val_idx]
    n_times = fields.shape[-1]
    train_blocks, test_blocks = train_test_blocks(n_times, block_len)

    def split(recs, blocks):
        # stack recordings on the sample axis, keeping the train (or test) blocks,
        # so each sample's params are fit and scored on disjoint leak-free blocks.
        # All recordings share n_blocks, so the stack is regular.
        return {"x": np.stack([r[blocks] for r in recs])}

    X_disc_train = split(disc_recs, train_blocks)
    X_disc_test = split(disc_recs, test_blocks)
    X_val_train = split(val_recs, train_blocks)
    X_val_test = split(val_recs, test_blocks)

    # X_eval: a few discover-train blocks per discover sample, for fingerprinting
    # and image feedback. _sample_indices maps each eval sample to its discover
    # sample's fitted params.
    rng = np.random.default_rng(random_seed)
    n_dt_blocks = X_disc_train["x"].shape[1]
    k = min(n_eval_samples, n_dt_blocks)
    eval_blocks = np.sort(rng.choice(n_dt_blocks, k, replace=False))
    X_eval = {
        "x": X_disc_train["x"][:, eval_blocks],
        "_sample_indices": np.arange(n_disc, dtype=int),
    }

    def to_jax(d):
        return {
            k: (jnp.array(v) if k != "_sample_indices" else v) for k, v in d.items()
        }

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
