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


def block_split(n_times: int, block_len: int = 200):
    """Leak-free train/test split by contiguous time blocks, dealt alternately.

    Chops the time axis into `block_len`-column blocks and deals whole blocks to
    train / test in strict alternation starting from train (block 0 -> train,
    block 1 -> test, block 2 -> train, ...), never splitting a block, so no training
    window can peek across a boundary into a test window. With an odd block count
    train keeps the extra block. Returns (train_cols, test_cols) as int arrays of 
    column indices into the (space, time) field.
    """
    n_blocks = n_times // block_len
    train_cols, test_cols = [], []
    for b in range(n_blocks):
        cols = np.arange(b * block_len, (b + 1) * block_len)
        (train_cols if b % 2 == 0 else test_cols).append(cols)
    train_cols = np.concatenate(train_cols) if train_cols else np.array([], int)
    test_cols = np.concatenate(test_cols) if test_cols else np.array([], int)
    return train_cols, test_cols

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
    is_tr = np.zeros(n_times, bool); is_tr[np.asarray(train_cols, int)] = True
    is_te = np.zeros(n_times, bool); is_te[np.asarray(test_cols, int)] = True

    def window_all(flag):
        m = np.ones(n_starts, bool)
        for k in range(h + 1):
            m &= flag[k : k + n_starts]
        return m

    return window_all(is_tr), window_all(is_te)


def forecast_mse(u_pred: np.ndarray, u_clean: np.ndarray) -> float:
    """Shared benchmark metric: MSE of a prediction against the CLEAN field.

    All methods are scored on this identical function so numbers are comparable:
    the target is always the noise-free ground truth, never the noisy observation.
    """
    u_pred = np.asarray(u_pred); u_clean = np.asarray(u_clean)
    assert u_pred.shape == u_clean.shape, (u_pred.shape, u_clean.shape)
    return float(np.mean((u_pred - u_clean) ** 2))


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


def teacher_forced_forecast(rhs, u_clean: np.ndarray, dtc: float, rollout_steps: int):
    """Teacher-forced restart forecast for the continuous-operator reference methods.

    Mirrors EDGAR's scoring protocol (`evaluate.py`): from every time column that
    has `rollout_steps` clean steps ahead, seed a classic-RK3 rollout at the TRUE
    state `u_clean[:, s]` and integrate `rollout_steps` steps on the model's own
    predictions (closed-loop within the window), then score every step against the
    clean field. Re-anchoring at each start — rather than a single full-horizon
    free-run — is exactly how EDGAR models are graded, so the reference methods and
    EDGAR share one protocol; it also stops a single diverging window from
    cascading into a global NaN.

    Both STENCIL-NET and SINDy learn a continuous-time RHS operator that is stepped
    externally by RK3, so this takes the RHS as a callable and owns the integrator,
    guaranteeing the two methods are stepped identically.

    Args:
        rhs: callable mapping a batched state `(n_starts, Lx)` and per-start
            physical times `(n_starts,)` to du/dt `(n_starts, Lx)`. Any known
            forcing must already be folded in by the caller.
        u_clean: clean ground-truth field, shape `(Lx, T)`.
        dtc: coarse timestep.
        rollout_steps: steps rolled per restart before scoring.

    Returns:
        (preds, targets), both `(n_starts, rollout_steps, Lx)` with
        `n_starts = T - rollout_steps`. The start of restart `i` is time column `i`.
    """
    u_clean = np.asarray(u_clean, dtype=float)
    Lx, T = u_clean.shape
    h = int(rollout_steps)
    n_starts = T - h
    starts = np.arange(n_starts)

    state = u_clean[:, starts].T.copy()           # (n_starts, Lx) at t = starts*dtc
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

    idx = starts[:, None] + 1 + np.arange(h)[None, :]   # (n_starts, h)
    targets = u_clean.T[idx]                            # (n_starts, h, Lx)
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

def _unforced_coarse_field(
    s_factor: int,
    t_factor: int,
    ic_seed: int | None = None,
    noise_level: float = 0.0,
    noise_seed: int = 0,
):
    """Simulate the autonomous (unforced) Burgers field and coarsen it.

    `ic_seed=None` uses the reference Gaussian-bump initial condition; an integer
    seed draws a random smooth IC (burgers_sim.draw_ic) — this is how the discover
    and validate splits get independent trajectories on the same attractor.

    `noise_level` adds the reference observation noise
    (noise_level * std(field) * N(0, 1)) to the coarse field. It is applied after
    the cache, not baked into it: the cached field is always the clean ground
    truth, so changing the noise level costs nothing and never invalidates a cache.

    Cached next to this file (one `.npz` per IC), because the fine simulation
    takes ~70 s. Delete the cache to force regeneration (e.g. after changing the
    simulator).
    """
    # File-based import: this module is exec'd from source by EDGAR (no package
    # context, so `from . import` would fail), but also imported normally by the
    # benchmark scripts. Loading by path works in both.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "burgers_sim", os.path.join(_HERE, "burgers_sim.py")
    )
    bs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bs)

    cache = os.path.join(
        _HERE,
        "burgers_unforced_coarse.npz"
        if ic_seed is None
        else f"burgers_unforced_coarse_ic{ic_seed}.npz",
    )
    if os.path.exists(cache):
        with np.load(cache) as z:
            if int(z["s_factor"]) == s_factor and int(z["t_factor"]) == t_factor:
                u_coarse, dxc, dtc = z["u_coarse"], float(z["dxc"]), float(z["dtc"])
                if noise_level:
                    u_coarse = bs.add_noise(u_coarse, noise_level, seed=noise_seed)
                return u_coarse, dxc, dtc

    sim = bs.simulate(forcing_seed=None, ic_seed=ic_seed)  # autonomous field
    u_coarse = np.asarray(bs.coarsen(sim, s_factor=s_factor, t_factor=t_factor)["u_coarse"])
    dxc = sim["L"] / u_coarse.shape[0]
    dtc = sim["dt"] * t_factor
    np.savez(
        cache,
        u_coarse=u_coarse, dxc=dxc, dtc=dtc,
        s_factor=s_factor, t_factor=t_factor,
    )
    if noise_level:
        u_coarse = bs.add_noise(u_coarse, noise_level, seed=noise_seed)
    return u_coarse, dxc, dtc


def load_data(
    data_path: str = "",
    s_factor: int = 4,
    t_factor: int = 20,
    block_len: int = 500,
    train_frac: float = 0.5,   # kept for API symmetry; split is deterministic below
    n_eval_samples: int = 4,
    random_seed: int = 0,
    n_recordings: int = 4,
    ic_seed_base: int = 0,
    noise_level: float = 0.0,
    noise_seed: int = 0,
):
    """Build EDGAR's sample tensors from N independent unforced coarse fields.

    Each *recording* is a full autonomous trajectory with its own initial
    condition; it becomes one EDGAR *sample* with its own independently-fitted
    parameters, while every sample shares the same evolved model. The first
    `n_recordings // 2` recordings form the discover samples, the rest the
    validate samples, so validate is a genuine generalisation test: unseen
    realisations of the same dynamics, not just other time-slices of a trajectory
    discover already saw.

    Recording 0 keeps the reference Gaussian-bump IC (so the README scoreboard
    posts stay comparable); recordings 1.. use random smooth ICs
    (burgers_sim.draw_ic, seed = ic_seed_base + i).

    Within each recording the time axis is cut into non-overlapping `block_len`
    blocks and alternate blocks are dealt to train / test (leak-free:
    autoregressive windows never cross a block boundary, so no test step is
    reachable from a train window, and no recording is shared across splits).
    Both train and test span the whole trajectory, so neither is only-transient
    or only-steady-state.

    `noise_level` is a fraction of each field's own standard deviation (the
    reference observation model, see burgers_sim.add_noise) and is applied to the
    whole field — train, test and eval blocks alike. Targets are therefore noisy
    too, so the achievable loss floor rises with noise_level: absolute losses are
    NOT comparable across noise levels, nor to the zero-noise scoreboard posts in
    the README.

    `data_path` is unused: the fields are generated deterministically by the
    simulator (autonomous; the only seeds that matter are the per-recording ICs
    and, when noise is on, `noise_seed`).

    Returns:
        (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval
        each dict has key 'x' of shape (n_samples, n_blocks, n_sensors, block_len),
        n_samples = n_recordings // 2.
    """

    def to_blocks(u):
        n_sensors, n_times = u.shape
        n_blocks = n_times // block_len
        field = u[:, : n_blocks * block_len]
        # (n_sensors, n_blocks, block_len) -> (n_blocks, n_sensors, block_len)
        return field.reshape(n_sensors, n_blocks, block_len).transpose(1, 0, 2)

    recordings = [
        to_blocks(
            _unforced_coarse_field(
                s_factor,
                t_factor,
                ic_seed=None if i == 0 else ic_seed_base + i,
                noise_level=noise_level,
                # distinct realisation per recording, else every sample sees the
                # same noise field and the map can fit the noise itself
                noise_seed=noise_seed + i,
            )[0]
        )
        for i in range(n_recordings)
    ]

    n_disc = n_recordings // 2
    disc_recs, val_recs = recordings[:n_disc], recordings[n_disc:]

    def split(recs, start):
        # stack recordings on the sample axis; alternate blocks -> train (start=0)
        # / test (start=1), so each sample's params are fit and scored on disjoint
        # leak-free blocks. All recordings share n_blocks, so the stack is regular.
        return {"x": np.stack([r[np.arange(start, r.shape[0], 2)] for r in recs])}

    X_disc_train = split(disc_recs, 0)
    X_disc_test = split(disc_recs, 1)
    X_val_train = split(val_recs, 0)
    X_val_test = split(val_recs, 1)

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
