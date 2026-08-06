"""State-space DSL testbed: stochastic van der Pol, observation of position only.

Contract (same as fhn_excitable / oscillator_ss):
    LLM writes ``model(state, y_prev, params) -> (new_state, mean)``.
    ``params`` includes ``log_sigma_obs`` (learnable observation noise) and any
    number of ``s0_*``-prefixed keys that declare initial-state values (framework
    strips the prefix and passes them as ``state``).

What's different from fhn_excitable:
    van der Pol is a NONLINEAR relaxation oscillator, not an excitable system.
    The observation ``x`` (position) is coupled to a hidden velocity ``u`` via
        dx/dt = u
        du/dt = mu * (1 - x^2) * u - x                       + process noise
    A model that captures only linear damping cannot fit VdP: the nonlinear
    (1 - x^2) * u term reverses sign at |x| > 1 and drives the relaxation
    "kick" of the limit cycle. Discovery story: evolution must find the
    nonlinear damping structure, not just track velocity.

    Note that unlike FHN's ``w``, the hidden velocity ``u`` is close to the
    finite-difference of ``x``. A crude AR(2)-style model can extract some of
    ``u``'s signal from lagged observations; the oracle is not that far above
    such a baseline. Proper Kalman-style joint inference of (x, u) with the
    correct nonlinear damping is what closes the last nat.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── module-level constants (baked into loss_fn's closure, jit-safe) ──

WARMUP_STEPS: int = 100
"""Leading predictions skipped by loss_fn. Same rationale as fhn_excitable: the
hidden state starts at its DEFAULT_PARAMS prior and needs ~1 period (~150
steps at mu=2, dt=0.05) to stabilise. 100 gives comfortable margin."""

TEST_WARMUP_STEPS: int = 0
"""Leading predictions ignored by ``loss_fn`` when evaluating on the *test*
window. Zero: ``roll_state`` scans the whole train window (>> WARMUP_STEPS) from
``s0_*``, so the latent carried across the boundary is already converged — the
first test prediction has no init transient to discard. The test dicts stamp this
as ``_eval_warmup``; the train (fitting) path has no such key and falls back to
``WARMUP_STEPS``. Kept symmetric with ``WARMUP_STEPS`` so both are tunable in one
place. Safe as a data-dict value because the test path is eager (never jitted),
unlike the fit loss — so no ``ConcretizationError``."""


# ── data synthesis: stochastic van der Pol ──


def _synth_vdp(
    rng: np.random.Generator,
    T: int,
    dt: float,
    mu: float,
    proc_noise_std: float,
    obs_noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One trajectory of Euler-Maruyama-integrated stochastic van der Pol.

    Returns (y, u_raw, x_raw, y_shift, y_scale) with y = normalise(x + obs noise)
    to zero-mean unit-std, and (y_shift, y_scale) the per-trajectory affine
    transform so raw-scale predictions map back:
        y_predicted = (raw_predicted - y_shift) / y_scale
    """
    x = np.zeros(T, dtype=np.float64)
    u = np.zeros(T, dtype=np.float64)
    x[0] = 2.0 + 0.1 * rng.standard_normal()   # start near the +x turnaround
    u[0] = 0.0 + 0.1 * rng.standard_normal()

    for t in range(1, T):
        # process noise on u only, scaled by sqrt(dt) → dt-invariant SDE variance
        eta = proc_noise_std * rng.standard_normal() / np.sqrt(dt)
        dx = u[t - 1]
        du = mu * (1.0 - x[t - 1] ** 2) * u[t - 1] - x[t - 1] + eta
        x[t] = x[t - 1] + dt * dx
        u[t] = u[t - 1] + dt * du

    y_raw = x + obs_noise_std * rng.standard_normal(T)
    y_shift = float(y_raw.mean())
    y_scale = float(y_raw.std() + 1e-8)
    y = (y_raw - y_shift) / y_scale
    return (y.astype(np.float32), u.astype(np.float32), x.astype(np.float32),
            y_shift, y_scale)


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    n_trajectories: int = 32,
    T: int = 2400,
    dt: float = 0.05,
    mu: float = 2.0,
    proc_noise_std: float = 0.3,
    obs_noise_std: float = 0.15,
    train_frac: float = 0.5,
    T_eval: int = 200,
    n_eval: int = 4,
    seed: int = 42,
):
    """Generate stochastic van der Pol trajectories with observation of x only.

    Trials (timesteps) are split into two contiguous chunks at
    ``split_t = round(T * train_frac)``: train = the first ``train_frac`` of the
    trajectory, test = the rest. Params are fit on the train window and
    cross-validated on the held-out test window; the latent state is carried
    across the boundary by ``roll_state``. The initial-state ``s0_*`` params are
    *not* cross-validated.
    """
    del data_path  # synthetic
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac ({train_frac}) must be in (0, 1).")
    split_t = int(round(T * train_frac))
    min_win = WARMUP_STEPS + 40
    # Both train and test must clear the warmup for a meaningful post-warmup loss
    # window (the test window includes the boundary sample, hence T - split_t + 1).
    if split_t < min_win or (T - split_t + 1) < min_win:
        raise ValueError(
            f"train_frac={train_frac}, T={T} -> train={split_t} / "
            f"test={T - split_t + 1} timesteps; each must exceed WARMUP_STEPS "
            f"({WARMUP_STEPS}) + 40 = {min_win}. Increase T or move train_frac "
            "toward 0.5."
        )

    rng = np.random.default_rng(seed)
    ys, us, xs, shifts, scales = [], [], [], [], []
    for _ in range(n_trajectories):
        y_i, u_i, x_i, sh, sc = _synth_vdp(
            rng, T, dt, mu, proc_noise_std, obs_noise_std
        )
        ys.append(y_i)
        us.append(u_i)
        xs.append(x_i)
        shifts.append(sh)
        scales.append(sc)
    all_y = np.stack(ys)
    all_u = np.stack(us)
    all_x = np.stack(xs)
    all_shift = np.asarray(shifts, dtype=np.float32)
    all_scale = np.asarray(scales, dtype=np.float32)

    split_rng = np.random.default_rng(seed + 1)
    perm = split_rng.permutation(n_trajectories)
    disc_idx = np.sort(perm[: n_trajectories // 2])
    val_idx = np.sort(perm[n_trajectories // 2 :])

    y_disc = jnp.asarray(all_y[disc_idx])
    y_val = jnp.asarray(all_y[val_idx])
    u_disc = jnp.asarray(all_u[disc_idx])
    u_val = jnp.asarray(all_u[val_idx])
    x_disc = jnp.asarray(all_x[disc_idx])
    x_val = jnp.asarray(all_x[val_idx])
    shift_disc = jnp.asarray(all_shift[disc_idx])
    shift_val = jnp.asarray(all_shift[val_idx])
    scale_disc = jnp.asarray(all_scale[disc_idx])
    scale_val = jnp.asarray(all_scale[val_idx])

    # ── temporal train/test split (contiguous chunks; split_t set above) ──
    def _train_win(y_full):
        return y_full[:, :split_t]

    def _test_win(arr):
        return arr[:, split_t - 1:]

    def _persistence_nll_per_trajectory(y_np: np.ndarray) -> np.ndarray:
        residuals = y_np[:, 1:] - y_np[:, :-1]
        sigma = np.maximum(residuals.std(axis=-1, keepdims=True), 1e-3)
        nll = np.log(sigma) + 0.5 * (residuals / sigma) ** 2
        return nll[:, TEST_WARMUP_STEPS:].mean(axis=-1).astype(np.float32)

    # Baseline computed on the test window (at TEST_WARMUP_STEPS) so it matches
    # discover.final, the model's now-warmup-free held-out test loss.
    pers_disc = _persistence_nll_per_trajectory(np.asarray(_test_win(y_disc)))
    pers_val = _persistence_nll_per_trajectory(np.asarray(_test_win(y_val)))

    X_disc_train = {"y": _train_win(y_disc)}
    X_disc_test = {
        "y": _test_win(y_disc),
        "_eval_warmup": TEST_WARMUP_STEPS,
        "_persistence_nll": jnp.asarray(pers_disc),
        # Oracle sidecars, sliced to the same test window as "y"; scoring path
        # never reads them (apply_model / loss_fn only touch "y").
        "_u_true": _test_win(u_disc),
        "_x_true": _test_win(x_disc),
        "_y_shift": shift_disc,
        "_y_scale": scale_disc,
    }
    X_val_train = {"y": _train_win(y_val)}
    X_val_test = {
        "y": _test_win(y_val),
        "_eval_warmup": TEST_WARMUP_STEPS,
        "_persistence_nll": jnp.asarray(pers_val),
        "_u_true": _test_win(u_val),
        "_x_true": _test_win(x_val),
        "_y_shift": shift_val,
        "_y_scale": scale_val,
    }

    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    T_eval_actual = int(min(T_eval, split_t))
    eval_pos = np.sort(split_rng.choice(len(disc_idx), n_eval_actual, replace=False))
    X_eval = {
        "y": y_disc[eval_pos, :T_eval_actual],
        "_sample_indices": eval_pos,
        "_fingerprint_only": True,
    }

    print(
        f"[vdp_relaxation] T={T}, dt={dt}, {n_trajectories} traj -> "
        f"disc/val={len(disc_idx)}/{len(val_idx)}; "
        f"split_t={split_t} (train {split_t} / test {T - split_t + 1} timesteps); "
        f"mu={mu}; proc/obs noise={proc_noise_std}/{obs_noise_std}; "
        f"WARMUP_STEPS={WARMUP_STEPS}; X_eval T={T_eval_actual}, n={n_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Strip s0_-prefixed keys → initial state dict; rest is dyn params."""
    init_state = {}
    dyn_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params


def _scan_one(model_fn, y_traj, init_state, dyn_params):
    """Scan ``model_fn`` over one trajectory, returning ``(means, final_state)``.
    We keep the final carry so ``roll_state`` can hand it to the test window.

    Returns:
        - means: (T-1,) array — the one-step-ahead prediction at each step.
        - final_state: pytree of the same structure as ``init_state``.
    """
    def scan_step(state, y_prev):
        new_state, mean = model_fn(state, y_prev, dyn_params)
        return new_state, mean

    final_state, means = jax.lax.scan(scan_step, init_state, y_traj[:-1])
    return means, final_state


def apply_model(model_fn, data, params):
    """State-space scan wrapper — identical to fhn_excitable's.

    On the test window we don't want the fitted ``s0_*`` init: instead ``roll_state``
    scans the train window and stores the per-sample final latent as
    ``data["_init_carry"]``, which the test scan uses as its initial state. Absent
    the key, behaviour is the plain-scan default (this is the train-window path).
    """
    y = data["y"]
    fingerprint_only = bool(data.get("_fingerprint_only", False))
    init_carry = data.get("_init_carry")
    # None broadcasts to every sample; a per-sample carry is mapped over axis 0.
    carry_axis = None if init_carry is None else 0

    def per_sample(y_traj, p, carry):
        init_state, dyn_params = _split_params_s0(p)
        if carry is not None:
            init_state = carry
        means, _ = _scan_one(model_fn, y_traj, init_state, dyn_params)
        if fingerprint_only:
            targets = y_traj[1:]
            return targets - means
        log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
        return jnp.stack([means, log_sigma], axis=-1)

    return jax.vmap(per_sample, in_axes=(0, 0, carry_axis))(y, params, init_carry)


def roll_state(model_fn, data, params):
    """Return the per-sample final latent carry after scanning the train trials.

    Registered by the scorer (via ``TaskSpec.rollout_fn``) as the train→test
    hand-off: the returned pytree becomes ``data["_init_carry"]`` for the test
    evaluation. Scans from each sample's ``s0_*`` params — the train window always
    starts from the learnable initial state.
    """
    y = data["y"]

    def per_sample(y_traj, p):
        init_state, dyn_params = _split_params_s0(p)
        _, final_state = _scan_one(model_fn, y_traj, init_state, dyn_params)
        return final_state

    return jax.vmap(per_sample, in_axes=(0, 0))(y, params)


def loss_fn(model_output, data):
    """Per-sample Gaussian NLL over the post-warmup horizon.

    ``warmup`` defaults to the fit-time ``WARMUP_STEPS``; the test dicts override
    it to ``TEST_WARMUP_STEPS`` (=0) via ``_eval_warmup``. Only the eager test path
    supplies the key, so the jitted fit loss keeps the static default.
    """
    warmup = int(data.get("_eval_warmup", WARMUP_STEPS))
    y = data["y"][:, 1:]
    means = model_output[:, warmup:, 0]
    log_sigmas = model_output[:, warmup:, 1]
    tgt = y[:, warmup:]
    nll = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    return jnp.mean(nll, axis=-1)


# ── debug / triage helper ──


def validate_step(model_fn, default_params: dict, program_code: str = "") -> None:
    """Eager sanity check for one program's model_fn."""
    assert isinstance(default_params, dict), (
        f"DEFAULT_PARAMS must be a dict, got {type(default_params).__name__}"
    )
    assert "log_sigma_obs" in default_params, (
        "DEFAULT_PARAMS must include 'log_sigma_obs' (observation-noise log-std). "
        f"Got keys: {sorted(default_params.keys())}"
    )

    init_state, dyn_params = _split_params_s0(default_params)
    init_state_j = jax.tree_util.tree_map(jnp.asarray, init_state)
    dyn_params_j = jax.tree_util.tree_map(jnp.asarray, dyn_params)

    try:
        new_state, mean = model_fn(init_state_j, jnp.asarray(0.5), dyn_params_j)
    except Exception as e:
        raise AssertionError(
            f"model_fn(init_state, y_prev=0.5, params) raised {type(e).__name__}: {e}\n"
            f"init_state={init_state}\ndyn_params={dyn_params}\n"
            f"---program code---\n{program_code}\n---end---"
        ) from e

    init_struct = jax.tree_util.tree_structure(init_state_j)
    new_struct = jax.tree_util.tree_structure(new_state)
    assert init_struct == new_struct, (
        f"model_fn returned a state with different pytree structure than init.\n"
        f"init state:   {init_state}\n"
        f"returned:     {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x).__name__, new_state)}\n"
        "lax.scan requires the carry structure to be invariant across iterations."
    )

    mean_arr = jnp.asarray(mean)
    assert mean_arr.shape == (), (
        f"model_fn must return a scalar mean, got shape {mean_arr.shape}"
    )
    assert bool(jnp.isfinite(mean_arr)), (
        f"model_fn returned non-finite mean: {mean_arr}"
    )
