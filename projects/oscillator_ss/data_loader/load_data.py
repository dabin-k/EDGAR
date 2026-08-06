"""State-space DSL testbed: noisy oscillator with slowly drifting frequency.

Contract:
    LLM writes ``model(state, y_prev, params) -> (new_state, mean)``.
    ``params`` includes ``log_sigma_obs`` (learnable observation noise) and any
    number of ``s0_*``-prefixed keys that declare the initial-state values for
    the scan carry (framework strips the prefix and passes them as ``state``).

Framework:
    ``apply_model()`` wraps the LLM's step in ``jax.lax.scan`` over the full
    trajectory. Because ``y[t]`` is never in scope inside ``model``, leakage is a
    scope-level impossibility. ``loss_fn`` computes Gaussian NLL over the
    post-warmup horizon using ``log_sigma_obs`` broadcast across time.

Data:
    Each trajectory is a damped harmonic oscillator whose natural frequency
    ω(t) drifts as a slow sinusoid, driven by process noise and observed with
    additive Gaussian noise. Full trajectory is used for both train and test
    (no within-trajectory time split — the scan carries state).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── module-level constants (baked into loss_fn's closure, jit-safe) ──

WARMUP_STEPS: int = 50
"""Number of leading predictions ignored by ``loss_fn``. Absorbs the transient
where the learnable ``s0_*`` prior has not yet been corrected by observations.
Edit the number here to change; there is no YAML surface for this because a
data-dict key would be traced by JIT and cause ``ConcretizationTypeError``."""

TEST_WARMUP_STEPS: int = 0
"""Leading predictions ignored by ``loss_fn`` when evaluating on the *test*
window. Zero: ``roll_state`` scans the whole train window (>> WARMUP_STEPS) from
``s0_*``, so the latent carried across the boundary is already converged — the
first test prediction has no init transient to discard. The test dicts stamp this
as ``_eval_warmup``; the train (fitting) path has no such key and falls back to
``WARMUP_STEPS``. Kept symmetric with ``WARMUP_STEPS`` so both are tunable in one
place. Safe as a data-dict value because the test path is eager (never jitted),
unlike the fit loss — so no ``ConcretizationTypeError``."""


# ── data synthesis: noisy oscillator with drifting frequency ──


def _noisy_drifting_oscillator(
    rng: np.random.Generator,
    T: int,
    dt: float,
    freq_mean: float,
    freq_drift_amp: float,
    freq_drift_period: float,
    damping: float,
    process_noise_std: float,
    obs_noise_std: float,
) -> np.ndarray:
    """Discrete damped harmonic oscillator with slowly drifting natural frequency.

    Underlying continuous system:
        x'' + 2*ζ*ω(t)*x' + ω(t)^2 * x = σ_p * η(t)
    where ω(t) = freq_mean + freq_drift_amp * sin(2π * t / (freq_drift_period * dt))
    and η is unit white noise. Observation: y = x + σ_obs * ε.

    Integrated via semi-implicit Euler for stability.
    """
    t_axis = np.arange(T) * dt
    omega_t = freq_mean + freq_drift_amp * np.sin(
        2 * np.pi * t_axis / (freq_drift_period * dt)
    )

    x = np.zeros(T, dtype=np.float64)
    v = np.zeros(T, dtype=np.float64)
    x[0] = 0.5 * rng.standard_normal()
    v[0] = 0.5 * rng.standard_normal()

    for t in range(1, T):
        w = omega_t[t]
        # process noise scaled by sqrt(dt) so the SDE variance is dt-invariant
        eta = process_noise_std * rng.standard_normal() / np.sqrt(dt)
        a = -2.0 * damping * w * v[t - 1] - w * w * x[t - 1] + eta
        v[t] = v[t - 1] + dt * a
        x[t] = x[t - 1] + dt * v[t]

    y = x + obs_noise_std * rng.standard_normal(T)
    # Normalize to zero-mean unit-std for stable NLL across trajectories.
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y.astype(np.float32)


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    n_trajectories: int = 32,
    T: int = 1024,
    dt: float = 0.05,
    freq_mean: float = 1.0,
    freq_drift_amp: float = 0.3,
    freq_drift_period: float = 400.0,
    damping: float = 0.05,
    process_noise_std: float = 0.15,
    obs_noise_std: float = 0.05,
    train_frac: float = 0.5,
    T_eval: int = 100,
    n_eval: int = 4,
    seed: int = 42,
):
    """Generate noisy drifting-frequency oscillator trajectories.

    Returns the standard EDGAR five-way split. Trials (timesteps) are split into
    two contiguous chunks at ``split_t = round(T * train_frac)``: train = the
    first ``train_frac`` of the trajectory, test = the rest. Params are fit on the
    train window and cross-validated on the held-out test window; the latent state
    is carried across the boundary by ``roll_state`` (so dynamics + noise params
    generalise across time). The initial-state ``s0_*`` params are *not*
    cross-validated.
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
    if T_eval < 10:
        raise ValueError(f"T_eval ({T_eval}) too small for a discriminative fingerprint")

    rng = np.random.default_rng(seed)
    all_y = np.stack(
        [
            _noisy_drifting_oscillator(
                rng, T, dt, freq_mean, freq_drift_amp, freq_drift_period,
                damping, process_noise_std, obs_noise_std,
            )
            for _ in range(n_trajectories)
        ]
    )

    split_rng = np.random.default_rng(seed + 1)
    perm = split_rng.permutation(n_trajectories)
    disc_idx = np.sort(perm[: n_trajectories // 2])
    val_idx = np.sort(perm[n_trajectories // 2 :])

    y_disc = jnp.asarray(all_y[disc_idx])
    y_val = jnp.asarray(all_y[val_idx])

    # ── temporal train/test split (contiguous chunks; split_t set above) ──
    def _train_win(y_full):
        return y_full[:, :split_t]

    def _test_win(arr):
        return arr[:, split_t - 1:]

    # Persistence-baseline Gaussian NLL per trajectory (diagnostic for score_seeds
    # etc.; mirrors the convention in dynamical_1d which stores _persistence_mse).
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
    }
    X_val_train = {"y": _train_win(y_val)}
    X_val_test = {
        "y": _test_win(y_val),
        "_eval_warmup": TEST_WARMUP_STEPS,
        "_persistence_nll": jnp.asarray(pers_val),
    }

    # X_eval: SHORT trajectories for fingerprint dedup (avoids cosine-collapse
    # of phase-locked long trajectories). Uses same generator with a fresh rng
    # slice so eval trajectories are independent draws.
    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    eval_y = np.stack(
        [
            _noisy_drifting_oscillator(
                split_rng, T_eval, dt, freq_mean, freq_drift_amp,
                freq_drift_period, damping, process_noise_std, obs_noise_std,
            )
            for _ in range(n_eval_actual)
        ]
    )
    eval_pos = np.sort(split_rng.choice(len(disc_idx), n_eval_actual, replace=False))
    X_eval = {
        "y": jnp.asarray(eval_y),
        "_sample_indices": eval_pos,
        "_fingerprint_only": True,
    }

    print(
        f"[oscillator_ss] T={T}, dt={dt}, {n_trajectories} traj -> "
        f"disc/val={len(disc_idx)}/{len(val_idx)}; "
        f"split_t={split_t} (train {split_t} / test {T - split_t + 1} timesteps); "
        f"ω_0={freq_mean}±{freq_drift_amp} "
        f"(drift period={freq_drift_period} samples); obs_noise={obs_noise_std}; "
        f"WARMUP_STEPS={WARMUP_STEPS}; X_eval T={T_eval}, n={n_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Strip ``s0_``-prefixed keys → initial state dict; rest is dyn params.

    Only strips keys of the form ``s0_<name>`` with ``<name>`` non-empty — bare
    ``s0`` or ``s0abc`` (no separating underscore) are left in dyn_params to
    avoid false-positive collisions.
    """
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
    """State-space scan wrapper.

    ``model_fn(state, y_prev, dyn_params) -> (new_state, mean)`` is scanned
    over the trajectory. ``y_prev`` at scan iteration i is ``y[i]``; the emitted
    mean is a prediction of ``y[i+1]``. ``y[t]`` is never in scope inside
    ``model_fn`` when it predicts y[t] — that's the causality guarantee.

    Returns:
        - Normal path: ``(n_samples, T-1, 2)`` with [..., 0] = means,
          [..., 1] = broadcast ``log_sigma_obs``. Consumed by ``loss_fn``.
        - Fingerprint path (``data["_fingerprint_only"]==True``): ``(n_samples, T-1)``
          of residuals. Consumed by ``_eval_fingerprint`` for dedup.

    On the test window we don't want the fitted ``s0_*`` init: instead ``roll_state``
    scans the train window and stores the per-sample final latent as
    ``data["_init_carry"]``, which the test scan uses as its initial state. Absent
    the key, behaviour is the plain-scan default (this is the train-window path).
    """
    y = data["y"]                                # (n_samples, T)
    # Python bool → baked into trace time when jitted; True only in X_eval which
    # never enters a jit-wrapped path (verified: _eval_fingerprint is eager).
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
            # Fingerprint = RESIDUALS (y_true - predicted mean), NOT raw means.
            # Raw predictions phase-lock to the driving signal → all models look
            # highly cosine-similar (~0.99). Residuals capture what makes each
            # model DIFFERENT (its systematic error pattern) and give dedup
            # actual discriminative power.
            targets = y_traj[1:]                  # y[1..T-1]
            return targets - means                # (T-1,) residuals
        log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
        return jnp.stack([means, log_sigma], axis=-1)   # (T-1, 2)

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

    ``model_output``: ``(n_samples, T-1, 2)`` (means and log_sigma stacked).
    ``data["y"]``:    ``(n_samples, T)`` targets; scan predicts y[1..T-1].
    ``WARMUP_STEPS`` is a module-level Python constant (jit-safe, cloudpickle-
    survivable). See its docstring for why not a config value. The test dicts
    override the skip to ``TEST_WARMUP_STEPS`` via ``_eval_warmup``; only the eager
    test path supplies the key, so the jitted fit loss keeps the static default.
    """
    warmup = int(data.get("_eval_warmup", WARMUP_STEPS))
    y = data["y"][:, 1:]                         # (n_samples, T-1)
    means      = model_output[:, warmup:, 0]
    log_sigmas = model_output[:, warmup:, 1]
    tgt        = y[:, warmup:]
    nll = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    return jnp.mean(nll, axis=-1)                # (n_samples,)


# ── debug / triage helper ──


def validate_step(model_fn, default_params: dict, program_code: str = "") -> None:
    """Eager sanity check for one program's model_fn.

    Called by ``leakage_check.py`` and ``scripts/debug_program.py`` — NOT by the
    scoring pipeline (which relies on ``_worker``'s exception → inf-loss path).
    Catches the common seed-authoring bugs (missing keys, wrong return
    structure, non-finite output) with clear error messages before they become
    cryptic scan tracebacks.

    Raises ``AssertionError`` with a descriptive message on failure.
    """
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
