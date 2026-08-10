"""Unit tests for the state-space scan mechanics.

These tests are self-contained references for the pattern used by
``projects/oscillator_ss/data_loader/load_data.py`` (and any future
state-space project). They exercise:

1. Scan produces the correct trajectory on a hand-computed AR(1) system.
2. ISOLATION invariant — perturbing ``y[t:]`` leaves ``pred[<t]`` bit-exact.
3. ``s0_`` prefix extraction round-trips cleanly, including collision-adjacent
   names like ``s0`` (bare, no underscore) and ``s0abc`` (no underscore between
   prefix and name).
4. The stacked ``(mean, log_sigma)`` output has the shape ``(n, T-1, 2)`` and
   supports both training-loss extraction and fingerprint slicing.
5. Warmup skip in ``loss_fn`` is bit-invariant to changes in ``y[:warmup]``.
6. Gaussian NLL matches a hand-computed value for a matched normal.
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── the pattern under test: state-space scan wrapper ──


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Strip ``s0_``-prefixed keys → initial state dict; rest is dyn params.

    Only strips keys of the form ``s0_<name>`` where ``<name>`` is a non-empty
    identifier (i.e. ``s0`` alone or ``s0abc`` without an underscore separator
    is NOT stripped, avoiding false-positive collisions).
    """
    init_state = {}
    dyn_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params


def _apply_model_scan(model_fn, y_traj, params):
    """Reference implementation of the state-space apply_model kernel.

    ``model_fn(state, y_prev, dyn_params)`` returns ``(new_state, mean)``.
    Returns ``(T-1, 2)`` array with column 0 = mean, column 1 = log_sigma_obs.
    """
    init_state, dyn_params = _split_params_s0(params)

    def scan_step(state, y_prev):
        new_state, mean = model_fn(state, y_prev, dyn_params)
        return new_state, mean

    _, means = jax.lax.scan(scan_step, init_state, y_traj[:-1])
    log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
    return jnp.stack([means, log_sigma], axis=-1)


# ── seed models used across tests ──


def _ar1_model(state, y_prev, params):
    """AR(1): mean at t = alpha * y_prev; state carries y_prev for next step."""
    new_state = {"y_last": y_prev}
    mean = params["alpha"] * y_prev
    return new_state, mean


def _persistence_model(state, y_prev, params):
    return {"y_last": y_prev}, y_prev


# ── tests ──


def test_scan_ar1_matches_hand_computed():
    """AR(1) scan output must equal ``alpha * y[t-1]`` for each t."""
    T = 8
    alpha = 0.7
    y = jnp.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.0])
    params = {"alpha": alpha, "log_sigma_obs": -1.0, "s0_y_last": 0.0}

    out = _apply_model_scan(_ar1_model, y, params)  # (T-1, 2)
    expected_means = alpha * y[:-1]  # (T-1,)

    assert out.shape == (T - 1, 2)
    assert jnp.allclose(out[:, 0], expected_means)
    assert jnp.all(out[:, 1] == -1.0)  # log_sigma broadcast


def test_isolation_invariant_persistence():
    """Perturbing ``y[t:]`` leaves ``pred[<t]`` bit-exact — for a model whose
    prediction at step i uses y[i] (persistence), pred[<t] means outputs for
    scan iterations where y_prev is y[k] with k < t. Since scan feeds y[:-1],
    scan iter i sees y[i]. So changing y[t:] should leave outputs[:t] unchanged.
    """
    T = 12
    y_orig = jnp.arange(T, dtype=jnp.float32)
    y_pert = y_orig.at[5:].set(999.0)  # perturb everything from t=5

    params = {"log_sigma_obs": 0.0, "s0_y_last": 0.0}
    out_orig = _apply_model_scan(_persistence_model, y_orig, params)
    out_pert = _apply_model_scan(_persistence_model, y_pert, params)

    # Scan iter i (i in [0, T-1)) sees y_prev = y[i]; output means[i] = y[i].
    # For i < 5, y[i] was not perturbed → means[i] must be bit-exact.
    assert jnp.array_equal(out_orig[:5, 0], out_pert[:5, 0])
    # For i >= 5 the perturbation propagates.
    assert not jnp.array_equal(out_orig[5:, 0], out_pert[5:, 0])


def test_isolation_invariant_stateful_ar1():
    """Same guarantee for a stateful model: state built from y[<t] cannot
    depend on y[≥t]. Verifies the scan boundary really enforces causality."""
    T = 12
    y_orig = jnp.arange(T, dtype=jnp.float32)
    y_pert = y_orig.at[7:].set(-1000.0)

    params = {"alpha": 0.5, "log_sigma_obs": 0.0, "s0_y_last": 0.0}
    out_orig = _apply_model_scan(_ar1_model, y_orig, params)
    out_pert = _apply_model_scan(_ar1_model, y_pert, params)

    # AR(1) prediction at scan iter i uses y[i] as y_prev.
    # Predictions for i < 7 all use y[<7] which was not perturbed.
    assert jnp.array_equal(out_orig[:7, 0], out_pert[:7, 0])
    assert not jnp.array_equal(out_orig[7:, 0], out_pert[7:, 0])


def test_s0_prefix_extraction_basic():
    params = {"alpha": 0.5, "log_sigma_obs": -2.0, "s0_x": 1.0, "s0_v": 2.0}
    init_state, dyn_params = _split_params_s0(params)
    assert init_state == {"x": 1.0, "v": 2.0}
    assert dyn_params == {"alpha": 0.5, "log_sigma_obs": -2.0}


def test_s0_prefix_no_state_keys():
    params = {"alpha": 0.5, "log_sigma_obs": -2.0}
    init_state, dyn_params = _split_params_s0(params)
    assert init_state == {}
    assert dyn_params == params


def test_s0_prefix_avoids_false_positives():
    """Bare 's0' or 's0abc' (no separating underscore) must NOT be stripped."""
    params = {"s0": 1.0, "s0abc": 2.0, "s0_x": 3.0}
    init_state, dyn_params = _split_params_s0(params)
    assert init_state == {"x": 3.0}
    assert dyn_params == {"s0": 1.0, "s0abc": 2.0}


def test_stacked_output_shape_and_axes():
    T = 6
    y = jnp.arange(T, dtype=jnp.float32)
    params = {"alpha": 1.0, "log_sigma_obs": -3.0, "s0_y_last": 0.0}
    out = _apply_model_scan(_ar1_model, y, params)
    assert out.shape == (T - 1, 2)
    means = out[..., 0]
    log_sigmas = out[..., 1]
    assert means.shape == (T - 1,)
    assert log_sigmas.shape == (T - 1,)
    assert jnp.all(log_sigmas == -3.0)


def test_gaussian_nll_matches_hand_computed():
    """For a matched normal (y = mean, log_sigma = 0), per-element NLL is
    log(sigma) + 0.5 * ((y - mean)/sigma)^2 = 0 + 0 = 0. Total mean is 0."""
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    means = y
    log_sigmas = jnp.zeros_like(y)
    nll = log_sigmas + 0.5 * ((y - means) / jnp.exp(log_sigmas)) ** 2
    assert jnp.allclose(nll, 0.0)

    # Now offset y by 1: NLL per element = 0 + 0.5 * 1^2 = 0.5
    y_off = y + 1.0
    nll_off = log_sigmas + 0.5 * ((y_off - means) / jnp.exp(log_sigmas)) ** 2
    assert jnp.allclose(nll_off, 0.5)


def test_warmup_skip_is_invariant_to_early_y():
    """A loss_fn using module-level WARMUP_STEPS must ignore y[:warmup] entirely."""
    WARMUP = 3
    T = 10
    y_orig = jnp.arange(T, dtype=jnp.float32)

    # Perturb only the warmup region. If loss correctly skips warmup, the loss
    # must be bit-exact on the two inputs. Predictions are also computed FROM
    # y_prev, so persistence-model outputs in the warmup region differ — but
    # loss_fn slices those away.
    y_pert = y_orig.at[:WARMUP].set(-999.0)

    params = {"log_sigma_obs": 0.0, "s0_y_last": 0.0}
    out_orig = _apply_model_scan(_persistence_model, y_orig, params)
    out_pert = _apply_model_scan(_persistence_model, y_pert, params)

    def loss_fn_with_warmup(output, y_full, warmup):
        y = y_full[1:]  # targets are y[1..T-1]
        means = output[warmup:, 0]
        log_sigmas = output[warmup:, 1]
        tgt = y[warmup:]
        return jnp.mean(log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2)

    # After the perturbation propagates through the persistence model and gets
    # sliced away by warmup, the two losses should be identical because the
    # remaining predictions and targets are all in the untouched region.
    # Specifically: after warmup=3, scan iters 3..T-2 use y[3..T-2] which are
    # unchanged; targets y[4..T-1] are also unchanged.
    loss_orig = loss_fn_with_warmup(out_orig, y_orig, WARMUP)
    loss_pert = loss_fn_with_warmup(out_pert, y_pert, WARMUP)
    assert jnp.allclose(loss_orig, loss_pert)


def test_scan_output_dtype_is_stable():
    """apply_model output must be float32 (JAX default); no accidental object dtype."""
    T = 5
    y = jnp.arange(T, dtype=jnp.float32)
    params = {"alpha": 0.5, "log_sigma_obs": -1.0, "s0_y_last": 0.0}
    out = _apply_model_scan(_ar1_model, y, params)
    assert out.dtype in (jnp.float32, jnp.float64)


def test_scan_vmap_over_samples():
    """apply_model wrapped in vmap over samples produces (n_samples, T-1, 2)."""
    n_samples = 4
    T = 6
    y_batch = jnp.arange(n_samples * T, dtype=jnp.float32).reshape(n_samples, T)
    params_batch = {
        "alpha": jnp.full((n_samples,), 0.5),
        "log_sigma_obs": jnp.full((n_samples,), -1.0),
        "s0_y_last": jnp.zeros((n_samples,)),
    }

    def per_sample(y_traj, p):
        return _apply_model_scan(_ar1_model, y_traj, p)

    out = jax.vmap(per_sample, in_axes=(0, 0))(y_batch, params_batch)
    assert out.shape == (n_samples, T - 1, 2)


def test_scan_edge_case_T2():
    """T=2: scan runs one iteration → one prediction of y[1] from y[0]."""
    y = jnp.array([1.0, 2.0])
    params = {"alpha": 0.5, "log_sigma_obs": 0.0, "s0_y_last": 0.0}
    out = _apply_model_scan(_ar1_model, y, params)
    assert out.shape == (1, 2)
    assert jnp.allclose(out[0, 0], 0.5 * 1.0)


def test_scan_rejects_T1():
    """T=1 means y[:-1] is empty; scan is well-defined but produces no output.

    This test documents the expected behavior: apply_model with T=1 returns
    a (0, 2) array. Data loaders must ensure T >= 2 to have any training signal.
    """
    y = jnp.array([1.0])
    params = {"alpha": 0.5, "log_sigma_obs": 0.0, "s0_y_last": 0.0}
    out = _apply_model_scan(_ar1_model, y, params)
    assert out.shape == (0, 2)


def test_isolation_is_bit_exact_not_approximate():
    """Structural leakage guarantee requires BIT-EXACT invariance, not just close.

    Any observable difference in pred[<t] under y[t:] perturbation would signal
    a broken abstraction. Using np.array_equal (not allclose) to enforce this.
    """
    T = 20
    y_orig = jnp.array(np.random.default_rng(0).standard_normal(T), dtype=jnp.float32)
    y_pert = y_orig.at[10:].set(-42.0)

    params = {"alpha": 0.3, "log_sigma_obs": -1.5, "s0_y_last": 0.0}
    out_orig = np.asarray(_apply_model_scan(_ar1_model, y_orig, params))
    out_pert = np.asarray(_apply_model_scan(_ar1_model, y_pert, params))

    assert np.array_equal(out_orig[:10], out_pert[:10])
