"""Objective F — autonomous multi-step rollout Gaussian NLL (deterministic state-space model).

The simplest probabilistic state-space model: a *deterministic* WC(S) forward run with a Gaussian
observation model on the readout,

    z_t   = F_theta(z_{t-1}, stim_{t-1})     (deterministic; no process noise)
    y_obs = H z_t + N(0, sigma^2)            (all residual attributed to observation noise)

scored by the negative log-likelihood of the observations. Unlike objective E (the extended
Kalman filter), there is no process noise, no covariance propagation and no data-driven latent
correction — the latent evolves purely from its own dynamics. This keeps the model fully
interpretable (readable dynamics, a single reported noise level per channel) at the cost of the
assumption that the dynamics are exact and only the *observations* are noisy. It is the
gain->0 endpoint of the EKF; comparing F against E tells us whether that assumption holds
(if E learns negligible process noise, F is justified).

**Not teacher-forced.** Like objectives B/C/D this reads ``pred_y_rollout`` — the free-running
autonomous rollout from each anchor (the model feeds its own predictions back, §7). Observed data
is used only to synchronise the latent at each anchor, never fed into the transition during the
scored window, so persistence is not an available solution (that was the failure of the
teacher-forced one-step objective A on smooth noisy PSTHs; see journal 2026-08-25).

**Profiled (concentrated) variance.** Rather than carry a learnable ``log_sigma_obs`` param (which
would need its own model-file variant and would inflate ``n_params``), we concentrate the Gaussian
variance out analytically: for a Gaussian with unknown variance the MLE is
``sigma_c^2 = mean(residual_c^2)`` per channel, and plugging it back gives the profile NLL
``0.5*(log(2*pi*sigma_c^2) + 1)``. This is exactly the optimum a free ``sigma`` would converge to,
and equals ``0.5*log(MSE_c)`` up to a constant — i.e. a *per-channel* log-MSE. The per-channel
variance normalisation is the concrete gain over objective B's raw summed MSE, which is otherwise
dominated by whichever of E/I has the larger magnitude. The fitted noise std is recoverable for
reporting as ``sqrt(mean(residual_c^2))``.

To switch to an explicit fixed or learnable ``sigma`` instead of profiling, replace ``sigma2``
below with the fixed value / a ``params``-supplied ``log_sigma`` (the latter also needs the key
added to the model's ``DEFAULT_PARAMS``).
"""
from __future__ import annotations

import jax.numpy as jnp

# Floor on the per-channel MLE variance: guards log(0) if a rollout fits a channel near-perfectly.
# Noisy neural data keeps the variance well above this; it only ever bites in degenerate cases.
_SIGMA2_FLOOR = 1e-8


def loss_F_rollout_nll(model_output, data):
    """Free-running rollout profile Gaussian NLL, per sample ``(n,)``.

    ``pred_y_rollout`` and ``target_y_future`` are both ``[n, n_stim, A, K, 2]`` (A anchors,
    K-step horizon, last axis (E, I)) — identical tensors to objective B, scored under a Gaussian
    observation model instead of MSE. The per-channel variance is the MLE over every non-sample,
    non-channel axis (conditions, anchors, horizon), so each sample reports its own E/I noise
    level. Returns the summed-over-channels NLL per sample (nats/bin).
    """
    pred = model_output["pred_y_rollout"]       # [n, n_stim, A, K, 2]
    target = data["target_y_future"]            # [n, n_stim, A, K, 2]

    resid2 = (pred - target) ** 2
    # MLE observation variance per (sample, channel): mean sq residual over (n_stim, A, K).
    reduce_axes = tuple(range(1, resid2.ndim - 1))         # every axis but sample (0) and channel (-1)
    sigma2 = jnp.mean(resid2, axis=reduce_axes)            # [n, 2]
    sigma2 = jnp.clip(sigma2, _SIGMA2_FLOOR, None)

    nll_per_channel = 0.5 * (jnp.log(2.0 * jnp.pi * sigma2) + 1.0)   # [n, 2]
    return jnp.sum(nll_per_channel, axis=-1)               # [n]
