"""Objective E — latent state-space marginal NLL via an extended Kalman filter.

Scientific question: once the transition is *not* handed the raw observation each step (so it
can't collapse to persistence), can the WCS dynamics explain the smooth PSTH while attributing
the trial-to-trial wriggle to observation noise? Unlike objectives A–D (all MSE variants on a
deterministic run of the transition), this objective fits a probabilistic state-space model

    z_t   = F_theta(z_{t-1}, stim_{t-1}) + process_noise      z = [E, I, S]
    y_obs = H z_t + obs_noise                                 H picks (E, I); S latent

and trains on the **marginal** negative log-likelihood of the observations, computed by the EKF
recursion in ``apply_model`` (``_apply_model_kalman``). The data enters only through the
innovations, never the forward transition — so persistence is not an available solution.

``model_output`` here carries the per-step innovations and their covariances (the prediction-
error decomposition of the log-likelihood), plus ``pred_y_1step`` for plotting / eval fingerprint.
The first ``EDGAR_WC_WARMUP_BINS`` steps are dropped from the loss, matching objective A.
"""
from __future__ import annotations

import os

import jax.numpy as jnp


def loss_E_kalman(model_output, data):
    """EKF marginal negative log-likelihood, per sample ``(n,)``.

    ``innovations`` is ``[n, n_stim, T-1, 2]`` (r_t = y_obs[t] - H m_t^-, aligned to the
    ``target_y[1:]`` slice like ``pred_y_1step``); ``innovation_cov`` is ``[n, n_stim, T-1, 2, 2]``
    (S_t = H P_t^- H^T + Sigma). The per-step NLL is 0.5*(r^T S^{-1} r + logdet(2*pi*S)); the
    logdet term is what makes the process/observation noise scales identifiable (without it the
    fit would drive the innovation covariance to zero). Reduced by mean over (conditions, time)
    so the magnitude is comparable to the MSE objectives and the reduction matches
    ``per_sample_mse`` (mean over every axis except the leading sample axis).
    """
    w = max(0, int(os.environ.get("EDGAR_WC_WARMUP_BINS", "0")))
    r = model_output["innovations"][:, :, w:, :]           # [n, n_stim, T-1-w, 2]
    S = model_output["innovation_cov"][:, :, w:, :, :]     # [n, n_stim, T-1-w, 2, 2]

    Sinv = jnp.linalg.inv(S)
    quad = jnp.einsum("...i,...ij,...j->...", r, Sinv, r)  # [n, n_stim, T-1-w]
    logdet = jnp.linalg.slogdet(2.0 * jnp.pi * S)[1]       # [n, n_stim, T-1-w]
    nll = 0.5 * (quad + logdet)                            # per-step NLL

    return jnp.mean(nll, axis=tuple(range(1, nll.ndim)))   # -> (n,)
