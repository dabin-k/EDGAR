"""Shared helpers for the four training objectives (plan §9, §10).

All objective ``loss_*`` functions consume the §8 ``model_output`` dict emitted by
``apply_model`` and the ``data`` dict from ``load_data``, and return **per-sample** losses
of shape ``(n,)`` (the engine wraps them in ``jnp.mean(loss_fn(...))`` and dedup/plots index
axis 0 per sample — same convention as ``fhn_excitable``). Every model-output / target tensor
carries the sample axis first: ``[n, n_stim, ...]``, so "per-sample" means reduce over every
axis except axis 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def mse_loss(pred, target):
    """Plain scalar MSE (plan §9). Kept for reference / non-batched callers."""
    return jnp.mean((pred - target) ** 2)


def per_sample_mse(pred, target):
    """MSE reduced over every axis except the leading sample axis → ``(n,)``."""
    diff = (pred - target) ** 2
    return jnp.mean(diff, axis=tuple(range(1, diff.ndim)))


def soft_max(x, temperature=20.0, axis=-1):
    """Differentiable soft maximum over ``axis`` (plan §10-D)."""
    w = jax.nn.softmax(temperature * x, axis=axis)
    return jnp.sum(w * x, axis=axis)


def _masked_mean(x, mask, axis=-1):
    """Mean of ``x`` over ``axis`` restricted to ``mask`` (jit-safe, no boolean indexing)."""
    mask = mask.astype(x.dtype)
    return jnp.sum(x * mask, axis=axis) / jnp.clip(jnp.sum(mask, axis=axis), 1.0)


def _masked_soft_max(x, mask, temperature=20.0, axis=-1):
    """``soft_max`` of ``x`` over ``axis`` restricted to ``mask`` (masked softmax)."""
    neg_inf = jnp.asarray(-1e30, dtype=x.dtype)
    logits = jnp.where(mask, temperature * x, neg_inf)
    w = jax.nn.softmax(logits, axis=axis)
    return jnp.sum(w * x, axis=axis)


SUPPRESSION_THRESHOLD = 0.2

# "Rebound-ish": amplitude above REBOUND_LOWER, softly gated to exclude pulse-driven peaks above
# REBOUND_UPPER. The upper gate is a logistic (centred at REBOUND_UPPER, sharpness REBOUND_GATE_SHARPNESS)
REBOUND_LOWER = 1.2
REBOUND_UPPER = 2.0
REBOUND_GATE_SHARPNESS = 10.0


def _rebound_ish(x, mask):
    """Mean over ``mask`` of amplitude-above-``REBOUND_LOWER``, soft-gated below ``REBOUND_UPPER``."""
    upper_gate = jax.nn.sigmoid(REBOUND_GATE_SHARPNESS * (REBOUND_UPPER - x))
    return _masked_mean(jax.nn.relu(x - REBOUND_LOWER) * upper_gate, mask, axis=-1)


def response_features(y, time_axis):
    """Macroscopic E/I response signatures (plan §10-D), jit-safe.

    ``y`` is ``[..., T, 2]`` (last axis (E, I)); ``time_axis`` is the shared ``[T]`` grid in
    seconds. Returns ``[..., 4]``: (E_suppression, E_rebound, I_suppression, I_rebound), on the
    post-stimulus window (``t >= 0``), baseline ~ 1.

    Revised 2026-08-25. The original ``late_offset`` was dropped (its ``t >= 0.8 s`` window is never
    populated by the ``chop``, max ~0.4 s, so it was identically 0). ``rebound`` is no longer a
    soft-max of the raw trace (which latched onto single-bin noise peaks); instead each channel
    contributes two features:
      * ``suppression`` — area driven below ``SUPPRESSION_THRESHOLD`` (deep silencing).
      * ``rebound``     — amplitude above ``REBOUND_LOWER``, softly gated below ``REBOUND_UPPER``
        so pulse-driven peaks are excluded (``_rebound_ish``). Amplitude-weighted (a stronger
        rebound counts more), with a smooth logistic upper cutoff.

    The plan's reference uses boolean mask indexing (``e[..., post]``), not traceable under
    ``jit``/``grad`` (data-dependent window length); we use masked reductions with the same
    semantics instead.
    """
    e = y[..., 0]                                    # [..., T]
    i = y[..., 1]                                    # [..., T]
    post = time_axis >= 0.0                          # [T]

    E_suppression_area = _masked_mean(jax.nn.relu(SUPPRESSION_THRESHOLD - e), post, axis=-1)
    E_rebound_ish = _rebound_ish(e, post)
    I_suppression_area = _masked_mean(jax.nn.relu(SUPPRESSION_THRESHOLD - i), post, axis=-1)
    I_rebound_ish = _rebound_ish(i, post)
    return jnp.stack([E_suppression_area, E_rebound_ish, I_suppression_area, I_rebound_ish], axis=-1)  # [..., 4]
