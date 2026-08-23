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


def response_features(y, time_axis):
    """Macroscopic E-response signatures (plan §10-D), jit-safe.

    ``y`` is ``[..., T, 2]`` (last axis (E, I)); ``time_axis`` is the shared ``[T]`` grid in
    seconds. Returns ``[..., 3]``: (suppression_area, rebound, late_offset) on the E trace,
    with baseline ~ 1. The plan's reference uses boolean mask indexing (``e[..., post]``),
    which is not traceable under ``jit``/``grad`` because the post/late windows have a
    data-dependent length; we use masked reductions with the same semantics instead.
    """
    e = y[..., 0]                                    # [..., T]

    post = time_axis >= 0.0                          # [T]
    late = time_axis >= 0.8

    suppression_area = _masked_mean(jax.nn.relu(1.0 - e), post, axis=-1)
    rebound = _masked_soft_max(e, post, axis=-1)
    late_offset = _masked_mean(e - 1.0, late, axis=-1)

    return jnp.stack([suppression_area, rebound, late_offset], axis=-1)
