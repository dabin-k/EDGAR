"""Shared helpers for the training objectives (plan §9, §10).

All objective ``loss_*`` functions consume the §8 ``model_output`` dict emitted by
``apply_model`` and the ``data`` dict from ``load_data``, and return **per-sample** losses
of shape ``(n,)`` (the engine wraps them in ``jnp.mean(loss_fn(...))`` and dedup/plots index
axis 0 per sample — same convention as ``fhn_excitable``). Every model-output / target tensor
carries the sample axis first: ``[n, n_stim, ...]``, so "per-sample" means reduce over every
axis except axis 0.
"""
from __future__ import annotations

import jax.numpy as jnp


def mse_loss(pred, target):
    """Plain scalar MSE (plan §9). Kept for reference / non-batched callers."""
    return jnp.mean((pred - target) ** 2)


def per_sample_mse(pred, target):
    """MSE reduced over every axis except the leading sample axis → ``(n,)``."""
    diff = (pred - target) ** 2
    return jnp.mean(diff, axis=tuple(range(1, diff.ndim)))
