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


def per_sample_masked_mse(pred, target, mask=None):
    """MSE over real rows only → ``(n,)``.

    ``pred``/``target`` are ``[n, C, ...]``; ``mask`` is ``[n, C]`` (1 = real trial, 0 = padding,
    see ``load_data._load_real``). Each sample averages over its own real rows, so padding never
    changes the loss. ``mask=None`` (synthetic data has no padding) is plain ``per_sample_mse``.
    """
    if mask is None:
        return per_sample_mse(pred, target)
    diff = (pred - target) ** 2
    m = mask.reshape(mask.shape + (1,) * (diff.ndim - mask.ndim))
    per_row = diff[0, 0].size
    return jnp.sum(diff * m, axis=tuple(range(1, diff.ndim))) / (jnp.sum(mask, axis=1) * per_row)


def per_population_masked_mse(pred, target, mask=None):
    """Masked MSE kept per population (last axis, = (E, I)) → ``(n, 2)``; otherwise as
    ``per_sample_masked_mse``."""
    diff = (pred - target) ** 2
    axes = tuple(range(1, diff.ndim - 1))
    if mask is None:
        return jnp.mean(diff, axis=axes)
    m = mask.reshape(mask.shape + (1,) * (diff.ndim - mask.ndim))
    per_row = diff[0, 0, ..., 0].size
    return jnp.sum(diff * m, axis=axes) / (jnp.sum(mask, axis=1)[:, None] * per_row)


def per_sample_loss(pred, target, data):
    """The objectives' per-sample loss ``(n,)``: each population's masked MSE over real trials,
    divided by that population's ``loss_scale``, averaged over E and I.

    ``loss_scale`` ``(n, 2)`` is the sample's mean per-trial variance over its TRAIN trials, per
    channel, carried on both the train and test dicts (``load_data._loss_scale``, which validates
    it). The loss is therefore the mean over E and I of each population's fraction of variance
    unexplained (0 = perfect, 1 = no better than each trial's own mean). Normalising per population
    keeps the higher-variance population (I) from dominating the fit.
    """
    fvu = per_population_masked_mse(pred, target, data.get("mask")) / data["loss_scale"]
    return jnp.mean(fvu, axis=-1)
