"""Autoregressive evaluation wrapper.

This is the single point through which a model is called. The model never receives the
data dict — it receives a window of M past values of one cell and must predict the next
one — so it cannot read its own answer out of the input, and the window/target alignment
is written once, here, instead of in every model.

    model(data, params) -> scalar
        data['x'] : (M,), the last M steps; index -1 is the most recent.

Within a block we stack every length-M window into a Toeplitz matrix and map the model
over its rows, so each window is scored against the single step that follows it. The
first M timepoints of a block have no full window behind them and are never predicted.

Predictions are one step ahead only. The dynamics here are chaotic, so rolling a model
forward on its own output amplifies any error exponentially and the resulting loss would
say more about the Lyapunov exponent than about the model.

Windows never cross a block boundary, so `load_data`'s train/test block split is
leak-free by construction.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


M = 20  # lags visible to the model


def evaluate(model_fn, data, params):
    """Run `model_fn` over every window of every block of every sample.

    Args:
        model_fn: the model, taking a single window dict and a single sample's params.
        data: batched data dict; data['x'] is (n_samples, n_blocks, block_len).
        params: batched params pytree, leading axis n_samples.

    Returns:
        (preds, targets), both (n_samples, n_blocks, block_len - M).
    """

    def per_sample(sample, sample_params):
        return jax.vmap(partial(_predict_block, model_fn, sample_params))(sample["x"])

    return jax.vmap(per_sample)(data, params)


def _predict_block(model_fn, params, block):
    """One-step predictions within one block. block: (block_len,)."""
    n_windows = block.shape[0] - M
    # row i is [x(i), ..., x(i+M-1)] and predicts x(i+M)
    idx = jnp.arange(n_windows)[:, None] + jnp.arange(M)[None, :]
    windows = block[idx]

    preds = jax.vmap(lambda w: model_fn({"x": w}, params))(windows)
    return preds, block[M:]
