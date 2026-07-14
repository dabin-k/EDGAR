"""Autoregressive evaluation wrapper.

This is the single point through which a model is called. The model never receives
the data dict — it receives a window of past activity and must predict the next
step — so it cannot read its own answer out of the input, and the window/target
alignment is written once, here, instead of in every model.

    model(data, params) -> (n_cells,)
        data['x'] : (n_cells, MAX_LENGTH), most recent step in the last column

From every start index we hand the model a true window, roll it forward
ROLLOUT_STEPS on its own predictions, and score all of them (teacher-forced
restarts). Rolling out more than one step is not a nicety: with a hidden slow
variable, process noise is correlated across steps in the observed variable, and a
one-step-ahead loss rewards models that exploit that correlation over models that
have the dynamics right.

Windows never cross a block boundary, so `load_data`'s train/test block split is
leak-free by construction.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


MAX_LENGTH = 2  # lags visible to the model
ROLLOUT_STEPS = 3  # H: steps rolled on the model's own predictions


def evaluate(model_fn, data, params):
    """Run `model_fn` autoregressively over every sample.

    Args:
        model_fn: the model, taking a single window dict and a single sample's params.
        data: batched data dict; data['x'] is (n_samples, n_blocks, n_cells, block_len).
        params: batched params pytree, leading axis n_samples.

    Returns:
        (preds, targets), both (n_samples, n_blocks, n_starts, ROLLOUT_STEPS, n_cells).
    """

    def per_sample(sample, sample_params):
        return jax.vmap(partial(_rollout_block, model_fn, sample_params))(sample["x"])

    return jax.vmap(per_sample)(data, params)


def _rollout_block(model_fn, params, block):
    """Teacher-forced restarts within one block. block: (n_cells, block_len)."""
    n_cells, block_len = block.shape
    # start s: needs history [s-MAX_LENGTH+1 .. s] and targets [s+1 .. s+ROLLOUT_STEPS]
    starts = jnp.arange(MAX_LENGTH - 1, block_len - ROLLOUT_STEPS)

    def rollout_from(s):
        window = jax.lax.dynamic_slice(
            block, (0, s - MAX_LENGTH + 1), (n_cells, MAX_LENGTH)
        )

        def step(w, _):
            pred = model_fn({"x": w}, params)
            return jnp.concatenate([w[:, 1:], pred[:, None]], axis=1), pred

        _, preds = jax.lax.scan(step, window, None, length=ROLLOUT_STEPS)
        targets = jax.lax.dynamic_slice(block, (0, s + 1), (n_cells, ROLLOUT_STEPS)).T
        return preds, targets  # both (ROLLOUT_STEPS, n_cells)

    return jax.vmap(rollout_from)(starts)
