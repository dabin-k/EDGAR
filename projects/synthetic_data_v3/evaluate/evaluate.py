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


def evaluate(model_fn, data, params, input_sequence_length, rollout_steps):
    """Run `model_fn` autoregressively over every sample.

    Args:
        model_fn: the model, taking a single window dict and a single sample's params.
        data: batched data dict; data['x'] is (n_samples, n_blocks, n_cells, block_len).
        params: batched params pytree, leading axis n_samples.

    Returns:
        (preds, targets), both (n_samples, n_blocks, n_starts, rollout_steps, n_cells).
    """

    def per_sample(sample, sample_params):
        predict = partial(
                    _rollout_block,
                    model_fn,
                    sample_params,
                    input_sequence_length=input_sequence_length,
                    rollout_steps=rollout_steps,
                )
        return jax.vmap(predict)(sample["x"])

    return jax.vmap(per_sample)(data, params)


def _rollout_block(model_fn, params, block, input_sequence_length, rollout_steps):
    """Teacher-forced restarts within one block. block: (n_cells, block_len)."""
    n_cells, block_len = block.shape
    # start s: needs history [s-MAX_LENGTH+1 .. s] and targets [s+1 .. s+ROLLOUT_STEPS]
    starts = jnp.arange(input_sequence_length - 1, block_len - rollout_steps)

    def rollout_from(s):
        window = jax.lax.dynamic_slice(
            block, (0, s - input_sequence_length + 1), (n_cells, input_sequence_length)
        )

        def step(w, _):
            pred = model_fn({"x": w}, params)
            return jnp.concatenate([w[:, 1:], pred[:, None]], axis=1), pred

        _, preds = jax.lax.scan(step, window, None, length=rollout_steps)
        targets = jax.lax.dynamic_slice(block, (0, s + 1), (n_cells, rollout_steps)).T
        return preds, targets  # both (rollout_steps, n_cells)

    return jax.vmap(rollout_from)(starts)
