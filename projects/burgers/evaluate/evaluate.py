"""Autoregressive evaluation wrapper for the sensor-array benchmark.

This is the single point through which a model is called. The model never sees
the data dict for the block it is predicting — it receives a short window of the
most recent steps and must predict the next step — so it cannot read its own
answer out of the input, and the window/target alignment is written once, here,
instead of in every model.

    model(data, params) -> (n_sensors,)
        data['x'] : (n_sensors, MAX_LENGTH), most recent step in the last column

Sensors sit in a row and wrap around (periodic): sensor 0's left neighbour is
sensor n_sensors-1. A model reaches a neighbour with `np.roll(u, 1)` /
`np.roll(u, -1)`; there is no separate boundary case.

From every start index we hand the model a *true* window, then roll it forward
ROLLOUT_STEPS on its own predictions and score all of them (teacher-forced
restarts). Rolling out more than one step is deliberate: it is the stress test
for stability. A map that is plausible one step ahead but has the dynamics
slightly wrong will drift over the rollout and be punished, whereas a one-step
loss would not see the drift. It is also what exposes the missing-information
problem — the observed sensor field is a coarse view of a finer system, so a
single lag is not a sufficient statistic and a model that only looks one step
back cannot be exact.

Windows never cross a block boundary, so `load_data`'s train/test block split is
leak-free by construction.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


MAX_LENGTH = 2  # lags visible to the model (columns of data['x'])
ROLLOUT_STEPS = 5  # H: steps rolled on the model's own predictions before scoring


def evaluate(model_fn, data, params):
    """Run `model_fn` autoregressively over every sample.

    Args:
        model_fn: the model, taking a single window dict and a single sample's params.
        data: batched data dict; data['x'] is (n_samples, n_blocks, n_sensors, block_len).
        params: batched params pytree, leading axis n_samples.

    Returns:
        (preds, targets), both (n_samples, n_blocks, n_starts, ROLLOUT_STEPS, n_sensors).
    """

    def per_sample(sample, sample_params):
        return jax.vmap(partial(_rollout_block, model_fn, sample_params))(sample["x"])

    return jax.vmap(per_sample)(data, params)


def _rollout_block(model_fn, params, block):
    """Teacher-forced restarts within one block. block: (n_sensors, block_len)."""
    n_sensors, block_len = block.shape
    # start s: needs history [s-MAX_LENGTH+1 .. s] and targets [s+1 .. s+ROLLOUT_STEPS]
    starts = jnp.arange(MAX_LENGTH - 1, block_len - ROLLOUT_STEPS)

    def rollout_from(s):
        window = jax.lax.dynamic_slice(
            block, (0, s - MAX_LENGTH + 1), (n_sensors, MAX_LENGTH)
        )

        def step(w, _):
            pred = model_fn({"x": w}, params)
            return jnp.concatenate([w[:, 1:], pred[:, None]], axis=1), pred

        _, preds = jax.lax.scan(step, window, None, length=ROLLOUT_STEPS)
        targets = jax.lax.dynamic_slice(block, (0, s + 1), (n_sensors, ROLLOUT_STEPS)).T
        return preds, targets  # both (ROLLOUT_STEPS, n_sensors)

    return jax.vmap(rollout_from)(starts)
