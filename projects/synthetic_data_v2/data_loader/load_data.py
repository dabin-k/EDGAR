"""Synthetic autoregressive benchmark: population activity with a hidden slow variable.

Generates data from a latent system in which each cell carries an unobserved slow
current. Only the activity is observed, so the dynamics are not Markovian in the
observed variable — a model must look back more than one step to do well.

Ground truth (NOT seen by the LLM; derivation and baselines in
journal/2026-07-13_ar_synth.py):

    x_i(t+1) = A x_i(t) + dt ( sum_j K(i-j) phi(x_j(t)) - g a_i(t) ) + eps
    a_i(t+1) = B a_i(t) + dt x_i(t)                                  [hidden]

Data layout. A *sample* is one recording; parameters are fitted per recording.
Each recording's time axis is cut into non-overlapping blocks, and blocks alternate
between the train and test splits, so both splits contain both the transient and the
steady-state regime. Autoregressive windows never cross a block boundary, so no test
timepoint can be reached from a train window.

    x: (n_samples, n_blocks, n_cells, block_len)

`data_path` is unused: the data is generated deterministically from `random_seed`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


TRUE_PARAMS = dict(tau_x=1.0, tau_a=8.0, g=0.6, J_e=20.0, sigma=0.6, J_i=3.0)


def load_data(
    data_path: str,
    n_cells: int = 32,
    n_times: int = 250,
    n_recordings: int = 16,
    dt: float = 0.1,
    process_noise: float = 0.005,
    obs_noise: float = 0.0,
    block_len: int = 50,
    random_seed: int = 0,
    n_eval_samples: int = 4,
):
    """Simulate the latent system and split it for EDGAR.

    Returns:
        (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval.
        Recordings split 50/50 into discover/validate; within a recording, time
        blocks alternate between train and test.
    """
    x = _simulate(
        n_cells, n_times, n_recordings, dt, process_noise, obs_noise, random_seed
    )

    n_blocks = n_times // block_len
    blocks = x[:, :, : n_blocks * block_len].reshape(
        n_recordings, n_cells, n_blocks, block_len
    )
    blocks = blocks.transpose(0, 2, 1, 3)  # (n_rec, n_blocks, n_cells, block_len)

    train_blocks = np.arange(0, n_blocks, 2)
    test_blocks = np.arange(1, n_blocks, 2)

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_recordings)
    disc_idx = np.sort(perm[: n_recordings // 2])
    val_idx = np.sort(perm[n_recordings // 2 :])

    X_disc_train = {"x": blocks[disc_idx][:, train_blocks]}
    X_disc_test = {"x": blocks[disc_idx][:, test_blocks]}
    X_val_train = {"x": blocks[val_idx][:, train_blocks]}
    X_val_test = {"x": blocks[val_idx][:, test_blocks]}

    eval_pos = np.sort(
        rng.choice(len(disc_idx), min(n_eval_samples, len(disc_idx)), replace=False)
    )
    X_eval = {
        "x": X_disc_train["x"][eval_pos],
        "_sample_indices": eval_pos,
    }

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def loss_fn(preds, targets):
    """Mean squared error per sample.

    `evaluate` has already aligned predictions with their targets, so this does no
    indexing of its own. Both arrays are (n_samples, n_blocks, n_starts, horizon,
    n_cells); reduce everything but the sample axis.
    """
    return jnp.mean((preds - targets) ** 2, axis=(1, 2, 3, 4))


# ── ground truth simulator ──


def _kernel(n_cells, J_e, sigma, J_i):
    """Mexican-hat coupling on a ring, mean-field scaled."""
    theta = 2 * np.pi * np.arange(n_cells) / n_cells
    d = theta[:, None] - theta[None, :]
    d = np.arctan2(np.sin(d), np.cos(d))
    return (J_e * np.exp(-(d**2) / (2 * sigma**2)) - J_i) / n_cells


def _simulate(n_cells, n_times, n_recordings, dt, process_noise, obs_noise, seed):
    """Recordings start from a random field and the transient is kept.

    Starting from a settled bump collapses the trajectories onto a low-dimensional
    manifold where the dynamics look linear, and an unconstrained VAR then beats the
    true equation. The transient is what exercises the nonlinearity.
    """
    p = TRUE_PARAMS
    rng = np.random.default_rng(seed)
    A = 1 - dt / p["tau_x"]
    B = 1 - dt / p["tau_a"]
    K = _kernel(n_cells, p["J_e"], p["sigma"], p["J_i"])

    out = np.zeros((n_recordings, n_cells, n_times))
    for r in range(n_recordings):
        x = 1.5 * rng.standard_normal(n_cells)
        a = np.zeros(n_cells)
        for t in range(n_times):
            out[r, :, t] = x
            phi = np.tanh(np.maximum(x, 0.0))
            x_next = (
                A * x
                + dt * (K @ phi - p["g"] * a)
                + process_noise * rng.standard_normal(n_cells)
            )
            a = B * a + dt * x
            x = x_next

    if obs_noise > 0:
        out = out + obs_noise * rng.standard_normal(out.shape)
    return out


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}
