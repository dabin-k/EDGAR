"""Synthetic autoregressive benchmark: a scalar non-linear map, one map per cell.

Ground truth (NOT seen by the LLM):

    x(t+1) = A cos(p x(t)) sin(q x(t)) + eps

A, p, q are drawn independently per cell, so every cell follows the same functional
form with its own coefficients. The parameter ranges put every cell on a bounded
chaotic attractor: the fixed point at x = 0 has slope f'(0) = A q > 1, so it repels,
and the trajectory wanders over the whole range of x instead of settling. That is what
makes the equation identifiable — a trajectory that collapses to a fixed point only
ever visits one x, and the shape of f there is indistinguishable from its tangent line.
The first `burn_in` steps are discarded so no block contains the initial transient and
every block is drawn from the same stationary distribution.

Data layout. A *sample* is one cell; parameters are fitted per cell. Each cell's time
axis is cut into non-overlapping blocks, and blocks alternate between the train and
test splits. Model windows never cross a block boundary, so no test timepoint can be
reached from a train window.

    x: (n_samples, n_blocks, block_len)

`data_path` is unused: the data is generated deterministically from `random_seed`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def load_data(
    data_path: str,
    n_cells: int = 32,
    n_times: int = 500,
    burn_in: int = 200,
    process_noise: float = 0.005,
    obs_noise: float = 0.0,
    block_len: int = 100,
    random_seed: int = 0,
    n_eval_samples: int = 4,
):
    """Simulate the per-cell maps and split them for EDGAR.

    Returns:
        (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval.
        Cells split 50/50 into discover/validate; within a cell, time blocks
        alternate between train and test.
    """
    x = _simulate(n_cells, n_times, burn_in, process_noise, obs_noise, random_seed)

    n_blocks = n_times // block_len
    blocks = x[:, : n_blocks * block_len].reshape(n_cells, n_blocks, block_len)

    train_blocks = np.arange(0, n_blocks, 2)
    test_blocks = np.arange(1, n_blocks, 2)

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_cells)
    disc_idx = np.sort(perm[: n_cells // 2])
    val_idx = np.sort(perm[n_cells // 2 :])

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

    Args:
        preds: (n_samples, n_blocks, n_windows), one-step predictions.
        targets: same shape, the observed next step.

    Returns:
        (n_samples,) — the mean over every block and window of one cell.
    """
    return jnp.mean((preds - targets) ** 2, axis=(1, 2))


# ── ground truth simulator ──


def _simulate(n_cells, n_times, burn_in, process_noise, obs_noise, seed):
    """Iterate x -> A cos(p x) sin(q x) once per cell, discarding the transient."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(1.8, 2.5, size=n_cells)
    p = rng.uniform(1.5, 3.0, size=n_cells)
    q = rng.uniform(1.5, 2.0, size=n_cells)

    out = np.zeros((n_cells, n_times))
    x = 0.5 * rng.standard_normal(n_cells)
    for t in range(-burn_in, n_times):
        if t >= 0:
            out[:, t] = x
        x = A * np.cos(p * x) * np.sin(q * x) + process_noise * rng.standard_normal(
            n_cells
        )

    if obs_noise > 0:
        out = out + obs_noise * rng.standard_normal(out.shape)
    return out


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}
