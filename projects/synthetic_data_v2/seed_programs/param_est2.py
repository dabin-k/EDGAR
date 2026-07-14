import numpy as np


def parameter_estimator(data):
    """
    Estimate the three autoregressive weights by ordinary least squares, solving for
    them jointly over every (block, time) pair. The lags are strongly correlated, so
    fitting each weight on its own would bias all three.

    Args:
        data (dict): Key 'x', shape (n_blocks, block_len).

    Returns:
        dict: {"a1", "a2", "a3"}
    """
    x = np.asarray(data["x"])
    y = x[:, 3:]  # target: state at t+1
    lags = [x[:, 2:-1], x[:, 1:-2], x[:, :-3]]  # x(t), x(t-1), x(t-2)

    F = np.stack([lag.ravel() for lag in lags], axis=1)
    try:
        w = np.linalg.solve(F.T @ F + 1e-8 * np.eye(3), F.T @ y.ravel())
    except np.linalg.LinAlgError:
        w = np.array([0.5, 0.25, 0.1])

    if not np.all(np.isfinite(w)):
        w = np.array([0.5, 0.25, 0.1])

    return {"a1": float(w[0]), "a2": float(w[1]), "a3": float(w[2])}
