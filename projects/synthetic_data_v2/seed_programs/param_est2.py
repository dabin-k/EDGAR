import numpy as np


def parameter_estimator(data):
    """
    Estimate the decay rate as the least-squares slope of x(t+1) on x(t), pooled over
    all timepoints within each block.

    Args:
        data (dict): Key 'x', shape (n_blocks, block_len).

    Returns:
        dict: {"decay"}
    """
    x = np.asarray(data["x"])
    current = x[:, :-1]
    following = x[:, 1:]
    denom = np.sum(current * current)
    decay = np.sum(current * following) / denom if denom > 0 else 1.0
    return {"decay": float(decay)}
