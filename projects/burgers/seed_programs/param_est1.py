import numpy as np


def parameter_estimator(data):
    """
    Estimate the constant rate c for du/dt = c under a forward-Euler step.

    Euler gives u(t+1) - u(t) = c, so the least-squares c is just the mean
    step-to-step increment over every (block, sensor, time) triple.

    data['x'] : (n_blocks, n_sensors, block_len). Axis 2 is time (consecutive steps).

    Returns:
        dict: {'c': mean increment}.
    """
    x = np.asarray(data["x"])
    inc = x[:, :, 1:] - x[:, :, :-1]
    c = float(np.mean(inc)) if inc.size else 0.0
    if not np.isfinite(c):
        c = 0.0
    return {"c": c}
