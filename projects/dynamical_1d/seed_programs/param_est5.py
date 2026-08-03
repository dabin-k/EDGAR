import numpy as np


def parameter_estimator(data):
    """Persistence-leaning SINDy init from window means only."""
    h = np.asarray(data["history"], dtype=float)
    mean_val = float(h.mean())
    return {
        "c0": 0.0,
        "c_x1": 1.0,
        "c_x2": 0.0,
        "c_x1_3": 0.0,
        "c_x1v": 0.0,
        "c_v": 0.0,
    }
