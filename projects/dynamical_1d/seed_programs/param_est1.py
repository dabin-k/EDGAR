import numpy as np


def parameter_estimator(data):
    """Estimate persistence parameters from past windows only."""
    h = np.asarray(data["history"], dtype=float)
    mean_val = float(h.mean())
    return {"a": 1.0, "b": 0.0}
