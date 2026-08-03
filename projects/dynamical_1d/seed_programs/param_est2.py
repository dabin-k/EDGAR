import numpy as np


def parameter_estimator(data):
    """Rough harmonic init from past windows only."""
    h = np.asarray(data["history"], dtype=float)
    W = h.shape[-1]
    mean_val = float(h.mean())
    return {
        "omega": float(2.0 * np.pi / max(W, 1)),
        "offset": mean_val,
        "A1": 0.05,
        "B1": 0.05,
        "A2": 0.02,
        "B2": 0.02,
    }
