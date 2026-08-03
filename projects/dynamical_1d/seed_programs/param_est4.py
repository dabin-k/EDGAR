import numpy as np


def parameter_estimator(data):
    """Conservative EDM init (continuous weights only; d=3, k=5 fixed in model)."""
    return {
        "eps": 1e-3,
        "tau": 0.05,
        "blend": 0.0,
    }
