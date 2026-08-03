import numpy as np


def model(window, params):
    """
    Persistence / AR(1) baseline for 1D next-value regression.

    Predicts x[t] from the most recent past value x[t-1] = window[-1].
    This is the trivial baseline the evolutionary search must beat on rollout.
    """
    a = params["a"]
    b = params["b"]
    return a * window[-1] + b


model.DEFAULT_PARAMS = {"a": 1.0, "b": 0.0}
