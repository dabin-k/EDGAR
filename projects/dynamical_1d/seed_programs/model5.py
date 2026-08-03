import numpy as np


def model(window, params):
    """
    SINDy-style discrete update with a small polynomial library.

    Uses finite-difference velocity v ≈ x[t-1] - x[t-2] and terms common in
    oscillator / van der Pol sparse models:

        x[t] ≈ c0 + c1*x1 + c2*x2 + c3*x1^3 + c4*x1*v + c5*v
    """
    x1 = window[-1]
    x2 = window[-2] if window.shape[0] >= 2 else window[-1]
    v = x1 - x2

    return (
        params["c0"]
        + params["c_x1"] * x1
        + params["c_x2"] * x2
        + params["c_x1_3"] * x1**3
        + params["c_x1v"] * x1 * v
        + params["c_v"] * v
    )


model.DEFAULT_PARAMS = {
    "c0": 0.0,
    "c_x1": 1.0,
    "c_x2": 0.0,
    "c_x1_3": 0.0,
    "c_x1v": 0.0,
    "c_v": 0.0,
}
