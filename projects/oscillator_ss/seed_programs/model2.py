import numpy as np


def model(state, y_prev, params):
    """AR(1): mean at t = alpha * y[t-1].

    Slight generalisation of persistence: a learnable decay ``alpha`` toward
    zero (or toward whatever equilibrium the time series has after
    normalisation). For a well-oscillating signal this should barely beat
    persistence — it's the second rung on the ladder.
    """
    alpha = params["alpha"]
    new_state = {"y_last": y_prev}
    mean = alpha * new_state["y_last"]
    return new_state, mean


model.DEFAULT_PARAMS = {
    "alpha": 0.9,
    "log_sigma_obs": 0.0,
    "s0_y_last": 0.0,
}
