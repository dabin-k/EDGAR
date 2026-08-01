import numpy as np


def model(state, y_prev, params):
    """Persistence baseline: predict y[t] = y[t-1], with learnable observation noise.

    State-space contract:
      * state carries the previous observation
      * update: state["y_last"] := y_prev
      * predict: mean of next observation = y_last (i.e., y_prev)

    y[t] is NEVER in scope — the scan feeds y_prev = y[t-1] and we predict y[t].
    Any real dynamical model in this project must beat this baseline.
    """
    new_state = {"y_last": y_prev}
    mean = new_state["y_last"]
    return new_state, mean


model.DEFAULT_PARAMS = {
    "log_sigma_obs": 0.0,   # observation noise std ≈ 1 initially
    "s0_y_last": 0.0,       # initial state: y_last = 0 before any observation
}
