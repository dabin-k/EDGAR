import numpy as np


def model(state, y_prev, params):
    """Persistence baseline: predict y[t] = y[t-1] with learnable observation noise.

    The trivial 1-D seed. Cannot represent limit-cycle dynamics at all — its
    predicted mean is always the last observation, so it pays a huge NLL penalty
    on the fast swings through zero and mispredicts the sign of change at
    turning points. Any real model must beat this.
    """
    new_state = {"y_last": y_prev}
    mean = new_state["y_last"]
    return new_state, mean


model.DEFAULT_PARAMS = {
    "log_sigma_obs": 0.0,
    "s0_y_last": 0.0,
}
