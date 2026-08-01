import numpy as np


def model(state, y_prev, params):
    """Damped harmonic oscillator with Kalman-style observation correction.

    Latent state (x, v) evolves under a fixed-omega damped oscillator; each new
    observation y_prev pulls the state toward what it should be if the last
    prediction had been correct. This captures the oscillator's rhythm but has
    no mechanism to track a drifting frequency.
    """
    x = state["x"]
    v = state["v"]

    omega = params["omega"]
    damping = params["damping"]
    dt = params["dt"]
    k_gain = params["k_gain"]                    # observation-correction gain

    # 1. Observation-driven correction: nudge x toward y_prev.
    innovation = y_prev - x
    x_corr = x + k_gain * innovation
    v_corr = v                                    # keep v unchanged during correction

    # 2. One semi-implicit Euler step of x'' + 2*damping*omega*x' + omega^2*x = 0.
    acc = -2.0 * damping * omega * v_corr - omega * omega * x_corr
    v_new = v_corr + dt * acc
    x_new = x_corr + dt * v_new

    new_state = {"x": x_new, "v": v_new}
    mean = x_new                                  # predicted next observation
    return new_state, mean


model.DEFAULT_PARAMS = {
    "omega": 1.0,
    "damping": 0.1,
    "dt": 0.05,
    "k_gain": 0.3,
    "log_sigma_obs": -1.0,
    "s0_x": 0.0,
    "s0_v": 0.0,
}
