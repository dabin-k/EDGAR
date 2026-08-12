import numpy as np

H = 1 / 30000  # integration step (matches the data-generating grid)


def model(state, y_prev, params):
    """Linear coupled E/I circuit (recurrent push-pull).

    Adds the recurrent weight matrix to the leaky integrators: E excites both
    populations, I inhibits both. This linear coupling can produce damped
    oscillation and rebound, so it captures the shape of the evoked response far
    better than the independent integrators. It is still LINEAR, though — the
    drive is not rectified and activity is unbounded, so it misses the
    saturation/threshold nonlinearity of the real circuit.
    """
    E_prev = y_prev["E_prev"]
    I_prev = y_prev["I_prev"]
    stim_E_prev = y_prev["stim_E_prev"]
    stim_I_prev = y_prev["stim_I_prev"]

    drive_E = (params["W_EE"] * E_prev - params["W_EI"] * I_prev
               + params["C_E"] + params["XE"] * stim_E_prev)
    drive_I = (params["W_IE"] * E_prev - params["W_II"] * I_prev
               + params["C_I"] + params["XI"] * stim_I_prev)

    E_dot = (-E_prev + drive_E) / params["tau_E"]
    I_dot = (-I_prev + drive_I) / params["tau_I"]

    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    return {}, (E, I)


model.DEFAULT_PARAMS = {
    "tau_E": 0.002,
    "tau_I": 0.004,
    "W_EE": 0.02,
    "W_EI": 0.01,
    "W_IE": 0.02,
    "W_II": 0.01,
    "C_E": 1.0,
    "C_I": 1.0,
    "XE": 3.0,
    "XI": 1.0,
    "log_noise_coef": -4.6052,  # log(0.01): fitted obs-noise coef, var = exp(·)·max(mean, EPS_MEAN)
}
