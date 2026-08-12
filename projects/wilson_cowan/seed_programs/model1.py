import numpy as np

H = 1 / 30000  # integration step (matches the data-generating grid)


def model(state, y_prev, params):
    """Leaky E/I integrators with a latent inhibition variable. 

    Each population relaxes toward a constant baseline drive with its own time
    constant. 
    """
    E_prev = y_prev["E_prev"]
    I_prev = y_prev["I_prev"]
    stim_E_prev = y_prev["stim_E_prev"]
    stim_I_prev = y_prev["stim_I_prev"]

    S_prev = state["S"]  # latent variable
    S_dot = (-S_prev + (E_prev - I_prev)) / params["tau_S"]
    S = S_prev + H * S_dot

    drive_E = params["C_E"] + params["XE"] * stim_E_prev - params["W_ES"] * S_prev
    drive_I = params["C_I"] + params["XI"] * stim_I_prev - params["W_IS"] * S_prev

    E_dot = (-E_prev + drive_E) / params["tau_E"]
    I_dot = (-I_prev + drive_I) / params["tau_I"]

    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    return {"S": S}, (E, I)

model.INITIAL_STATE = {'S': 1.0}

model.DEFAULT_PARAMS = {
    "tau_E": 0.002,
    "tau_I": 0.004,
    "C_E": 1.0,
    "C_I": 1.0,
    "XE": 3.0,
    "XI": 1.0,
    "tau_S": 0.01,
    "W_ES": 0.5,
    "W_IS": 0.5,
    "log_noise_coef": -4.6052,  # log(0.01): fitted obs-noise coef, var = exp(·)·max(mean, EPS_MEAN)
}
