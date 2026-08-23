import numpy as np


def parameter_estimator(data):
    """Baseline/amplitude heuristic + small positive coupling weights.

    Same resting-drive and stimulus-gain estimates as the independent model;
    the recurrent weights start small and positive and are refined by gradient
    descent (their sign convention is fixed inside the model).
    """
    target_y = np.asarray(data["target_y"], dtype=np.float64)  # (n_stim, T, 2), last axis (E, I)
    E = target_y[..., 0]                          # (n_stim, T)
    I = target_y[..., 1]
    b = max(1, E.shape[1] // 10)                  # pre-stimulus window

    E_base = float(E[:, :b].mean())
    I_base = float(I[:, :b].mean())
    E_amp = max(float(E.max() - E_base), 0.1)
    I_amp = max(float(I.max() - I_base), 0.1)

    return {
        "tau_E": 60.0,
        "tau_I": 120.0,
        "W_EE": 0.02,
        "W_EI": 0.01,
        "W_IE": 0.02,
        "W_II": 0.01,
        "C_E": E_base,
        "C_I": I_base,
        "XE": E_amp,
        "XI": I_amp,
    }
