import numpy as np


def model(window, params):
    """
    Discrete harmonic / 2D latent-state seed for next-value regression.

    Treats x[t-1], x[t-2] as noisy observations of position and lagged position.
    Predicts x[t] via a marginally stable AR(2) with coefficients tied to omega:

        x[t] ≈ (2 cos(omega) + da1) * x[t-1] + (-1 + da2) * x[t-2] + offset

    This is the one-step map of an undamped 2D rotation (harmonic oscillator).
    """
    omega = np.clip(params["omega"], 0.01, np.pi)
    da1 = params["da1"]
    da2 = params["da2"]
    offset = params["offset"]

    x1 = window[-1]
    x2 = window[-2] if window.shape[0] >= 2 else window[-1]

    a1 = 2.0 * np.cos(omega) + da1
    a2 = -1.0 + da2
    return a1 * x1 + a2 * x2 + offset


model.DEFAULT_PARAMS = {
    "omega": 0.13,
    "da1": 0.0,
    "da2": 0.0,
    "offset": 0.0,
}
