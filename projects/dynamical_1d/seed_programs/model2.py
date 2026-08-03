import numpy as np


def model(window, params):
    """
    Harmonic extrapolation from a strictly-past window.

    Combines a DC offset with fundamental + second-harmonic sinusoids over
    fixed lag indices k = W..1 (window[-1] is lag 1). Captures oscillatory
    phase/velocity structure that persistence cannot.
    """
    W = window.shape[0]
    omega = np.clip(params["omega"], 0.01, np.pi)
    offset = params["offset"]
    A1 = params["A1"]
    B1 = params["B1"]
    A2 = params["A2"]
    B2 = params["B2"]

    k = np.arange(W, 0, -1, dtype=float)
    basis = (
        A1 * np.cos(omega * k)
        + B1 * np.sin(omega * k)
        + A2 * np.cos(2.0 * omega * k)
        + B2 * np.sin(2.0 * omega * k)
    )
    return offset + np.dot(window, basis) / W


model.DEFAULT_PARAMS = {
    "omega": 0.08,
    "offset": 0.5,
    "A1": 0.05,
    "B1": 0.05,
    "A2": 0.02,
    "B2": 0.02,
}
