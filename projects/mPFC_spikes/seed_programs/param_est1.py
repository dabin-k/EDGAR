import numpy as np


def parameter_estimator(data):
    """
    Estimate power-law Hawkes parameters from the neuron's PAST windows.

    Uses only data['history'] (past spike-count windows, shape (A, W)); never the
    held-out target. The baseline mu0 is set so softplus(mu0) matches the
    empirical mean rate (mean spikes/bin over all past bins), and the
    self-excitation gain / tail start small and are refined by gradient descent.
    """
    h = np.asarray(data["history"], dtype=float)  # (A, W) strictly-past windows
    mean_rate = max(float(h.mean()), 1e-6)
    mu0 = float(np.log(np.expm1(mean_rate)))  # inverse-softplus of the mean rate
    return {
        "mu0": mu0,
        "K": 0.1,
        "gamma": 0.5,
        "u0": 1.0,
    }
