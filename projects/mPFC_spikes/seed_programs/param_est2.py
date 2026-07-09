import numpy as np


def parameter_estimator(data):
    """
    Estimate leaky self-excitation parameters from the neuron's PAST windows.

    Uses only data['history'] (past spike-count windows, shape (A, W)); never the
    held-out target. The baseline is set so softplus(base) matches the empirical
    mean rate (mean spikes/bin over all past bins); the gain starts small and the
    memory timescale starts at ~100 ms, both refined by gradient descent.
    """
    h = np.asarray(data["history"], dtype=float)  # (A, W) strictly-past windows
    mean_rate = max(float(h.mean()), 1e-6)
    base = float(np.log(np.expm1(mean_rate)))  # inverse-softplus of the mean rate
    return {
        "base": base,
        "w": 0.1,
        "tau": 20.0,
    }
