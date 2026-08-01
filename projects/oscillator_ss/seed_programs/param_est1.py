import numpy as np


def parameter_estimator(data):
    """Estimate persistence params from a single trajectory's summary statistics."""
    y = np.asarray(data["y"], dtype=np.float64)
    residuals = y[1:] - y[:-1]                          # persistence-model residuals
    sigma = max(float(residuals.std()), 1e-3)
    return {
        "log_sigma_obs": float(np.log(sigma)),
        "s0_y_last": float(y[0]),
    }
