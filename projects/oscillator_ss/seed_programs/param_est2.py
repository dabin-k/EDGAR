import numpy as np


def parameter_estimator(data):
    """AR(1) params: fit alpha by least squares on y[t] vs y[t-1]; sigma from
    resulting residuals.
    """
    y = np.asarray(data["y"], dtype=np.float64)
    y0 = y[:-1]
    y1 = y[1:]
    denom = float((y0 * y0).sum()) + 1e-8
    alpha = float((y0 * y1).sum() / denom)
    residuals = y1 - alpha * y0
    sigma = max(float(residuals.std()), 1e-3)
    return {
        "alpha": alpha,
        "log_sigma_obs": float(np.log(sigma)),
        "s0_y_last": float(y[0]),
    }
