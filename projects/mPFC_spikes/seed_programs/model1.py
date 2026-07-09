import numpy as np


def model(window, params):
    """
    Power-law self-exciting point process (Hawkes-style), windowed form.

    Given a strictly-past window of spike counts `window` (shape (W,)), predict
    the expected spike count `mu` in the NEXT bin. Window orientation:
    `window[-1]` is the most recent past bin x[t-1]; `window[0]` is the oldest
    x[t-W]. The target bin x[t] is NEVER part of the input, so the prediction is
    causal by construction.

        mu = softplus( mu0 + K * sum_{j=1..W} kernel[j] * x[t-j] )
        kernel[j] = 1 / (j + u0) ** (1 + gamma)     # heavy-tailed decay

    Each past spike adds a bump decaying as a power law, capturing burstiness and
    long-tailed inter-spike intervals.

    params: mu0 (baseline, pre-softplus), K (gain), gamma (tail exponent), u0 (offset)
    Returns: a scalar (expected spike count in the next bin).
    """
    W = window.shape[0]

    mu0 = params["mu0"]
    K = params["K"]
    gamma = np.clip(params["gamma"], 0.0, 3.0)
    u0 = np.clip(params["u0"], 1.0, 100.0)

    # lag j for each window position: window[-1] -> j=1, window[0] -> j=W
    lags = np.arange(W, 0, -1)
    kernel = 1.0 / (lags + u0) ** (1.0 + gamma)

    mu_lin = mu0 + K * np.dot(window, kernel)
    return np.logaddexp(0.0, mu_lin) + 1e-6  # softplus -> strictly positive


model.DEFAULT_PARAMS = {
    "mu0": -5.5,   # softplus(-5.5) ~ 0.004 spikes/bin ~ empirical baseline
    "K": 0.0,      # start at homogeneous Poisson; gradient descent grows it
    "gamma": 0.5,
    "u0": 1.0,
}
