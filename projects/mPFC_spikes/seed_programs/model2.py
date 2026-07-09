import numpy as np


def model(window, params):
    """
    Leaky self-exciting process with exponential history, windowed form.

    Given a strictly-past window of spike counts `window` (shape (W,)), predict
    the expected spike count `mu` in the NEXT bin. Window orientation:
    `window[-1]` is the most recent past bin x[t-1]; `window[0]` is the oldest
    x[t-W]. The target bin x[t] is NEVER part of the input (causal by design).

    A single exponentially-decaying memory of past spikes drives the intensity:

        h  = sum_{k=1..W} decay**(k-1) * x[t-k],   decay = exp(-1 / tau)
        mu = softplus( base + w * h )

    The most recent past bin (k=1) gets weight decay**0 = 1; older bins decay
    geometrically. This is the recurrent counterpart to the convolutional Hawkes
    seed, written here as a closed-form weighted sum over the fixed window.

    params: base (pre-softplus baseline), w (self-excitation gain), tau (decay in bins)
    Returns: a scalar (expected spike count in the next bin).
    """
    W = window.shape[0]

    base = params["base"]
    w = params["w"]
    tau = np.clip(params["tau"], 1.0, 5000.0)
    decay = np.exp(-1.0 / tau)

    # weight for window position i: lag k = W - i, so window[-1] (k=1) -> decay**0
    i = np.arange(W)
    weights = decay ** (W - 1 - i)
    h = np.dot(window, weights)

    return np.logaddexp(0.0, base + w * h) + 1e-6  # softplus, uses past only


model.DEFAULT_PARAMS = {
    "base": -5.5,  # softplus(-5.5) ~ 0.004 spikes/bin ~ empirical baseline
    "w": 0.0,      # start at homogeneous Poisson; gradient descent grows it
    "tau": 20.0,   # 100 ms memory at 5 ms bins
}
