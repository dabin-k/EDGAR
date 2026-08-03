import numpy as np

# Fixed topology for JAX tracing.
_EMBED_DIM = 3
_K_NN = 5


def model(window, params):
    """
    EDM-style local forecast in delay coordinates (simplex / S-map lite).

    Fixed 3D delay embedding and k=5 neighbors (JAX-safe). Candidate states
    are all length-d slices inside the window; the current state is the last
    slice. Predicts the next value as a distance-weighted average of the
    values that followed each neighbor. Tunable: eps, tau, blend.
    """
    d = _EMBED_DIM
    k_nn = _K_NN
    W = window.shape[0]
    eps = np.maximum(params["eps"], 1e-8)
    tau = np.maximum(params["tau"], 1e-8)
    blend = np.clip(params["blend"], 0.0, 1.0)

    n = W - d
    if n <= 0:
        return window[-1]

    curr = window[-d:]
    idx = np.arange(d)[None, :] + np.arange(n)[:, None]
    states = window[idx]
    nxts = window[np.arange(n) + d]
    dists = np.sqrt(np.mean((states - curr) ** 2, axis=1)) + eps

    order = np.argsort(dists)[:k_nn]
    weights = np.exp(-dists[order] / tau)
    weights = weights / (np.sum(weights) + 1e-12)
    pred = np.sum(weights * nxts[order])
    return blend * window[-1] + (1.0 - blend) * pred


model.DEFAULT_PARAMS = {
    "eps": 1e-3,
    "tau": 0.05,
    "blend": 0.0,
}
