import numpy as np


def parameter_estimator(data):
    """
    Closed-form least-squares estimate of the relaxation rate k for du/dt = -k u
    under a 2-step Adams-Bashforth step.

    AB2 gives u(t+1) - u(t) = 1.5 RHS(u(t)) - 0.5 RHS(u(t-1)) = -k (1.5 u(t) - 0.5 u(t-1)),
    so k is the single-feature least-squares slope of the increment on
    -(1.5 u(t) - 0.5 u(t-1)) over every (block, sensor, time) triple.

    data['x'] : (n_blocks, n_sensors, block_len). Axis 0 block (never differenced
    across), axis 1 sensor (periodic), axis 2 time (consecutive steps).

    Returns:
        dict: {'k': relaxation rate}.
    """
    x = np.asarray(data["x"])
    y = x[:, :, 2:]                          # state at t+1
    u0, u1 = x[:, :, 1:-1], x[:, :, :-2]     # state at t and t-1
    f = -(1.5 * u0 - 0.5 * u1).ravel()
    target = (y - u0).ravel()
    denom = float(f @ f)
    k = float(f @ target / denom) if denom > 0 else 0.1
    if not np.isfinite(k):
        k = 0.1
    return {"k": k}
