import numpy as np


def parameter_estimator(data):
    """
    Closed-form least-squares estimate of the smoothing and velocity weights.

    Builds one design matrix over every (block, sensor, time) triple and solves
    the normal equations jointly, so the two correlated features do not fight.

    data['x'] : (n_blocks, n_sensors, block_len). Axis 0 is block (never
    differenced across), axis 1 is sensor (periodic; np.roll(., axis=1)), axis 2
    is time (consecutive steps).
    """
    x = np.asarray(data["x"])
    y = x[:, :, 2:]                                  # target: state at t+1
    u0, u1 = x[:, :, 1:-1], x[:, :, :-2]             # state at t and t-1
    lap = 0.5 * (np.roll(u0, 1, axis=1) + np.roll(u0, -1, axis=1)) - u0
    vel = u0 - u1
    try:
        F = np.stack([(y - u0).ravel() * 0 + lap.ravel(), vel.ravel()], axis=1)
        rhs = (y - u0).ravel()
        w = np.linalg.solve(F.T @ F + 1e-8 * np.eye(2), F.T @ rhs)
        blend, velocity = float(w[0]), float(w[1])
    except np.linalg.LinAlgError:
        blend, velocity = 0.3, 0.5
    if not np.isfinite(blend):
        blend = 0.3
    if not np.isfinite(velocity):
        velocity = 0.5
    return {"blend": blend, "velocity": velocity}
