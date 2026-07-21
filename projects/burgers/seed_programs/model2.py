import numpy as np


def model(data, params):
    """
    Local smoothing with linear temporal extrapolation.

    Averages each sensor with its two ring neighbours (a smoothing kernel that
    reflects the strong correlation between neighbouring sensors), then adds a
    fraction of the last observed step-to-step change as a velocity term. Both
    the smoothing weight and the velocity weight are free scalars.

    data['x'] : (n_sensors, max_length); column -1 most recent, sensors periodic.

    Returns:
        np.ndarray: predicted activity at the next step, (n_sensors,).
    """
    u0 = data["x"][:, -1]
    u1 = data["x"][:, -2]
    smooth = (1.0 - params["blend"]) * u0 + params["blend"] * 0.5 * (
        np.roll(u0, 1) + np.roll(u0, -1)
    )
    return smooth + params["velocity"] * (u0 - u1)


model.DEFAULT_PARAMS = {"blend": 0.3, "velocity": 0.5}
