import numpy as np


def model(data, params):
    """Affine retinotopic map: cortical (x, y) -> (azimuth, elevation).

    A first-order (linear + offset) approximation of the cortex->visual-field map,
    shared across every pixel of the recording:

        azimuth   = a_x * x + a_y * y + a_0
        elevation = e_x * x + e_y * y + e_0

    data['cortical_pos']: shape (n_trials, 2), columns (x, y).

    params:
        a_x, a_y, a_0: azimuth coefficients.
        e_x, e_y, e_0: elevation coefficients.

    Returns:
        np.ndarray: Predicted (azimuth, elevation), shape (n_trials, 2).
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    az = params["a_x"] * x + params["a_y"] * y + params["a_0"]
    el = params["e_x"] * x + params["e_y"] * y + params["e_0"]
    return np.stack([az, el], axis=-1)


model.DEFAULT_PARAMS = {
    "a_x": 1.0,
    "a_y": 0.0,
    "a_0": 0.0,
    "e_x": 0.0,
    "e_y": 1.0,
    "e_0": 0.0,
}
