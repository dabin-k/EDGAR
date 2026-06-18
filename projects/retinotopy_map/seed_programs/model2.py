import numpy as np


def model(data, params):
    """Quadratic retinotopic map: cortical (x, y) -> (azimuth, elevation).

    Extends the affine seed with second-order terms (x^2, y^2, x*y) to capture the
    curvature of the cortical magnification, still shared across every pixel:

        coord = c_2x * x^2 + c_2y * y^2 + c_xy * x*y + c_x * x + c_y * y + c_0

    for each of azimuth and elevation.

    data['cortical_pos']: shape (n_trials, 2), columns (x, y).

    Returns:
        np.ndarray: Predicted (azimuth, elevation), shape (n_trials, 2).
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]

    def _quad(prefix):
        return (
            params[f"{prefix}_2x"] * x**2
            + params[f"{prefix}_2y"] * y**2
            + params[f"{prefix}_xy"] * x * y
            + params[f"{prefix}_x"] * x
            + params[f"{prefix}_y"] * y
            + params[f"{prefix}_0"]
        )

    return np.stack([_quad("a"), _quad("e")], axis=-1)


model.DEFAULT_PARAMS = {
    "a_2x": 0.0, "a_2y": 0.0, "a_xy": 0.0, "a_x": 1.0, "a_y": 0.0, "a_0": 0.0,
    "e_2x": 0.0, "e_2y": 0.0, "e_xy": 0.0, "e_x": 0.0, "e_y": 1.0, "e_0": 0.0,
}
