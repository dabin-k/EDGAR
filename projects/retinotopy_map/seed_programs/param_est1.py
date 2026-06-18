import numpy as np


def parameter_estimator(data):
    """Closed-form OLS fit of the affine cortex->visual-field map.

    Solves the two independent least-squares problems (azimuth and elevation each
    regressed on [x, y, 1]) pooled over all training pixels of this recording.

    Args:
        data (dict): data['cortical_pos'] shape (n_trials, 2) = (x, y);
                     data['visual_field'] shape (n_trials, 2) = (az, el).

    Returns:
        dict: {"a_x", "a_y", "a_0", "e_x", "e_y", "e_0"}.
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=-1)  # (n_trials, 3)
    coef, *_ = np.linalg.lstsq(design, data["visual_field"], rcond=None)  # (3, 2)
    return {
        "a_x": float(coef[0, 0]),
        "a_y": float(coef[1, 0]),
        "a_0": float(coef[2, 0]),
        "e_x": float(coef[0, 1]),
        "e_y": float(coef[1, 1]),
        "e_0": float(coef[2, 1]),
    }
