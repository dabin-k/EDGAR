import numpy as np


def parameter_estimator(data):
    """Closed-form OLS fit of the quadratic cortex->visual-field map.

    Regresses azimuth and elevation each on the quadratic feature basis
    [x^2, y^2, x*y, x, y, 1], pooled over all training pixels of this recording.

    Args:
        data (dict): data['cortical_pos'] shape (n_trials, 2) = (x, y);
                     data['visual_field'] shape (n_trials, 2) = (az, el).

    Returns:
        dict: quadratic coefficients for azimuth ("a_*") and elevation ("e_*").
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    design = np.stack([x**2, y**2, x * y, x, y, np.ones_like(x)], axis=-1)  # (n_trials, 6)
    coef, *_ = np.linalg.lstsq(design, data["visual_field"], rcond=None)  # (6, 2)
    names = ["_2x", "_2y", "_xy", "_x", "_y", "_0"]
    out = {}
    for i, suffix in enumerate(names):
        out[f"a{suffix}"] = float(coef[i, 0])
        out[f"e{suffix}"] = float(coef[i, 1])
    return out
