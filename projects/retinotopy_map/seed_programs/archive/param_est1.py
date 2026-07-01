import numpy as np


def parameter_estimator(data):
    """Initial guess for the inverse-monopole cortex->visual-field map.

    Reads the fovea straight off the data and seeds a near-linear map there:
      - p_f = cortical position of the pixel whose measured (az, el) is closest to the
        visual-field origin (the fovea).
      - complex slope m = a/K from a least-squares fit of z = az + i*el against
        (w - p_f) over the near-foveal pixels; a is taken from the eccentricity range
        and K = a/m, giving a gentle (near-linear) starting map that gradient descent
        can then curve.

    Args:
        data (dict): data['cortical_pos'] shape (n_trials, 2) = (x, y);
                     data['visual_field'] shape (n_trials, 2) = (az, el).

    Returns:
        dict: {"f_x", "f_y", "K_re", "K_im", "log_a"}.
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    w = x + 1j * y
    z = data["visual_field"][:, 0] + 1j * data["visual_field"][:, 1]
    r = np.abs(z)  # eccentricity

    p_f = w[int(np.argmin(r))]
    dw = w - p_f

    # complex slope m = a/K from a fit z ~ m * dw over near-foveal pixels.
    r_cut = np.quantile(r, 0.3)
    near = (r <= r_cut) & (np.abs(dw) > 0)
    if near.sum() < 3:
        near = np.abs(dw) > 0
    m = np.sum(np.conj(dw[near]) * z[near]) / np.sum(np.abs(dw[near]) ** 2)
    if not np.isfinite(m) or m == 0:
        m = 1.0 + 0j

    a = float(np.max(r)) or 1.0
    K = a / m

    return {
        "f_x": float(p_f.real),
        "f_y": float(p_f.imag),
        "K_re": float(K.real),
        "K_im": float(K.imag),
        "log_a": float(np.log(a)),
    }
