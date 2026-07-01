import numpy as np


def model(data, params):
    """Inverse monopole retinotopic map: cortical (x, y) -> (azimuth, elevation).

    Biologically-inspired analogue of Schwartz's complex-log cortical map. Schwartz
    models visual-field -> cortex as w = k*log(z + a); our data runs the other way
    (cortex -> visual field), so this implements the analytic inverse, re-anchored so
    the cortical location of the fovea is an explicit parameter:

        w = x + i*y                       (cortical pixel, complex)
        z = a * (exp((w - p_f) / K) - 1)  (visual-field point, complex)
        (azimuth, elevation) = (Re z, Im z)

    At w = p_f the prediction is z = 0, so p_f = (f_x, f_y) is the cortical location of
    the fovea. Eccentricity grows exponentially with cortical distance from p_f,
    reproducing cortical magnification (much cortex near the fovea, little in the
    periphery). exp is single-valued, so unlike the forward log map there is no
    branch-cut / hemifield ambiguity.

    data['cortical_pos']: shape (n_trials, 2), columns (x, y).

    params:
        f_x, f_y: cortical location of the fovea.
        K_re, K_im: complex magnification scale K = K_re + i*K_im; |K| is the cortical
            length-scale, arg(K) the rotation between cortical and visual-field axes.
        log_a: log of the eccentricity scale a (fit in log-space to keep a > 0).

    Returns:
        np.ndarray: Predicted (azimuth, elevation), shape (n_trials, 2).
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    w = x + 1j * y
    p_f = params["f_x"] + 1j * params["f_y"]
    K = params["K_re"] + 1j * params["K_im"]
    a = np.exp(params["log_a"])

    z = a * (np.exp((w - p_f) / K) - 1.0)
    return np.stack([z.real, z.imag], axis=-1)


model.DEFAULT_PARAMS = {
    "f_x": 0.5, "f_y": 0.5,
    "K_re": 0.2, "K_im": 0.0,
    "log_a": 0.0,
}
