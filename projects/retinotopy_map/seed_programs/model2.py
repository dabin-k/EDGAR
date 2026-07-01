import numpy as np


def model(data, params):
    """Inverse dipole retinotopic map: cortical (x, y) -> (azimuth, elevation).

    Dipole generalization of the inverse monopole (model3): the analytic inverse of
    Schwartz's w = k*log((z + a)/(z + b)), re-anchored at the cortical fovea p_f:

        w = x + i*y
        E = exp((w - p_f) / K)
        z = a*b*(1 - E) / (a*E - b)
        (azimuth, elevation) = (Re z, Im z)

    The extra parameter b shapes the far periphery (where the monopole over-compresses).
    As b -> infinity this reduces exactly to the monopole, so the dipole strictly nests
    it. At w = p_f the prediction is z = 0, so p_f = (f_x, f_y) is the cortical fovea.

    data['cortical_pos']: shape (n_trials, 2), columns (x, y).

    params:
        f_x, f_y: cortical location of the fovea.
        K_re, K_im: complex magnification scale K = K_re + i*K_im (see model3).
        log_a: log of the near-fovea scale a (a > 0).
        log_b: log of the peripheral scale b (b > 0); b -> inf recovers the monopole.

    Returns:
        np.ndarray: Predicted (azimuth, elevation), shape (n_trials, 2).
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    w = x + 1j * y
    p_f = params["f_x"] + 1j * params["f_y"]
    K = params["K_re"] + 1j * params["K_im"]
    a = np.exp(params["log_a"])
    b = np.exp(params["log_b"])

    E = np.exp((w - p_f) / K)
    z = a * b * (1.0 - E) / (a * E - b + 1e-12)
    return np.stack([z.real, z.imag], axis=-1)


model.DEFAULT_PARAMS = {
    "f_x": 0.5, "f_y": 0.5,
    "K_re": 0.2, "K_im": 0.0,
    "log_a": 0.0, "log_b": 4.0,
}
