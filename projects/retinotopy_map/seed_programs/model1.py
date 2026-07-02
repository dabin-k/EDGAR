import numpy as np  # numpy, JAX-clean; np->jnp later


def model(data, params):
    """Affine cortex->visual-field map. An affine map has a CONSTANT Jacobian, 
    so its area magnification |det J| = |a_x*e_y - a_y*e_x| is the SAME at every pixel. 

    azimuth   = a_x*x + a_y*y + a_0
    elevation = e_x*x + e_y*y + e_0

    data['cortical_pos']: shape (n_trials, 2), columns (x, y).

    params:
        a_x, a_y, a_0: azimuth coefficients.
        e_x, e_y, e_0: elevation coefficients.
    
    Returns : 
        np.ndarray: Predicted (azimuth, elevation, log_area_magnification), shape (n_trials, 3).
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    az = params["a_x"] * x + params["a_y"] * y + params["a_0"]
    el = params["e_x"] * x + params["e_y"] * y + params["e_0"]

    detJ = np.abs(params["a_x"] * params["e_y"] - params["a_y"] * params["e_x"])
    log_mag = np.log(detJ + 1e-30) * np.ones_like(x)
    return np.stack([az, el, log_mag], axis=-1)


model.DEFAULT_PARAMS = {
    "a_x": 1.0, "a_y": 0.0, "a_0": 0.0,
    "e_x": 0.0, "e_y": 1.0, "e_0": 0.0,
}
