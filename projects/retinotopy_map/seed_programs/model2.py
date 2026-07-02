import numpy as np  # written in numpy; JAX-clean (elementwise ops, static defaults) -> np->jnp later


def model(data, params):
    """Inverse-dipole cortex->visual-field map.

    Cortex (raw pixels) -> visual field (absolute degrees) + local magnification.

    log_a, log_b   dipole shape (near-fovea / far-periphery poles)
    gamma          point-image coupling exponent (1 = constant point image)

    f_x, f_y       cortical fovea == camera translation (anchor: sign-map border)
    K_re, K_im     |K| == pixel scale (anchor: known um/px);
                    arg(K) == camera rotation (anchor: anatomical axis)
    z0_az, z0_el   visual-field origin / gaze offset (real; screen fixes the rest)

    aniso          y(elevation)-axis pixel anisotropy (1 = isotropic)
    shear          in-plane pixel shear (0 = none)
    ------------------------------------------------------------------------------

    data['cortical_pos'] : (n, 2) raw cortical (x, y) in pixels (NOT normalised).

    Returns : 
        np.ndarray: shape (n, 3) predicted (azimuth, elevation, log_area_magnification). 
            The 3rd is the model's analytic Jacobian area element |det J|, in log space,
            to be scored against the empirical magnification computed from the data.

    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]

    aniso = params["aniso"]
    shear = params["shear"]
    xw = x + shear * y
    yw = aniso * y
    w = xw + 1j * yw

    p_f = params["f_x"] + 1j * params["f_y"]
    K = params["K_re"] + 1j * params["K_im"]
    a = np.exp(params["log_a"])
    b = np.exp(params["log_b"])

    E = np.exp((w - p_f) / K)
    denom = a * E - b
    dipole = a * b * (1.0 - E) / denom
    z = (params["z0_az"] + 1j * params["z0_el"]) + dipole

    # --- analytic Jacobian of the dipole part: dz/dw (verified formula) ---
    dzdw = a * b * (b - a) / denom ** 2 * E / K
    # area magnification of the FULL map = |dz/dw|^2 * |det(TierA)|,  det(TierA)=aniso
    # gamma lets the search test the point-image law instead of asserting it.
    area_mag = (np.abs(dzdw) ** 2) * np.abs(aniso)
    log_mag = params["gamma"] * np.log(area_mag + 1e-30)

    return np.stack([z.real, z.imag, log_mag], axis=-1)


model.DEFAULT_PARAMS = {
    "log_a": np.log(3.0), "log_b": np.log(150.0), "gamma": 1.0, 
    "f_x": 0.0, "f_y": 0.0, "K_re": 100.0, "K_im": 0.0,
    "z0_az": 0.0, "z0_el": 0.0,  
    "aniso": 1.0, "shear": 0.0,
}
