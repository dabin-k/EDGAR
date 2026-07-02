import numpy as np


def parameter_estimator(data):
    """Initial guess for the extended inverse-dipole map.

      * read the visual-field origin z0 from a robust centre of the RF cloud
        instead of implicitly assuming it is (0, 0);
      * seeds p_f from the RF nearest that estimated origin, not argmin|z|;
      * RF-reliability weighting (~1/size^2) so noisy peripheral RFs do not
        dominate the near-fovea slope fit (rec #5);
      * seeds log_P from the observed size/|Jacobian| ratio if sizes are present.
    """
    x = data["cortical_pos"][:, 0]
    y = data["cortical_pos"][:, 1]
    # normalise cortex the same way the model does (keeps the seed self-consistent)
    cx, cy = float(np.mean(x)), float(np.mean(y))
    s = 1.0 / (np.std(x) + 1e-9)
    w = (x - cx) * s + 1j * (y - cy) * s

    az = data["visual_field"][:, 0]
    el = data["visual_field"][:, 1]
    z = az + 1j * el

    size = data.get("rf_size", None)
    wgt = 1.0 / (np.asarray(size) ** 2 + 1e-9) if size is not None else np.ones_like(x)

    # --- visual-field origin = reliability-weighted centre of RF cloud ---
    z0 = np.sum(wgt * z) / np.sum(wgt)
    zc = z - z0                      # eccentricity measured from the fitted origin
    r = np.abs(zc)

    # seed p_f from the cortical location of the most foveal (smallest ecc) RFs
    k = max(3, int(0.05 * len(r)))
    idx = np.argsort(r)[:k]
    p_f = np.sum(wgt[idx] * w[idx]) / np.sum(wgt[idx])
    dw = w - p_f

    # near-fovea complex slope m = a/K, reliability-weighted
    r_cut = np.quantile(r, 0.3)
    near = (r <= r_cut) & (np.abs(dw) > 0)
    if near.sum() < 3:
        near = np.abs(dw) > 0
    ww = wgt[near]
    m = np.sum(ww * np.conj(dw[near]) * zc[near]) / np.sum(ww * np.abs(dw[near]) ** 2)
    if not np.isfinite(m) or m == 0:
        m = 1.0 + 0j

    a = float(np.max(r)) or 1.0
    K = a / m

    params = {
        "f_x": float(p_f.real), "f_y": float(p_f.imag),
        "K_re": float(K.real), "K_im": float(K.imag),
        "log_a": float(np.log(a)), "log_b": float(np.log(50.0 * a)),
        "z0_az": float(z0.real), "z0_el": float(z0.imag),
        "aniso": 1.0, "shear": 0.0, "gamma": 1.0,
    }

    # --- seed log_P from observed size vs local magnification ---
    if size is not None:
        E = np.exp((w - p_f) / K)
        dzdw = a * (50.0 * a) * ((50.0 * a) - a) / (a * E - 50.0 * a + 1e-12) ** 2 * E / K
        ratio = np.asarray(size) / (np.abs(dzdw) + 1e-9)
        params["log_P"] = float(np.log(np.median(ratio) + 1e-9))
    else:
        params["log_P"] = 0.0
    return params
