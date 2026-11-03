import numpy as np


def parameter_estimator(data):
    """Cheap initial guess for the multiplicative model for a single plant.

    K_L and K_N are set to the median of L and N (a typical half-saturation
    scale); the linear coefficients are then a single least-squares solve.

    Args:
        data (dict): Data for one plant with "L", "N" and "growth_rate", each of
            shape (n_trials,).

    Returns:
        dict: Estimated "K_L", "K_N", "light_nitrogen_coef" and "bias".
    """
    L = np.asarray(data["L"])
    N = np.asarray(data["N"])
    y = np.asarray(data["growth_rate"])

    K_L = np.median(L)
    K_N = np.median(N)

    light = L / (K_L + L)
    nitrogen = N / (K_N + N)
    A = np.column_stack([light * nitrogen, np.ones_like(y)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return {
        "K_L": float(K_L),
        "K_N": float(K_N),
        "light_nitrogen_coef": float(a),
        "bias": float(b),
    }
