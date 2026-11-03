import numpy as np


def model(data: dict, params: dict, components: dict) -> np.ndarray:
    """
    Linear model combining light and nitrogen sub-components.

    Args:
        data: Dict with "L" (light) and "N" (nitrogen), each of shape (n_trials,).
        params: Dict of scalar per-plant parameters "K_L", "K_N", "light_coef", "nitrogen_coef", and "bias".
        components: Dict of fixed sub-component functions,
            `components["g_light"](L, K_L)` and `components["g_nitrogen"](N, K_N)`.

    Returns:
        Predicted growth rate, shape (n_trials,).
    """
    L = data["L"]
    N = data["N"]

    K_L = params["K_L"]
    K_N = params["K_N"]
    a = params["light_coef"]
    b = params["nitrogen_coef"]
    c = params["bias"]

    g_light = components["g_light"]
    g_nitrogen = components["g_nitrogen"]

    return a * g_light(L, K_L) + b * g_nitrogen(N, K_N) + c


model.DEFAULT_PARAMS = {
    "K_L": 1.0,
    "K_N": 1.0,
    "light_coef": 1.0,
    "nitrogen_coef": 1.0,
    "bias": 0.0,
}
