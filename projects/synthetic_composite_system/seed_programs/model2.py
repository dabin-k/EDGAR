import numpy as np

def model(data: dict, params: dict, components: dict) -> np.ndarray:
    r"""
    Multiplicative model combining light and nitrogen sub-components.
    Equation:
        $$f = a_{\mathrm{light,nitrogen}} \cdot g_{\mathrm{light}}(L, K_L) \cdot g_{\mathrm{nitrogen}}(N, K_N) + \mathrm{bias}$$

    Args:
        data: Dict with "L" (light) and "N" (nitrogen), each of shape (n_trials,).
        params: Dict of scalar per-plant parameters "K_L", "K_N", "light_nitrogen_coef", and "bias".
        components: Dict of fixed sub-component functions,
            `components["g_light"](L, K_L)` and `components["g_nitrogen"](N, K_N)`.

    Returns:
        Predicted growth rate, shape (n_trials,).
    """
    L = data["L"]
    N = data["N"]

    K_L = params["K_L"]
    K_N = params["K_N"]
    a = params["light_nitrogen_coef"]
    b = params["bias"]

    g_light = components["g_light"]
    g_nitrogen = components["g_nitrogen"]

    return a * g_light(L, K_L) * g_nitrogen(N, K_N) + b


model.DEFAULT_PARAMS = {
    "K_L": 1.0,
    "K_N": 1.0,
    "light_nitrogen_coef": 1.0,
    "bias": 0.0,
}
