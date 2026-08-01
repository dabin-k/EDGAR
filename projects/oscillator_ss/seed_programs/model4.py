import numpy as np


def model(state, y_prev, params):
    """Kalman-lite oscillator with online frequency tracking.

    Extension of the damped harmonic oscillator (model3) that also updates a
    running estimate of the natural frequency ω(t) from each new observation.
    Direct match for the target use case: a noisy oscillator whose frequency
    drifts slowly over time.

    State: (x, v, freq_est, freq_drift). Each observation supplies an
    innovation ``y_prev - x`` that drives corrections on x, v, AND freq_est.
    A slow low-pass drift term tracks freq_est over long timescales.
    """
    x = state["x"]
    v = state["v"]
    freq_est = state["freq_est"]
    freq_drift = state["freq_drift"]

    damping = params["damping"]
    dt = params["dt"]
    k_x = params["k_x"]
    k_v = params["k_v"]
    k_freq = params["k_freq"]                    # gain for freq_est update
    k_drift = params["k_drift"]                  # gain for freq_drift low-pass

    # 1. Observation-driven corrections on x, v, and freq_est.
    innovation = y_prev - x
    x_corr = x + k_x * innovation
    v_corr = v + k_v * innovation
    freq_corr = freq_est + k_freq * innovation * v_corr    # cross-term captures ω-effect on v
    freq_drift_new = freq_drift + k_drift * (freq_corr - freq_est)
    freq_new = freq_corr + freq_drift_new * dt

    # 2. One step of the oscillator ODE at current omega estimate.
    omega = freq_new
    acc = -2.0 * damping * omega * v_corr - omega * omega * x_corr
    v_new = v_corr + dt * acc
    x_new = x_corr + dt * v_new

    new_state = {
        "x": x_new,
        "v": v_new,
        "freq_est": freq_new,
        "freq_drift": freq_drift_new,
    }
    mean = x_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "damping": 0.05,
    "dt": 0.05,
    "k_x": 0.3,
    "k_v": 0.1,
    "k_freq": 0.02,
    "k_drift": 0.001,
    "log_sigma_obs": -1.5,
    "s0_x": 0.0,
    "s0_v": 0.0,
    "s0_freq_est": 1.0,
    "s0_freq_drift": 0.0,
}
