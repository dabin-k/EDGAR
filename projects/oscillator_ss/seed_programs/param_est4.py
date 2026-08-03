import numpy as np


def parameter_estimator(data):
    """Initial ω from FFT peak; correction gains hardcoded to sensible values
    that gradient descent can refine.
    """
    y = np.asarray(data["y"], dtype=np.float64)
    y = y - y.mean()
    T = y.shape[0]
    dt = 0.05                                        # must match model4's dt
    if T < 8:
        omega_est = 1.0
    else:
        spectrum = np.abs(np.fft.rfft(y))
        spectrum[0] = 0.0
        k_peak = int(np.argmax(spectrum))
        # rfft bin k → f=k/T cycles/sample → angular freq per sample = 2π·k/T,
        # which must be divided by dt to get radians per second (what model uses).
        omega_est = max(2.0 * np.pi * k_peak / (T * dt), 0.05)
    residual_std = max(float(np.diff(y).std()), 1e-3)
    return {
        "damping": 0.05,
        "dt": 0.05,
        "k_x": 0.3,
        "k_v": 0.1,
        "k_freq": 0.02,
        "k_drift": 0.001,
        "log_sigma_obs": float(np.log(residual_std)),
        "s0_x": float(y[0]),
        "s0_v": 0.0,
        "s0_freq_est": float(omega_est),
        "s0_freq_drift": 0.0,
    }
