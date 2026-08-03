import numpy as np


def parameter_estimator(data):
    """Estimate oscillator frequency from the dominant FFT peak of y.

    Peak frequency in cycles/sample → ω = 2π * f_peak. Falls back to 1.0 if
    the trajectory is too short or flat.
    """
    y = np.asarray(data["y"], dtype=np.float64)
    y = y - y.mean()
    T = y.shape[0]
    dt = 0.05                                        # must match model3's dt
    if T < 8:
        omega_est = 1.0
    else:
        spectrum = np.abs(np.fft.rfft(y))
        spectrum[0] = 0.0                            # ignore DC
        k_peak = int(np.argmax(spectrum))
        # rfft bin k → f=k/T cycles/sample → angular freq per sample = 2π·k/T,
        # which must be divided by dt to get radians per second (what model uses).
        omega_est = max(2.0 * np.pi * k_peak / (T * dt), 0.05)
    residual_std = max(float(np.diff(y).std()), 1e-3)
    return {
        "omega": float(omega_est),
        "damping": 0.1,
        "dt": 0.05,
        "k_gain": 0.3,
        "log_sigma_obs": float(np.log(residual_std)),
        "s0_x": float(y[0]),
        "s0_v": 0.0,
    }
