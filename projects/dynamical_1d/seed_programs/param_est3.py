import numpy as np


def parameter_estimator(data):
    """Rough harmonic frequency from past windows only (FFT peak on one anchor)."""
    h = np.asarray(data["history"], dtype=float)
    row = h[0] - h[0].mean()
    n = row.shape[0]
    if n < 4:
        return {"omega": 0.13, "da1": 0.0, "da2": 0.0, "offset": float(h.mean())}

    spec = np.abs(np.fft.rfft(row))[1:]
    if spec.size == 0 or spec.max() <= 0:
        omega = 0.13
    else:
        k = int(np.argmax(spec)) + 1
        omega = float(2.0 * np.pi * k / n)
        omega = float(np.clip(omega, 0.01, np.pi))

    return {
        "omega": omega,
        "da1": 0.0,
        "da2": 0.0,
        "offset": float(h.mean()),
    }
