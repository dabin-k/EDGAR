import numpy as np


def parameter_estimator(data):
    """Baseline -> constant drive; evoked amplitude -> stimulus gain.

    Pre-stimulus mean estimates the resting drive C_*; the peak excursion above
    baseline estimates the stimulus gain X_*. Time constants are left as small
    positive defaults for gradient descent to refine.
    """
    target_y = np.asarray(data["target_y"], dtype=np.float64)  # (n_stim, T, 2), last axis (E, I)
    E = target_y[..., 0]                          # (n_stim, T)
    I = target_y[..., 1]
    b = max(1, E.shape[1] // 10)                  # pre-stimulus window

    E_base = float(E[:, :b].mean())
    I_base = float(I[:, :b].mean())
    E_amp = max(float(E.max() - E_base), 0.1)
    I_amp = max(float(I.max() - I_base), 0.1)

    return {
        "tau_E": 60.0,
        "tau_I": 120.0,
        "C_E": E_base,
        "C_I": I_base,
        "XE": E_amp,
        "XI": I_amp,
        # Latent-inhibition params (model1 reads these) — small positive defaults, GD refines.
        "tau_S": 300.0,
        "W_ES": 0.001,
        "W_IS": 0.001,
        # Initial value of the latent S — learnable (s0_ prefix); GD refines from here.
        "s0_S": 1.0,
    }
