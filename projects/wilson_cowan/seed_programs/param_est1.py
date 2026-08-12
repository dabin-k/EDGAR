import numpy as np


def _smooth_hamming(x: np.ndarray, dt_ms: float, bandwidth_ms: float = 40.0) -> np.ndarray:
    win = int(round(bandwidth_ms / dt_ms))
    if win < 2:
        return x
    if win % 2 == 0:
        win += 1
    w = np.hamming(win)
    w = w / w.sum()
    norm = np.convolve(np.ones(x.shape[-1]), w, mode="same")
    return np.apply_along_axis(lambda v: np.convolve(v, w, mode="same") / norm, -1, x)


def parameter_estimator(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Estimate parameters of the Wilson-Cowan model from time series data.

    Estimates tau_E from robust decay rates over high-activity post-stimulus windows,
    sets tau_I to the same value, estimates E_max and I_max based on observed peak responses, 
    solves for weight/constant parameters using 4-parameter linear regression over active time-steps.

    Args:
        data: Dictionary of time series data arrays containing:
            - "target_y": Observed activity array of shape (C, T, 2); last axis = (E, I)
            - "stim_E": Excitatory stimulus array of shape (C, T)
            - "stim_I": Inhibitory stimulus array of shape (C, T)
          where C is the number of stimulus conditions.

    Returns:
        Estimated model parameters dictionary containing keys:
            'tau_E', 'tau_I', 'W_EE', 'W_EI', 'W_IE', 'W_II',
            'E_max', 'I_max', 'C_E', 'C_I', 'XE', 'XI'
    """
    # Ensure arrays are float64 for least-squares numerical stability
    target_y = np.asarray(data["target_y"], dtype=np.float64)  # (C, T, 2)
    stim_E = np.asarray(data["stim_E"], dtype=np.float64)  # (C, T)
    stim_I = np.asarray(data["stim_I"], dtype=np.float64)

    # Smooth the E and I rates first - otherwise we get a lot of noise in the decay rate estimates 
    def _smooth_hamming(x: np.ndarray, dt_ms: float, bandwidth_ms: float = 40.0):
        win = int(round(bandwidth_ms / dt_ms))
        # keep smoothing zero-phase 
        if win % 2 == 0:
            win += 1
        w = np.hamming(win)
        w = w / w.sum()
        norm = np.convolve(np.ones(x.shape[-1]), w, mode="same")
        return np.apply_along_axis(lambda v: np.convolve(v, w, mode="same") / norm, -1, x)

    # dt_ms = 1.0 in data - fine to hardcode 
    E = _smooth_hamming(target_y[..., 0], dt_ms=1.0, bandwidth_ms=40.0)   # (C, T)
    I = _smooth_hamming(target_y[..., 1], dt_ms=1.0, bandwidth_ms=40.0)

    # 1. Estimate E_max and I_max from observed peaks
    # This gives rather good estimates
    max_obs_E = np.max(E)
    max_obs_I = np.max(I)
    E_max = max_obs_E * 1.1
    I_max = max_obs_I * 1.1

    # Concatenate data across all stimulus conditions into single (C*(T-1),) series for E and I
    E_prev = E[:, :-1].reshape(-1)
    E_next = E[:, 1:].reshape(-1)
    dE = E_next - E_prev
    stim_E_prev = stim_E[:, :-1].reshape(-1)

    # tau_E: select post-stimulus steps with high activity where the system is decaying
    mask_E_decay = (stim_E_prev == 0) & (E_prev > 0.5 * max_obs_E) & (dE < 0)
    ratio_E_est = np.percentile(dE[mask_E_decay] / E_prev[mask_E_decay], 10) #use the minimum 10 percent of E decay, avoids outliers from noise
    tau_E = -1.0 / ratio_E_est
    tau_I = tau_E #just assume the same, hard to extract good value due to low SNR and inhibitory response not really entering decaying mode

    # 3. 4-parameter linear regression for Excitatory parameters
    # LHS_E = (dE/dt + E_prev/tau_E) / (E_max - E_prev)
    I_prev = I[:, :-1].reshape(-1)
    mask_E_active = (E_prev > 0.5)

    LHS_E = (dE[mask_E_active] + E_prev[mask_E_active] / tau_E) / (E_max - E_prev[mask_E_active])
    X_E = np.column_stack([
        E_prev[mask_E_active],
        -I_prev[mask_E_active],
        np.ones_like(E_prev[mask_E_active]),
        stim_E_prev[mask_E_active]
    ])
    w, _, _, _ = np.linalg.lstsq(X_E, LHS_E, rcond=None)
    W_EE = max(w[0] * tau_E, 0.0)
    W_EI = max(w[1] * tau_E, 0.0)
    C_E  = max(w[2] * tau_E, 0.0)
    XE   = max(w[3] * tau_E, 0.0)

    # 4. 4-parameter linear regression for Inhibitory parameters
    # LHS_I = (dI/dt + I_prev/tau_I) / (I_max - I_prev)
    I_prev_I = I[:, :-1].reshape(-1)
    I_next_I = I[:, 1:].reshape(-1)
    dI = I_next_I - I_prev_I
    E_prev_I = E[:, :-1].reshape(-1)
    stim_I_prev = stim_I[:, :-1].reshape(-1)

    mask_I_active = (I_prev_I > 0.5)
    LHS_I = (dI[mask_I_active] + I_prev_I[mask_I_active] / tau_I) / (I_max - I_prev_I[mask_I_active])
    X_I = np.column_stack([
        E_prev_I[mask_I_active],
        -I_prev_I[mask_I_active],
        np.ones_like(I_prev_I[mask_I_active]),
        stim_I_prev[mask_I_active]
    ])
    g, _, _, _ = np.linalg.lstsq(X_I, LHS_I, rcond=None)
    W_IE = max(g[0] * tau_I, 0.0)
    W_II = max(g[1] * tau_I, 0.0)
    C_I  = max(g[2] * tau_I, 0.0)
    XI   = max(g[3] * tau_I, 0.0)

    return {
        'tau_E': float(tau_E),
        'tau_I': float(tau_I),
        'W_EE': float(W_EE),
        'W_IE': float(W_IE),
        'W_EI': float(W_EI),
        'W_II': float(W_II),
        'E_max': float(E_max),
        'I_max': float(I_max),
        'C_E': float(C_E),
        'C_I': float(C_I),
        'XE': float(XE),
        'XI': float(XI)
    }