from __future__ import annotations
from typing import Dict, Any
import numpy as np

def parameter_estimator(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Estimate parameters of the Wilson-Cowan-with-slow-inhibition (WCS) model.

    Same linear-regression backbone as the base WC estimator (param_est1): tau_E from
    robust post-stimulus decay, tau_I tied to it, E_max/I_max from peaks, and the
    weight/constant/drive parameters from a linear regression over active time-steps.

    The WCS model adds a hidden slow-inhibition variable S -- a low-pass of the
    observed I: S[t] = S[t-1] + (-S[t-1] + I[t-1]) / tau_S -- that feeds back on both
    populations via -W_ES*S_prev (excitatory drive) and -W_IS*S_prev (inhibitory
    drive). S is not observed, so we grid-search tau_S over the slow regime: for each
    candidate we reconstruct S from I and refit the E regression with a -S_prev
    column, keeping the tau_S that minimises the E-equation residual (the high-SNR E
    response carries the slow-adaptation signal; the low-SNR I equation is
    uninformative for tau_S and only adds a spurious fast-collinear optimum). The
    chosen S then enters both the E and I fits; W_ES / W_IS are read off the -S_prev
    columns. tau_S is only weakly identifiable from a single trace (its time constant
    spans much of the recording), so treat it as a starting point for GD refinement.
    s0_S (the learnable initial S that seeds the scan carry) is taken from the
    pre-stimulus baseline of I.

    Args:
        data: Dictionary of time series data arrays containing:
            - "target_y": Observed activity array of shape (C, T, 2); last axis = (E, I)
            - "stim_E": Excitatory stimulus array of shape (C, T)
            - "stim_I": Inhibitory stimulus array of shape (C, T)
          where C is the number of stimulus conditions.

    Returns:
        Estimated model parameters dictionary containing keys:
            'tau_E', 'tau_I', 'W_EE', 'W_EI', 'W_IE', 'W_II', 'E_max', 'I_max',
            'C_E', 'C_I', 'tau_S', 'W_ES', 'W_IS', 'XE', 'XI', 's0_S'
        plus the objective-E EKF noise/init log-variances (inert for A-D; see load_data._kf_hyper):
            'kf_log_q_E', 'kf_log_q_I', 'kf_log_q_S', 'kf_log_sig_E', 'kf_log_sig_I',
            'kf_log_p0_E', 'kf_log_p0_I', 'kf_log_p0_S'
    """
    # Ensure arrays are float64 for least-squares numerical stability
    target_y = np.asarray(data["target_y"], dtype=np.float64)  # (C, T, 2)
    stim_E = np.asarray(data["stim_E"], dtype=np.float64)  # (C, T)
    stim_I = np.asarray(data["stim_I"], dtype=np.float64)

    # Smooth the E and I rates first - otherwise we get a lot of noise in the decay rate estimates 
    # 2026-08-27 update : actually, smoothing seems harmful on synthetic data at least. Let default be no smoothing.
    def _smooth_hamming(x: np.ndarray, dt_ms: float, bandwidth_ms: float = 0.0):
        win = int(round(bandwidth_ms / dt_ms))
        # keep smoothing zero-phase 
        if win % 2 == 0:
            win += 1
        w = np.hamming(win)
        w = w / w.sum()
        norm = np.convolve(np.ones(x.shape[-1]), w, mode="same")
        return np.apply_along_axis(lambda v: np.convolve(v, w, mode="same") / norm, -1, x)

    # dt_ms = 1.0 in data - fine to hardcode
    E_raw = target_y[..., 0]   # (C, T)
    I_raw = target_y[..., 1]
    E = _smooth_hamming(E_raw, dt_ms=1.0)   # (C, T)
    I = _smooth_hamming(I_raw, dt_ms=1.0)

    C, T = E.shape

    # Objective-E (EKF) noise/init inits, as log-variances (consumed by load_data._kf_hyper).
    # Observation noise = variance of the high-frequency residual the smoother removes: exactly the
    # trial-to-trial wriggle the state-space model attributes to measurement noise. E/I initial
    # covariance equals it (the filter seeds m0's E/I straight from y[0], so we know them to ~one
    # observation). Process noise starts an order below obs noise (a "trust the dynamics" prior; the
    # NLL logdet term + GD refine it), and latent S — the smoothest variable — gets 0.1x that again.
    # p0_S (S seed uncertainty) is set below from the pre-stim I baseline spread.
    _floor = 1e-8
    sig_E = max(float(np.var(E_raw - E)), _floor)
    sig_I = max(float(np.var(I_raw - I)), _floor)

    # 1. Estimate E_max and I_max from observed peaks
    max_obs_E = np.max(E)
    max_obs_I = np.max(I)
    E_max = max_obs_E * 1.1
    I_max = max_obs_I * 1.1

    # Concatenate data across all stimulus conditions into single (C*(T-1),) series
    E_prev = E[:, :-1].reshape(-1)
    E_next = E[:, 1:].reshape(-1)
    dE = E_next - E_prev
    stim_E_prev = stim_E[:, :-1].reshape(-1)

    I_prev = I[:, :-1].reshape(-1)
    I_next = I[:, 1:].reshape(-1)
    dI = I_next - I_prev
    stim_I_prev = stim_I[:, :-1].reshape(-1)

    # tau_E: post-stimulus steps with high activity where the system is decaying
    mask_E_decay = (stim_E_prev == 0) & (E_prev > 0.5 * max_obs_E) & (dE < 0)
    ratio_E_est = np.percentile(dE[mask_E_decay] / E_prev[mask_E_decay], 10)  # min 10% -> robust to noise
    tau_E = -1.0 / ratio_E_est
    tau_I = tau_E  # just assume the same, hard to extract good value due to low SNR 

    # 2. Take average pre-stim onset of I across conditions as the initial S for the scan carry.
    stim_any = (stim_E > 0) | (stim_I > 0)                      # (C, T)
    onsets = [np.argmax(stim_any[c]) if stim_any[c].any() else T for c in range(C)]
    base_len = max(1, min(min(onsets), T))
    s0_per_cond = I[:, :base_len].mean(axis=1)                  # (C,)
    s0_S = float(s0_per_cond.mean())
    # S is seeded from the pre-stim I baseline (a softer estimate than the direct y[0] seed of
    # E/I), so its initial covariance is the spread of that baseline across conditions/time.
    p0_S = max(float(np.var(I_raw[:, :base_len])), _floor)

    # tau_S is the slow-inhibition constant: search the slow regime (well above the
    # fast tau_E) up to a couple of trace lengths, log-spaced.
    tau_S_grid = np.geomspace(5.0 * tau_E, max(20.0 * tau_E, 2.0 * T), 10)
    a = (1.0 - 1.0 / tau_S_grid)[:, None]                       # (G, 1)
    b = (1.0 / tau_S_grid)[:, None]                             # (G, 1)
    S = np.empty((tau_S_grid.size, C, T))
    S[:, :, 0] = s0_per_cond[None, :]
    for t in range(1, T):
        S[:, :, t] = a * S[:, :, t - 1] + b * I[None, :, t - 1]
    # S_prev flattened across all conditions, matching E_prev/I_prev layout: (G, C*(T-1))
    S_prev_grid = S[:, :, :-1].reshape(tau_S_grid.size, -1)

    # 3. Fixed design points (masks depend only on activity, not on tau_S).
    mask_E_active = (E_prev > 0.5)
    mask_I_active = (I_prev > 0.5)
    LHS_E = (dE[mask_E_active] + E_prev[mask_E_active] / tau_E) / (E_max - E_prev[mask_E_active])
    LHS_I = (dI[mask_I_active] + I_prev[mask_I_active] / tau_I) / (I_max - I_prev[mask_I_active])
    base_E = np.column_stack([
        E_prev[mask_E_active], -I_prev[mask_E_active],
        np.ones(int(mask_E_active.sum())), stim_E_prev[mask_E_active],
    ])
    base_I = np.column_stack([
        E_prev[mask_I_active], -I_prev[mask_I_active],
        np.ones(int(mask_I_active.sum())), stim_I_prev[mask_I_active],
    ])

    def _fit(X, y):
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        return coef, float(resid @ resid)

    # Grid search tau_S using the high-SNR E equation only: the -S_prev column
    # captures the slow-adaptation structure and its shape identifies tau_S.
    best = None
    for gi in range(tau_S_grid.size):
        w, rE = _fit(np.column_stack([base_E, -S_prev_grid[gi, mask_E_active]]), LHS_E)
        if best is None or rE < best[0]:
            best = (rE, tau_S_grid[gi], w, gi)

    _, tau_S, w, gi_best = best
    # Fit the I equation at the same S (its own W_IS off the -S_prev column).
    g, _ = _fit(np.column_stack([base_I, -S_prev_grid[gi_best, mask_I_active]]), LHS_I)

    # 4. Map regression coefficients back to model parameters (coef * tau).
    W_EE = max(w[0] * tau_E, 0.0)
    W_EI = max(w[1] * tau_E, 0.0)
    C_E  = max(w[2] * tau_E, 0.0)
    XE   = max(w[3] * tau_E, 0.0)
    W_ES = max(w[4] * tau_E, 0.0)

    W_IE = max(g[0] * tau_I, 0.0)
    W_II = max(g[1] * tau_I, 0.0)
    C_I  = max(g[2] * tau_I, 0.0)
    XI   = max(g[3] * tau_I, 0.0)
    W_IS = max(g[4] * tau_I, 0.0)

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
        'tau_S': float(tau_S),
        'W_ES': float(W_ES),
        'W_IS': float(W_IS),
        'XE': float(XE),
        'XI': float(XI),
        's0_S': float(s0_S),
        # Objective-E EKF noise/init (log-variances); inert for objectives A-D.
        'kf_log_q_E': float(np.log(0.1 * sig_E)),
        'kf_log_q_I': float(np.log(0.1 * sig_I)),
        'kf_log_q_S': float(np.log(0.01 * sig_I)),
        'kf_log_sig_E': float(np.log(sig_E)),
        'kf_log_sig_I': float(np.log(sig_I)),
        'kf_log_p0_E': float(np.log(sig_E)),
        'kf_log_p0_I': float(np.log(sig_I)),
        'kf_log_p0_S': float(np.log(p0_S)),
    }
