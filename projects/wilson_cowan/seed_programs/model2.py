import numpy as np
import jax.numpy as jnp

# Wilson-Cowan model WITH slow inhibition (WCS).
#
# Adds a hidden slow-inhibition variable S (from TRN activity) that low-pass filters the
# inhibitory activity and feeds back onto both populations. apply_model seeds S from the
# learnable `log_s0_S` param below and integrates all of E, I, S itself.
def model(z, u, params):
    """Right-hand side of the WCS ODE: time derivatives of E, I and the hidden S (per second).

    Equations:
        tau_S * dS/dt = -S + I
        tau_E * dE/dt = -E + (E_max - E) * max(W_EE * E - W_EI * I - W_ES * S + C_E + XE * stim_E, 0)
        tau_I * dI/dt = -I + (I_max - I) * max(W_IE * E - W_II * I - W_IS * S + C_I + XI * stim_I, 0)

    tau_E, tau_I, tau_S are in seconds. The framework integrates these derivatives over each bin.

    Args
    ----
    z : dict {'E','I','S'} — current values of the observed E, I and the hidden S.
    u : dict {'stim_E','stim_I'} — the stimulus of the bin.
    params : dict of model parameters in natural units (adds tau_S, W_ES, W_IS over the base WC set).

    Returns
    -------
    {'E': dE/dt, 'I': dI/dt, 'S': dS/dt} — same keys as z.
    """
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']  # slow inhibition onto excitatory population
    W_IS = params['W_IS']  # slow inhibition onto inhibitory population

    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']

    C_E = params['C_E']  # constant input to the excitatory population
    C_I = params['C_I']  # constant input to the inhibitory population

    XE = params['XE']  # transient input to the excitatory population
    XI = params['XI']  # transient input to the inhibitory population

    E = z['E']
    I = z['I']
    S = z['S']
    stim_E = u['stim_E']
    stim_I = u['stim_I']

    dS = (-S + I) / tau_S
    dE = (-E + (E_max - E) * np.maximum(W_EE * E - W_EI * I - W_ES * S + C_E + XE * stim_E, 0)) / tau_E
    dI = (-I + (I_max - I) * np.maximum(W_IE * E - W_II * I - W_IS * S + C_I + XI * stim_I, 0)) / tau_I

    return {'E': dE, 'I': dI, 'S': dS}


# Positive parameters are fitted in log space
model.DEFAULT_PARAMS = {
    'log_tau_E': np.log(0.3),   # time constant for excitatory population (s)
    'log_tau_I': np.log(0.2),   # time constant for inhibitory population (s)
    'log_W_EE': np.log(0.01),   # weight of excitatory to excitatory connections
    'log_W_IE': np.log(0.01),   # weight of inhibitory to excitatory connections
    'log_W_EI': np.log(0.01),   # weight of excitatory to inhibitory connections
    'log_W_II': np.log(0.01),   # weight of inhibitory to inhibitory connections
    'log_E_max': np.log(20.0),
    'log_I_max': np.log(20.0),
    'C_E': 0.001,
    'C_I': 0.001,
    'log_tau_S': np.log(0.3),   # slow-inhibition time constant (s)
    'log_W_ES': np.log(0.001),
    'log_W_IS': np.log(0.001),
    'log_XE': np.log(1.0),
    'log_XI': np.log(1.0),
    'log_s0_S': np.log(1.0),  # learnable initial value of the latent S (> 0; seeds the scan carry)
}
