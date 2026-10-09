import numpy as np
import jax.numpy as jnp


# Wilson-Cowan model 
def model(z, u, params):
    """Right-hand side of the WC ODE: time derivatives of E and I (per second).

    Equations:
        tau_E * dE/dt = -E + (E_max - E) * max(W_EE*E - W_EI*I + C_E + XE*stim_E, 0)
        tau_I * dI/dt = -I + (I_max - I) * max(W_IE*E - W_II*I + C_I + XI*stim_I, 0)

        tau_E, tau_I in seconds. The framework integrates these derivatives over each bin.
        C_E and C_I : constant baseline drive to the excitatory and inhibitory populations, respectively,
        stim_E and stim_I : binary variable that is 1 when the stimulus is present and 0 otherwise,

    Args:
        z: dict {'E','I'} — current values of the variables (no hidden variables in this model)
        u: dict {'stim_E','stim_I'} — the stimulus of the bin
        params: dictionary of model parameters in natural units

    Returns:
        {'E': dE/dt, 'I': dI/dt} — same keys as z
    """
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    tau_E = params['tau_E']
    tau_I = params['tau_I']

    C_E = params['C_E'] # constant input to the excitatory population
    C_I = params['C_I'] # constant input to the inhibitory population    

    XE = params['XE'] # transient input to the excitatary population 
    XI = params['XI'] # transient input to the inhibitory population
    
    ## p27 : inhibitory pulse is considered slower and is applied with an alpha function in the paper. For now use a boxcar function for both
    # tau_alpha = parameters['tau_alpha'] # time constant for alpha function for inhibitory input
    # XI_drive = XI * _alpha_drive(stimuli[:, 1], tau_alpha, h)

    E = z['E']
    I = z['I']
    stim_E = u['stim_E']
    stim_I = u['stim_I']

    dE = (-E + (E_max - E) * np.maximum(W_EE * E - W_EI * I + C_E + XE * stim_E, 0)) / tau_E
    dI = (-I + (I_max - I) * np.maximum(W_IE * E - W_II * I + C_I + XI * stim_I, 0)) / tau_I

    return {'E': dE, 'I': dI}

# Positive parameters are fitted in log space
model.DEFAULT_PARAMS = {
    "log_tau_E": np.log(0.06),   # time constants in seconds
    "log_tau_I": np.log(0.12),
    "log_W_EE": np.log(0.01),
    "log_W_EI": np.log(0.01),
    "log_W_IE": np.log(0.01),
    "log_W_II": np.log(0.01),
    "log_E_max": np.log(20.0),
    "log_I_max": np.log(20.0),
    # With E = I = 1 the W terms cancel, so the fixed point is E_max * C / (1 + C) ~ 1: the
    # baseline-normalised data level.
    "C_E": 0.05,
    "C_I": 0.05,
    "log_XE": np.log(3.0),
    "log_XI": np.log(1.0),
}
