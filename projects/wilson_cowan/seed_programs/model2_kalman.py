import numpy as np
import jax.numpy as jnp

# Wilson-Cowan model WITH slow inhibition (WCS) — OBJECTIVE-E (Kalman) VARIANT.
#
# Identical dynamics to model2.py, but the EKF noise/init hyperparameters (kf_*) are
# folded straight into DEFAULT_PARAMS instead of living on a separate
# `model.KF_DEFAULT_PARAMS` attribute. Reason: an attribute is lost on every LLM
# offspring (compile_model rebuilds model_fn purely from the translated code string,
# which never re-declares the attribute), whereas DEFAULT_PARAMS travels through the
# ModelSchema.default_params field and survives translation. Folding the kf_ keys in
# makes them exactly as durable as W_EE / s0_S.
#
# Swap this file onto `model2.py` when running objective E; use the no-kf model2.py
# for objectives A–D. load_data._split_params_kalman peels the kf_/s0_ keys back out,
# and config's param_penalty_exclude_prefixes: ["s0_","kf_"] keeps them out of the
# parsimony penalty.
def model(state_prev, y_prev, params):
    """Network with hidden slow-inhibition dynamics update.

    Equation :

        tau_S * dS/dt = -S + I

        tau_E * dE/dt = -E_prev + (E_max - E_prev) * max(W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev, 0)
        tau_I * dI/dt = -I_prev + (I_max - I_prev) * max(W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev, 0)
        E = E_prev + dE/dt
        I = I_prev + dI/dt

    Args
    ----
    state_prev : dict with the previous hidden state {'S': S_prev}.
    y_prev : dict {'E_prev','I_prev','stim_E_prev','stim_I_prev'} — the previous
        observation bundled with the previous stimulus.
    params : dict of model parameters (adds tau_S, W_ES, W_IS over the base WC set).
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

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    S_prev = state_prev['S']

    S_dot = (-S_prev + I_prev) / tau_S
    S = S_prev + S_dot

    E_dot = -E_prev + (E_max - E_prev) * np.maximum((W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev), 0)
    E_dot /= tau_E
    I_dot = -I_prev + (I_max - I_prev) * np.maximum((W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev), 0)
    I_dot /= tau_I
    E = E_prev + E_dot
    I = I_prev + I_dot

    new_state = {'S': S}
    return new_state, (E, I)


model.DEFAULT_PARAMS = {
    'tau_E': 300.,  # time constant for excitatory population
    'tau_I': 200.,  # time constant for inhibitory population
    'W_EE': 0.01,   # weight of excitatory to excitatory connections
    'W_IE': 0.01,   # weight of inhibitory to excitatory connections
    'W_EI': 0.01,   # weight of excitatory to inhibitory connections
    'W_II': 0.01,   # weight of inhibitory to inhibitory connections
    'E_max': 20.0,
    'I_max': 20.0,
    'C_E': 0.001,
    'C_I': 0.001,
    'tau_S': 300,
    'W_ES': 0.001,
    'W_IS': 0.001,
    'XE': 1.0,
    'XI': 1.0,
    's0_S': 1.0,  # learnable initial value of the latent S (GD-fit; seeds the scan carry)
    # Objective-E (EKF) noise/init hyperparameters, LOG-VARIANCES for positivity. Folded in here
    # (not a separate attribute) so they survive LLM translation. param_est2 overwrites these with
    # data-driven inits (obs noise from the smoothing residual, etc.); these are the durable
    # fallback. Consumed by load_data._split_params_kalman -> _kf_hyper. Excluded from the
    # parsimony penalty via config's param_penalty_exclude_prefixes.
    'kf_log_q_E': float(np.log(1e-3)),    # process-noise variance on E
    'kf_log_q_I': float(np.log(1e-3)),    # process-noise variance on I
    'kf_log_q_S': float(np.log(1e-4)),    # process-noise variance on latent S
    'kf_log_sig_E': float(np.log(1e-1)),  # observation-noise variance on E
    'kf_log_sig_I': float(np.log(1e-1)),  # observation-noise variance on I
    'kf_log_p0_E': float(np.log(1e-1)),   # initial-state variance on E
    'kf_log_p0_I': float(np.log(1e-1)),   # initial-state variance on I
    'kf_log_p0_S': float(np.log(1e-1)),   # initial-state variance on S
}


def model_jax(state_prev, y_prev, params):
    """JAX mirror, differentiable under jax.grad + lax.scan.

    Identical equations to the numpy `model` above with `jnp.maximum` in place of
    `np.maximum`, so it drops straight into `apply_model` for gradient-descent fitting.
    """
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']
    W_IS = params['W_IS']

    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']

    C_E = params['C_E']
    C_I = params['C_I']

    XE = params['XE']
    XI = params['XI']

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    S_prev = state_prev['S']

    S_dot = (-S_prev + I_prev) / tau_S
    S = S_prev + S_dot

    E_dot = -E_prev + (E_max - E_prev) * jnp.maximum((W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev), 0)
    E_dot /= tau_E
    I_dot = -I_prev + (I_max - I_prev) * jnp.maximum((W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev), 0)
    I_dot /= tau_I
    E = E_prev + E_dot
    I = I_prev + I_dot

    new_state = {'S': S}
    return new_state, (E, I)


model_jax.DEFAULT_PARAMS = model.DEFAULT_PARAMS
