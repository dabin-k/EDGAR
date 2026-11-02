import jax.numpy as jnp
from model2 import model

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

model_jax.DEFAULT_PARAMS = {
    'tau_E': 30.,   # time constant for excitatory population (in 10 ms bins)
    'tau_I': 20.,   # time constant for inhibitory population (in 10 ms bins)
    'W_EE': 0.01,   # weight of excitatory to excitatory connections
    'W_IE': 0.01,   # weight of inhibitory to excitatory connections
    'W_EI': 0.01,   # weight of excitatory to inhibitory connections
    'W_II': 0.01,   # weight of inhibitory to inhibitory connections
    'E_max': 20.0,
    'I_max': 20.0,
    'C_E': 0.001,
    'C_I': 0.001,
    'tau_S': 30.,   # slow-inhibition time constant (in 10 ms bins)
    'W_ES': 0.001,
    'W_IS': 0.001,
    'XE': 1.0,
    'XI': 1.0,
    's0_S': 1.0,  # learnable initial value of the latent S (GD-fit; seeds the scan carry)
}