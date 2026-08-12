import numpy as np
import jax.numpy as jnp


# Wilson-Cowan model 
def model(state_prev, y_prev, params):
    ''' Dynamics update function with non-linearity 

    Equation : 
        tau_E * dE/dt = -E + (E_max - E) * max(W_EE*E - W_EI*I + C_E + XE*stim_E, 0)
        tau_I * dI/dt = -I + (I_max - I) * max(W_IE*E - W_II*I + C_I + XI*stim_I, 0)

        C_E and C_I : constant baseline drive to the excitatory and inhibitory populations, respectively,
        stim_E and stim_I : binary variable that is 1 when the stimulus is present and 0 otherwise,

    Args:
        state_prev: a dictionary representing the previous state of the system -- not used in this function, but included for compatibility with the EDGAR framework
        y_prev: tuple of ((E_prev, I_prev), (stim_E_prev, stim_I_prev)) representing the previous state and previous stimuli
        params: dictionary of model parameters

    '''
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

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    E_dot = -E_prev + (E_max - E_prev)* np.maximum((W_EE * E_prev - W_EI * I_prev + C_E + XE * stim_E_prev), 0)
    I_dot = -I_prev + (I_max - I_prev)* np.maximum((W_IE * E_prev - W_II * I_prev + C_I + XI * stim_I_prev), 0)
    E = E_prev + E_dot/tau_E
    I = I_prev + I_dot/tau_I

    # hard code new state as empty state 
    new_state = {}
    return new_state, (E, I)

model.DEFAULT_PARAMS = {
    "tau_E": 60.0,
    "tau_I": 120.0,
    "C_E": 1.0,
    "C_I": 1.0,
    "XE": 3.0,
    "XI": 1.0,
    "tau_S": 300.0,
    "W_ES": 0.5,
    "W_IS": 0.5,
    "s0_S": 1.0,  # learnable initial value of the latent S (GD-fit per sample; seeds the scan carry)
}
