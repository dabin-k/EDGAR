import numpy as np
import matplotlib.pyplot as plt
plt.style.use('scientific')

# Wilson-Cowan model 
def model(state_prev, y_prev, params):
    ''' Wilson-Cowan model dynamics update function. 
    Args
    ----
    state: a dictionary representing the previous state of the system -- not used in this function, but included for compatibility with the EDGAR framework
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

if __name__ == "__main__":
    # Generate unnormalized data
    params = {
    'tau_E' : 10.0, # time constant for excitatory population
    'tau_I' : 20.0, # time constant for inhibitory population
    'W_EE' : 0.15,  # weight of excitatory to excitatory connections
    'W_IE' : 0.1,  # weight of inhibitory to excitatory connections
    'W_EI' : 0.15,  # weight of excitatory to inhibitory connections
    'W_II' : 0.1,  # weight of inhibitory to inhibitory connections
    'E_max' : 10.0,
    'I_max' : 10.0,
    'C_E' : 0.01,
    'C_I' : 0.01,
    'XE' : 0.1,
    'XI' : 0.1
    }
    tmax = 1000
    stim_t = 50
    stim_E = np.zeros(tmax)
    stim_I = np.zeros(tmax)
    stim_E[stim_t] = 1
    stim_I[stim_t] = 1

    ts = np.arange(tmax)
    Et = np.zeros(tmax)
    It = np.zeros(tmax)
    Et[0], It[0] = 0.1, 0.1
    for t in range(1, tmax):
        y_prev = {
            'E_prev': Et[t-1],
            'I_prev': It[t-1],
            'stim_E_prev': stim_E[t-1],
            'stim_I_prev': stim_I[t-1]
        }
        _, (E, I) = model({}, y_prev, params)
        Et[t] = E
        It[t] = I

    #Normalize data so t_max steady state is 1
    E_ss = Et[-1]
    I_ss = It[-1]
    Et_norm = Et / E_ss
    It_norm = It / I_ss

    #Apply normalization to parameters, should generate same dynamics as above
    params_norm = params.copy()
    params_norm['W_EE'] *= E_ss
    params_norm['W_IE'] *= E_ss
    params_norm['W_EI'] *= I_ss
    params_norm['W_II'] *= I_ss
    params_norm['E_max'] /= E_ss
    params_norm['I_max'] /= I_ss
    Et_norm2 = np.zeros(tmax)
    It_norm2 = np.zeros(tmax)
    Et_norm2[0], It_norm2[0] = 0.1 / E_ss, 0.1 / I_ss
    for t in range(1, tmax):
        y_prev = {
            'E_prev': Et_norm2[t-1],
            'I_prev': It_norm2[t-1],
            'stim_E_prev': stim_E[t-1],
            'stim_I_prev': stim_I[t-1]
        }
        _, (E, I) = model({}, y_prev, params_norm)
        Et_norm2[t] = E
        It_norm2[t] = I

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    axes[0, 0].plot(ts, Et, label='E(t)', color = 'C0')
    axes[0, 1].plot(ts, It, label='I(t)', color = 'C1')
    axes[1, 0].plot(ts, Et_norm2, label='E(t) (Normalized)', color = 'black')
    axes[1, 1].plot(ts, It_norm2, label='I(t) (Normalized)', color = 'black')
    axes[1, 0].plot(ts, Et_norm, label='E(t) (Normalized)', color = 'C0', linestyle = ':')
    axes[1, 1].plot(ts, It_norm, label='I(t) (Normalized)', color = 'C1',linestyle=':')
    plt.savefig('wc_dynamics.png', dpi=150)