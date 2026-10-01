import matplotlib.pyplot as plt
import numpy as np

def lin_model(state_prev, y_prev, params):
    """
        Lin's model t -> t+1 update function. 
        Returns new state and new output (E, I) at time t+1, given state, previous y at time t and params.
    """
    #State variables
    S = state_prev['S']
    L = state_prev['L']
    R = state_prev['R']
    B = state_prev['B']
    T = state_prev['T']

    #Parameters
    E_max = params['E_max']
    I_max = params['I_max']
    L_max = params['L_max']

    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']
    W_IS = params['W_IS']
    W_EL = params['W_EL']
    W_IL = params['W_IL']
    W_LB = params['W_LB']
    W_LT = params['W_LT']
    W_LE = params['W_LE']
    W_RE = params['W_RE']
    W_RL = params['W_RL']
    
    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']
    tau_R = params['tau_R']
    tau_B = params['tau_B']

    C_E = params['C_E'] # constant input to the excitatory population
    C_I = params['C_I'] # constant input to the inhibitory population    
    C_L = params['C_L']
    C_R = params['C_R']

    XE = params['XE'] # transient input to the excitatory population 
    XI = params['XI'] # transient input to the inhibitory population

    #E, I and stims
    E = y_prev['E']
    I = y_prev['I']
    stim_E = y_prev['stim_E']
    stim_I = y_prev['stim_I']

    #Update equations
    E_upd = E + 1/tau_E *(-E + (E_max-E)*np.maximum((W_EE*E - W_EI*I + C_E + XE*stim_E - W_ES*S + W_EL*L), 0))
    I_upd = I + 1/tau_I *(-I + (I_max-I)*np.maximum((W_IE*E - W_II*I + C_I + XI*stim_I - W_IS*S + W_IL*L), 0))
    S_upd = S + 1/tau_S * (-S + I)
    L_upd = L + 1/tau_E * (-L + (L_max-L)*np.maximum((W_LE*E + C_L + W_LB*L*B - W_LT*T), 0))
    B_upd = B + 1/tau_B * (-B + (1-B)*np.maximum((1 - L), 0))
    R_upd = R + 1/tau_R * (-R + (1-R)*np.maximum((W_RE*E + W_RL*L - C_R), 0))
    T_upd = T + 1/tau_S * (-T + R)

    new_state = {'S': S_upd, 'L': L_upd, 'R': R_upd, 'B': B_upd, 'T': T_upd}

    return new_state, (E_upd, I_upd)

def init_lin_steadystate(params: dict, E_0: float = 1.0, I_0: float = 1.0, L_0: float = 0.5, R_0: float = 0.5) -> tuple[dict, tuple[float, float], dict]:
    """Initialize the Lin's model for E, I, S, L, B, R, T steady-state values.

    Args:
        params: Dictionary of model parameters.
        E_0: Target steady-state value for the excitatory population.
        I_0: Target steady-state value for the inhibitory population.
        L_0: Target steady-state value for the L population.
        R_0: Target steady-state value for the R population.

    Returns:
        tuple: (initial_state, initial_y, updated_params)
            - initial_state (dict): Initial values of S, L, R, B, T.
            - initial_y (tuple): Target steady state values (E_0, I_0).
            - updated_params (dict): Parameters with C_E, C_I, C_L, C_R set.
    """
    if L_0 > 1.0:
        raise ValueError("L_0 must be < 1.0 to ensure positive B_0.")
    # Derive dependent steady-state variables
    S_0 = I_0
    T_0 = R_0
    
    # B_0 is uniquely determined by L_0
    B_0 = (1.0 - L_0) / (2.0 - L_0)

    # Extract model parameters
    E_max = params['E_max']
    I_max = params['I_max']
    L_max = params['L_max']

    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']
    W_IS = params['W_IS']
    W_EL = params['W_EL']
    W_IL = params['W_IL']
    W_LB = params['W_LB']
    W_LT = params['W_LT']
    W_LE = params['W_LE']
    W_RE = params['W_RE']
    W_RL = params['W_RL']

    # Choose constant inputs to satisfy steady-state equations
    params['C_E'] = E_0 / (E_max - E_0) - W_EE * E_0 + (W_EI + W_ES) * I_0 - W_EL * L_0
    params['C_I'] = I_0 / (I_max - I_0) - W_IE * E_0 + (W_II + W_IS) * I_0 - W_IL * L_0
    params['C_L'] = L_0 / (L_max - L_0) - W_LE * E_0 - W_LB * L_0 * B_0 + W_LT * R_0
    params['C_R'] = W_RE * E_0 + W_RL * L_0 - R_0 / (1.0 - R_0)

    initial_state = {
        'S': S_0,
        'L': L_0,
        'R': R_0,
        'B': B_0,
        'T': T_0
    }
    return initial_state, (E_0, I_0), params

def generate_data(params: dict, tmax: int, stim_designs: tuple[np.ndarray, np.ndarray], initial_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for the Lin model

    Args:
        params (dict): Dictionary of model parameters.
        tmax (int): Total number of time steps.
        stim_designs (tuple[np.ndarray, np.ndarray]): Tuple containing stimulus arrays for excitatory and inhibitory populations.
        initial_state (dict): Initial state values for E, I, L, R populations.

    Returns:
        tuple: (Et, It) - time series for excitatory and inhibitory populations.
    """
    state, (E0, I0), params = init_lin_steadystate(params, initial_state['E0'], initial_state['I0'], initial_state['L0'], initial_state['R0'])
    stim_E_design, stim_I_design = stim_designs
    ts = np.arange(tmax)
    Et = np.zeros(tmax)
    It = np.zeros(tmax)
    Et[0], It[0] = E0, I0
    for t in range(1, tmax):
        y_prev = {
            'E': Et[t-1],
            'I': It[t-1],
            'stim_E': stim_E_design[t-1],
            'stim_I': stim_I_design[t-1]
        }
        state, (E, I) = lin_model(state, y_prev, params)
        Et[t] = E
        It[t] = I

    return Et, It

if __name__ == "__main__":
    #To compare to experimental units, t here = 1 corresponds to 1ms
    seed = 22
    rng = np.random.default_rng(seed)
    params_lin = {
    'tau_E' : rng.normal(6.0, 0.1), # time constant for excitatory population
    'tau_I' : rng.normal(5.0, 0.1), # time constant for inhibitory population
    'tau_S' : rng.normal(100, 2), # time constant for slow inhibition
    'W_EE' : rng.normal(0.2, 0.01),  # weight of excitatory to excitatory connections
    'W_IE' : rng.normal(0.2, 0.01),  # weight of inhibitory to excitatory connections
    'W_EI' : rng.normal(0.1, 0.005),  # weight of excitatory to inhibitory connections
    'W_II' : rng.normal(0.15, 0.005),  # weight of inhibitory to inhibitory connections
    'W_ES' : rng.normal(0.05, 0.005),  # weight of slow inhibition to excitatory connections
    'W_IS' : rng.normal(0.05, 0.005),  # weight of slow inhibition to inhibitory connections
    'E_max' : 10.0,
    'I_max' : 10.0,
    'XE' : rng.normal(5, 0.1),
    'XI' : rng.normal(1, 0.1),
    'tau_R' : rng.normal(50, 5.0), # time constant for R population
    'tau_B' : rng.normal(150, 5.0), # time constant for B population
    'W_EL' : rng.normal(0.08, 0.01),  # Further adjusted weight for L to excitatory connections
    'W_IL' : rng.normal(0.08, 0.01),  # Further adjusted weight for L to inhibitory connections
    'W_LB' : rng.normal(0.08, 0.01),  # weight of B to L connections
    'W_LT' : rng.normal(0.5, 0.02),  # Kept weight of T to L connections
    'W_LE' : rng.normal(0.08, 0.01),  # Further adjusted weight for E to L connections
    'W_RE' : rng.normal(0.08, 0.01),  # weight of E to R connections
    'W_RL' : rng.normal(0.08, 0.01),  # weight of L to R connections
    'L_max' : 2.0,
    }

    #Initial conditions
    initial_state = {
        'E0': 1.0,  # Initial value for excitatory population
        'I0': 1.0,  # Initial value for inhibitory population
        'L0': 1.0,  # Initial value for L population
        'R0': 0.0,  # Initial value for R population
    }

    print(f"Params: {params_lin}")
    print(f"Initial conditions: {initial_state}")
    tmax = 1510
    stim_t = 10
    delta_t = 1
    interpulse_ts = (5, 50, 100, 200)
    stimE_design = np.zeros((len(interpulse_ts), tmax))
    stimI_design = np.zeros((len(interpulse_ts), tmax))

    #Stim design
    pulse_type = "EI" # Options: "E", "I", "EI", "IE"
    for i, ip_t in enumerate(interpulse_ts):
        if pulse_type.startswith("E"):
            stimE_design[i, stim_t:stim_t+delta_t] = 1

        elif pulse_type.startswith("I"):
            stimI_design[i, stim_t:stim_t+delta_t] = 1

        if pulse_type.endswith("E"):
            stimE_design[i, ip_t:ip_t+delta_t] = 1
        elif pulse_type.endswith("I"):
            stimI_design[i, ip_t:ip_t+delta_t] = 1


    Ets, Its = [], []
    for i in range(len(interpulse_ts)):
        stim_designs = (stimE_design[i], stimI_design[i])
        Et, It = generate_data(params_lin, tmax, stim_designs, initial_state)
        Ets.append(Et)
        Its.append(It)

    #Plotting
    fig, axes = plt.subplots(nrows = 1, ncols = len(interpulse_ts), figsize=(4*len(interpulse_ts), 4), tight_layout=True)
    for i in range(len(interpulse_ts)):
        axes[i].plot(Ets[i], label='E(t)', color = 'C0')
        axes[i].plot(Its[i], label='I(t)', color = 'C1')
        axes[i].set_title(f"Interpulse time: {interpulse_ts[i]}")
        axes[i].legend()

    plt.savefig(f"simulated_dynamics_pulse{pulse_type}_E0{initial_state['E0']}_I0{initial_state['I0']}_L0{initial_state['L0']}_R0{initial_state['R0']}.png", dpi=150)