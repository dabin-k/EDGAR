import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_params(random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    #Model parameters
    params_wcs = {
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
    'XI' : rng.normal(1, 0.1)
    }
    params_lin = params_wcs.copy()
    params_lin.update({
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
    })
    return params_wcs, params_lin

def WCS_model(state_prev, y_prev, params):
    """
        Wilson-Cowan model + slow inhibitiom t -> t+1 update function.
        Returns new state and new output (E, I) at time t+1, given state, previous y at time t and params.
    """
    #State variables
    S = state_prev['S']

    #Parameters
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

    C_E = params['C_E'] # constant input to the excitatory population
    C_I = params['C_I'] # constant input to the inhibitory population    

    XE = params['XE'] # transient input to the excitatory population 
    XI = params['XI'] # transient input to the inhibitory population

    #E, I and stims
    E = y_prev['E']
    I = y_prev['I']
    stim_E = y_prev['stim_E']
    stim_I = y_prev['stim_I']

    #Update equations
    E_upd = E + 1/tau_E *(-E + (E_max-E)*np.maximum((W_EE*E - W_EI*I + C_E + XE*stim_E - W_ES*S), 0))
    I_upd = I + 1/tau_I *(-I + (I_max-I)*np.maximum((W_IE*E - W_II*I + C_I + XI*stim_I - W_IS*S), 0))
    S_upd = S + 1/tau_S * (-S + I)

    new_state = {'S': S_upd}

    return new_state, (E_upd, I_upd)

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

def init_WCS_steadystate(params: dict, E_0: float = 1.0, I_0: float = 1.0) -> tuple[dict, tuple[float, float], dict]:
    """Initialize the WCS model for E, I, S steady-state values.

    Args:
        params: Dictionary of model parameters.
        E_0: Target steady-state value for the excitatory population.
        I_0: Target steady-state value for the inhibitory population.

    Returns:
        tuple: (initial_state, initial_y, updated_params)
            - initial_state (dict): Initial values of S.
            - initial_y (tuple): Target steady state values (E_0, I_0).
            - updated_params (dict): Parameters with C_E, C_I set.
    """
    S_0 = I_0 #S0 steady-state value

    E_max, I_max, W_EE, W_EI, W_IE, W_II, W_ES, W_IS = params['E_max'], params['I_max'], params['W_EE'], params['W_EI'], params['W_IE'], params['W_II'], params['W_ES'], params['W_IS']
    # Choose C_E, C_I to satisfy steady-state equations
    params['C_E'] = E_0 / (E_max - E_0) - W_EE * E_0 + (W_EI + W_ES) * I_0
    params['C_I'] = I_0 / (I_max - I_0) - W_IE * E_0 + (W_II + W_IS) * I_0


    return {'S': S_0}, (E_0, I_0), params


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
    if L_0 >= 1.0:
        raise ValueError("L_0 must be less than 1.0 to ensure non-zero B_0.")
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

def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to the data.

    Args:
        data (np.ndarray): Input data array.
        noise_level (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy data.
    """
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

def generate_model_data(model_name: str, params: dict, tmax: int, stim_designs: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for the specified model.

    Args:
        model_name (str): Name of the model ('wcs' or 'lin').
        params (dict): Dictionary of model parameters.
        tmax (int): Total number of time steps.
        stim_designs (tuple[np.ndarray, np.ndarray]): Tuple containing stimulus arrays for excitatory and inhibitory populations.

    Returns:
        tuple: (Et, It) - time series for excitatory and inhibitory populations.
    """
    if model_name == "wcs":
        model = WCS_model
        init_steadystate = init_WCS_steadystate
    elif model_name == "lin":
        model = lin_model
        init_steadystate = init_lin_steadystate
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'wcs' or 'lin'.")

    state, (E0, I0), params = init_steadystate(params)
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
        state, (E, I) = model(state, y_prev, params)
        Et[t] = E
        It[t] = I

    return Et, It

def plot_simulation_results(
    save_path: Path, 
    n_samples: int, 
    interpulse_ts: tuple[int], 
    pulse_type_labels: list[str],
    all_E_lin_clean: np.ndarray, 
    all_I_lin_clean: np.ndarray, 
    all_E_wcs_clean: np.ndarray, 
    all_I_wcs_clean: np.ndarray,
    all_E_lin_folded: np.ndarray,
    all_I_lin_folded: np.ndarray,
    all_E_wcs_folded: np.ndarray,
    all_I_wcs_folded: np.ndarray,
    time: np.ndarray,
):
    """
    Generates and saves plots comparing Lin and WCS model simulations.
    Overlays folded repeats on top of clean Lin and WCS data using faded lines (small alpha).
    """
    n_interpulse_ts = len(interpulse_ts)
    n_folds = all_E_lin_folded.shape[2]
    for sample_idx in range(n_samples):
        for p_idx, p_type in enumerate(pulse_type_labels):
            for ip_idx, ip_t in enumerate(interpulse_ts):
                stim_condition_idx = p_idx * n_interpulse_ts + ip_idx

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), tight_layout=True)

                # 1. Plot Lin Model
                # Plot folded noisy repeats first with small alpha so they are faded
                for f in range(n_folds):
                    axes[0].plot(
                        time, 
                        all_E_lin_folded[sample_idx, stim_condition_idx, f, :], 
                        color='C0', 
                        alpha=0.3, 
                        linewidth=1.0,
                        label='E(t) Folded' if f == 0 else ""
                    )
                    axes[0].plot(
                        time, 
                        all_I_lin_folded[sample_idx, stim_condition_idx, f, :], 
                        color='C1', 
                        alpha=0.3, 
                        linewidth=1.0,
                        label='I(t) Folded' if f == 0 else ""
                    )

                # Plot Clean/deterministic trace on top with a solid line
                axes[0].plot(
                    time, 
                    all_E_lin_clean[sample_idx, stim_condition_idx, 0, :], 
                    label='E(t) Lin Clean', 
                    color='C0', 
                    linewidth=2.0
                )
                axes[0].plot(
                    time, 
                    all_I_lin_clean[sample_idx, stim_condition_idx, 0, :], 
                    label='I(t) Lin Clean', 
                    color='C1', 
                    linewidth=2.0
                )
                axes[0].set_title(f"Lin Model - Pulse Type: {p_type}, Interpulse: {ip_t}ms")
                axes[0].legend()
                axes[0].set_xlabel("Time (ms)")
                axes[0].set_ylabel("Activity")

                # 2. Plot WCS Model
                # Plot folded noisy repeats first with small alpha so they are faded
                for f in range(n_folds):
                    axes[1].plot(
                        time, 
                        all_E_wcs_folded[sample_idx, stim_condition_idx, f, :], 
                        color='C0', 
                        alpha=0.3, 
                        linewidth=1.0,
                        linestyle='--',
                        label='E(t) Folded' if f == 0 else ""
                    )
                    axes[1].plot(
                        time, 
                        all_I_wcs_folded[sample_idx, stim_condition_idx, f, :], 
                        color='C1', 
                        alpha=0.3, 
                        linewidth=1.0,
                        linestyle='--',
                        label='I(t) Folded' if f == 0 else ""
                    )

                # Plot Clean/deterministic trace on top with a solid line
                axes[1].plot(
                    time, 
                    all_E_wcs_clean[sample_idx, stim_condition_idx, 0, :], 
                    label='E(t) WCS Clean', 
                    color='C0', 
                    linestyle='--', 
                    linewidth=2.0
                )
                axes[1].plot(
                    time, 
                    all_I_wcs_clean[sample_idx, stim_condition_idx, 0, :], 
                    label='I(t) WCS Clean', 
                    color='C1', 
                    linestyle='--', 
                    linewidth=2.0
                )
                axes[1].set_title(f"WCS Model - Pulse Type: {p_type}, Interpulse: {ip_t}ms")
                axes[1].legend()
                axes[1].set_xlabel("Time (ms)")
                axes[1].set_ylabel("Activity")

                plt.savefig(save_path / f"sample_{sample_idx}_pulse_{p_type}_interpulse_{ip_t}.png", dpi=150)
                plt.close(fig) # Close the figure to free memory


def _generate_stimulus_designs(p_type: str, stim_t: int, stim_dur: int, ip_t: int, tmax: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates stimulus design arrays for excitatory and inhibitory populations.

    Args:
        p_type (str): Pulse type string (e.g., "EE", "EI", "IE", "II").
        stim_t (int): Time step at which the first stimulus is applied.
        stim_dur (int): Duration of each stimulus pulse.
        ip_t (int): Interpulse time for the second stimulus.
        tmax (int): Total number of time steps.

    Returns:
        tuple[np.ndarray, np.ndarray]: (stimE_design_current, stimI_design_current)
    """
    stimE_design_current = np.zeros(tmax)
    stimI_design_current = np.zeros(tmax)

    if p_type.startswith("E"):
        stimE_design_current[stim_t:stim_t+stim_dur] = 1
    elif p_type.startswith("I"):
        stimI_design_current[stim_t:stim_t+stim_dur] = 1

    if p_type.endswith("E"):
        stimE_design_current[ip_t:ip_t+stim_dur] = 1
    elif p_type.endswith("I"):
        stimI_design_current[ip_t:ip_t+stim_dur] = 1
        
    return stimE_design_current, stimI_design_current

def generate_synthetic_dataset(
    save_path: Path, 
    n_samples: int = 10, 
    tmax: int = 1510, 
    stim_t: int = 10, 
    stim_dur: int = 1, 
    interpulse_ts: tuple[int] = (5, 50, 100, 200), 
    noise_level: float = 0.05, 
    n_folds: int = 5,
    repeats_per_fold: int = 10,
    delta: float = 0.5, 
    max_retries: int = 10
):
    """
    Generate synthetic datasets for Lins and WCS models, saving clean and folded variants,
    and plotting comparison results with fold overlays.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    pulse_type_labels = ["EE", "EI", "IE", "II"]
    n_pulse_types = len(pulse_type_labels)
    n_interpulse_ts = len(interpulse_ts)
    n_stim_conditions = n_pulse_types * n_interpulse_ts

    # 1. Clean Dataset Arrays (nfolds=1)
    all_E_lin_clean = np.zeros((n_samples, n_stim_conditions, 1, tmax))
    all_I_lin_clean = np.zeros((n_samples, n_stim_conditions, 1, tmax))
    all_E_wcs_clean = np.zeros((n_samples, n_stim_conditions, 1, tmax))
    all_I_wcs_clean = np.zeros((n_samples, n_stim_conditions, 1, tmax))

    # 2. Folded Dataset Arrays (nfolds=n_folds)
    all_E_lin_folded = np.zeros((n_samples, n_stim_conditions, n_folds, tmax))
    all_I_lin_folded = np.zeros((n_samples, n_stim_conditions, n_folds, tmax))
    all_E_wcs_folded = np.zeros((n_samples, n_stim_conditions, n_folds, tmax))
    all_I_wcs_folded = np.zeros((n_samples, n_stim_conditions, n_folds, tmax))
    
    # E_design and I_design are now (n_stim_conditions, tmax)
    all_E_design = np.zeros((n_stim_conditions, tmax))
    all_I_design = np.zeros((n_stim_conditions, tmax))
    
    # pulse_type is a fixed array of strings, independent of samples, repeats or interpulse times
    formatted_pulse_type_labels = [f"paired_{pt}" for pt in pulse_type_labels]
    all_pulse_type_array = np.array(formatted_pulse_type_labels)

    time_array = np.arange(-stim_t, tmax - stim_t)  # Time values, with 0 corresponding to first stimulus onset

    # Generate E_design and I_design once, as they are independent of samples
    for p_idx, p_type in enumerate(pulse_type_labels):
        for ip_idx, ip_t in enumerate(interpulse_ts):
            stim_condition_idx = p_idx * n_interpulse_ts + ip_idx

            stimE_design_current, stimI_design_current = _generate_stimulus_designs(
                p_type, stim_t, stim_dur, ip_t, tmax
            )
            
            all_E_design[stim_condition_idx, :] = stimE_design_current
            all_I_design[stim_condition_idx, :] = stimI_design_current

    for sample_idx in range(n_samples):
        valid_sample = False
        for retry_count in range(max_retries):
            params_wcs, params_lin = get_params(random_seed=sample_idx + retry_count * 1000) # Vary seed for retries
            
            # Temporary storage to hold simulated data before we confirm the sample is valid
            temp_E_lin = np.zeros((n_stim_conditions, tmax))
            temp_I_lin = np.zeros((n_stim_conditions, tmax))
            temp_E_wcs = np.zeros((n_stim_conditions, tmax))
            temp_I_wcs = np.zeros((n_stim_conditions, tmax))
            
            sample_ok = True
            for p_idx, p_type in enumerate(pulse_type_labels):
                for ip_idx, ip_t in enumerate(interpulse_ts):
                    stim_condition_idx = p_idx * n_interpulse_ts + ip_idx

                    # Retrieve the pre-generated stimulus design
                    stimE_design_current = all_E_design[stim_condition_idx, :]
                    stimI_design_current = all_I_design[stim_condition_idx, :]
                    stim_designs = (stimE_design_current, stimI_design_current)
                    
                    Et_lin, It_lin = generate_model_data("lin", params_lin.copy(), tmax, stim_designs)
                    Et_wcs, It_wcs = generate_model_data("wcs", params_wcs.copy(), tmax, stim_designs)
                    
                    # Verification check: check if E and I of the lin data are within delta of 1 at tmax (index -1)
                    if abs(Et_lin[-1] - 1.0) > delta or abs(It_lin[-1] - 1.0) > delta:
                        sample_ok = False
                        break
                    
                    temp_E_lin[stim_condition_idx, :] = Et_lin
                    temp_I_lin[stim_condition_idx, :] = It_lin
                    temp_E_wcs[stim_condition_idx, :] = Et_wcs
                    temp_I_wcs[stim_condition_idx, :] = It_wcs
                if not sample_ok:
                    break
            
            if sample_ok:
                # 1. Clean deterministic values
                all_E_lin_clean[sample_idx, :, 0, :] = temp_E_lin
                all_I_lin_clean[sample_idx, :, 0, :] = temp_I_lin
                all_E_wcs_clean[sample_idx, :, 0, :] = temp_E_wcs
                all_I_wcs_clean[sample_idx, :, 0, :] = temp_I_wcs

                # 2. Noisy trial-averaged fold values
                for stim_cond_idx in range(n_stim_conditions):
                    for f in range(n_folds):
                        repeats_E_lin = [add_noise(temp_E_lin[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                        repeats_I_lin = [add_noise(temp_I_lin[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                        repeats_E_wcs = [add_noise(temp_E_wcs[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                        repeats_I_wcs = [add_noise(temp_I_wcs[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                        
                        all_E_lin_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_E_lin, axis=0)
                        all_I_lin_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_I_lin, axis=0)
                        all_E_wcs_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_E_wcs, axis=0)
                        all_I_wcs_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_I_wcs, axis=0)
                        
                valid_sample = True
                break
        
        if not valid_sample:
            # Fallback in case no valid sample is generated after max_retries
            print(f"Warning: Could not find parameters satisfying the stability condition within {max_retries} retries for sample {sample_idx}.")
            # Store fallback deterministic
            all_E_lin_clean[sample_idx, :, 0, :] = temp_E_lin
            all_I_lin_clean[sample_idx, :, 0, :] = temp_I_lin
            all_E_wcs_clean[sample_idx, :, 0, :] = temp_E_wcs
            all_I_wcs_clean[sample_idx, :, 0, :] = temp_I_wcs
            
            # Store fallback noisy folds
            for stim_cond_idx in range(n_stim_conditions):
                for f in range(n_folds):
                    repeats_E_lin = [add_noise(temp_E_lin[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                    repeats_I_lin = [add_noise(temp_I_lin[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                    repeats_E_wcs = [add_noise(temp_E_wcs[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                    repeats_I_wcs = [add_noise(temp_I_wcs[stim_cond_idx, :], noise_level) for _ in range(repeats_per_fold)]
                    
                    all_E_lin_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_E_lin, axis=0)
                    all_I_lin_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_I_lin, axis=0)
                    all_E_wcs_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_E_wcs, axis=0)
                    all_I_wcs_folded[sample_idx, stim_cond_idx, f, :] = np.mean(repeats_I_wcs, axis=0)
                    
    # Save clean datasets
    np.savez(
        save_path / "synthetic_data_clean.npz",
        E=all_E_lin_clean,
        I=all_I_lin_clean,
        E_design=all_E_design,
        I_design=all_I_design,
        pulse_type=all_pulse_type_array,
        time=time_array
    )
    np.savez(
        save_path / "synthetic_data_clean_wcs.npz",
        E=all_E_wcs_clean,
        I=all_I_wcs_clean,
        E_design=all_E_design,
        I_design=all_I_design,
        pulse_type=all_pulse_type_array,
        time=time_array
    )

    # Save folded datasets
    np.savez(
        save_path / "synthetic_data_folded.npz",
        E=all_E_lin_folded,
        I=all_I_lin_folded,
        E_design=all_E_design,
        I_design=all_I_design,
        pulse_type=all_pulse_type_array,
        time=time_array
    )
    np.savez(
        save_path / "synthetic_data_folded_wcs.npz",
        E=all_E_wcs_folded,
        I=all_I_wcs_folded,
        E_design=all_E_design,
        I_design=all_I_design,
        pulse_type=all_pulse_type_array,
        time=time_array
    )

    # Call plotting function comparing Lin clean/folded and WCS clean/folded
    plot_simulation_results(
        save_path=save_path,
        n_samples=n_samples,
        interpulse_ts=interpulse_ts,
        pulse_type_labels=pulse_type_labels,
        all_E_lin_clean=all_E_lin_clean,
        all_I_lin_clean=all_I_lin_clean,
        all_E_wcs_clean=all_E_wcs_clean,
        all_I_wcs_clean=all_I_wcs_clean,
        all_E_lin_folded=all_E_lin_folded,
        all_I_lin_folded=all_I_lin_folded,
        all_E_wcs_folded=all_E_wcs_folded,
        all_I_wcs_folded=all_I_wcs_folded,
        time=time_array
    )

if __name__ == "__main__":
    generate_synthetic_dataset(
        save_path=Path("synthetic"), 
        n_samples=10, 
        tmax=1510, 
        stim_t=10, 
        stim_dur=1, 
        interpulse_ts=(5, 50, 100, 200), 
        noise_level=0.2, 
        n_folds=5,
        repeats_per_fold=10,
        delta=0.5, 
        max_retries=10
    )