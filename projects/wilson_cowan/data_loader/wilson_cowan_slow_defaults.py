# Defaults for the Wilson-Cowan model WITH slow inhibition (WCS).
# Same simulation grid as the base WC model; adds a hidden slow-inhibition
# variable S (from TRN activity) with its own time constant and weights.

N_SAMPLES = 8          # cells / animals; parameters are fit per sample
N_REPEATS = 12         # noisy repeats per (sample, stim condition)

# p26 : times considered : -50ms to 600ms. I suspect stimulus was shown at 0ms
DT = 1/30 # 1/30ms
N_TIMES_MS = 350
N_TIMES = int(N_TIMES_MS / DT)
STIM_ONSET_MS = 50
STIM_DUR_MS = 2 # one of 1, 2, 4

STIM_ONSET = int(STIM_ONSET_MS / DT)
STIM_DUR = int(STIM_DUR_MS / DT)

H = DT / 1000 # in ms

# Wilson-Cowan with slow inhibition model parameters - p23 of paper.
# Per-sample parameters are drawn from a normal centred on the median with
# std = 0.2 * MAD (see simulate_data._generate_parameters).
PARAM_MEDIAN = {
    'tau_E' : 0.0011, # time constant for excitatory population
    'tau_I' : 0.0026, # time constant for inhibitory population
    'W_EE' : 0.0101,  # weight of excitatory to excitatory connections
    'W_IE' : 0.0098,  # weight of inhibitory to excitatory connections
    'W_EI' : 0.0012,  # weight of excitatory to inhibitory connections
    'W_II' : 0.0020,  # weight of inhibitory to inhibitory connections
    'E_max' : 115.6,
    'I_max' : 113.8,
    'C_E' : 0.0014,
    'C_I' : 0.0030,
    'tau_S' : 0.0586, # time constant of the slow inhibition variable S
    'W_ES' : 0.0018,  # slow inhibition onto excitatory population
    'W_IS' : 0.0016,  # slow inhibition onto inhibitory population
    'XE' : 1.64,
    'XI' : 0.16,
}

PARAM_MAD = {
    'tau_E' : 0.0002,
    'tau_I' : 0.0011, # time constant for inhibitory population
    'W_EE' : 0.0068,  # weight of excitatory to excitatory connections
    'W_IE' : 0.0065,  # weight of inhibitory to excitatory connections
    'W_EI' : 0.0008,  # weight of excitatory to inhibitory connections
    'W_II' : 0.0019,  # weight of inhibitory to inhibitory connections
    'E_max' : 75.3,
    'I_max' : 64.8,
    'C_E' : 0.0013,
    'C_I' : 0.0022,
    'tau_S' : 0.0349, # time constant of the slow inhibition variable S
    'W_ES' : 0.0017,  # slow inhibition onto excitatory population
    'W_IS' : 0.0015,  # slow inhibition onto inhibitory population
    'XE' : 1.25,
    'XI' : 0.12,
}

# # Placeholder per-sample spread: 1% of each median. Tune against real data.
# PARAM_MAD = {k: 0.01 * abs(v) for k, v in PARAM_MEDIAN.items()}

# Hard-coded initial state for discovery (E, I observed; S hidden).
INITIAL_STATE = {
    'E' : 1.0,
    'I' : 1.0,
    'S' : 1.0,
}
