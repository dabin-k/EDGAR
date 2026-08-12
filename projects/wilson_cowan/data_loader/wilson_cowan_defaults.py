N_SAMPLES = 8          # cells / animals; parameters are fit per sample
N_REPEATS = 12         # noisy repeats per (sample, stim condition)

DT = 1 / 30            # sampling interval, ms per bin (1/30 ms)
N_TIMES_MS = 350       # simulate 0-350ms (shortened from 650 to speed up GD)
N_TIMES = int(N_TIMES_MS / DT)
STIM_ONSET_MS = 50
STIM_DUR_MS = 2        # boxcar pulse width, ms

STIM_ONSET = int(STIM_ONSET_MS / DT)
STIM_DUR = int(STIM_DUR_MS / DT)

H = DT / 1000          # integration step used by the generator (ms)

# Per-sample parameters are drawn uniformly on [median - MAD, median + MAD].
PARAM_MEDIAN = {
    'tau_E': 0.0011,
    'tau_I': 0.0065,
    'W_EE': 0.0396,
    'W_IE': 0.0277,
    'W_EI': 0.0074,
    'W_II': 0.0014,
    'E_max': 29.5,
    'I_max': 39.9,
    'C_E': 0.0018,
    'C_I': 0.0134,
    'XE': 3.51,
    'XI': 1.26,
}

PARAM_MAD = {
    'tau_E': 0.0002,
    'tau_I': 0.0019,
    'W_EE': 0.0162,
    'W_IE': 0.005, # make this small
    'W_EI': 0.0033,
    'W_II': 0.0115,
    'E_max': 2.0, # make this small
    'I_max': 2.0, # make this small
    'C_E': 0.0016,
    'C_I': 0.0115,
    'XE': 2.10,
    'XI': 0.86,
}

# Initial state shared across samples (E and I fully observed, no hidden variable).
INITIAL_STATE = {
    'E': 1.0,
    'I': 1.0,
}
