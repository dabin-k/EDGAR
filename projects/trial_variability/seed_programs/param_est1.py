import numpy as np

def parameter_estimator(data):
    """
        data['signal'] shape (n_trials, n_cells)
    """
    signal = data['signal']
    gain = np.mean(signal, axis=1) #take average response over cells for that trial
    return {"multiplicative_gain": gain.astype(float)}