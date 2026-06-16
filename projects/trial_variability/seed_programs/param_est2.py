import numpy as np

def parameter_estimator(data):
    """
        data['signal'] shape (n_trials, n_cells)
    """
    signal = data['signal']
    gain = np.mean(signal, axis=1) #take average response over cells for that trial
    additive_offset = np.std(signal, axis=1) #take std response over cells for that trial
    coupling = np.mean(signal, axis=0) #take average response over trials for each cell
    return {"multiplicative_gain": gain.astype(float), "additive_offset": additive_offset.astype(float), "coupling_factor": coupling.astype(float)}