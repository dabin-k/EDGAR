import numpy as np

def model(data, params):
    """
        Affine shared variability model, consisting of gain modulation + additive offset.

        Equation : f(t,c) = multiplicative_gain(t) * r(s(t),c) + additive_offset(t) * coupling_factor(c)
        where r(s(t),c) is the signal, the averaged response of neuron c over all trials presented with stimulus s.

        data['signal'] = signal #shape (n_trials, n_cells), where signal[i,c] is the average response of neuron c to binned stimulus s[i].
    
        params:
            multiplicative_gain: shape (n_trials,)
            additive_offset: shape (n_trials,)
            coupling_factor: shape (n_cells,)

        Returns:
            np.ndarray: Response, shape (n_trials, n_cells).
    """
    signal = data['signal']
    gain = params["multiplicative_gain"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]
    return gain[:, np.newaxis] * signal + np.outer(offset, coupling)

#Data is shaped (n_samples, n_trials, n_cells) when not vmapped
model.DEFAULT_PARAMS = lambda data: {
    "multiplicative_gain": np.ones(data['response'].shape[-2]),
    "additive_offset": np.zeros(data['response'].shape[-2]),
    "coupling_factor": np.ones(data['response'].shape-[1])
}
    