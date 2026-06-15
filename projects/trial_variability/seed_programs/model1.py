import numpy as np

def model(data, params):
    """
        Multiplicative gain modulation model for trial-to-trial variability in neural responses.

        Equation : f(t,c) = multiplicative_gain(t) * r(s(t),c)
        where r(s(t),c) is the signal, the averaged response of neuron c over all trials presented with stimulus s.

        data['signal'] = signal #shape (n_trials, n_cells), where signal[i,c] is the average response of neuron c to binned stimulus s[i].
    
        params:
            multiplicative_gain: shape (n_trials,)

        Returns:
            np.ndarray: Response, shape (n_trials, n_cells).
    """
    signal = data['signal']
    gain = params["multiplicative_gain"]

    return gain[:, np.newaxis] * signal

#Data is shaped (n_samples, n_trials, n_cells) when not vmapped
model.DEFAULT_PARAMS = lambda data: {
    "multiplicative_gain": np.ones(data['response'].shape[-2])
}