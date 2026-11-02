import numpy as np


def parameter_estimator(data):
    """Estimates model parameters for a single sample.

    This function takes a single sample of data and attempts to estimate
    the optimal parameters for the `model` function. This is often used
    to provide good initial parameter guesses for gradient descent optimization.

    Args:
        data (dict): A dictionary containing data for one sample,
                     e.g., `data['stimulus']` and `data['response']`,
                     each with shape `(n_trials,)`.

    Returns:
        dict: A dictionary of estimated parameters, with keys matching
              those defined in `model.DEFAULT_PARAMS`.
    """
    raise NotImplementedError
