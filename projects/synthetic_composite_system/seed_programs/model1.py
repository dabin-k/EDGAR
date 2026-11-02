import numpy as np


def model(data, params):
    """Evaluates the model for a single sample given data and parameters.

    This function represents the core mathematical or algorithmic model.
    It takes a single sample of data and a set of parameters, and
    produces predictions.

    Args:
        data (dict): A dictionary containing data for one sample,
                     e.g., `data['stimulus']` with shape `(n_trials,)`.
        params (dict): A dictionary of model parameters, with keys
                       matching those defined in `model.DEFAULT_PARAMS`.

    Returns:
        np.ndarray: The model's predictions for the given sample,
                    typically with shape `(n_trials,)`.
    """
    raise NotImplementedError


model.DEFAULT_PARAMS = {
    # "param_name": initial_value,
}
