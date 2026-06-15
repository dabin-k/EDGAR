import numpy as np


def model(data, params):
    """y = a * x"""
    x = data["x"]
    a = params["a"]
    return a * x


# batched data is passed
model.DEFAULT_PARAMS = lambda data: {"a": np.ones(data["x"].shape[-1])}
