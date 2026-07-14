def model(data, params):
    """
    Linear autoregression: the next value is a weighted sum of the three most recent
    values.

    data['x'] = recent activity of one cell, shape (max_length,).
        Index -1 is the most recent step, index -2 the one before it.

    params:
        a1: Weight on the most recent value, x(t).
        a2: Weight on x(t-1).
        a3: Weight on x(t-2).

    Returns:
        float: predicted activity at the next step.
    """
    x = data["x"]
    return params["a1"] * x[-1] + params["a2"] * x[-2] + params["a3"] * x[-3]


model.DEFAULT_PARAMS = {"a1": 0.5, "a2": 0.25, "a3": 0.1}
