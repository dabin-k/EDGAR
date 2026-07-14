def model(data, params):
    """
    Leaky decay: the cell's activity decays towards zero at a constant rate.

    data['x'] = recent activity of one cell, shape (max_length,).
        Index -1 is the most recent step, index -2 the one before it.

    params:
        decay: Fraction of the current activity retained at the next step.

    Returns:
        float: predicted activity at the next step.
    """
    return params["decay"] * data["x"][-1]


model.DEFAULT_PARAMS = {"decay": 0.9}
