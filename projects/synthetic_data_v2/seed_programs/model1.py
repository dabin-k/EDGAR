def model(data, params):
    """
    Persistence: the cell's activity is assumed not to change.

    data['x'] = recent activity of one cell, shape (max_length,).
        Index -1 is the most recent step, index -2 the one before it.

    Returns:
        float: predicted activity at the next step.
    """
    return data["x"][-1]


model.DEFAULT_PARAMS = {}
