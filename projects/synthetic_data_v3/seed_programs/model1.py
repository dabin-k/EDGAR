def model(data, params):
    """
    Persistence: the population state is assumed not to change.

    data['x'] = recent activity, shape (n_cells, max_length).
        Column -1 is the most recent step, column -2 the one before it.

    Returns:
        np.ndarray: predicted activity at the next step, shape (n_cells,).
    """
    return data["x"][:, -1]


model.DEFAULT_PARAMS = {}
