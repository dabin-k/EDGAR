def model(data, params):
    """
    Leaky decay: each cell decays towards zero at a single shared rate, with no
    interaction between cells.

    data['x'] = recent activity, shape (n_cells, max_length).
        Column -1 is the most recent step, column -2 the one before it.

    params:
        decay: Fraction of the current activity retained at the next step.

    Returns:
        np.ndarray: predicted activity at the next step, shape (n_cells,).
    """
    return params["decay"] * data["x"][:, -1]


model.DEFAULT_PARAMS = {"decay": 0.9}
