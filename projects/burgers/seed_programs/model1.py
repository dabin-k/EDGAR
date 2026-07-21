def model(data, params):
    """
    Persistence: the sensor field is assumed not to change over one step.

    data['x'] = recent activity, shape (n_sensors, max_length).
        Column -1 is the most recent step, column -2 the one before it.
        Sensors wrap around (periodic): np.roll reaches a neighbour.

    Returns:
        np.ndarray: predicted activity at every sensor at the next step, (n_sensors,).
    """
    return data["x"][:, -1]


model.DEFAULT_PARAMS = {}
