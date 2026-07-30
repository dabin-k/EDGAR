def model(data, params):
    """
    Constant-rate RHS integrated with forward Euler.

    A discrete propagator is a continuous law du/dt = RHS(u) plus a time stepper.
    As null hypothesis, we assume no spatial structure. 

      * RHS: a constant rate, du/dt = c. 

      * Integrator: forward Euler, u(t) = u(t-1) + RHS. First order, single lag
        The Euler dt is folded into c.

    data['x'] : (n_sensors, input_sequence_length); column -1 most recent, sensors periodic.
    This model uses only the most recent column

    Returns:
        np.ndarray: predicted field at the next step, (n_sensors,).
    """
    u0 = data["x"][:, -1]

    def rhs(u):
        return params["c"]

    return u0 + rhs(u0)


model.DEFAULT_PARAMS = {"c": 0.0}
