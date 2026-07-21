def model(data, params):
    """
    Linear-relaxation RHS integrated with 2-step Adams-Bashforth.

      * RHS: linear relaxation du/dt = -k u 

      * Integrator: Adams-Bashforth 2 (AB2), a second-order *two-lag* stepper,
            u(t) = u(t-1) + 1.5 * RHS(u(t-1)) - 0.5 * RHS(u(t-2))
            The step dt is folded into k.

    data['x'] : (n_sensors, max_length); column -1 most recent, -2 previous, periodic.

    Returns:
        np.ndarray: predicted field at the next step, (n_sensors,).
    """
    u0 = data["x"][:, -1]
    u1 = data["x"][:, -2]

    def rhs(u):
        return -params["k"] * u

    return u0 + 1.5 * rhs(u0) - 0.5 * rhs(u1)


model.DEFAULT_PARAMS = {"k": 0.1}
