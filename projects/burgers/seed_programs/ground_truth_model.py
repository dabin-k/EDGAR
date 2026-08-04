import numpy as np

def model(data, params):
    u0 = data["x"][:, -1]

    # use RK4 approximation for dervivatives     
    def dudx(u):
        # return (np.roll(u, -1) - np.roll(u, 1)) / 2.0
        return (np.roll(u, -2) - 8 * np.roll(u, -1) + 8 * np.roll(u, 1) - np.roll(u, 2)) / 12.0

    def d2udx2(u):
        # return np.roll(u, -1) - 2.0 * u + np.roll(u, 1)
        return (-np.roll(u, -2) + 16 * np.roll(u, -1) - 30 * u + 16 * np.roll(u, 1) - np.roll(u, 2)) / 12.0

    def rhs(u):
        advection = u * dudx(u) 
        laplacian = d2udx2(u)
        return advection + params["D"] * laplacian

    return u0 + rhs(u0)  # forward Euler; dtc folded into a, b


model.DEFAULT_PARAMS = {"D": 0.01}

def model_jax(data, params):
    import jax.numpy as jnp

    u0 = data["x"][:, -1]

    def dudx(u):
        return (jnp.roll(u, -2) - 8 * jnp.roll(u, -1) + 8 * jnp.roll(u, 1) - jnp.roll(u, 2)) / 12.0

    def d2udx2(u):
        return (-jnp.roll(u, -2) + 16 * jnp.roll(u, -1) - 30 * u + 16 * jnp.roll(u, 1) - jnp.roll(u, 2)) / 12.0

    def rhs(u):
        advection = u * dudx(u)
        laplacian = d2udx2(u)
        return advection + params["D"] * laplacian

    return u0 + rhs(u0)
