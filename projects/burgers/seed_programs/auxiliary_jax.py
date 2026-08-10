"""JAX twins of the helpers in auxiliary.py -- keep the two files in step.

Fourth-order central differences on a periodic row of unit-spaced sensors.
"""

import jax.numpy as jnp


def dudx_fn(u, axis=-1):
    """Fourth-order central first derivative of u along the sensor axis."""
    return (
        -jnp.roll(u, -2, axis=axis)
        + 8 * jnp.roll(u, -1, axis=axis)
        - 8 * jnp.roll(u, 1, axis=axis)
        + jnp.roll(u, 2, axis=axis)
    ) / 12.0


def d2udx2_fn(u, axis=-1):
    """Fourth-order central second derivative of u along the sensor axis."""
    return (
        -jnp.roll(u, -2, axis=axis)
        + 16 * jnp.roll(u, -1, axis=axis)
        - 30 * u
        + 16 * jnp.roll(u, 1, axis=axis)
        - jnp.roll(u, 2, axis=axis)
    ) / 12.0
