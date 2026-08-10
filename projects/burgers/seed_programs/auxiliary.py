"""Spatial derivative helpers for models of the sensor row.

Fourth-order central differences on a periodic row of unit-spaced sensors, so
`dudx_fn(u)` is du/dx in sensor-index units; any physical spacing is absorbed
into the free parameters. `np.roll(u, 1)` brings in the left neighbour u[i-1]
and `np.roll(u, -1)` the right neighbour u[i+1].

`axis` selects the sensor axis. The default suits a model, whose `u` is the
1-D state over sensors; a parameter estimator working on
(n_blocks, n_sensors, block_len) must pass `axis=1`.
"""

import numpy as np


def dudx_fn(u, axis=-1):
    """Fourth-order central first derivative of u along the sensor axis."""
    return (
        -np.roll(u, -2, axis=axis)
        + 8 * np.roll(u, -1, axis=axis)
        - 8 * np.roll(u, 1, axis=axis)
        + np.roll(u, 2, axis=axis)
    ) / 12.0


def d2udx2_fn(u, axis=-1):
    """Fourth-order central second derivative of u along the sensor axis."""
    return (
        -np.roll(u, -2, axis=axis)
        + 16 * np.roll(u, -1, axis=axis)
        - 30 * u
        + 16 * np.roll(u, 1, axis=axis)
        - np.roll(u, 2, axis=axis)
    ) / 12.0
