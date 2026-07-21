"""Oracle propagator: the true Burgers operator + a first-order (Euler) integrator.

NOT loaded as a seed -- TaskSpec discovers seeds via glob("model*.py"), so this
file (named ground_truth_*) is ignored by the run and kept only as a reference for
what the agnostic seeds should evolve toward. The "o1" tag = order-1 in time
(forward Euler); an AB2 / higher-order variant would be the order-2 oracle.

Derivation (method of lines on the unforced field u_t = -u u_x + D u_xx):

  * time, forward Euler over one coarse step dtc:
        u(t) = u(t-1) + dtc * RHS(u(t-1))
  * space, central differences on the coarse grid (spacing h = dxc), via np.roll:
        u_x  ~ (roll(u,-1) - roll(u,1)) / (2h)
        u_xx ~ (roll(u,-1) - 2u + roll(u,1)) / h^2

  => u(t) = u(t-1) + a * u*(roll(u,-1)-roll(u,1)) + b * (roll(u,-1)-2u+roll(u,1))
     with a = -dtc/(2h) ~ -0.204  (advection, -u u_x)
          b =  dtc*D/h^2 ~ 0.083  (diffusion,  D u_xx)

  a, b absorb the grid constants (dtc, h, D) and are fit freely in sensor-index
  space. See journal/2026-07-21.md: the fitted b runs high (~0.13) because it also
  soaks up sub-grid closure error and the anti-diffusive forward-Euler bias.
"""

import numpy as np


def model(data, params):
    u0 = data["x"][:, -1]

    def rhs(u):
        advection = u * (np.roll(u, -1) - np.roll(u, 1))     # ~ -u u_x   (a < 0)
        laplacian = np.roll(u, -1) - 2.0 * u + np.roll(u, 1)  # ~  u_xx    (b > 0)
        return params["a"] * advection + params["b"] * laplacian

    return u0 + rhs(u0)  # forward Euler; dtc folded into a, b


model.DEFAULT_PARAMS = {"a": -0.204, "b": 0.083}
