"""Oracle propagator: the true Burgers operator + a second-order Taylor step.

NOT loaded as a seed -- TaskSpec discovers seeds via glob("model*.py"), so this
file (named ground_truth_*) is ignored by the run and kept only as a reference
for what the agnostic seeds should evolve toward. "o2" = order-2 in time. This is
the SINGLE-LAG order-2 oracle (Lax-Wendroff style); ground_truth_model_o2_ab2.py
is the two-lag version, and ground_truth_model_o1.py the forward-Euler one.

Derivation (method of lines on the unforced field u_t = -u u_x + D u_xx):

Space is discretised exactly as in the o1 oracle, so write one Euler increment as
G(u) = a*u*(roll(u,-1)-roll(u,1)) + b*(roll(u,-1)-2u+roll(u,1)) ~ dtc*RHS. Taylor
in time needs u_tt, which the PDE supplies (Cauchy-Kovalevskaya): differentiate
u_t = N(u) in t and substitute u_t = N(u),

    u_tt = N'(u)[N(u)],   N'(u)[v] = -(u v)_x + D v_xx

    => u_tt = d_x(u^2 u_x) - D d_x(u u_xx) - D d_xx(u u_x) + D^2 u_xxxx
            = 2 u u_x^2 + u^2 u_xx - 4D u_x u_xx - 2D u u_xxx + D^2 u_xxxx

(verified symbolically; residual exactly 0). Discretising that with central
differences gives five correction terms whose coefficients all collapse onto a
and b -- a^2, 2a^2, 2ab, ab, b^2/2 -- so order 2 costs no extra parameters. But
that expansion Taylors the continuum and *then* discretises, which disagrees with
discretise-then-Taylor at O(h^2) and stalls at rate ~2.2 in practice.

The clean form applies the Taylor step to the semi-discrete system directly, so
the second-order term is exactly (1/2) G'(u)[G(u)] with G' the Frechet derivative

    G'(u)[v] = a*(v*D1(u) + u*D1(v)) + b*D2(v)

  => u(t) = u(t-1) + G + 0.5*G'(u)[G],   G = G(u(t-1))

This is what the code below implements: 3-point stencil, 2 parameters, exactly
consistent with the spatial discretisation. Verified: one-step error against a
DOP853 solve of the semi-discrete system converges at rate 3.00 (o1 gives 2.00),
and at the run's dtc=0.04 it is 11.5x more accurate than forward Euler -- the most
accurate of the three oracles per unit of temporal truncation error.

MEASURED on the CLEAN discover split (noise_level=0.0, rollout_steps=2), scored
at the analytic a, b above -- no fitting:

    persistence (do-nothing)        2.2053e-07
    o1 Euler                        1.5068e-10
    o2 AB2 (two-lag)                8.0085e-11
    o2 Taylor (this)                8.1563e-11     1.85x better than o1

This form and AB2 agree to 2%, as two routes to the same order should, so on
clean data the second lag buys nothing beyond order 2 -- worth knowing, since the
x4 spatial coarsening makes the coarse field non-Markovian (see
data_loader/load_data.py) and might have been expected to favour the two-lag
scheme. Keep this file as the single-lag control that separates "order 2" from
"second lag".

Order 2 in time is worth 1.85x, not the ~11x the one-step truncation analysis
suggests, because it fixes only the *temporal* error: the residual is dominated
by sub-grid closure error from the spatial coarsening, which no time integrator
can remove.
"""

import numpy as np


def model(data, params):
    u0 = data["x"][:, -1]
    a, b = params["a"], params["b"]

    def d1(u):
        return np.roll(u, -1) - np.roll(u, 1)          # ~ 2h u_x

    def d2(u):
        return np.roll(u, -1) - 2.0 * u + np.roll(u, 1)  # ~ h^2 u_xx

    g = a * u0 * d1(u0) + b * d2(u0)                    # ~ dtc * RHS
    # (1/2) G'(u0)[g], the (dtc^2/2) u_tt term of the semi-discrete system
    correction = 0.5 * (a * (g * d1(u0) + u0 * d1(g)) + b * d2(g))

    return u0 + g + correction


model.DEFAULT_PARAMS = {"a": -0.204, "b": 0.083}
