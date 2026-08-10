"""Oracle propagator: the true Burgers operator + a second-order (AB2) integrator.

NOT loaded as a seed -- TaskSpec discovers seeds via glob("model*.py"), so this
file (named ground_truth_*) is ignored by the run and kept only as a reference
for what the agnostic seeds should evolve toward. "o2" = order-2 in time; see
ground_truth_model_o1.py for the forward-Euler version and
ground_truth_model_o2_taylor.py for the single-lag order-2 variant.

Derivation (method of lines on the unforced field u_t = -u u_x + D u_xx):

Space is discretised exactly as in the o1 oracle (central differences, folded
into a and b), so the only change is the time integrator. Write one Euler
increment as G(u) = a*u*(roll(u,-1)-roll(u,1)) + b*(roll(u,-1)-2u+roll(u,1)),
i.e. G ~ dtc * RHS. Taylor in time:

    u(t+dtc) = u + dtc*u_t + (dtc^2/2)*u_tt + O(dtc^3)

Forward Euler drops the u_tt term; that is its entire leading error. Recover it
from the second lag instead of computing it, using dN/dt = u_tt:

    N(t-dtc) = N(t) - dtc*u_tt + O(dtc^2)
    => dtc*(alpha*N(t) + beta*N(t-dtc))
         = (alpha+beta)*dtc*u_t - beta*dtc^2*u_tt + O(dtc^3)

Matching term by term against the Taylor series:

    dtc^1:  alpha + beta = 1
    dtc^2:         -beta = 1/2      =>  beta = -1/2, alpha = 3/2

  => u(t) = u(t-1) + 1.5*G(u(t-1)) - 0.5*G(u(t-2))          [Adams-Bashforth 2]

More generally the two-lag explicit schemes of order 2 form a one-parameter
family, u+ = (1-g)*u_n + g*u_n-1 + dtc*[(g+3)/2*N_n + (g-1)/2*N_n-1]; g=0 is AB2,
g=1 is leapfrog. Imposing third order as well forces g=5, which is zero-unstable
(Dahlquist barrier), so AB2 is the natural choice.

a and b are unchanged from the o1 oracle -- second order in time costs zero extra
parameters. Verified: one-step error against a DOP853 solve of the semi-discrete
system converges at rate 3.00 (o1 gives 2.00), and at the run's dtc=0.04 AB2 is
4.3x more accurate than forward Euler.

MEASURED on the CLEAN discover split (noise_level=0.0, rollout_steps=2), scored
at the analytic a, b above -- no fitting:

    persistence (do-nothing)        2.2053e-07
    o1 Euler                        1.5068e-10
    o2 AB2 (this)                   8.0085e-11     1.88x better than o1
    o2 Taylor (single-lag)          8.1563e-11

AB2 and the single-lag Taylor form agree to 2%, as two routes to the same order
should, and both fit inside the current window (input_sequence_length is 5, so
neither is constrained by the number of lags on offer). The choice between them is
a convenience, not an accuracy argument -- the clean data does not show the second
lag buying anything beyond order 2 (contrast the note in
ground_truth_model_o2_taylor.py; with 5 lags available that note is about what the
extra lags are worth, not about whether they exist).

Order 2 in time is worth 1.88x, not the ~4x the one-step truncation analysis
suggests, because it fixes only the *temporal* error: the residual is dominated
by sub-grid closure error from the x4 spatial coarsening, which no time
integrator can remove.
"""

import numpy as np


def model(data, params):
    u1 = data["x"][:, -1]  # u(t-1)
    u2 = data["x"][:, -2]  # u(t-2)

    def rhs(u):
        advection = u * (np.roll(u, -1) - np.roll(u, 1))     # ~ -u u_x   (a < 0)
        laplacian = np.roll(u, -1) - 2.0 * u + np.roll(u, 1)  # ~  u_xx    (b > 0)
        return params["a"] * advection + params["b"] * laplacian

    return u1 + 1.5 * rhs(u1) - 0.5 * rhs(u2)  # AB2; dtc folded into a, b


model.DEFAULT_PARAMS = {"a": -0.204, "b": 0.083}
