"""Forced viscous Burgers generator for the EDGAR noise benchmark (Step 1).

The physics here is a faithful port of the STENCIL-NET reference implementation
(Maddu et al., Sci. Rep. 13:12787, 2023; github.com/mosaic-group/STENCIL-NET,
`utils.py` and `ForcedBurgersSimulation.ipynb`). The four private functions
`_flux`, `_dflux`, `_weno_scheme`, `_burgers_rk3` are copied verbatim from that
codebase (only renamed and lightly commented) so the benchmark's ground truth is
exactly the field STENCIL-NET was designed for. Do not "improve" them — fidelity
to the reference is the point.

Equation (periodic domain, length L):

    u_t + (u^2/2)_x = D u_xx + f(x, t)

    f(x,t) = sum_k A_k sin(w_k t + 2 pi l_k x / L + phi_k)   (smooth random forcing)

Integrated with an SSP-RK3 time step; the advective flux uses a 5th-order WENO
reconstruction and diffusion a 2nd-order central stencil. Reference parameters:
Lx=256, dx=L/(Lx-1) with L=2*pi, D=0.02, dt=0.002, N=20 forcing modes.

The observation model matches the reference exactly: coarse-grained (sub-sampled)
field plus *data-dependent* Gaussian noise,

    u_obs = u_coarse + noise_level * std(u_coarse) * N(0, 1)

so `noise_level` is a fraction of the field's own standard deviation (their KdV
denoising demo used 0.3; our sweep spans 0 .. 0.3).
"""

from __future__ import annotations

import numpy as np


# ── vendored physics (verbatim from STENCIL-NET/utils.py) ──


def _flux(u):
    return 0.5 * (u ** 2)


def _dflux(u):
    return u


def _weno_scheme(w, dx):
    """5th-order WENO flux divergence for the Burgers flux u^2/2 (periodic).

    Verbatim from STENCIL-NET/utils.py::WENO_scheme with sol='burgers'.
    np.roll(a, [0, k]) shifts right by k with periodic wrap, so `np.roll(a,[0,1])`
    is the i-1 neighbour and `np.roll(a,[0,-1])` the i+1 neighbour.
    """
    a = max(abs(_dflux(w)))
    v = 0.5 * (_flux(w) + a * w)
    u = np.roll(0.5 * (_flux(w) - a * w), [0, -1])  # flux splitting

    vmm = np.roll(v, [0, 2])
    vm = np.roll(v, [0, 1])
    vp = np.roll(v, [0, -1])
    vpp = np.roll(v, [0, -2])

    p0n = (2.0 * vmm - 7.0 * vm + 11.0 * v) / 6.0
    p1n = (-vm + 5.0 * v + 2.0 * vp) / 6.0
    p2n = (2.0 * v + 5.0 * vp - vpp) / 6.0

    B0n = (13.0 / 12) * ((vmm - 2.0 * vm + v) ** 2) + (1.0 / 4) * ((vmm - 4.0 * vm + 3.0 * v) ** 2)
    B1n = (13.0 / 12) * ((vm - 2.0 * v + vp) ** 2) + (1.0 / 4) * ((vm - vp) ** 2)
    B2n = (13.0 / 12) * ((v - 2.0 * vp + vpp) ** 2) + (1.0 / 4) * ((3.0 * v - 4.0 * vp + vpp) ** 2)

    d0n, d1n, d2n = 1.0 / 10, 6.0 / 10, 3.0 / 10
    epsilon = 1e-6
    alpha0n = d0n / ((epsilon + B0n) ** 2)
    alpha1n = d1n / ((epsilon + B1n) ** 2)
    alpha2n = d2n / ((epsilon + B2n) ** 2)
    alphasumn = alpha0n + alpha1n + alpha2n
    w0n = alpha0n / alphasumn
    w1n = alpha1n / alphasumn
    w2n = alpha2n / alphasumn
    Rplus = w0n * p0n + w1n * p1n + w2n * p2n

    umm = np.roll(u, [0, 2])
    um = np.roll(u, [0, 1])
    up = np.roll(u, [0, -1])
    upp = np.roll(u, [0, -2])
    p0p = (-umm + 5 * um + 2 * u) / 6.0
    p1p = (2 * um + 5 * u - up) / 6.0
    p2p = (11 * u - 7 * up + 2 * upp) / 6.0
    B0p = (13.0 / 12) * ((umm - 2.0 * um + u) ** 2) + (1.0 / 4) * ((umm - 4.0 * um + 3.0 * u) ** 2)
    B1p = (13.0 / 12) * ((um - 2.0 * u + up) ** 2) + (1.0 / 4) * ((um - up) ** 2)
    B2p = (13.0 / 12) * ((u - 2.0 * up + upp) ** 2) + (1.0 / 4) * ((3.0 * u - 4.0 * up + upp) ** 2)
    d0p, d1p, d2p = 3.0 / 10, 6.0 / 10, 1.0 / 10
    alpha0p = d0p / ((epsilon + B0p) ** 2)
    alpha1p = d1p / ((epsilon + B1p) ** 2)
    alpha2p = d2p / ((epsilon + B2p) ** 2)
    alphasump = alpha0p + alpha1p + alpha2p
    w0p = alpha0p / alphasump
    w1p = alpha1p / alphasump
    w2p = alpha2p / alphasump
    Rminus = w0p * p0p + w1p * p1p + w2p * p2p

    return (Rplus + Rminus - np.roll(Rplus, [0, 1]) - np.roll(Rminus, [0, 1])) / dx


def _forcing_at(A, w, phi, l, L, x, t):
    """Sinusoidal forcing sum_k A_k sin(w_k t + 2 pi l_k x/L + phi_k)."""
    f = np.zeros_like(x)
    for k in range(len(A)):
        f = f + A[k] * np.sin(w[k] * t + 2.0 * np.pi * l[k] * (x / L) + phi[k])
    return f


def _burgers_rk3(Tsim, Lx, x, D, dt, A, w, phi, l, L, u0=None):
    """Integrate forced Burgers with SSP-RK3; verbatim structure from the reference.

    Returns (u, phase): u is (Lx, Tsim); phase is the RHS diffusion-minus-advection
    term stored per step (kept for parity with the reference; not needed downstream).
    `u0` overrides the initial condition; None uses the reference Gaussian bump.
    """
    dx = x[1] - x[0]
    N = len(A)
    u = np.zeros((Lx, Tsim))
    phase = np.zeros((Lx, Tsim))
    u[:, 0] = np.exp(-((x - 3) ** 2)) if u0 is None else u0  # reference initial condition
    t = 0.0
    for j in range(0, Tsim - 1):
        forcing = _forcing_at(A, w, phi, l, L, x, t)
        um = np.roll(u[:, j], [0, 1]); up = np.roll(u[:, j], [0, -1])
        diff = D * (up - 2.0 * u[:, j] + um) / (dx * dx)
        phase[:, j] = diff - _weno_scheme(u[:, j], dx)
        k1 = dt * phase[:, j] + dt * forcing
        temp = u[:, j] + 0.5 * k1

        forcing = _forcing_at(A, w, phi, l, L, x, t + 0.5 * dt)
        um = np.roll(temp, [0, 1]); up = np.roll(temp, [0, -1])
        diff = D * (up - 2.0 * temp + um) / (dx * dx)
        k2 = dt * diff - dt * _weno_scheme(temp, dx) + dt * forcing
        temp = u[:, j] - k1 + 2.0 * k2

        forcing = _forcing_at(A, w, phi, l, L, x, t + dt)
        um = np.roll(temp, [0, 1]); up = np.roll(temp, [0, -1])
        diff = D * (up - 2.0 * temp + um) / (dx * dx)
        k3 = dt * diff - dt * _weno_scheme(temp, dx) + dt * forcing
        u[:, j + 1] = u[:, j] + (1.0 / 6.0) * (k1 + 4.0 * k2 + k3)
        t = t + dt
    return u, phase


# ── public generator API ──


def draw_forcing(N=20, Al=-0.1, Ar=0.1, wl=-0.4, wr=0.4, seed=0):
    """Draw the N random forcing modes exactly as the reference notebook does."""
    rng = np.random.RandomState(seed)  # matches np.random.seed(0) + rand() order
    A = np.zeros(N); w = np.zeros(N); phi = np.zeros(N); l = np.zeros(N)
    for k in range(N):
        A[k] = (Ar - Al) * rng.rand() + Al
        w[k] = (wr - wl) * rng.rand() + wl
        phi[k] = 2.0 * np.pi * rng.rand()
        l[k] = rng.randint(2, 5)
    return A, w, phi, l


def draw_ic(x, L, seed, n_modes=2):
    # """Draw a random smooth periodic initial condition u0(x).

    # A sum of the first `n_modes` Fourier modes with seeded random amplitudes and
    # phases, normalised to unit peak amplitude so it sits in the same range as the
    # reference bump. Used to give the validate split a genuinely independent
    # initial condition while staying on the same attractor (see load_data).
    # """
    # rng = np.random.RandomState(seed)
    # u0 = np.zeros_like(x)
    # for k in range(1, n_modes + 1):
    #     a = rng.uniform(0.8, 1.1)
    #     ph = 2.0 * np.pi * rng.rand()
    #     u0 += a * np.sin(2.0 * np.pi * k * x / L + ph)
    # return u0 / (np.max(np.abs(u0)) + 1e-12)
    rng = np.random.RandomState(seed)
    center = rng.uniform(0, L)
    amp = rng.uniform(0.7, 1.3)
    return amp * np.exp(-((x - center) ** 2))

def simulate(
    Lx=256,
    L=2.0 * np.pi,
    D=0.02,
    dt=0.002,
    Tsim=80001,
    N=20,
    forcing_seed=0,
    ic_seed=None,
):
    """Simulate the fine-grid forced-Burgers reference field.

    Returns dict with u (Lx, Tsim), x (Lx,), and all parameters needed to
    reproduce the field and the forcing (for downstream EDGAR forcing terms).

    If forcing_seed = None, apply no forcing (f=0) and return the unforced Burgers field.
    If ic_seed is not None, use a random smooth initial condition (draw_ic) instead
    of the reference Gaussian bump.
    """
    x = np.linspace(0, L, Lx)
    if forcing_seed is None:
        A = np.zeros(N); w = np.zeros(N); phi = np.zeros(N); l = np.zeros(N)
    else:
        A, w, phi, l = draw_forcing(N=N, seed=forcing_seed)
    u0 = None if ic_seed is None else draw_ic(x, L, ic_seed)
    u, phase = _burgers_rk3(Tsim, Lx, x, D, dt, A, w, phi, l, L, u0=u0)
    return {
        "u": u, "x": x, "L": L, "D": D, "dt": dt, "Lx": Lx, "Tsim": Tsim,
        "A": A, "w": w, "phi": phi, "l": l, "N": N, "forcing_seed": forcing_seed,
        "ic_seed": ic_seed,
    }


def coarsen(sim, s_factor=4, t_factor=20):
    """Sub-sample the fine field to the coarse observation grid (STENCIL-NET style)."""
    u = sim["u"]
    coarse_x = np.arange(0, sim["Lx"], s_factor)
    coarse_t = np.arange(0, sim["Tsim"], t_factor)
    u_coarse = u[np.ix_(coarse_x, coarse_t)].astype(float)
    return {
        "u_coarse": u_coarse,
        "x_coarse": sim["x"][coarse_x],
        "dxc": s_factor * (sim["x"][1] - sim["x"][0]),
        "dtc": t_factor * sim["dt"],
        "s_factor": s_factor, "t_factor": t_factor,
    }


def add_noise(u_coarse, noise_level, seed=0):
    """Data-dependent Gaussian noise: noise_level * std(field) * N(0,1).

    Exactly the reference observation model. noise_level is a fraction of the
    field's own standard deviation, so it is comparable across fields.
    """
    rng = np.random.RandomState(seed)
    sigma = noise_level * np.std(u_coarse)
    return u_coarse + sigma * rng.randn(*u_coarse.shape)
