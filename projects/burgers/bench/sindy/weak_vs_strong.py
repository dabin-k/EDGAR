"""Weak (integral) vs strong (derivative) SINDy on Burgers -- Step 3 stretch.

Answers: does the integral/weak formulation (the noise remedy the Champion paper cites)
help this benchmark? Tested on two independent axes, on UNFORCED Burgers so the weak
form applies cleanly (WeakPDELibrary builds its own weak-form u_t and cannot use the
forcing-subtraction trick from runner.py -- see runner.fit_pdefind).

Findings (see journal/2026-07-20_weak_sindy.md, figure weak_sindy_comparison.png):
  * NOISE (well-resolved grid, Nx=256): weak wins decisively. Strong collapses by
    sigma=0.01 (u u_x -> -0.14, u_xx gone); weak holds ~0.02 u_xx - 1.0 u u_x to sigma=0.1.
  * COARSE-GRAINING (zero noise): weak does NOT rescue it -- strong and weak track each
    other. Coarse-graining is a closure failure, not a derivative-noise failure.
    (A decaying unforced field coarsens gracefully to Nx=64; both break only at Nx=32.
    The catastrophic Nx=64 failure is specific to the continuously forced field.)

Run: python weak_vs_strong.py    (regenerates fields, ~15 s; writes results/weak_vs_strong.json)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pysindy as ps
from pysindy.feature_library import PDELibrary, WeakPDELibrary, PolynomialLibrary

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "data_loader"))
import burgers_sim as bs  # noqa: E402

D = 0.02


def _unforced(u0, x, dt, nsteps):
    weno = bs._weno_scheme; dx = x[1] - x[0]; u = u0.copy(); snaps = [u.copy()]
    def rhs(w_):
        um = np.roll(w_, 1); up = np.roll(w_, -1)
        return D * (up - 2 * w_ + um) / dx ** 2 - weno(w_, dx)
    for _ in range(nsteps):
        k1 = dt * rhs(u); k2 = dt * rhs(u + 0.5 * k1); k3 = dt * rhs(u - k1 + 2 * k2)
        u = u + (k1 + 4 * k2 + k3) / 6; snaps.append(u.copy())
    return np.array(snaps).T


def _st_grid(xg, dt, nt):
    tg = np.arange(nt) * dt
    XX, TT = np.meshgrid(xg, tg, indexing="ij")
    return np.stack([XX, TT], axis=-1)


def fit_strong(u, xg, dt, thr=0.02):
    lib = PDELibrary(function_library=PolynomialLibrary(degree=2, include_bias=False),
                     derivative_order=2, spatial_grid=xg, include_bias=True,
                     is_uniform=True, periodic=True)
    m = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=thr, alpha=1e-5))
    m.fit(u[:, :, None], t=dt, feature_names=["u"]); return m


def fit_weak(u, xg, dt, thr=0.02, K=200, seed=0):
    np.random.seed(seed)
    lib = WeakPDELibrary(function_library=PolynomialLibrary(degree=2, include_bias=False),
                         derivative_order=2, spatiotemporal_grid=_st_grid(xg, dt, u.shape[1]),
                         include_bias=True, is_uniform=True, periodic=True, K=K)
    m = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=thr, alpha=1e-5))
    m.fit(u[:, :, None], t=dt, feature_names=["u"]); return m


def _coef(m, name):
    for n, c in zip(m.get_feature_names(), m.coefficients().ravel()):
        if n.replace(" ", "") == name:
            return float(c)
    return 0.0


def main():
    out_dir = os.path.join(_HERE, "results"); os.makedirs(out_dir, exist_ok=True)
    Lx = 256; L = 2 * np.pi; x = np.linspace(0, L, Lx); dt = 0.001
    u_fine = _unforced(np.sin(x), x, dt, 4000)
    us = u_fine[:, ::10]; dte = dt * 10; std = us.std()
    rng = np.random.default_rng(0)

    res = {"noise": {"levels": [], "S_uu": [], "S_xx": [], "W_uu": [], "W_xx": []},
           "coarsen": {"Nx": [], "S_uu": [], "S_xx": [], "W_uu": [], "W_xx": []}}

    for nl in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        un = us + nl * std * rng.standard_normal(us.shape)
        ms, mw = fit_strong(un, x, dte), fit_weak(un, x, dte)
        res["noise"]["levels"].append(nl)
        res["noise"]["S_uu"].append(_coef(ms, "uu_1")); res["noise"]["S_xx"].append(_coef(ms, "u_11"))
        res["noise"]["W_uu"].append(_coef(mw, "uu_1")); res["noise"]["W_xx"].append(_coef(mw, "u_11"))

    # multiscale IC -> spectrally rich decaying field (still coarsens gracefully to Nx=64).
    # Draw order (integers, uniform, uniform) is load-bearing: it fixes the exact field
    # used for figure weak_sindy_comparison.png. Changing the order reseeds a different,
    # sometimes degenerate field -- keep as-is.
    rng2 = np.random.default_rng(1); u0 = np.zeros_like(x)
    for k in range(12):
        lk = rng2.integers(1, 9); ak = rng2.uniform(0.2, 0.6) * (-1) ** k; ph = rng2.uniform(0, 2 * np.pi)
        u0 += ak * np.sin(lk * x + ph)
    u0 *= 0.6 / np.max(np.abs(u0))
    u_multi = _unforced(u0, x, dt, 4000)
    for s in [1, 2, 4, 8]:
        xc = x[::s]; uc = u_multi[::s, ::10]
        ms, mw = fit_strong(uc, xc, dte), fit_weak(uc, xc, dte)
        res["coarsen"]["Nx"].append(len(xc))
        res["coarsen"]["S_uu"].append(_coef(ms, "uu_1")); res["coarsen"]["S_xx"].append(_coef(ms, "u_11"))
        res["coarsen"]["W_uu"].append(_coef(mw, "uu_1")); res["coarsen"]["W_xx"].append(_coef(mw, "u_11"))

    with open(os.path.join(out_dir, "weak_vs_strong.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("NOISE (Nx=256)  levels:", res["noise"]["levels"])
    print("  strong u u_x:", [round(v, 2) for v in res["noise"]["S_uu"]])
    print("  weak   u u_x:", [round(v, 2) for v in res["noise"]["W_uu"]])
    print("COARSEN (0 noise) Nx:", res["coarsen"]["Nx"])
    print("  strong u u_x:", [round(v, 2) for v in res["coarsen"]["S_uu"]])
    print("  weak   u u_x:", [round(v, 2) for v in res["coarsen"]["W_uu"]])


if __name__ == "__main__":
    main()
