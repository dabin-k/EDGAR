"""Coarsening diagnostic for SINDy on Burgers (Step 3 finding).

Establishes *why* SINDy fails on the benchmark field, by separating three effects:
  1. Is the SINDy config correct?      -> test on unforced, well-resolved Burgers.
  2. Does the known forcing break it?  -> test forced fine grid, forcing subtracted.
  3. Does coarse-graining break it?    -> sweep spatial sub-sampling 1x .. 8x.

Result (see journal/2026-07-20_sindy_findings.md): the config is correct and forcing
subtraction works; SINDy recovers u_t = 0.02 u_xx - 1.0 u u_x on the fine grid at zero
noise. It FAILS on the coarse benchmark grid (s_factor=4, Nx=64) even at zero noise --
a coarse-graining/closure failure, not a noise failure.

Run:  python coarsening_diagnostic.py       (regenerates the fine field, ~60-90 s)
Writes results/coarsening_diagnostic.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pysindy as ps
from pysindy.feature_library import PDELibrary, PolynomialLibrary
from pysindy.differentiation import FiniteDifference

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "data_loader"))
import burgers_sim as bs  # noqa: E402
from runner import forcing_field, fit_pdefind  # noqa: E402


def _unforced(u0, x, D, dt, nsteps):
    weno = bs._weno_scheme; dx = x[1] - x[0]; u = u0.copy(); snaps = [u.copy()]
    def rhs(w_):
        um = np.roll(w_, 1); up = np.roll(w_, -1)
        return D * (up - 2 * w_ + um) / dx ** 2 - weno(w_, dx)
    for _ in range(nsteps):
        k1 = dt * rhs(u); k2 = dt * rhs(u + 0.5 * k1); k3 = dt * rhs(u - k1 + 2 * k2)
        u = u + (k1 + 4 * k2 + k3) / 6; snaps.append(u.copy())
    return np.array(snaps).T


def main():
    out_dir = os.path.join(_HERE, "results"); os.makedirs(out_dir, exist_ok=True)
    results = {}

    # 1. config check: unforced, well-resolved
    Lx = 256; L = 2 * np.pi; x = np.linspace(0, L, Lx); dt = 0.001; D = 0.02
    uu = _unforced(np.sin(x), x, D, dt, 4000)[:, ::10]
    m = fit_pdefind(uu, x, dt * 10, forcing=None, threshold=0.005)
    results["1_unforced_fine"] = {"Nx": Lx, "equation": m.equations()[0]}

    # 2 & 3: forced reference field, coarsen spatially
    sim = bs.simulate()
    u_fine = sim["u"]; L = sim["L"]; dt = sim["dt"]
    A, w, phi, l, N = sim["A"], sim["w"], sim["phi"], sim["l"], int(sim["N"])
    x_fine = np.linspace(0, L, sim["Lx"])
    for sfac in [1, 2, 4, 8]:
        xc = x_fine[::sfac]; uc = u_fine[::sfac, ::20]; dte = dt * 20
        F = forcing_field(A, w, phi, l, N, L, xc, dte, uc.shape[1])
        m = fit_pdefind(uc, xc, dte, forcing=F, threshold=0.01)
        results[f"s_factor_{sfac}"] = {"Nx": len(xc), "dx": float(xc[1] - xc[0]),
                                       "equation": m.equations()[0]}

    with open(os.path.join(out_dir, "coarsening_diagnostic.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("target: u_t = 0.02 u_11 - 1.0 uu_1\n")
    for k, v in results.items():
        print(f"{k:20s} (Nx={v['Nx']:3d}): (u)' = {v['equation']}")


if __name__ == "__main__":
    main()
