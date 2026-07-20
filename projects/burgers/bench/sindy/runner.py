"""SINDy (PDE-FIND) runner for the Burgers benchmark (Step 3).

Uses PySINDy's PDE-FIND: a PDELibrary of spatial-derivative + polynomial candidate
terms {1, u, u^2, u_x, u u_x, u_xx, ...} fit by STLSQ sparse regression. This is the
`SINDy` variant we settled on for this benchmark (the field is already in the right
coordinates, so no autoencoder — see journal/burgers_benchmark_plan.md, Decision 1).

Fairness with STENCIL-NET: STENCIL-NET is *given* the known forcing f(x,t) (it enters
its loss as the fc terms). So SINDy must get the same known forcing. We subtract the
analytic f(x,t) from the estimated u_t before regression, so SINDy fits the homogeneous
operator u_t - f = N(u, u_x, u_xx, ...). Without this the forcing corrupts the whole fit.

Two library variants (see fit_pdefind):
  * strong (default): PDELibrary, differentiates the data to build u_t and the spatial
    derivative terms. Cheap, but derivative estimation amplifies noise.
  * weak / integral (--weak): WeakPDELibrary, integrates the data against smooth test
    functions instead of differentiating it. This is the noise-robust "integral SINDy"
    the Champion paper cites, and previews the lesson EDGAR borrows.

Key validated facts (see coarsening_diagnostic.py, weak_vs_strong.py, and the journal
notes 2026-07-20 + 2026-07-20_weak_sindy.md):
  * On a WELL-RESOLVED grid (fine, Nx=256) strong SINDy recovers
    u_t = 0.02 u_xx - 1.0 u u_x essentially exactly at zero noise (any time-subsampling).
  * NOISE: strong SINDy collapses fast -- by sigma=0.01 on the fine grid it loses u_xx
    and u u_x drops to ~ -0.14. The WEAK form is dramatically more robust: it holds
    ~ 0.02 u_xx - 1.0 u u_x all the way to sigma=0.1. => use --weak as the noise-robust
    SINDy baseline in the Step-4 head-to-head.
  * COARSE-GRAINING: this is a *closure* failure, not a derivative-noise failure, so the
    weak form does NOT rescue it -- strong and weak track each other as the grid coarsens.
    Nuance worth keeping: coarsening a *decaying* (unforced) field is forgiving (recovery
    stays near -1.0 down to Nx=64; both break only at Nx=32). The catastrophic Nx=64
    failure is specific to the *continuously forced* benchmark field: the persistent
    forcing sustains under-resolved shock structure everywhere, and the relative residual
    ||(u_t - f) - N(u)|| on the coarse grid climbs 0.17 (Nx=256) -> 0.71 (Nx=64). The
    forcing modes themselves are low (l_k in {2,3,4}), so it is not forcing
    under-resolution -- it is the coarse u u_x / u_xx stencils failing on sustained shocks.

CLI:
    python runner.py --noise 0.0            # strong fit on shared coarse bundle
    python runner.py --sweep                # full noise sweep, writes results/
    python runner.py --sweep --weak         # integral-SINDy noise sweep
Runs locally on CPU (seconds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pysindy as ps
from pysindy.feature_library import PDELibrary, PolynomialLibrary
from pysindy.differentiation import FiniteDifference

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "data_loader"))
import load_data as ld  # noqa: E402


def forcing_field(A, w, phi, l, N, L, xg, dt, nt):
    """Analytic forcing f(x,t) = sum_k A_k sin(w_k t + 2 pi l_k x/L + phi_k) on a grid.

    Matches burgers_sim._forcing_at exactly (verified bit-identical draw)."""
    tg = np.arange(nt) * dt
    XX, TT = np.meshgrid(xg, tg, indexing="ij")
    F = np.zeros_like(XX)
    for k in range(int(N)):
        F += A[k] * np.sin(w[k] * TT + 2.0 * np.pi * l[k] * (XX / L) + phi[k])
    return F


def fit_pdefind(u, xg, dt, forcing=None, threshold=0.05, alpha=1e-5,
                poly_degree=2, deriv_order=2, weak=False):
    """Fit PDE-FIND to a (Nx, Nt) field. Returns the fitted pysindy model.

    forcing : (Nx, Nt) array or None. If given, subtracted from u_t (known-input case).
              NOTE: only the STRONG path honours this. WeakPDELibrary builds its own
              weak-form u_t (integrated against test functions) and IGNORES x_dot, so
              forcing subtraction cannot be applied in the weak path -- see the warning
              below. The weak-form results in the journal were therefore obtained on
              *unforced* fields, which isolates the noise/coarsening question cleanly.
              Fitting weak SINDy on the forced benchmark field without forcing handling
              would be corrupted by the forcing (a known open item for Step 4).
    weak    : if True, use WeakPDELibrary (integral formulation) -- the noise-robust
              "integral SINDy" variant the Champion paper cites (benchmark stretch).
    """
    U = u[:, :, None]
    if weak and forcing is not None:
        import warnings
        warnings.warn(
            "fit_pdefind(weak=True): WeakPDELibrary builds its own weak-form u_t and "
            "ignores x_dot, so the supplied forcing is NOT subtracted. On the forced "
            "benchmark field this fit will be corrupted by the forcing. Use weak SINDy "
            "on unforced data, or add a weak-form forcing term (open item for Step 4).",
            RuntimeWarning, stacklevel=2,
        )
    if forcing is not None and not weak:
        ut = FiniteDifference(axis=1)._differentiate(u, dt)
        x_dot = (ut - forcing)[:, :, None]
    else:
        x_dot = None  # let pysindy estimate u_t itself
    if weak:
        from pysindy.feature_library import WeakPDELibrary
        lib = WeakPDELibrary(
            function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
            derivative_order=deriv_order,
            spatiotemporal_grid=_spatiotemporal_grid(xg, dt, u.shape[1]),
            include_bias=True, is_uniform=True, periodic=True, K=500,
        )
    else:
        lib = PDELibrary(
            function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
            derivative_order=deriv_order, spatial_grid=xg,
            include_bias=True, is_uniform=True, periodic=True,
        )
    model = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=threshold, alpha=alpha))
    model.fit(U, t=dt, x_dot=x_dot, feature_names=["u"])
    return model


def _spatiotemporal_grid(xg, dt, nt):
    tg = np.arange(nt) * dt
    XX, TT = np.meshgrid(xg, tg, indexing="ij")
    return np.stack([XX, TT], axis=-1)


def run(noise_level=0.0, threshold=0.05, weak=False, out_dir=None):
    out_dir = out_dir or os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    b = ld.load_bundle()
    u = ld.noise_field(b, noise_level)
    xg = b["x_coarse"].astype(float); dt = b["dtc"]
    F = forcing_field(b["A"], b["w"], b["phi"], b["l"], b["N"], b["L"], xg, dt, u.shape[1])
    model = fit_pdefind(u, xg, dt, forcing=F, threshold=threshold, weak=weak)
    eqn = model.equations()[0]
    coefs = model.coefficients().tolist()
    names = model.get_feature_names()
    result = {
        "method": "sindy_weak" if weak else "sindy",
        "noise_level": noise_level, "threshold": threshold, "grid": "coarse_s4",
        "n_x": int(u.shape[0]), "equation": eqn,
        "feature_names": names, "coefficients": coefs,
    }
    tag = ("weak_" if weak else "") + f"noise{noise_level}"
    with open(os.path.join(out_dir, f"sindy_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--weak", action="store_true", help="use WeakPDELibrary (integral SINDy)")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    levels = [0.0, 0.01, 0.05, 0.1, 0.3] if args.sweep else [args.noise]
    for nl in levels:
        r = run(noise_level=nl, threshold=args.threshold, weak=args.weak)
        print(f"noise={nl}: (u)' = {r['equation']}")
