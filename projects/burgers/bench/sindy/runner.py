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
  * COARSE-GRAINING: the closure gap is real, but part of the earlier *catastrophic*
    Nx=64 failure of STRONG SINDy on the forced field was the noise-amplifying
    finite-difference u_t estimate, not closure alone. With the weak-form forcing term
    (fit_pdefind now handles forcing in BOTH paths), weak SINDy on the forced coarse grid
    (Nx=64) recovers ~0.02 u_xx - 1.0 u u_x + small spurious terms -- the collapse is gone,
    though residual closure error remains. The relative residual ||(u_t - f) - N(u)|| on
    the coarse grid still climbs 0.17 (Nx=256) -> 0.71 (Nx=64); the forcing modes are low
    (l_k in {2,3,4}), so it is not forcing under-resolution.

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
                poly_degree=2, deriv_order=2, weak=False, K=500):
    """Fit PDE-FIND to a (Nx, Nt) field. Returns a dict {equation, names, coefs}.

    Both the strong and weak paths honour a known forcing f(x,t):

    forcing : (Nx, Nt) array or None. If given, it is removed so SINDy fits the
              homogeneous operator u_t - f = N(u, u_x, u_xx, ...). This is the fair
              known-input setup (STENCIL-NET is *given* the same forcing).
              * STRONG path: subtract f from the finite-difference u_t before regression.
              * WEAK path: WeakPDELibrary computes a weak-form target <u_t, phi> and does
                NOT accept an x_dot argument. But the weak equation is
                    <u_t, phi> = sum_j Xi_j <theta_j(u), phi> + <f, phi>,
                and <f, phi> = <F_t, phi> for F the time-antiderivative of f, so by the
                linearity of convert_u_dot_integral we can move it to the LHS:
                    target = convert_u_dot_integral(u) - convert_u_dot_integral(F).
                We assemble the weak regression by hand (pysindy's Optimizer.fit rejects
                the raw 2D weak target via its AxesArray bookkeeping) with a small STLSQ.
    weak    : if True, use WeakPDELibrary (integral formulation) -- the noise-robust
              "integral SINDy" variant the Champion paper cites. Validated: with the
              forcing correction it recovers 0.02 u_xx - 1.0 u u_x on the FORCED fine
              grid, and stays robust to sigma=0.1 (see journal 2026-07-20_weak_sindy.md).
    """
    if weak:
        return _fit_weak(u, xg, dt, forcing, threshold, alpha, poly_degree, deriv_order, K)
    return _fit_strong(u, xg, dt, forcing, threshold, alpha, poly_degree, deriv_order)


def _model_to_dict(model):
    names = list(model.get_feature_names())
    coefs = np.asarray(model.coefficients()).ravel().tolist()
    return {"equation": model.equations()[0], "names": names, "coefs": coefs}


def _fit_strong(u, xg, dt, forcing, threshold, alpha, poly_degree, deriv_order):
    U = u[:, :, None]
    if forcing is not None:
        ut = FiniteDifference(axis=1)._differentiate(u, dt)
        x_dot = (ut - forcing)[:, :, None]
    else:
        x_dot = None  # let pysindy estimate u_t itself
    lib = PDELibrary(
        function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
        derivative_order=deriv_order, spatial_grid=xg,
        include_bias=True, is_uniform=True, periodic=True,
    )
    model = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=threshold, alpha=alpha))
    model.fit(U, t=dt, x_dot=x_dot, feature_names=["u"])
    return _model_to_dict(model)


def _fit_weak(u, xg, dt, forcing, threshold, alpha, poly_degree, deriv_order, K):
    from pysindy.feature_library import WeakPDELibrary
    from pysindy.utils import AxesArray
    lib = WeakPDELibrary(
        function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
        derivative_order=deriv_order,
        spatiotemporal_grid=_spatiotemporal_grid(xg, dt, u.shape[1]),
        include_bias=True, is_uniform=True, periodic=True, K=K,
    )
    axes = {"ax_spatial": [0], "ax_time": 1, "ax_coord": [2]}
    U = AxesArray(u[:, :, None], axes)
    lib.fit([U])
    Theta = np.array(lib.transform([U])[0], subok=False, dtype=float)          # (K, n_feat)
    ydot = np.array(lib.convert_u_dot_integral(U), subok=False, dtype=float).ravel()
    if forcing is not None:
        # F = time-antiderivative of f (cumulative trapezoid along time), so <f,phi>=<F_t,phi>
        F = np.zeros_like(forcing)
        F[:, 1:] = np.cumsum(0.5 * (forcing[:, 1:] + forcing[:, :-1]) * dt, axis=1)
        Fa = AxesArray(F[:, :, None], axes)
        ydot = ydot - np.array(lib.convert_u_dot_integral(Fa), subok=False, dtype=float).ravel()
    xi = _stlsq(Theta, ydot, threshold, alpha)
    names = list(lib.get_feature_names(["u"]))
    terms = [f"{c:.3f} {n}" for n, c in zip(names, xi) if abs(c) > 0]
    eqn = " + ".join(terms) if terms else "0"
    return {"equation": eqn, "names": names, "coefs": xi.tolist()}


def _stlsq(Theta, y, threshold=0.02, alpha=1e-5, n_iter=20):
    """Sequentially-thresholded ridge least squares (PDE-FIND's STLSQ), plain-numpy."""
    Theta = np.asarray(Theta, float); y = np.asarray(y, float).ravel()
    n_feat = Theta.shape[1]

    def ridge(X, yy):
        return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ yy)

    xi = np.zeros(n_feat)
    big = np.ones(n_feat, bool)
    xi[big] = ridge(Theta, y)
    for _ in range(n_iter):
        small = np.abs(xi) < threshold
        if np.array_equal(small, ~big):  # converged (support unchanged)
            pass
        xi[small] = 0.0
        big = ~small
        if big.sum() == 0:
            break
        xi[big] = ridge(Theta[:, big], y)
    return xi


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
    fit = fit_pdefind(u, xg, dt, forcing=F, threshold=threshold, weak=weak)
    result = {
        "method": "sindy_weak" if weak else "sindy",
        "noise_level": noise_level, "threshold": threshold, "grid": "coarse_s4",
        "n_x": int(u.shape[0]), "equation": fit["equation"],
        "feature_names": fit["names"], "coefficients": fit["coefs"],
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
