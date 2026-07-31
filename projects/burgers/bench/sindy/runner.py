"""SINDy (PDE-FIND) runner for the Burgers benchmark (Step 3).

Uses PySINDy's PDE-FIND: a PDELibrary of spatial-derivative + polynomial candidate
terms {1, u, u^2, u_x, u u_x, u_xx, ...} fit by STLSQ sparse regression. This is the
`SINDy` variant we settled on for this benchmark (the field is already in the right
coordinates, so no autoencoder — see journal/burgers_benchmark_plan.md, Decision 1).

`run()` reads the shared *unforced* datasets (`ic_seed_*_nl_*.npz`), one sample at a
time. Each sample has its own viscosity D, so the target operator is
u_t = D u_xx - u u_x and coefficient recovery is checked against a per-sample truth.
The fit is horizon-independent (SINDy recovers a continuous operator), so ONE fit is
scored at every rollout horizon -- unlike STENCIL-NET, which retrains per horizon.

Forcing is still supported by fit_pdefind for the older forced bundle. Fairness with
STENCIL-NET there: STENCIL-NET is *given* the known forcing f(x,t) (it enters its loss
as the fc terms). So SINDy must get the same known forcing. We subtract the analytic
f(x,t) from the estimated u_t before regression, so SINDy fits the homogeneous operator
u_t - f = N(u, u_x, u_xx, ...). Without this the forcing corrupts the whole fit.

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

NOTE on the STLSQ threshold: the true u_xx coefficient IS the sample's D, as small as
0.005, so the old default of 0.01 thresholded the true diffusion term away *by
construction* on the low-D samples. The default is now 0.002, below every true D.

CLI:
    python runner.py --data-path .../ic_seed_0_nl_0.0.npz --sample-idx 1
    python runner.py --data-path .../ic_seed_0_nl_0.1.npz --sample-idx 1 --weak
    python sweep.py                         # 3 noise x 4 samples x {strong, weak}
Runs locally on CPU (seconds per fit).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pysindy as ps
from pysindy.feature_library import PDELibrary, PolynomialLibrary
from pysindy.differentiation import FiniteDifference

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "data_loader"))
import load_data as ld  # noqa: E402

_DEFAULT_DATA_DIR = "/home/dabin/data/burgers_simulated"
MSE_CAP = 1e3


def forcing_field(A, w, phi, l, N, L, xg, dt, nt):
    """Analytic forcing f(x,t) = sum_k A_k sin(w_k t + 2 pi l_k x/L + phi_k) on a grid.

    Matches burgers_sim._forcing_at exactly (verified bit-identical draw)."""
    tg = np.arange(nt) * dt
    XX, TT = np.meshgrid(xg, tg, indexing="ij")
    F = np.zeros_like(XX)
    for k in range(int(N)):
        F += A[k] * np.sin(w[k] * TT + 2.0 * np.pi * l[k] * (XX / L) + phi[k])
    return F

def fit_pdefind(blocks, xg, dt, forcing=None, threshold=0.05, alpha=1e-5,
                poly_degree=2, deriv_order=2, weak=False, K=500, seed=0):
    """Fit PDE-FIND jointly over a list of time-blocks. Returns {equation, names, coefs}.

    blocks : a list of (Nx, nt_b) contiguous time-blocks (a single field is just a
        length-1 list). All blocks are fit JOINTLY -- one equation for all of them --
        by stacking each block's regression rows (u_t / weak-form target and library
        columns) and running a single sparse regression. Derivatives are estimated PER
        BLOCK so no finite difference / weak-integral window straddles a block boundary;
        this is how SINDy is cross-validated on the leak-free alternating blocks of
        load_data.block_split (fit on train blocks, score on test blocks).

    Both the strong and weak paths honour a known forcing f(x,t):

    forcing : a list of per-block (Nx, nt_b) arrays matching `blocks`, or None. If given,
              it is removed so SINDy fits the homogeneous operator
              u_t - f = N(u, u_x, u_xx, ...). This is the fair known-input setup
              (STENCIL-NET is *given* the same forcing).
              * STRONG path: subtract f from the finite-difference u_t before regression.
              * WEAK path: WeakPDELibrary computes a weak-form target <u_t, phi> and does
                NOT accept an x_dot argument. But the weak equation is
                    <u_t, phi> = sum_j Xi_j <theta_j(u), phi> + <f, phi>,
                and <f, phi> = <F_t, phi> for F the time-antiderivative of f, so by the
                linearity of convert_u_dot_integral we can move it to the LHS:
                    target = convert_u_dot_integral(u) - convert_u_dot_integral(F).
                F is a per-block cumulative-trapezoid antiderivative; its arbitrary additive
                constant is annihilated by <1, phi_t> = 0, so restarting it at each block is
                exact. We assemble the weak regression by hand (pysindy's Optimizer.fit
                rejects the raw 2D weak target via its AxesArray bookkeeping) with a small STLSQ.
    weak    : if True, use WeakPDELibrary (integral formulation) -- the noise-robust
              "integral SINDy" variant the Champion paper cites. Validated: with the
              forcing correction it recovers 0.02 u_xx - 1.0 u u_x on the FORCED fine
              grid, and stays robust to sigma=0.1 (see journal 2026-07-20_weak_sindy.md).
    seed    : WEAK path only. WeakPDELibrary picks its K integration subdomains from the
              global np.random and exposes no seed, so repeat fits on identical data
              return different coefficients. This pins the draw. The scatter it pins is
              real, not numerical: over repeat draws the diffusion coefficient is stable
              to ~1-2% at high D but the advective one moves by tens of percent at low D
              (see journal 2026-07-29), so single-draw coefficients should not be quoted
              to more precision than that.
    """
    if not isinstance(blocks, (list, tuple)):
        raise TypeError("blocks must be a list of (Nx, nt_b) time-blocks (use [field] for one)")
    fblocks = [None] * len(blocks) if forcing is None else list(forcing)
    if len(fblocks) != len(blocks):
        raise ValueError(f"forcing has {len(fblocks)} blocks but blocks has {len(blocks)}")
    if weak:
        return _fit_weak(blocks, xg, dt, fblocks, threshold, alpha, poly_degree, deriv_order,
                         K, seed)
    return _fit_strong(blocks, xg, dt, fblocks, threshold, alpha, poly_degree, deriv_order)


def _model_to_dict(model):
    names = list(model.get_feature_names())
    coefs = np.asarray(model.coefficients()).ravel().tolist()
    return {"equation": model.equations()[0], "names": names, "coefs": coefs}


def _fit_strong(blocks, xg, dt, fblocks, threshold, alpha, poly_degree, deriv_order):
    # One (Nx, nt_b, 1) trajectory per block; u_t is finite-differenced within each
    # block (never across a boundary) then forcing-corrected. pysindy stacks the
    # per-trajectory library rows and solves a single STLSQ over all blocks.
    Us, Xdots = [], []
    for u, f in zip(blocks, fblocks):
        Us.append(u[:, :, None])
        if f is not None:
            ut = FiniteDifference(axis=1)._differentiate(u, dt)
            Xdots.append((ut - f)[:, :, None])
        else:
            Xdots.append(None)
    x_dot = Xdots if all(x is not None for x in Xdots) else None  # else pysindy estimates u_t
    lib = PDELibrary(
        function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
        derivative_order=deriv_order, spatial_grid=xg,
        include_bias=True, is_uniform=True, periodic=True,
    )
    model = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=threshold, alpha=alpha))
    model.fit(Us, t=dt, x_dot=x_dot, feature_names=["u"])
    return _model_to_dict(model)


def _fit_weak(blocks, xg, dt, fblocks, threshold, alpha, poly_degree, deriv_order, K,
              seed=0):
    from pysindy.feature_library import WeakPDELibrary
    from pysindy.utils import AxesArray
    nt = blocks[0].shape[1]
    # All blocks share one grid, so build the weak library once (its K random
    # spatiotemporal integration subdomains are fixed at construction) and push every
    # block through the SAME subdomains, then stack the K rows per block.
    #
    # WeakPDELibrary draws those subdomain centres from the GLOBAL np.random and has no
    # seed argument of its own, so two runs on identical data return different
    # coefficients. Seed around the construction (restoring the caller's RNG state
    # afterwards, so this stays a local effect) to make the fit reproducible. Note this
    # pins the draw; it does not remove the underlying subdomain-draw variance, which is
    # real and largest for the advective coefficient at low D.
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        lib = WeakPDELibrary(
            function_library=PolynomialLibrary(degree=poly_degree, include_bias=False),
            derivative_order=deriv_order,
            spatiotemporal_grid=_spatiotemporal_grid(xg, dt, nt),
            include_bias=True, is_uniform=True, periodic=True, K=K,
        )
    finally:
        np.random.set_state(rng_state)
    axes = {"ax_spatial": [0], "ax_time": 1, "ax_coord": [2]}
    lib.fit([AxesArray(blocks[0][:, :, None], axes)])
    Thetas, ydots = [], []
    for u, f in zip(blocks, fblocks):
        if u.shape[1] != nt:
            raise ValueError("weak SINDy blocks must share block_len (one shared grid)")
        U = AxesArray(u[:, :, None], axes)
        Thetas.append(np.array(lib.transform([U])[0], subok=False, dtype=float))   # (K, n_feat)
        yd = np.array(lib.convert_u_dot_integral(U), subok=False, dtype=float).ravel()
        if f is not None:
            # F = per-block time-antiderivative of f (cumulative trapezoid), so <f,phi>=<F_t,phi>
            F = np.zeros_like(f)
            F[:, 1:] = np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dt, axis=1)
            Fa = AxesArray(F[:, :, None], axes)
            yd = yd - np.array(lib.convert_u_dot_integral(Fa), subok=False, dtype=float).ravel()
        ydots.append(yd)
    Theta = np.vstack(Thetas)
    ydot = np.concatenate(ydots)
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


# ----------------------------------------------------------------------------
# recovered-PDE -> RHS, with a shock-capturing integrator. Moved here from
# compare_sindy_stencilnet.py: this runner needs them to score its own fits, and
# the compare script re-exports them.
# ----------------------------------------------------------------------------
def parse_term(name):
    """PDELibrary feature name -> (poly_power_outside_deriv, deriv_order).

    'uu_1'->(1,1), 'u^2u_11'->(2,2), 'u'->(1,0), 'u^2'->(2,0), '1'->(0,0)."""
    name = name.strip()
    if name in ("1", ""):
        return (0, 0)
    dorder = 0
    base = name
    if "_" in name:
        base, ones = name.rsplit("_", 1)
        dorder = len(ones)
    E = 0
    for tok in re.findall(r"u(?:\^(\d+))?", base):
        E += int(tok) if tok else 1
    poly = E - (1 if dorder > 0 else 0)  # one u lives inside the derivative
    return (poly, dorder)


def make_rhs(names, coefs, dx, scheme="central"):
    """u_t = N(u): reaction (order0, pointwise) + advection (order1, LF flux)
       + diffusion (order2, central). Order-1 term c u^p u_x = d/dx[c/(p+1) u^(p+1)]
       is integrated conservatively so shock-forming advection stays stable. 'scheme' sets
       the interface dissipation alpha

           'central' : alpha=0. No added dissipation.
           'lax' : Lax-Friedrichs (alpha = max wave speed) - adds dissipation to stabilize shocks.
    """
    react, flux, diff = [], [], []
    for n, c in zip(names, coefs):
        c = float(c)
        if c == 0:
            continue
        p, d = parse_term(n)
        (react if d == 0 else flux if d == 1 else diff).append((c, p))

    # np.roll on axis=-1 (the spatial axis) so N works on both a single field (Lx,)
    # and a batch of restart states (n_starts, Lx) — teacher_forced_forecast steps
    # all restarts at once. alpha (local wave speed) is per-row via keepdims.
    def dx2(v):
        return (np.roll(v, -1, axis=-1) - 2 * v + np.roll(v, 1, axis=-1)) / dx ** 2

    def N(v):
        out = np.zeros_like(v)
        for c, p in react:
            out += c * v ** p
        for c, p in diff:
            out += c * (v ** p) * dx2(v)
        if flux:
            f = -sum(c / (p + 1) * v ** (p + 1) for c, p in flux)  # u_t = -f_x
            if scheme == "central":
                alpha = 0.0
            else:
                alpha = np.max(np.abs(sum(c * v ** p for c, p in flux)), axis=-1, keepdims=True) + 1e-12
            Fp = 0.5 * (f + np.roll(f, -1, axis=-1)) - 0.5 * alpha * (np.roll(v, -1, axis=-1) - v)
            out += -(Fp - np.roll(Fp, 1, axis=-1)) / dx
        return out
    return N


def _score(rhs, field, dtc, train_cols, test_cols, rollout_steps):
    """Forecast MSE + persistence floor on the shared teacher-forced restarts.

    `field` is the OBSERVED (noisy) field: it seeds every restart and supplies every
    target, so the fit is graded on exactly what EDGAR is graded on. Seeding from
    u_clean would hand SINDy a denoised initial condition EDGAR never receives.
    """
    T = field.shape[1]
    preds, targets = ld.teacher_forced_forecast(rhs, field, dtc, rollout_steps)
    stable = bool(np.all(np.isfinite(preds)))
    preds = np.nan_to_num(preds, nan=1e30, posinf=1e30, neginf=-1e30)
    p_preds, p_targets = ld.teacher_forced_forecast(
        lambda state, t_arr: np.zeros_like(state), field, dtc, rollout_steps)
    tr, te = ld.split_start_masks(train_cols, test_cols, T, rollout_steps)
    mse_tr = min(ld.forecast_mse(preds[tr], targets[tr]), MSE_CAP)
    mse_te = min(ld.forecast_mse(preds[te], targets[te]), MSE_CAP)
    pers_tr = ld.forecast_mse(p_preds[tr], p_targets[tr])
    pers_te = ld.forecast_mse(p_preds[te], p_targets[te])
    return {
        "rollout_steps": rollout_steps, "stable": stable,
        "n_train_starts": int(tr.sum()), "n_test_starts": int(te.sum()),
        "forecast_mse_train": mse_tr, "forecast_mse_test": mse_te,
        "persistence_mse_train": pers_tr, "persistence_mse_test": pers_te,
        "skill_test": 1.0 - mse_te / pers_te,
    }


def run(data_path, sample_idx, weak=False, threshold=0.002, rollout_steps=(1, 2, 4),
        block_len=200, out_dir=None):
    """Fit one sample of a shared benchmark dataset and score it at several horizons.

    The fit uses only the TRAIN blocks (leak-free ld.block_split), as a list of
    contiguous sub-fields so no derivative / weak-integral window straddles a
    boundary, then is scored on the disjoint TEST blocks it never saw.

    Unlike STENCIL-NET, the fit does NOT depend on rollout_steps -- SINDy recovers a
    continuous operator, and the horizon only changes how far it is rolled at score
    time. So one fit is scored at every horizon rather than refitted per horizon.

    The true operator is u_t = D u_xx - u u_x with D the sample's own viscosity, so
    coefficient recovery is checked against a target that moves per sample.
    """
    ds = ld.load_dataset(data_path)
    out_dir = out_dir or os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    sample_idx = int(sample_idx)

    xg = ds["x_coarse"].astype(float)
    dtc = ds["dtc"]
    dx = float(xg[1] - xg[0])
    D = float(ds["D"][sample_idx])
    noise_level = float(ds["noise_level"])
    ic_seed = int(ds["ic_seed"])
    if ds["forced"]:
        raise ValueError(f"{data_path} is forced; this runner handles the unforced datasets")
    u_obs = ds["u_noisy"][sample_idx]
    u_clean = ds["u_clean"][sample_idx]
    T = u_clean.shape[1]

    train_cols, test_cols = ld.block_split(T, block_len=block_len)
    fit = fit_pdefind(ld.contiguous_blocks(u_obs, train_cols), xg, dtc, forcing=None,
                      threshold=threshold, weak=weak)
    cmap = dict(zip(fit["names"], fit["coefs"]))
    N = make_rhs(fit["names"], fit["coefs"], dx)
    rhs = lambda state, t_arr: N(state)  # noqa: E731  (unforced: no time dependence)

    horizons = [_score(rhs, u_obs, dtc, train_cols, test_cols, int(h))
                for h in rollout_steps]

    coef_u_xx = float(cmap.get("u_11", 0.0))
    coef_uu_x = float(cmap.get("uu_1", 0.0))
    result = {
        "method": "sindy_weak" if weak else "sindy_strong",
        "data_path": data_path, "ic_seed": ic_seed, "sample_idx": sample_idx, "D": D,
        "noise_level": noise_level, "threshold": threshold, "block_len": block_len,
        "n_x": int(u_clean.shape[0]),
        "equation": fit["equation"], "feature_names": fit["names"],
        "coefficients": fit["coefs"],
        "n_terms": int(sum(1 for c in fit["coefs"] if c != 0)),
        # truth is u_t = D u_xx - u u_x, so the u_xx target moves with the sample
        "coef_u_xx": coef_u_xx, "true_u_xx": D,
        "coef_uu_x": coef_uu_x, "true_uu_x": -1.0,
        "rel_err_u_xx": (coef_u_xx - D) / D,
        "rel_err_uu_x": (coef_uu_x + 1.0) / 1.0,
        "horizons": horizons,
    }
    tag = f"{'weak' if weak else 'strong'}_ic{ic_seed}_nl{noise_level}_s{sample_idx}"
    with open(os.path.join(out_dir, f"sindy_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, dest="data_path",
                    help="ic_seed_*_nl_*.npz written by data_loader/generate_datasets.py")
    ap.add_argument("--sample-idx", type=int, required=True, dest="sample_idx")
    ap.add_argument("--threshold", type=float, default=0.002,
                    help="STLSQ threshold; must sit below the smallest true D (0.005)")
    ap.add_argument("--weak", action="store_true", help="use WeakPDELibrary (integral SINDy)")
    ap.add_argument("--rollout-steps", type=int, nargs="+", default=[1, 2, 4],
                    dest="rollout_steps")
    args = ap.parse_args()
    r = run(data_path=args.data_path, sample_idx=args.sample_idx,
            threshold=args.threshold, weak=args.weak, rollout_steps=args.rollout_steps)
    print(f"D={r['D']}  nl={r['noise_level']}  {r['method']}")
    print(f"  (u)' = {r['equation']}")
    print(f"  u_xx {r['coef_u_xx']:.5f} (true {r['true_u_xx']:.3f})  "
          f"uu_x {r['coef_uu_x']:.3f} (true -1.0)")
    for h in r["horizons"]:
        print(f"  h={h['rollout_steps']}: mse_test {h['forecast_mse_test']:.3e} "
              f"pers {h['persistence_mse_test']:.3e} skill {h['skill_test']:.5f} "
              f"stable={h['stable']}")
