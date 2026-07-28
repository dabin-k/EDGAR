"""Standalone head-to-head: SINDy (PDE-FIND) vs STENCIL-NET on forced Burgers.

Both methods are scored on the SAME shared metric -- forecast MSE against the
CLEAN ground-truth field (load_data.forecast_mse) -- swept over observation
noise sigma. This is Step 4 of the benchmark plan.

The two methods produce different objects, so we report two complementary views:

  1. COEFFICIENT RECOVERY (integrator-independent, SINDy only): how close the
     recovered coefficients of the two true Burgers terms (0.02 u_xx, -1.0 u u_x)
     are to truth. STENCIL-NET has no symbolic form, so it has no entry here.

  2. FORECAST MSE (shared, all methods): teacher-forced RK3 restarts scored vs the
     clean field with the KNOWN forcing -- the SAME protocol EDGAR is graded on
     (load_data.teacher_forced_forecast / evaluate.py): from every true state, roll
     `rollout_steps` on the model's own predictions and score every step.
       * STENCIL-NET's numbers come from its own runner (read from
         results/stencilnet_*.json), which uses the identical shared scorer.
       * SINDy recovers a continuous PDE u_t = N(u; Xi) + f, integrated with a
         shock-capturing Lax-Friedrichs flux for the advective (order-1) terms +
         central differences for diffusion + RK3 in time -- the same stepper and
         known forcing STENCIL-NET uses. A recovered PDE that is inaccurate or
         carries spurious anti-diffusive terms (e.g. u^2 u_xx) can still blow up
         within a rollout window; we flag that (stable=False) and cap the MSE.
         This instability is a genuine, reportable SINDy weakness -- exactly the
         failure mode the benchmark set out to probe.

  An ORACLE line (the TRUE 0.02 u_xx - 1.0 u u_x scored with the same restarts and
  integrator) is the forecast-MSE floor: it shows how much error is the coarse-grid
  integrator itself vs the recovered dynamics.

Usage:
    python compare_sindy_stencilnet.py                      # default sweep + plot
    python compare_sindy_stencilnet.py --threshold 0.01     # SINDy STLSQ threshold
    python compare_sindy_stencilnet.py --levels 0.0 0.05 0.1

Writes results/compare_results.json and results/sindy_vs_stencilnet.png.
STENCIL-NET rows are merged in automatically if results/stencilnet_noise*.json
exist (produced on the GPU box); otherwise the SINDy side + oracle are reported
alone and STENCIL-NET is drawn as "pending".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SINDY = os.path.join(_HERE, "sindy")
_SN_RESULTS = os.path.join(_HERE, "stencilnet", "results")
_OUT = os.path.join(_HERE, "results")

sys.path.insert(0, _SINDY)
sys.path.insert(0, os.path.join(_HERE, "..", "data_loader"))
import load_data as ld  # noqa: E402
import runner as sindy_runner  # noqa: E402  (bench/sindy/runner.py)

TRUE = {"u_11": 0.02, "uu_1": -1.0}  # target Burgers operator (D=0.02)


# ----------------------------------------------------------------------------
# recovered-PDE -> RHS, with a shock-capturing integrator (see module docstring)
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


def make_rhs(names, coefs, dx):
    """u_t = N(u): reaction (order0, pointwise) + advection (order1, LF flux)
       + diffusion (order2, central). Order-1 term c u^p u_x = d/dx[c/(p+1) u^(p+1)]
       is integrated conservatively so shock-forming advection stays stable."""
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
            alpha = np.max(np.abs(sum(c * v ** p for c, p in flux)), axis=-1, keepdims=True) + 1e-12
            Fp = 0.5 * (f + np.roll(f, -1, axis=-1)) - 0.5 * alpha * (np.roll(v, -1, axis=-1) - v)
            out += -(Fp - np.roll(Fp, 1, axis=-1)) / dx
        return out
    return N


# ----------------------------------------------------------------------------
def sindy_sweep(levels, threshold, weak, bundle, forcing_batched, F, train_cols,
                rollout_steps, MSE_CAP=1e3):
    xg = bundle["x_coarse"].astype(float)
    dtc = bundle["dtc"]
    dx = xg[1] - xg[0]
    u_clean = bundle["u_coarse"]
    rows = []
    for nl in levels:
        u = ld.noise_field(bundle, nl)
        fit = sindy_runner.fit_pdefind([u], xg, dtc, forcing=[F], threshold=threshold, weak=weak)
        cmap = dict(zip(fit["names"], fit["coefs"]))
        N = make_rhs(fit["names"], fit["coefs"], dx)
        # teacher-forced restarts, identical to STENCIL-NET and EDGAR. A recovered
        # PDE with spurious anti-diffusive terms can still blow up within a rollout
        # window (stable=False); we cap the MSE, but re-anchoring keeps one bad
        # window from NaN-ing the whole forecast.
        rhs = lambda state, t: N(state) + forcing_batched(t)  # noqa: E731
        preds, targets = ld.teacher_forced_forecast(rhs, u_clean, dtc, rollout_steps)
        stable = bool(np.all(np.isfinite(preds)))
        preds_g = np.nan_to_num(preds, nan=1e30, posinf=1e30, neginf=-1e30)
        heldout = np.arange(preds.shape[0]) >= train_cols
        mse_full = min(ld.forecast_mse(preds_g, targets), MSE_CAP)
        mse_hld = min(ld.forecast_mse(preds_g[heldout], targets[heldout]), MSE_CAP)
        rows.append({
            "method": "sindy_weak" if weak else "sindy_strong",
            "noise_level": nl, "threshold": threshold,
            "equation": fit["equation"],
            "coef_u_xx": float(cmap.get("u_11", 0.0)),
            "coef_uu_x": float(cmap.get("uu_1", 0.0)),
            "forecast_mse_full": mse_full, "forecast_mse_heldout": mse_hld,
            "stable": stable,
        })
    return rows


def oracle_row(bundle, forcing_batched, rollout_steps, train_cols):
    xg = bundle["x_coarse"].astype(float)
    dtc = bundle["dtc"]
    dx = xg[1] - xg[0]
    u_clean = bundle["u_coarse"]
    N = make_rhs(["u_11", "uu_1"], [TRUE["u_11"], TRUE["uu_1"]], dx)
    rhs = lambda state, t: N(state) + forcing_batched(t)  # noqa: E731
    preds, targets = ld.teacher_forced_forecast(rhs, u_clean, dtc, rollout_steps)
    heldout = np.arange(preds.shape[0]) >= train_cols
    return {
        "forecast_mse_full": ld.forecast_mse(preds, targets),
        "forecast_mse_heldout": ld.forecast_mse(preds[heldout], targets[heldout]),
        "stable": bool(np.all(np.isfinite(preds))),
    }


def load_stencilnet(min_epochs=30000):
    """Read converged STENCIL-NET runs from the GPU box. Un-converged smoke runs
    (epochs < min_epochs) are skipped so a partial run does not pollute the plot."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_SN_RESULTS, "stencilnet_noise*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("epochs", 0) < min_epochs:
            print(f"  [skip] {os.path.basename(path)}: epochs={r.get('epochs')} "
                  f"< {min_epochs} (not converged)")
            continue
        rows.append({
            "method": "stencilnet", "noise_level": r["noise_level"],
            "forecast_mse_full": r.get("forecast_mse_full"),
            "forecast_mse_heldout": r.get("forecast_mse_heldout"),
        })
    return rows


# ----------------------------------------------------------------------------
def make_plot(sindy_s, sindy_w, sn_rows, oracle, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel A: coefficient recovery (SINDy only)
    levs = [r["noise_level"] for r in sindy_s]
    ax0.axhline(TRUE["uu_1"], ls="--", c="0.6", lw=1, label="true u·u_x (-1.0)")
    ax0.axhline(TRUE["u_11"], ls=":", c="0.6", lw=1, label="true u_xx (0.02)")
    ax0.plot(levs, [r["coef_uu_x"] for r in sindy_s], "o-", c="C0", label="strong: u·u_x")
    ax0.plot(levs, [r["coef_uu_x"] for r in sindy_w], "s-", c="C1", label="weak: u·u_x")
    ax0.plot(levs, [r["coef_u_xx"] for r in sindy_s], "o--", c="C0", alpha=0.5, label="strong: u_xx")
    ax0.plot(levs, [r["coef_u_xx"] for r in sindy_w], "s--", c="C1", alpha=0.5, label="weak: u_xx")
    ax0.set_xlabel("noise level σ (fraction of u_std)")
    ax0.set_ylabel("recovered coefficient")
    ax0.set_title("A. SINDy coefficient recovery")
    ax0.legend(fontsize=7, ncol=2)

    # panel B: forecast MSE (shared metric)
    def _plot(ax, rows, style, label, color):
        xs = [r["noise_level"] for r in rows]
        ys = [r["forecast_mse_full"] for r in rows]
        ax.plot(xs, ys, style, color=color, label=label)
        for r in rows:  # mark unstable SINDy fits
            if not r.get("stable", True):
                ax.plot(r["noise_level"], r["forecast_mse_full"], "x", color="red", ms=9, mew=2)

    ax1.axhline(oracle["forecast_mse_full"], ls="--", c="0.5", lw=1,
                label=f"oracle floor ({oracle['forecast_mse_full']:.2e})")
    _plot(ax1, sindy_s, "o-", "SINDy strong", "C0")
    _plot(ax1, sindy_w, "s-", "SINDy weak", "C1")
    if sn_rows:
        _plot(ax1, sorted(sn_rows, key=lambda r: r["noise_level"]), "^-", "STENCIL-NET", "C2")
    else:
        ax1.plot([], [], "^-", color="C2", label="STENCIL-NET (pending GPU run)")
    ax1.set_yscale("log")
    ax1.set_xlabel("noise level σ (fraction of u_std)")
    ax1.set_ylabel("forecast MSE vs clean field")
    ax1.set_title("B. Forecast MSE (shared metric)")
    ax1.legend(fontsize=7)
    ax1.text(0.02, 0.02, "red × = unstable integration (capped)", transform=ax1.transAxes,
             fontsize=6.5, color="red", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def main(levels, threshold):
    warnings.filterwarnings("ignore")
    os.makedirs(_OUT, exist_ok=True)
    b = ld.load_bundle()
    xg = b["x_coarse"].astype(float)
    dtc = b["dtc"]
    A, w, phi, l, N, L = b["A"], b["w"], b["phi"], b["l"], int(b["N"]), b["L"]
    F = sindy_runner.forcing_field(A, w, phi, l, N, L, xg, dtc, b["u_coarse"].shape[1])

    def forcing_batched(t_arr):
        """Known forcing at physical times t_arr (n_starts,) -> (n_starts, Lx)."""
        tt = np.asarray(t_arr)[:, None]
        return sum(A[k] * np.sin(w[k] * tt + 2 * np.pi * l[k] * (xg[None, :] / L) + phi[k])
                   for k in range(N))

    train_cols = 1001  # matches STENCIL-NET training window
    rollout_steps = ld.benchmark_rollout_steps()  # same horizon as EDGAR + STENCIL-NET

    sindy_s = sindy_sweep(levels, threshold, False, b, forcing_batched, F, train_cols, rollout_steps)
    sindy_w = sindy_sweep(levels, threshold, True, b, forcing_batched, F, train_cols, rollout_steps)
    oracle = oracle_row(b, forcing_batched, rollout_steps, train_cols)
    sn_rows = load_stencilnet()

    results = {"threshold": threshold, "levels": list(levels), "oracle": oracle,
               "sindy_strong": sindy_s, "sindy_weak": sindy_w, "stencilnet": sn_rows}
    with open(os.path.join(_OUT, "compare_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    png = make_plot(sindy_s, sindy_w, sn_rows, oracle, os.path.join(_OUT, "sindy_vs_stencilnet.png"))

    # console summary table
    print(f"\nForecast MSE vs clean (threshold={threshold}), oracle floor="
          f"{oracle['forecast_mse_full']:.3e}")
    hdr = f"{'noise':>6} | {'strong':>11} {'st?':>3} | {'weak':>11} {'st?':>3} | {'STENCIL-NET':>12}"
    print(hdr); print("-" * len(hdr))
    sn_by = {r["noise_level"]: r for r in sn_rows}
    for rs, rw in zip(sindy_s, sindy_w):
        nl = rs["noise_level"]
        sn = sn_by.get(nl, {}).get("forecast_mse_full")
        sn_s = f"{sn:12.3e}" if sn is not None else f"{'pending':>12}"
        print(f"{nl:6.3f} | {rs['forecast_mse_full']:11.3e} "
              f"{'Y' if rs['stable'] else 'N':>3} | {rw['forecast_mse_full']:11.3e} "
              f"{'Y' if rw['stable'] else 'N':>3} | {sn_s}")
    print(f"\nwrote {_OUT}/compare_results.json and {png}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="SINDy STLSQ threshold (0.01 keeps the true 0.02 u_xx term)")
    args = ap.parse_args()
    main(args.levels, args.threshold)
