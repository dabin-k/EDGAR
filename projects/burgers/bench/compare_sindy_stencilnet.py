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
import warnings

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SINDY = os.path.join(_HERE, "sindy")
_SN_RESULTS = os.path.join(_HERE, "stencilnet", "results")
_OUT = os.path.join(_HERE, "results")

sys.path.insert(0, _HERE)
sys.path.insert(0, _SINDY)
sys.path.insert(0, os.path.join(_HERE, "..", "data_loader"))
import load_data as ld  # noqa: E402
import runner as sindy_runner  # noqa: E402  (bench/sindy/runner.py)

TRUE = {"u_11": 0.02, "uu_1": -1.0}  # target Burgers operator (D=0.02)
BLOCK_LEN = 200

def contiguous_blocks(field: np.ndarray, cols: np.ndarray) -> list[np.ndarray]:
    """Split `field[:, cols]` back into its maximal runs of consecutive columns.

    `block_split` returns the train (or test) columns as one concatenated index
    array; this recovers the individual contiguous time-blocks, so a per-block
    consumer (SINDy's joint fit, STENCIL-NET's rollout loss) never forms a
    finite-difference / rollout window that straddles a block boundary.
    """
    cols = np.asarray(cols, int)
    if cols.size == 0:
        return []
    breaks = np.where(np.diff(cols) != 1)[0] + 1
    return [field[:, run] for run in np.split(cols, breaks)]


split_start_masks = ld.split_start_masks   # shared with the STENCIL-NET/SINDy runners


# recovered-PDE -> RHS with a shock-capturing integrator; these live in the SINDy
# runner now (it needs them to score its own fits) and are re-exported here so this
# script's call sites are unchanged.
parse_term = sindy_runner.parse_term
make_rhs = sindy_runner.make_rhs


# ----------------------------------------------------------------------------
def sindy_sweep(levels, threshold, weak, bundle, forcing_batched, F, train_cols,
                train_mask, test_mask, rollout_steps, MSE_CAP=1e3, preds_save_outdir=None):
    xg = bundle["x_coarse"].astype(float)
    dtc = bundle["dtc"]
    dx = xg[1] - xg[0]
    u_clean = bundle["u_coarse"]
    rows = []
    for nl in levels:
        u = ld.noise_field(bundle, nl)
        # Cross-validated fit: SINDy sees only the TRAIN blocks, as a list of contiguous
        # sub-fields (one joint equation; derivatives stay within a block). Then it is
        # scored on the disjoint TEST blocks it never saw.
        u_blocks = contiguous_blocks(u, train_cols)
        F_blocks = contiguous_blocks(F, train_cols)
        fit = sindy_runner.fit_pdefind(u_blocks, xg, dtc, forcing=F_blocks,
                                       threshold=threshold, weak=weak)
        cmap = dict(zip(fit["names"], fit["coefs"]))
        N = make_rhs(fit["names"], fit["coefs"], dx)
        # teacher-forced restarts, identical to STENCIL-NET and EDGAR. A recovered
        # PDE with spurious anti-diffusive terms can still blow up within a rollout
        # window (stable=False); we cap the MSE, but re-anchoring keeps one bad
        # window from NaN-ing the whole forecast.
        rhs = lambda state, t: N(state) + forcing_batched(t)  # noqa: E731
        preds, targets = ld.teacher_forced_forecast(rhs, u_clean, dtc, rollout_steps)
        if preds_save_outdir:
            weak_or_strong = "sindy_weak" if weak else "sindy_strong"
            preds_save_outpath = os.path.join(preds_save_outdir, f"{weak_or_strong}_noise_level_{nl}.npz")
            np.savez(preds_save_outpath, preds=preds, targets=targets)
        stable = bool(np.all(np.isfinite(preds)))
        preds_g = np.nan_to_num(preds, nan=1e30, posinf=1e30, neginf=-1e30)
        mse_tr = min(ld.forecast_mse(preds_g[train_mask], targets[train_mask]), MSE_CAP)
        mse_te = min(ld.forecast_mse(preds_g[test_mask], targets[test_mask]), MSE_CAP)
        rows.append({
            "method": "sindy_weak" if weak else "sindy_strong",
            "noise_level": nl, "threshold": threshold,
            "equation": fit["equation"],
            "rollout_steps": rollout_steps,
            "coef_u_xx": float(cmap.get("u_11", 0.0)),
            "coef_uu_x": float(cmap.get("uu_1", 0.0)),
            "forecast_mse_train": mse_tr, "forecast_mse_test": mse_te,
            "stable": stable,
        })
    return rows


def oracle_row(bundle, forcing_batched, rollout_steps, train_mask, test_mask, preds_save_outdir=None):
    xg = bundle["x_coarse"].astype(float)
    dtc = bundle["dtc"]
    dx = xg[1] - xg[0]
    u_clean = bundle["u_coarse"]
    N = make_rhs(["u_11", "uu_1"], [TRUE["u_11"], TRUE["uu_1"]], dx)
    rhs = lambda state, t: N(state) + forcing_batched(t)  # noqa: E731
    preds, targets = ld.teacher_forced_forecast(rhs, u_clean, dtc, rollout_steps)
    if preds_save_outdir:
        preds_save_outpath = os.path.join(preds_save_outdir, f"oracle_preds_targets.npz")
        np.savez(preds_save_outpath, preds=preds, targets=targets)
    return {
        "forecast_mse_train": ld.forecast_mse(preds[train_mask], targets[train_mask]),
        "forecast_mse_test": ld.forecast_mse(preds[test_mask], targets[test_mask]),
        "stable": bool(np.all(np.isfinite(preds))),
    }


def load_stencilnet(rollout_steps, min_epochs=30000):
    """Read converged STENCIL-NET runs from the GPU box. Un-converged smoke runs
    (epochs < min_epochs) are skipped so a partial run does not pollute the plot.
    The runner must score on the SAME block split (forecast_mse_test); rows missing
    that key are pre-CV runs and are skipped."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_SN_RESULTS, f"rollout{rollout_steps}/stencilnet_noise*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("epochs", 0) < min_epochs:
            print(f"  [skip] {os.path.basename(path)}: epochs={r.get('epochs')} "
                  f"< {min_epochs} (not converged)")
            continue
        if "forecast_mse_test" not in r:
            print(f"  [skip] {os.path.basename(path)}: no forecast_mse_test "
                  f"(pre-CV run; re-run stencilnet/runner.py)")
            continue
        rows.append({
            "method": "stencilnet", "noise_level": r["noise_level"], "rollout_steps": rollout_steps,
            "forecast_mse_train": r.get("forecast_mse_train"),
            "forecast_mse_test": r.get("forecast_mse_test"),
        })
    return rows


# ----------------------------------------------------------------------------
def make_sindy_coefficients_plot(sindy_s, sindy_w, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))

    # panel A: coefficient recovery (SINDy only)
    levs = [r["noise_level"] for r in sindy_s]
    plt.axhline(TRUE["uu_1"], ls="--", c="0.6", lw=1, label="true u·u_x (-1.0)")
    plt.axhline(TRUE["u_11"], ls=":", c="0.6", lw=1, label="true u_xx (0.02)")
    plt.plot(levs, [r["coef_uu_x"] for r in sindy_s], "o-", c="C0", label="strong: u·u_x")
    plt.plot(levs, [r["coef_uu_x"] for r in sindy_w], "s-", c="C1", label="weak: u·u_x")
    plt.plot(levs, [r["coef_u_xx"] for r in sindy_s], "o--", c="C0", alpha=0.5, label="strong: u_xx")
    plt.plot(levs, [r["coef_u_xx"] for r in sindy_w], "s--", c="C1", alpha=0.5, label="weak: u_xx")
    plt.xlabel("noise level σ (fraction of u_std)")
    plt.ylabel("recovered coefficient")
    plt.title("SINDy coefficient recovery")
    plt.legend(fontsize=7, ncol=2)

    plt.savefig(out_path, dpi=150)
    return out_path


def make_mse_plot(results_list, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rollouts = len(results_list)
    fig, axs = plt.subplots(1, n_rollouts, figsize=(16, 4.2))

    # panel B: held-out forecast MSE (shared metric, cross-validated on TEST blocks)
    def _plot(ax, rows, style, label, color):
        xs = [r["noise_level"] for r in rows]
        ys = [r["forecast_mse_test"] for r in rows]
        ax.plot(xs, ys, style, color=color, label=label)
        for r in rows:  # mark unstable SINDy fits
            if not r.get("stable", True):
                ax.plot(r["noise_level"], r["forecast_mse_test"], "x", color="red", ms=9, mew=2)
    
    for i, results in enumerate(results_list):
        rollout_steps = results["rollout_steps"]
        sindy_s = results["sindy_strong"]
        sindy_w = results["sindy_weak"]
        sn_rows = results.get("stencilnet", [])
        oracle = results["oracle"]

        ax = axs[i]
        ax.set_title(f"Rollout Steps: {rollout_steps}")
        ax.axhline(oracle["forecast_mse_test"], ls="--", c="0.5", lw=1,
                    label=f"oracle floor ({oracle['forecast_mse_test']:.2e})")
        _plot(ax, sindy_s, "o-", "SINDy strong", "C0")
        _plot(ax, sindy_w, "s-", "SINDy weak", "C1")
        if sn_rows:
            _plot(ax, sorted(sn_rows, key=lambda r: r["noise_level"]), "^-", "STENCIL-NET", "C2")
        else:
            ax.plot([], [], "^-", color="C2", label="STENCIL-NET (pending GPU run)")
        ax.set_yscale("log")
        ax.set_xlabel("noise level σ (fraction of u_std)")
        ax.set_ylabel("held-out forecast MSE vs clean field")
        ax.set_title(f"Rollout steps {rollout_steps}")
        ax.legend(fontsize=7)
        ax.text(0.02, 0.02, "red × = unstable integration (capped)", transform=ax.transAxes,
                 fontsize=6.5, color="red", va="bottom")

    fig.suptitle("Held-out forecast MSE for different values of rollout_steps", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path

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

    # panel B: held-out forecast MSE (shared metric, cross-validated on TEST blocks)
    def _plot(ax, rows, style, label, color):
        xs = [r["noise_level"] for r in rows]
        ys = [r["forecast_mse_test"] for r in rows]
        ax.plot(xs, ys, style, color=color, label=label)
        for r in rows:  # mark unstable SINDy fits
            if not r.get("stable", True):
                ax.plot(r["noise_level"], r["forecast_mse_test"], "x", color="red", ms=9, mew=2)

    ax1.axhline(oracle["forecast_mse_test"], ls="--", c="0.5", lw=1,
                label=f"oracle floor ({oracle['forecast_mse_test']:.2e})")
    _plot(ax1, sindy_s, "o-", "SINDy strong", "C0")
    _plot(ax1, sindy_w, "s-", "SINDy weak", "C1")
    if sn_rows:
        _plot(ax1, sorted(sn_rows, key=lambda r: r["noise_level"]), "^-", "STENCIL-NET", "C2")
    else:
        ax1.plot([], [], "^-", color="C2", label="STENCIL-NET (pending GPU run)")
    ax1.set_yscale("log")
    ax1.set_xlabel("noise level σ (fraction of u_std)")
    ax1.set_ylabel("held-out forecast MSE vs clean field")
    ax1.set_title("B. Held-out forecast MSE (shared metric)")
    ax1.legend(fontsize=7)
    ax1.text(0.02, 0.02, "red × = unstable integration (capped)", transform=ax1.transAxes,
             fontsize=6.5, color="red", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def main(levels, threshold, rollouts):
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

    # Shared leak-free CV split: SINDy and STENCIL-NET fit/train and score on the SAME
    # alternating train/test blocks rollout_steps stays tied to EDGAR's horizon via load_data.
    T = b["u_coarse"].shape[1]
    results_list = []
    for rollout_steps in rollouts:
        print(f"\n=== rollout_steps={rollout_steps} ===")
        train_cols, test_cols = ld.block_split(T, block_len=BLOCK_LEN)
        train_mask, test_mask = split_start_masks(train_cols, test_cols, T, rollout_steps)

        # only save down preds if rollout_steps -- this is just to visualise the predictions for different noise levels
        preds_save_outdir = os.path.join(_SINDY, f"results") if rollout_steps == 1 else None
        sindy_s = sindy_sweep(levels, threshold, False, b, forcing_batched, F,
                            train_cols, train_mask, test_mask, rollout_steps, preds_save_outdir=preds_save_outdir)
        sindy_w = sindy_sweep(levels, threshold, True, b, forcing_batched, F,
                            train_cols, train_mask, test_mask, rollout_steps, preds_save_outdir=preds_save_outdir)
        oracle = oracle_row(b, forcing_batched, rollout_steps, train_mask, test_mask, preds_save_outdir=preds_save_outdir)
        sn_rows = load_stencilnet(rollout_steps)

        results = {"threshold": threshold, "levels": list(levels), "oracle": oracle, "rollout_steps": rollout_steps,
                "split": {"block_len": BLOCK_LEN, "rollout_steps": rollout_steps,
                            "n_train_starts": int(train_mask.sum()),
                            "n_test_starts": int(test_mask.sum())},
                "sindy_strong": sindy_s, "sindy_weak": sindy_w, "stencilnet": sn_rows}
        results_list.append(results)
        with open(os.path.join(_OUT, "compare_results.json"), "w") as fh:
            json.dump(results_list, fh, indent=2)

        # png = make_plot(sindy_s, sindy_w, sn_rows, oracle, os.path.join(_OUT, f"sindy_vs_stencilnet_n_rollout{rollout_steps}.png"))

        # console summary table — held-out (TEST-block) forecast MSE, the cross-validated metric
        print(f"\nHeld-out forecast MSE vs clean (threshold={threshold}), (n_rollout={rollout_steps}), oracle floor="
            f"{oracle['forecast_mse_test']:.3e}  "
            f"[{int(train_mask.sum())} train / {int(test_mask.sum())} test restarts]")
        hdr = f"{'noise':>6} | {'strong':>11} {'st?':>3} | {'weak':>11} {'st?':>3} | {'STENCIL-NET':>12}"
        print(hdr); print("-" * len(hdr))
        sn_by = {r["noise_level"]: r for r in sn_rows}
        for rs, rw in zip(sindy_s, sindy_w):
            nl = rs["noise_level"]
            sn = sn_by.get(nl, {}).get("forecast_mse_test")
            sn_s = f"{sn:12.3e}" if sn is not None else f"{'pending':>12}"
            print(f"{nl:6.3f} | {rs['forecast_mse_test']:11.3e} "
                f"{'Y' if rs['stable'] else 'N':>3} | {rw['forecast_mse_test']:11.3e} "
                f"{'Y' if rw['stable'] else 'N':>3} | {sn_s}")

    # SINDy coefficinets remain the same across rollout_steps, so only plot once (the first rollout_steps)
    png = make_sindy_coefficients_plot(sindy_s, sindy_w, os.path.join(_OUT, f"sindy_coefficients.png"))

    # create rows containing noise_levels and forecast_test_mse for each method, and concatenate as a list
    png = make_mse_plot(results_list, os.path.join(_OUT, f"sindy_vs_stencilnet_mse.png"))

    return results_list


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3])
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="SINDy STLSQ threshold (0.01 keeps the true 0.02 u_xx term)")
    ap.add_argument("--rollout_steps", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()
    main(args.levels, args.threshold, args.rollout_steps)
