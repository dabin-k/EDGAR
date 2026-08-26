#!/usr/bin/env python
"""Smoke-check that the EDGAR fit pipeline is correctly wired for objectives A, B, E, F.

Rather than launching a full EDGAR evolution, this fits ONE sample (the M150605 session) with
model2 (WCS), initialised from param_est2, under each of the four objectives in turn, using the
optimisation settings from ``config.yaml``. For every objective it reports the param-estimator
init loss, the post-GD train loss, and the held-out (k-fold) test loss, and flags whether the
optimiser actually reduced the training loss and stayed finite. It exercises exactly the scoring
path EDGAR uses (``edgar.scoring.scoring._get_params`` / ``_optimize`` / ``_eval_loss`` +
the project's ``apply_model`` / ``loss_fn``), so a green row means that objective's wiring —
data build, param_est init, model transition, apply_model mode, and loss — runs end to end and
is trainable on real data.

Objective routing (see data_loader/load_data.py):
  * A  one-step teacher-forced MSE     — model2,        apply_model teacher-forced path
  * B  autonomous rollout MSE          — model2,        apply_model rollout path
  * E  EKF latent state-space NLL      — model2_kalman, apply_model Kalman path (needs kf_* params)
  * F  rollout Gaussian NLL (det. SSM) — model2,        apply_model rollout path

"One sample" is built directly from ``neural_data.build_cv_samples`` (k_fold, held_out_fold=0),
the same building block ``load_data._load_real`` uses per mouse×fold; the mouse-split ``load_data``
real path proper needs >=2 sessions and does not apply to a single-sample check.

Usage:
    python check_objective_setup.py                     # A,B,E,F with config's GD settings
    python check_objective_setup.py --objectives A,E     # subset
    python check_objective_setup.py --max-iter 200       # faster smoke (overrides config max_iter)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]                             # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))

# The one sample: the M150605 session the config's data_path glob resolves to.
DATA_FILE = Path("/home/dabin/code/ichun_opto/results/population_rates_M150605_ICTP1_s1_trimmed.npz")

# model module per objective: E uses the Kalman variant (kf_* folded into DEFAULT_PARAMS); the
# rest use the plain WCS model. Both share identical dynamics.
MODEL_BY_OBJECTIVE = {"A": "model2", "B": "model2", "E": "model2_kalman", "F": "model2"}
OBJECTIVES = ["A", "B", "E", "F"]

# Fitted params land here (kept next to this script so downstream free-run / analysis scripts can
# load them). One npz per objective + a combined summary; see _save_params for the schema.
PARAMS_DIR = WC / "scripts" / "fitted_params"

import jax  # noqa: E402  (import after sys.path is set up)
import jax.numpy as jnp  # noqa: E402


def _load_config() -> dict:
    with open(WC / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _build_data(split, time_axis: np.ndarray) -> dict:
    """One CVSplit (target_y (C,T,2), stim (C,T,2)) -> the n=1 apply_model data dict.

    Mirrors ``load_data._load_real._build`` for a single sample: adds the leading n=1 axis every
    array carries and assembles the autonomous-rollout targets from the config anchors/horizon.
    """
    from load_data import _rollout_anchors  # reads EDGAR_WC_* env set below

    T = time_axis.shape[0]
    anchor_starts, K = _rollout_anchors(T)
    target_y = jnp.asarray(split.target_y)[None]              # (1, C, T, 2)
    stim = np.asarray(split.stim)                            # (C, T, 2)
    sE = jnp.asarray(stim[..., 0])[None]                    # (1, C, T)
    sI = jnp.asarray(stim[..., 1])[None]
    target_y_future = jnp.stack(
        [target_y[:, :, a + 1: a + 1 + K, :] for a in anchor_starts], axis=2
    )                                                        # (1, C, A, K, 2)
    time = jnp.broadcast_to(jnp.asarray(time_axis)[None, :], (1, T))
    return {"target_y": target_y, "stim_E": sE, "stim_I": sI,
            "target_y_future": target_y_future, "time": time}


def _scalar_params(params: dict, keys: list[str]) -> dict:
    """Pull the single-sample (n=1) params pytree down to a plain ``{name: float}`` dict."""
    return {k: float(np.asarray(params[k]).reshape(-1)[0]) for k in keys}


def _save_params(result: dict, meta: dict, suffix: str = "") -> Path:
    """Write one objective's fitted params + init + reconstruction metadata to an npz.

    Schema (readable back with ``np.load(path, allow_pickle=True)``):
        param_keys       (P,) str   — the model's DEFAULT_PARAMS keys, in order
        param_values     (P,) f64   — the GD-fitted params (use these for the free run)
        init_values      (P,) f64   — the param_est2 init (before GD), same key order
        train_loss / test_loss / init_train_loss  f64
        objective, model, data_file, cv_type       str
        held_out_fold, warmup_bins, rollout_k, anchor_stride  int
        dt_s, chop (2,)                             f64
        gd_learning_rate, gd_max_iter, gd_gradient_clip_norm
    To rebuild the model for a free run: import the named ``model`` module, zip
    ``param_keys``/``param_values`` into a dict, and drive ``model.model_jax`` from it.
    """
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    keys = result["param_keys"]
    out = PARAMS_DIR / f"check_fit_{result['objective']}{suffix}.npz"
    np.savez(
        out,
        param_keys=np.array(keys),
        param_values=np.array([result["fit"][k] for k in keys], dtype=np.float64),
        init_values=np.array([result["init"][k] for k in keys], dtype=np.float64),
        train_loss=np.float64(result["train"]),
        test_loss=np.float64(result["test"]),
        init_train_loss=np.float64(result["init_train"]),
        objective=np.str_(result["objective"]),
        model=np.str_(result["model"]),
        data_file=np.str_(meta["data_file"]),
        cv_type=np.str_(meta["cv_type"]),
        held_out_fold=np.int64(meta["held_out_fold"]),
        warmup_bins=np.int64(meta["warmup_bins"]),
        rollout_k=np.int64(meta["rollout_k"]),
        anchor_stride=np.int64(meta["anchor_stride"]),
        dt_s=np.float64(meta["dt_s"]),
        chop=np.array(meta["chop"], dtype=np.float64),
        gd_learning_rate=np.float64(meta["gd"]["learning_rate"]),
        gd_max_iter=np.int64(meta["gd"]["max_iter"]),
        gd_gradient_clip_norm=np.float64(meta["gd"]["gradient_clip_norm"]),
    )
    return out


def _save_init(cv, time_axis: np.ndarray, meta: dict, suffix: str = "") -> Path:
    """Compute and save the shared param_est2 init (the pre-GD starting point).

    param_est2 is a data-only estimator (no gradient descent), so its output is identical for
    every objective — the same dynamics params, plus the objective-E ``kf_*`` noise/init. Saved
    ONCE here (all keys) as ``check_init_params.npz`` so downstream analysis can compare any
    fitted objective against the common start. Schema mirrors ``_save_params`` minus the
    train/test-loss and gd fields (there is no fit here):
        param_keys (P,) str, param_values (P,) f64, plus the reconstruction metadata.
    """
    import param_est2

    X_train = _build_data(cv.train, time_axis)
    # Single-sample dict (drop the leading n=1 axis load_data adds) for the estimator.
    sample = {k: np.asarray(X_train[k])[0] for k in ("target_y", "stim_E", "stim_I")}
    init = param_est2.parameter_estimator(sample)
    keys = list(init.keys())

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = PARAMS_DIR / f"check_init_params{suffix}.npz"
    np.savez(
        out,
        param_keys=np.array(keys),
        param_values=np.array([float(init[k]) for k in keys], dtype=np.float64),
        estimator=np.str_("param_est2"),
        data_file=np.str_(meta["data_file"]),
        cv_type=np.str_(meta["cv_type"]),
        held_out_fold=np.int64(meta["held_out_fold"]),
        chop=np.array(meta["chop"], dtype=np.float64),
        dt_s=np.float64(meta["dt_s"]),
    )
    return out


def _fit_objective(obj: str, cv, time_axis: np.ndarray, gd: dict) -> dict:
    """Init from param_est2, GD-fit ONE sample under objective ``obj``, return the loss summary."""
    import param_est2
    from load_data import apply_model, loss_fn
    from edgar.scoring.scoring import _get_params, _optimize, _eval_loss

    model_mod = __import__(MODEL_BY_OBJECTIVE[obj])
    param_keys = list(model_mod.model.DEFAULT_PARAMS.keys())

    os.environ["EDGAR_WC_OBJECTIVE"] = obj   # apply_model / loss_fn read this lazily per call

    X_train = _build_data(cv.train, time_axis)
    X_test = _build_data(cv.test, time_axis)

    # param_est init (data-driven; identical across objectives — it depends only on X_train).
    params_init = _get_params(param_est2.parameter_estimator,
                              model_mod.model.DEFAULT_PARAMS, X_train)
    init_train = _eval_loss(model_mod.model_jax, loss_fn, params_init, X_train, apply_model)

    fit = _optimize(model_mod.model_jax, loss_fn, params_init, X_train, gd, apply_model)
    train_loss = _eval_loss(model_mod.model_jax, loss_fn, fit, X_train, apply_model)
    test_loss = _eval_loss(model_mod.model_jax, loss_fn, fit, X_test, apply_model)

    finite = all(np.isfinite(v) for v in (init_train, train_loss, test_loss))
    improved = np.isfinite(train_loss) and train_loss <= init_train + 1e-9
    return {
        "objective": obj, "model": MODEL_BY_OBJECTIVE[obj], "n_params": len(param_keys),
        "param_keys": param_keys,
        "init": _scalar_params(params_init, param_keys),
        "fit": _scalar_params(fit, param_keys),
        "init_train": init_train, "train": train_loss, "test": test_loss,
        "finite": finite, "improved": improved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default=",".join(OBJECTIVES),
                    help="comma-separated subset of A,B,E,F (default: all)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="override config's scoring.gradient_descent.max_iter (for a fast smoke run)")
    ap.add_argument("--data-file", default=str(DATA_FILE), help="the one-sample session npz")
    ap.add_argument("--tag", default="",
                    help="suffix for the saved npz filenames (e.g. --tag rollout40_stride1 writes "
                         "check_fit_{obj}_rollout40_stride1.npz), so a re-fit under different config "
                         "does not overwrite the existing check_fit_{obj}.npz copies.")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    run_objs = [o.strip().upper() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in run_objs if o not in OBJECTIVES]
    if bad:
        raise SystemExit(f"unknown objective(s) {bad}; valid: {OBJECTIVES}")
    data_file = Path(args.data_file)
    if not data_file.exists():
        raise SystemExit(f"data file not found: {data_file}")

    cfg = _load_config()
    pp = cfg["project_params"]
    gd_cfg = cfg["scoring"]["gradient_descent"]
    gd = {
        "learning_rate": float(gd_cfg["learning_rate"]),
        "max_iter": int(args.max_iter if args.max_iter is not None else gd_cfg["max_iter"]),
        "gradient_clip_norm": float(gd_cfg["gradient_clip_norm"]),
    }
    chop = (float(pp["chop_pre_ms"]) / 1000.0, float(pp["chop_post_ms"]) / 1000.0)

    # Build the one sample (M150605), config's k_fold / chop.
    from neural_data import build_cv_samples
    cv = build_cv_samples(str(data_file), cv_type="k_fold", held_out_fold=0, chop=chop)
    time_axis = np.asarray(cv.time)
    T = time_axis.shape[0]
    dt_s = float(time_axis[1] - time_axis[0]) if T > 1 else float(pp["dt_seconds"])

    # Publish config's rollout / warmup / dt settings to env (load_data normally does this; our
    # single-sample build bypasses it). apply_model / loss_fn / _rollout_anchors read these.
    warmup_bins = int(round((float(pp["warmup_steps_ms"]) / 1000.0) / dt_s))
    os.environ["EDGAR_WC_ROLLOUT_K"] = str(int(pp["rollout_k"]))
    os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(int(pp["anchor_stride"]))
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(warmup_bins)
    os.environ["EDGAR_WC_DT"] = str(dt_s)

    print(f"data:  {data_file.name}  (1 sample: C={cv.train.n} conditions, T={T}, dt={dt_s*1000:.2f} ms)")
    print(f"chop:  {chop} s   warmup_bins={warmup_bins}   rollout_k={pp['rollout_k']}  "
          f"anchor_stride={pp['anchor_stride']}")
    print(f"gd:    lr={gd['learning_rate']}  max_iter={gd['max_iter']}  "
          f"clip={gd['gradient_clip_norm']}")
    print(f"fitting objectives {run_objs} (init = param_est2)\n")

    save_meta = {
        "data_file": str(data_file), "cv_type": "k_fold", "held_out_fold": 0,
        "warmup_bins": warmup_bins, "rollout_k": int(pp["rollout_k"]),
        "anchor_stride": int(pp["anchor_stride"]), "dt_s": dt_s, "chop": chop, "gd": gd,
    }

    # Shared param_est2 init (no GD; data-only, so identical across objectives — carries all 24
    # keys incl. the objective-E kf_* noise/init). Saved once as the common starting point.
    init_path = _save_init(cv, time_axis, save_meta, suffix)
    print(f"[saved] shared init params -> {init_path}\n")

    results = []
    for obj in run_objs:
        r = _fit_objective(obj, cv, time_axis, gd)
        out = _save_params(r, save_meta, suffix)
        results.append(r)
        status = "OK " if (r["finite"] and r["improved"]) else "!! "
        print(f"  [{status}] obj {obj} ({r['model']}, n_params={r['n_params']:2d}):  "
              f"init_train={r['init_train']:.5g}  train={r['train']:.5g}  test={r['test']:.5g}  "
              f"-> {out.name}")

    print("\n" + "=" * 78)
    print(f"{'obj':<4}{'model':<15}{'n_par':<7}{'init_train':<14}{'train':<14}{'test':<14}{'status'}")
    print("-" * 78)
    for r in results:
        status = "OK" if (r["finite"] and r["improved"]) else "FAIL"
        print(f"{r['objective']:<4}{r['model']:<15}{r['n_params']:<7}"
              f"{r['init_train']:<14.5g}{r['train']:<14.5g}{r['test']:<14.5g}{status}")
    print("=" * 78)
    n_ok = sum(r["finite"] and r["improved"] for r in results)
    print(f"\n{n_ok}/{len(results)} objectives fit cleanly (finite losses, train loss reduced).")
    print("NOTE: loss values are NOT comparable across objectives — each minimises a different "
          "quantity (MSE vs NLL). Compare init->train within a row, not across rows.")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
