"""Structural anti-leakage self-test for the state-space DSL.

Under the ``model(state, y_prev, params) → (new_state, mean)`` contract,
leakage is impossible *by construction*: ``y[t]`` is never in scope inside
``model_fn``. The scan feeds ``y_prev = y[t-1]`` at each iteration and asks
for a prediction of ``y[t]``.

This module asserts the invariants that make that guarantee real, exercises
the four seed programs end-to-end on real synthetic data, and doubles as the
no-LLM local gate for the pipeline:

  1. **ISOLATION** — perturbing ``y[t:]`` leaves every ``pred[<t]`` bit-exact.
  2. **SHAPE** — ``model`` only ever receives a scalar ``y_prev`` and returns
     ``(state, scalar_mean)``; enforced via ``validate_step()``.
  3. **NLL sanity** — seed programs load → apply_model → optimize → produce
     finite, O(1) losses on real data.
  4. **Baseline gap** — the Kalman-lite seed (model4) should beat persistence
     (model1) on validation NLL after gradient descent, verifying that the
     evolutionary loop has real signal to work with.

Run:  ``python projects/oscillator_ss/leakage_check.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml   # noqa: E402

from projects.oscillator_ss.data_loader.load_data import (   # noqa: E402
    load_data, apply_model, loss_fn, validate_step, WARMUP_STEPS,
)
from edgar.scoring.scoring import _optimize, _eval_loss   # noqa: E402


# ── seed-program loading (mirrors what the pipeline does) ──


def _numpy_to_jax_source(src: str) -> str:
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def _load_seed(seed_num: int) -> tuple[callable, callable, dict, str]:
    seed_dir = Path(__file__).parent / "seed_programs"
    model_src = (seed_dir / f"model{seed_num}.py").read_text()
    param_est_src = (seed_dir / f"param_est{seed_num}.py").read_text()

    ns_m = {}
    exec(_numpy_to_jax_source(model_src), ns_m)
    model_fn = ns_m["model"]
    default_params = dict(model_fn.DEFAULT_PARAMS)

    ns_p = {}
    exec(param_est_src, ns_p)
    param_est_fn = ns_p["parameter_estimator"]

    return model_fn, param_est_fn, default_params, model_src


# ── invariants ──


def check_isolation(model_fn, data_train: dict, params: dict) -> tuple[bool, dict]:
    """Perturbing y[t:] must not change any pred[<t] (bit-exact)."""
    y = np.asarray(data_train["y"])
    t_perturb = y.shape[1] // 2
    y_pert = y.copy()
    y_pert[:, t_perturb:] += 100.0

    out_orig = np.asarray(apply_model(model_fn, {"y": jnp.asarray(y)}, params))
    out_pert = np.asarray(apply_model(model_fn, {"y": jnp.asarray(y_pert)}, params))

    means_orig = out_orig[..., 0]      # (n_samples, T-1)
    means_pert = out_pert[..., 0]

    # Perturbed y[t_perturb:] means scan iters where y_prev = y[i] for i >= t_perturb
    # are affected. Iters where i < t_perturb must be bit-exact.
    prefix_bitexact = np.array_equal(means_orig[:, :t_perturb], means_pert[:, :t_perturb])
    suffix_differs  = not np.array_equal(means_orig[:, t_perturb:], means_pert[:, t_perturb:])

    return (prefix_bitexact and suffix_differs), {
        "prefix_bitexact (must be True)": prefix_bitexact,
        "suffix_differs (must be True)":  suffix_differs,
        "t_perturb": t_perturb,
    }


def check_shape(model_fn, default_params: dict, source: str) -> tuple[bool, dict]:
    """validate_step catches shape / structure / finiteness bugs."""
    try:
        validate_step(model_fn, default_params, program_code=source)
        return True, {"validate_step": "OK"}
    except AssertionError as e:
        return False, {"validate_step": f"FAILED: {e}"}


def check_scoring(model_fn, param_est_fn, default_params, data_train, data_test) -> tuple[bool, dict]:
    """Seed loads → param_est → apply_model → optimize → finite O(1) loss."""
    n = data_train["y"].shape[0]

    # per-sample param estimation
    y_np = np.asarray(data_train["y"])
    per_sample = [param_est_fn({"y": y_np[i]}) for i in range(n)]
    params_init = {
        k: jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]
    }

    L0 = _eval_loss(model_fn, loss_fn, params_init, data_test, apply_model)
    params = _optimize(
        model_fn, loss_fn, params_init, data_train,
        gd_config={"max_iter": 100, "learning_rate": 0.005, "gradient_clip_norm": 5.0},
        apply_model_fn=apply_model,
    )
    Lf = _eval_loss(model_fn, loss_fn, params, data_test, apply_model)

    finite = bool(np.isfinite(L0) and np.isfinite(Lf))
    improved = bool(Lf <= L0 + 1e-3)
    reasonable = bool(abs(Lf) < 100.0)

    return (finite and improved and reasonable), {
        "L_init": float(L0),
        "L_final": float(Lf),
        "improved (final <= init)": improved,
        "reasonable (|L| < 100)": reasonable,
    }


# ── driver ──


def _self_test() -> int:
    root = Path(__file__).parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))

    (Xd_tr, Xd_te), _, _ = load_data(**pp)
    print(f"[leakage_check] loaded discover train/test: {Xd_tr['y'].shape} / {Xd_te['y'].shape}")

    ok_all = True

    print("\n== structural invariants ==")
    for i in [1, 2, 3, 4]:
        model_fn, pe_fn, default_params, source = _load_seed(i)
        n = Xd_tr["y"].shape[0]
        params_stacked = {
            k: jnp.full((n,), float(v)) for k, v in default_params.items()
        }

        shp_ok, shp = check_shape(model_fn, default_params, source)
        iso_ok, iso = check_isolation(model_fn, Xd_tr, params_stacked)
        print(f"  seed {i} ({Path('model' + str(i) + '.py').stem}): "
              f"shape={'OK' if shp_ok else 'BAD'}  isolation={'OK' if iso_ok else 'LEAK'} {iso}")
        ok_all &= (shp_ok and iso_ok)

    print("\n== scoring (load → optimize → loss) ==")
    seed_losses = {}
    for i in [1, 2, 3, 4]:
        model_fn, pe_fn, default_params, source = _load_seed(i)
        ok, info = check_scoring(model_fn, pe_fn, default_params, Xd_tr, Xd_te)
        seed_losses[i] = info["L_final"]
        print(
            f"  seed {i}: L_init={info['L_init']:+.4f}  L_final={info['L_final']:+.4f}  "
            f"{'OK' if ok else 'FAIL'}"
        )
        ok_all &= ok

    print("\n== baseline gap (Kalman-lite vs persistence) ==")
    gap = seed_losses[1] - seed_losses[4]        # positive if seed 4 is better (lower NLL)
    gap_ok = gap >= 0.1                          # relaxed from 0.3 for the tiny 100-iter opt used here
    print(
        f"  seed 4 (Kalman-lite) NLL improvement over seed 1 (persistence): "
        f"{gap:+.4f} nat  {'OK' if gap_ok else 'WEAK'}  (target >= 0.1 nat with 100 GD steps)"
    )
    ok_all &= gap_ok

    print("\nSelf-test:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
