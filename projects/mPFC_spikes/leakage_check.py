"""Structural anti-leakage invariants + local on-project harness (Tier B).

Under the windowed next-bin contract, leakage is impossible *by construction*:
``apply_model`` hands each program's ``model(window, params)`` a single
strictly-past window ``(W,)`` taken from ``data["history"]`` and NEVER the
held-out ``data["target_y"]``. There is no time index to mis-shift and no
current bin in scope, so the old perturb-the-timeseries causality gate is
obsolete.

This module instead asserts the invariants that make that guarantee real, and
doubles as the no-LLM local gate for the mPFC pipeline:

  1. ISOLATION  — perturbing ``target_y`` never changes any model output
     (the target is not in the model's scope), while perturbing ``history``
     does. This is the structural anti-leakage guarantee.
  2. SHAPE      — the model is only ever called on a 1-D ``(W,)`` window.
  3. GUARD BAND — no train window shares a bin with any test target (the
     contiguous split + W guard band), and window orientation is correct
     (``history[..., -1]`` is the bin immediately before the target).
  4. SCORING    — the real seeds load -> apply_model -> optimize -> loss on real
     mPFC data, giving finite, O(1) losses that gradient descent lowers.

Run:  ``python projects/mPFC_spikes/leakage_check.py``  (uses real data if the
config's data_path resolves, else a synthetic sparse series).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


# ── seed models as JAX (what the translator produces from seed_programs/*.py) ──


def _seed_models():
    def hawkes(window, params):
        W = window.shape[0]
        gamma = jnp.clip(params["gamma"], 0.0, 3.0)
        u0 = jnp.clip(params["u0"], 1.0, 100.0)
        lags = jnp.arange(W, 0, -1)
        kernel = 1.0 / (lags + u0) ** (1.0 + gamma)
        return jnp.logaddexp(0.0, params["mu0"] + params["K"] * jnp.dot(window, kernel)) + 1e-6

    def leaky(window, params):
        W = window.shape[0]
        tau = jnp.clip(params["tau"], 1.0, 5000.0)
        decay = jnp.exp(-1.0 / tau)
        i = jnp.arange(W)
        weights = decay ** (W - 1 - i)
        return jnp.logaddexp(0.0, params["base"] + params["w"] * jnp.dot(window, weights)) + 1e-6

    return [
        ("Hawkes", hawkes, {"mu0": -5.5, "K": 1.0, "gamma": 0.5, "u0": 1.0}),
        ("Leaky", leaky, {"base": -5.5, "w": 1.0, "tau": 20.0}),
    ]


def _stack(params, n):
    return {k: jnp.full((n,), float(v)) for k, v in params.items()}


# ── invariants ──


def check_isolation(model_fn, params_stacked, data, atol=1e-6) -> tuple[bool, dict]:
    """Perturbing target_y must NOT change model output; perturbing history MUST.
    This is the structural anti-leakage guarantee for the windowed contract."""
    from projects.mPFC_spikes.data_loader.load_data import apply_model

    base = np.asarray(apply_model(model_fn, data, params_stacked))

    d_y = {**data, "target_y": data["target_y"] + 1.0}
    out_y = np.asarray(apply_model(model_fn, d_y, params_stacked))
    target_leaks = bool(np.max(np.abs(out_y - base)) > atol)

    d_h = {**data, "history": data["history"] + 1.0}
    out_h = np.asarray(apply_model(model_fn, d_h, params_stacked))
    history_used = bool(np.max(np.abs(out_h - base)) > atol)

    ok = (not target_leaks) and history_used
    return ok, {
        "target_leaks (must be False)": target_leaks,
        "history_used (must be True)": history_used,
    }


def check_shape(model_fn, params_stacked, data) -> tuple[bool, dict]:
    """The model must only ever be handed a 1-D (W,) window."""
    seen = {}

    def probe(window, p):
        seen["ndim"] = window.ndim
        seen["shape"] = tuple(window.shape)
        return model_fn(window, p)

    from projects.mPFC_spikes.data_loader.load_data import apply_model

    _ = np.asarray(apply_model(probe, data, params_stacked))
    W = data["history"].shape[-1]
    ok = seen.get("ndim") == 1 and seen.get("shape") == (W,)
    return ok, seen


def check_guard_band(data_path: str, pp: dict) -> tuple[bool, dict]:
    """No train window overlaps any test target; orientation is correct.

    Re-derives the anchor grid load_data uses and checks the time separation
    directly on bin indices (load_data also asserts orientation internally)."""
    from projects.mPFC_spikes.data_loader import load_data as LD

    W = int(pp.get("window_W", 300))
    A = int(pp.get("anchors_per_neuron", 4000))
    sall, ws = LD._resolve_paths(data_path)
    counts, _ = LD._bin_counts(LD._load_spike_trains(sall), LD._sws_segments(ws, pp.get("sws_mode", "longest")), pp.get("bin_ms", 5.0) / 1000.0)
    T = counts.shape[1]
    half = T // 2
    train_anchors = LD._pick_anchors(W, half, A, min_stride=W)
    test_anchors = LD._pick_anchors(half + W, T, A, min_stride=W)
    # train windows span [t-W, t-1] for t in train_anchors -> max index < half.
    train_window_max = int(train_anchors.max())  # last bin used is t-1 < half
    # test targets and their windows start at half+W-W = half.
    test_window_min = int(test_anchors.min() - W)
    ok = train_window_max <= half - 1 and test_window_min >= half
    return ok, {
        "train_window_max_bin": train_window_max,
        "half": int(half),
        "test_window_min_bin": test_window_min,
        "separated": ok,
    }


# ── driver ──


def _load_windowed_data(data_path: str, pp: dict):
    from projects.mPFC_spikes.data_loader.load_data import load_data

    (Xd_tr, Xd_te), _, _ = load_data(data_path, **pp)
    return Xd_tr, Xd_te


def _synthetic_data(n=6, A=1000, W=300, seed=0):
    rng = np.random.default_rng(seed)
    history = (rng.random((n, A, W)) < 0.03).astype(np.float32)
    target_y = (rng.random((n, A)) < 0.03).astype(np.float32)
    return {"history": jnp.asarray(history), "target_y": jnp.asarray(target_y)}


def _self_test() -> int:
    import yaml

    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))
    data_path = cfg["io"]["data_path"]

    # Prefer real data; fall back to a synthetic sparse series off-cluster.
    used_real = True
    try:
        Xd_tr, Xd_te = _load_windowed_data(data_path, pp)
    except Exception as e:
        print(f"[leakage_check] real data unavailable ({e}); using synthetic series")
        used_real = False
        Xd_tr = _synthetic_data(W=int(pp.get("window_W", 300)))
        Xd_te = _synthetic_data(W=int(pp.get("window_W", 300)))

    n = Xd_tr["history"].shape[0]
    ok_all = True

    print("== structural invariants ==")
    for name, model_fn, params in _seed_models():
        ps = _stack(params, n)
        iso_ok, iso = check_isolation(model_fn, ps, Xd_te)
        shp_ok, shp = check_shape(model_fn, ps, Xd_te)
        print(f"  {name}: isolation={'OK' if iso_ok else 'LEAK'} {iso}; shape={'OK' if shp_ok else 'BAD'} {shp}")
        ok_all &= iso_ok and shp_ok

    if used_real:
        gb_ok, gb = check_guard_band(data_path, pp)
        print(f"== guard band ==\n  {'OK' if gb_ok else 'OVERLAP'} {gb}")
        ok_all &= gb_ok

    print("== scoring (load -> apply_model -> optimize -> loss) ==")
    from edgar.scoring.scoring import _optimize, _eval_loss
    from projects.mPFC_spikes.data_loader.load_data import apply_model, loss_fn

    gd = cfg.get("scoring", {}).get("gradient_descent", {"learning_rate": 0.02, "max_iter": 100})
    for name, model_fn, params in _seed_models():
        p0 = _stack({**params, **({"K": 0.0} if "K" in params else {}), **({"w": 0.0} if "w" in params else {})}, n)
        L0 = _eval_loss(model_fn, loss_fn, p0, Xd_te, apply_model)
        pf = _optimize(model_fn, loss_fn, p0, Xd_tr, gd, apply_model)
        Lf = _eval_loss(model_fn, loss_fn, pf, Xd_te, apply_model)
        finite = np.isfinite(L0) and np.isfinite(Lf)
        improved = Lf <= L0 + 1e-6
        print(f"  {name}: loss {L0:.4f} -> {Lf:.4f}  finite={finite} improved={improved}")
        ok_all &= bool(finite and improved)

    print("\nSelf-test:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
