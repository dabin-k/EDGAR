"""Unit tests for the mPFC windowed next-bin scoring contract."""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
LOAD_DATA_PATH = REPO / "projects" / "mPFC_spikes" / "data_loader" / "load_data.py"


def _load_mpfc_module():
    spec = importlib.util.spec_from_file_location("mpfc_load_data", LOAD_DATA_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpfc_load_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mpfc():
    return _load_mpfc_module()


def _synthetic_windowed(n=4, A=12, W=8, seed=0):
    rng = np.random.default_rng(seed)
    history = (rng.random((n, A, W)) < 0.08).astype(np.float32)
    target_y = (rng.random((n, A)) < 0.08).astype(np.float32)
    return {"history": jnp.asarray(history), "target_y": jnp.asarray(target_y)}


def test_pick_anchors_enforces_min_stride(mpfc):
    anchors = mpfc._pick_anchors(10, 510, n=1000, min_stride=50)
    assert anchors.min() >= 10
    assert anchors.max() < 510
    assert np.all(np.diff(anchors) >= 50)
    assert len(anchors) <= (510 - 10) // 50


def test_train_test_guard_band_separation(mpfc):
    W = 10
    half = 100
    T = 200
    train_anchors = mpfc._pick_anchors(W, half, n=50, min_stride=W)
    test_anchors = mpfc._pick_anchors(half + W, T, n=50, min_stride=W)
    assert train_anchors.max() < half
    assert test_anchors.min() >= half + W
    assert int(train_anchors.max()) - W < half
    assert int(test_anchors.min()) - W >= half


def test_apply_model_nested_vmap_shapes(mpfc):
    data = _synthetic_windowed(n=3, A=7, W=6)

    def model_fn(window, params):
        return jnp.dot(window, params["w"]) + params["b"]

    params = {
        "w": jnp.full((3, 6), 0.1),
        "b": jnp.full((3,), 0.01),
    }
    mu = mpfc.apply_model(model_fn, data, params)
    assert mu.shape == (3, 7)
    assert jnp.all(mu > 0)


def test_loss_fn_denominator_floor(mpfc):
    data = _synthetic_windowed(n=2, A=4, W=3)
    # neuron 0: all-zero targets -> floor denominator at 1 spike
    data = {
        "history": data["history"],
        "target_y": jnp.zeros((2, 4), dtype=jnp.float32),
    }
    mu = jnp.full((2, 4), 0.5)
    losses = np.asarray(mpfc.loss_fn(mu, data))
    assert np.all(np.isfinite(losses))
    assert losses[0] == pytest.approx(2.0)  # sum(mu)=2, denom floored to 1


def test_constant_rate_baseline_formula(mpfc):
    y = jnp.array([[0.0, 1.0, 0.0, 2.0]])
    rate = max(float(y.mean()), 1e-8)
    baseline = 1.0 - np.log(rate)
    assert baseline == pytest.approx(1.0 - np.log(0.75))


def test_resolve_scoring_data_roundtrip(monkeypatch):
    from edgar.scoring import scoring as sc
    from edgar.scoring.scoring import _ScoringDataStore, _resolve_scoring_data

    monkeypatch.setattr(sc, "_SCORING_DATA_NPZ_THRESHOLD_BYTES", 1)

    train = {"history": jnp.ones((2, 3, 4)), "target_y": jnp.zeros((2, 3))}
    test = {"history": jnp.ones((2, 3, 4)) * 2, "target_y": jnp.ones((2, 3))}
    store = _ScoringDataStore((train, test))
    ref = store.ref()
    try:
        assert isinstance(ref, str)
        rt_train, rt_test = _resolve_scoring_data(ref)
        assert rt_train["history"].shape == (2, 3, 4)
        assert rt_test["target_y"].shape == (2, 3)
    finally:
        store.cleanup()
