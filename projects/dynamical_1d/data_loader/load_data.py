"""Windowed next-value regression on synthetic 1D dynamical trajectories.

Generates driven van der Pol oscillator traces (fixed ODE params, random ICs),
splits into EDGAR discover / validate / eval, and exposes strictly-past windows
for autoregressive learning.

    sample axis 0 = trajectory (one param set per trajectory)
    data["history"][n, a] = x[n, t-W : t]   # (W,) strictly past
    data["target_y"][n, a] = x[n, t+H-1]    # scalar target (H = predict_horizon)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

_OVERSAMPLE = 4


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def _downsample(x: np.ndarray, n_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[0]
    if n_in == n_out:
        return x
    if n_in > n_out and n_in % n_out == 0:
        factor = n_in // n_out
        return x.reshape(n_out, factor).mean(axis=1)
    xp = np.linspace(0, 1, n_in)
    xq = np.linspace(0, 1, n_out)
    return np.interp(xq, xp, x)


def _simulate_ode(
    rhs,
    y0,
    *,
    t_signal: float,
    n_out: int,
    obs: int = 0,
    t_burn_frac: float = 0.4,
    max_step: float | None = None,
) -> np.ndarray:
    n_sim = n_out * _OVERSAMPLE
    n_burn = int(n_sim * t_burn_frac)
    t_burn = t_signal * t_burn_frac
    t_total = t_burn + t_signal
    t_eval = np.linspace(0, t_total, n_burn + n_sim)
    sol = solve_ivp(
        rhs,
        (0, t_total),
        np.asarray(y0, dtype=np.float64),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=max_step if max_step is not None else np.inf,
    )
    y = np.asarray(sol.y[obs], dtype=np.float64)
    need = n_burn + n_sim
    if y.shape[0] < need:
        y = np.pad(y, (0, need - y.shape[0]), mode="edge")
    y = y[n_burn:need]
    return _downsample(y, n_out)


def _driven_vdp(
    rng: np.random.Generator, n: int, mu: float, amp: float, w_drive: float
) -> np.ndarray:
    def rhs(t, s):
        x, y = s
        return [y, mu * (1 - x * x) * y - x + amp * np.cos(w_drive * t)]

    y0 = rng.uniform(-2, 2, size=2)
    return _simulate_ode(rhs, y0, t_signal=110, n_out=n, obs=0, max_step=0.1)


def _normalize_trainstats(x: np.ndarray, half: int) -> np.ndarray:
    """Min-max normalize using train region only; apply to full trace."""
    train = x[:half]
    lo, hi = float(train.min()), float(train.max())
    return ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _generate_vdp_trajectory(
    rng: np.random.Generator,
    n: int,
    mu: float,
    amp: float,
    w_drive: float,
    noise_level: float,
    half: int,
) -> np.ndarray:
    clean = _driven_vdp(rng, n, mu, amp, w_drive).astype(np.float64)
    signal_std = float(clean.std()) + 1e-8
    noisy = clean + noise_level * signal_std * rng.standard_normal(n)
    return _normalize_trainstats(noisy, half)


def _simulate_system(
    rng: np.random.Generator,
    system: str,
    n: int,
    half: int,
    *,
    vdp_mu: float,
    vdp_amp: float,
    vdp_w_drive: float,
    noise_level: float,
) -> np.ndarray:
    if system == "driven_vdp":
        return _generate_vdp_trajectory(
            rng, n, vdp_mu, vdp_amp, vdp_w_drive, noise_level, half
        )
    raise NotImplementedError(f"system {system!r} not ported yet")


def _pick_anchors(lo: int, hi: int, n: int) -> np.ndarray:
    span = hi - lo
    if span <= 0:
        raise ValueError(f"empty anchor region [{lo}, {hi}); widen T or shrink W/H")
    n = int(min(n, span))
    return lo + np.unique(np.linspace(0, span - 1, n).astype(np.int64))


def _windows(x_sub: np.ndarray, anchors: np.ndarray, W: int, H: int) -> dict:
    # Anchor t indexes prediction target x[t+H-1]; history is x[t-W:t].
    idx = anchors[:, None] + np.arange(-W, 0)[None, :]
    history = x_sub[:, idx]
    target_y = x_sub[:, anchors + H - 1]
    assert np.array_equal(history[:, :, -1], x_sub[:, anchors - 1])
    return {
        "history": history.astype(np.float32),
        "target_y": target_y.astype(np.float32),
    }


def _persistence_mse_per_sample(windows: dict) -> np.ndarray:
    pred = windows["history"][:, :, -1]
    y = windows["target_y"]
    return np.mean((pred - y) ** 2, axis=-1).astype(np.float32)


def _rollout_extras(x_sub: np.ndarray, W: int, half: int, rollout_H: int) -> dict:
    t0 = half + W
    if t0 + rollout_H > x_sub.shape[1]:
        rollout_H = x_sub.shape[1] - t0
    return {
        "_rollout_seed": x_sub[:, t0 - W : t0].astype(np.float32),
        "_rollout_true": x_sub[:, t0 : t0 + rollout_H].astype(np.float32),
    }


def load_data(
    data_path: str = "",
    system: str = "driven_vdp",
    vdp_mu: float = 2.0,
    vdp_amp: float = 1.0,
    vdp_w_drive: float = 1.0,
    n_trajectories: int = 64,
    num_timepoints: int = 1024,
    noise_level: float = 0.05,
    window_W: int = 96,
    anchors_per_sample: int = 512,
    predict_horizon: int = 1,
    rollout_H: int = 200,
    n_eval: int = 4,
    eval_anchors: int = 200,
    seed: int = 42,
):
    del data_path  # synthetic; unused

    W = int(window_W)
    H = int(predict_horizon)
    T = int(num_timepoints)
    half = T // 2

    if half <= W:
        raise ValueError(f"half ({half}) <= window_W ({W}); shrink window_W or lengthen T")
    if H < 1:
        raise ValueError(f"predict_horizon must be >= 1, got {H}")
    if half - H < W:
        raise ValueError(
            f"train region too short for W={W}, H={H}; reduce window_W or predict_horizon"
        )

    rng = np.random.default_rng(seed)
    x = np.empty((n_trajectories, T), dtype=np.float32)
    for i in range(n_trajectories):
        x[i] = _simulate_system(
            rng,
            system,
            T,
            half,
            vdp_mu=vdp_mu,
            vdp_amp=vdp_amp,
            vdp_w_drive=vdp_w_drive,
            noise_level=noise_level,
        )

    train_hi = half - H + 1
    test_hi = T - H + 1
    train_anchors = _pick_anchors(W, train_hi, anchors_per_sample)
    test_anchors = _pick_anchors(half + W, test_hi, anchors_per_sample)

    assert train_anchors.min() >= W
    assert train_anchors.max() + H - 1 < half
    assert test_anchors.min() >= half + W
    assert test_anchors.max() + H - 1 < T

    split_rng = np.random.default_rng(seed + 1)
    perm = split_rng.permutation(n_trajectories)
    disc_idx = np.sort(perm[: n_trajectories // 2])
    val_idx = np.sort(perm[n_trajectories // 2 :])
    x_disc = x[disc_idx]
    x_val = x[val_idx]

    X_disc_train = _windows(x_disc, train_anchors, W, H)
    X_disc_test = _windows(x_disc, test_anchors, W, H)
    X_val_train = _windows(x_val, train_anchors, W, H)
    X_val_test = _windows(x_val, test_anchors, W, H)

    pers_disc = _persistence_mse_per_sample(X_disc_test)
    X_disc_test["_persistence_mse"] = pers_disc
    X_disc_test.update(_rollout_extras(x_disc, W, half, rollout_H))
    X_val_test["_persistence_mse"] = _persistence_mse_per_sample(X_val_test)
    X_val_test.update(_rollout_extras(x_val, W, half, rollout_H))

    n_eval = int(min(max(1, n_eval), len(disc_idx)))
    eval_pos = np.sort(split_rng.choice(len(disc_idx), n_eval, replace=False))
    eval_anchor_sub = _pick_anchors(W, train_hi, eval_anchors)
    X_eval = _windows(x_disc[eval_pos], eval_anchor_sub, W, H)
    X_eval["_sample_indices"] = eval_pos

    print(
        f"[load_data] {system}: {n_trajectories} trajectories -> "
        f"disc/val={len(disc_idx)}/{len(val_idx)}; T={T}; W={W}; H={H}; "
        f"anchors train/test={len(train_anchors)}/{len(test_anchors)}; "
        f"persistence MSE disc-test mean={pers_disc.mean():.6f} "
        f"median={np.median(pers_disc):.6f}"
    )

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def apply_model(model_fn, data, params):
    """Nested vmap: one param set per trajectory, scalar prediction per window."""
    H_hist = data["history"]

    def per_sample(h, p):
        return jax.vmap(lambda w: model_fn(w, p))(h)

    return jax.vmap(per_sample, in_axes=(0, 0))(H_hist, params)


def loss_fn(model_output, data):
    y = data["target_y"]
    return jnp.mean((model_output - y) ** 2, axis=-1)
