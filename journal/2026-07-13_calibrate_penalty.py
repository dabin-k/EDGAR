"""Calibrate scoring.param_penalty_weight for projects/synthetic_data_v2.

Scores are what EDGAR actually ranks on:  score = test_loss + w * n_params,
with n_params counted elementwise (edgar/evolution/program.py:339).

Parameters are fitted the way scoring._optimize fits them: per sample, Adam on the
*train* rollout loss, keeping the best iterate; then evaluated on the test blocks.
Pooled least-squares (as in 2026-07-13_ar_synth.py) understates what a VAR can do
here, so the penalty must be calibrated against this, not against that.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

PROJ = Path("projects/synthetic_data_v2")
sys.path.insert(0, str(PROJ / "evaluate"))
sys.path.insert(0, str(PROJ / "data_loader"))
from evaluate import evaluate  # noqa: E402
from load_data import load_data, loss_fn  # noqa: E402

N_CELLS = 32
DT = 0.1


# ── candidate models, in the project's calling convention ──


def persistence(data, params):
    return data["x"][:, -1]


def leaky_decay(data, params):
    return params["decay"] * data["x"][:, -1]


def var1(data, params):
    return params["W1"] @ data["x"][:, -1]


def var2(data, params):
    return params["W1"] @ data["x"][:, -1] + params["W2"] @ data["x"][:, -2]


def ground_truth(data, params):
    """The eliminated AR(2): ring coupling at both lags, second lag = hidden adaptation."""
    x_t, x_tm1 = data["x"][:, -1], data["x"][:, -2]
    A = 1 - DT / params["tau_x"]
    B = 1 - DT / params["tau_a"]
    theta = 2 * jnp.pi * jnp.arange(N_CELLS) / N_CELLS
    d = theta[:, None] - theta[None, :]
    d = jnp.arctan2(jnp.sin(d), jnp.cos(d))
    K = (
        params["J_e"] * jnp.exp(-(d**2) / (2 * params["sigma"] ** 2)) - params["J_i"]
    ) / N_CELLS
    phi = lambda z: jnp.tanh(jnp.maximum(z, 0.0))  # noqa: E731
    return (
        (A + B) * x_t
        - (A * B + params["g"] * DT**2) * x_tm1
        + DT * (K @ phi(x_t) - B * (K @ phi(x_tm1)))
    )


MODELS = [
    ("persistence", persistence, {}, "seed"),
    ("leaky decay", leaky_decay, {"decay": 0.9}, "seed"),
    ("VAR(1)", var1, {"W1": np.eye(N_CELLS)}, "baseline"),
    (
        "VAR(2)",
        var2,
        {"W1": np.eye(N_CELLS), "W2": np.zeros((N_CELLS, N_CELLS))},
        "baseline",
    ),
    (
        "ground truth",
        ground_truth,
        dict(tau_x=1.0, tau_a=8.0, g=0.6, J_e=20.0, sigma=0.6, J_i=3.0),
        "truth",
    ),
    # A structured coupling with the right form but a free radial profile: what we
    # would be happy for EDGAR to find. Kernel is a free vector over ring distance.
]


def n_params(default):
    return sum(np.asarray(v).size for v in default.values())


def batched(default, n):
    return jax.tree_util.tree_map(
        lambda v: jnp.stack([jnp.asarray(v, float)] * n), dict(default)
    )


def fit(model, default, train, max_iter=600, lr=0.01):
    """Mirror scoring._optimize: Adam on the train rollout loss, keep the best iterate."""
    n = train["x"].shape[0]
    params = batched(default, n)
    if not default:
        return params
    flat, unflatten = ravel_pytree(params)

    def total(f):
        preds, targets = evaluate(model, train, unflatten(f))
        return jnp.mean(loss_fn(preds, targets))

    grad = jax.jit(jax.value_and_grad(total))
    opt = optax.adam(lr)
    state = opt.init(flat)
    best, best_flat = float("inf"), flat
    for _ in range(max_iter):
        loss, g = grad(flat)
        if not jnp.isfinite(loss):
            break
        if float(loss) < best:
            best, best_flat = float(loss), flat
        updates, state = opt.update(g, state, flat)
        flat = optax.apply_updates(flat, updates)
    return unflatten(best_flat)


def test_loss(model, params, test):
    preds, targets = evaluate(model, test, params)
    return float(jnp.mean(loss_fn(preds, targets)))


(disc_train, disc_test), _, _ = load_data("unused")

rows = []
for name, model, default, role in MODELS:
    params = fit(model, default, disc_train)
    rows.append((name, role, n_params(default), test_loss(model, params, disc_test)))
    print(f"  fitted {name:<13} raw test loss {rows[-1][3]:.6f}")

print(f"\n{'model':<14}{'role':<10}{'n_par':>6}{'raw loss':>11}", end="")
weights = [0.01, 1e-3, 1e-4, 1e-5, 1e-6]
for w in weights:
    print(f"{'w=' + format(w, '.0e'):>11}", end="")
print()
for name, role, npar, loss in rows:
    print(f"{name:<14}{role:<10}{npar:>6}{loss:>11.6f}", end="")
    for w in weights:
        print(f"{loss + w * npar:>11.4f}", end="")
    print()

print("\nwinner (lowest score) by weight:")
for w in weights:
    best = min(rows, key=lambda r: r[3] + w * r[2])
    runner = sorted(rows, key=lambda r: r[3] + w * r[2])[1]
    b, r = best[3] + w * best[2], runner[3] + w * runner[2]
    print(
        f"  w={w:<8.0e} {best[0]:<14} (score {b:.4f})   "
        f"runner-up {runner[0]} ({r:.4f}), margin {r / b:.1f}x"
    )
