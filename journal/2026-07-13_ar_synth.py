"""Synthetic autoregressive benchmark: ring attractor with spike-frequency adaptation.

Latent system (what the brain does), N cells on a ring:

    x_i(t+1) = A x_i(t) + dt * ( sum_j K(i-j) phi(x_j(t)) - g a_i(t) ) + eps_i(t)
    a_i(t+1) = B a_i(t) + dt * x_i(t)

    A = 1 - dt/tau_x,  B = 1 - dt/tau_a
    K(d) = (J_e exp(-d^2 / 2 sigma^2) - J_i) / n_cells   (Mexican hat, mean-field scaled)
    phi  = tanh(relu(.))                                 (rectified, saturating)

We observe ONLY x. The adaptation current a is hidden. Because the a-update is a
one-pole linear filter, a can be eliminated exactly, giving the observed-form
ground truth:

    x(t+1) = (A + B) x(t) - (A B + g dt^2) x(t-1)
             + dt * [ C(t) - B C(t-1) ],      C(t) = K * phi(x(t))

An exact AR(2) whose second lag *is* the unobserved adaptation current. This is
the equation EDGAR should discover. Six parameters (tau_x, tau_a, g, J_e, sigma,
J_i) against 2*N^2 for an unconstrained VAR(2).

Process noise becomes MA(1) under elimination: eps(t) - B eps(t-1). The
deterministic map stays exact, but the true equation is no longer the
one-step-ahead MSE optimum -- least squares can partially cancel eps(t-1) from
the observed lags, and the true model cannot. Rolling out H>1 steps removes that
advantage (see __main__).

Recordings start from a random field rather than a settled bump, so the transient
sweeps a wide region of state space and the rectification is actually exercised.
Without that, trajectories collapse onto a low-dimensional bump manifold, the
dynamics look linear, and an unconstrained VAR(2) beats the truth.
"""

from __future__ import annotations

import numpy as np


# ── ground truth ──

TRUE = dict(tau_x=1.0, tau_a=8.0, g=0.6, J_e=20.0, sigma=0.6, J_i=3.0)
DT = 0.1


def phi(x):
    return np.tanh(np.maximum(x, 0.0))


def kernel(n_cells, J_e, sigma, J_i):
    """Mexican-hat coupling on a ring, mean-field scaled. Returns (n_cells, n_cells).

    The 1/n_cells keeps total synaptic input O(1) as the population grows, so phi
    stays in its curved regime instead of saturating into a step function.
    """
    theta = 2 * np.pi * np.arange(n_cells) / n_cells
    d = theta[:, None] - theta[None, :]
    d = np.arctan2(np.sin(d), np.cos(d))  # wrapped angular distance
    return (J_e * np.exp(-(d**2) / (2 * sigma**2)) - J_i) / n_cells


def simulate(
    n_cells=32,
    n_times=250,
    n_recordings=16,
    params=TRUE,
    dt=DT,
    process_noise=0.005,
    obs_noise=0.0,
    init="random",
    burn_in=0,
    seed=0,
):
    """Simulate the latent system, observing only x.

    Args:
        init: "random" starts from a random field (rich transient, then bump);
            "bump" starts from a settled bump (use with burn_in > 0).
        process_noise: dynamical noise, enters the state and is remembered.
        obs_noise: measurement noise added post hoc. Biases one-step AR fitting
            (errors-in-variables) -- keep at 0 until the basic case works.

    Returns:
        x, shape (n_recordings, n_cells, n_times).
    """
    rng = np.random.default_rng(seed)
    A = 1 - dt / params["tau_x"]
    B = 1 - dt / params["tau_a"]
    g = params["g"]
    K = kernel(n_cells, params["J_e"], params["sigma"], params["J_i"])
    theta = 2 * np.pi * np.arange(n_cells) / n_cells

    out = np.zeros((n_recordings, n_cells, n_times))
    for r in range(n_recordings):
        if init == "random":
            x = 1.5 * rng.standard_normal(n_cells)
        else:
            centre = rng.uniform(0, 2 * np.pi)
            d = np.arctan2(np.sin(theta - centre), np.cos(theta - centre))
            x = np.exp(-(d**2) / (2 * 0.5**2))
        a = np.zeros(n_cells)

        traj = np.zeros((n_cells, burn_in + n_times))
        for t in range(burn_in + n_times):
            traj[:, t] = x
            eps = process_noise * rng.standard_normal(n_cells)
            x_next = A * x + dt * (K @ phi(x) - g * a) + eps
            a = B * a + dt * x
            x = x_next
        out[r] = traj[:, burn_in:]

    if obs_noise > 0:
        out = out + obs_noise * rng.standard_normal(out.shape)
    return out


# ── observed-form ground truth: the equation EDGAR should discover ──


def ar2_true(x_t, x_tm1, params=TRUE, dt=DT):
    """One step of the eliminated AR(2) map. x_t, x_tm1: (n_cells,)."""
    A = 1 - dt / params["tau_x"]
    B = 1 - dt / params["tau_a"]
    g = params["g"]
    K = kernel(x_t.shape[0], params["J_e"], params["sigma"], params["J_i"])
    return (
        (A + B) * x_t
        - (A * B + g * dt**2) * x_tm1
        + dt * (K @ phi(x_t) - B * (K @ phi(x_tm1)))
    )


def check_elimination_is_exact():
    """The AR(2) form must reproduce the latent sim exactly when noise = 0."""
    x = simulate(n_recordings=1, n_times=200, process_noise=0.0, seed=1)[0]
    return max(
        np.abs(ar2_true(x[:, t], x[:, t - 1]) - x[:, t + 1]).max()
        for t in range(1, x.shape[1] - 1)
    )


# ── candidate models ──
#
# Every model has EDGAR's shape: step(x_t, x_tm1, params) -> x_{t+1}, plus a
# DEFAULT_PARAMS dict, plus a fit() estimating params from data (this stands in
# for EDGAR's parameter_estimator + gradient descent). Roles:
#
#   SEED      what we hand EDGAR to start from. Both are single-lag, so neither
#             can express the hidden adaptation current -- that is the point.
#   BASELINE  the strong dumb models EDGAR must beat, and must beat for the right
#             reason. VAR = vector autoregression: an unconstrained matrix of every
#             pairwise coupling, assuming no ring, no kernel, no nonlinearity.
#   TRUTH     the equation we want discovered.
#
# NOTE these are NOT the same thing. VAR(1) is a full n_cells x n_cells coupling
# matrix; the AR(1) seed is a single scalar leak applied to each cell separately.
# Seeding with VAR(1) would hand over the coupling structure for free.

N_CELLS = 32


def persistence(x_t, x_tm1, params):
    """SEED. x(t+1) = x(t). Nothing is learned; the bump is assumed frozen."""
    return x_t


persistence.DEFAULT_PARAMS = {}


def ar1_leak(x_t, x_tm1, params):
    """SEED. x_i(t+1) = alpha x_i(t). One scalar, per-cell, no coupling."""
    return params["alpha"] * x_t


ar1_leak.DEFAULT_PARAMS = {"alpha": 0.9}


def var1(x_t, x_tm1, params):
    """BASELINE. x(t+1) = W1 x(t). Every pairwise coupling, one lag. n_cells^2 params."""
    return params["W1"] @ x_t


var1.DEFAULT_PARAMS = {"W1": np.eye(N_CELLS)}


def var2(x_t, x_tm1, params):
    """BASELINE. x(t+1) = W1 x(t) + W2 x(t-1). Two lags. 2 n_cells^2 params."""
    return params["W1"] @ x_t + params["W2"] @ x_tm1


var2.DEFAULT_PARAMS = {"W1": np.eye(N_CELLS), "W2": np.zeros((N_CELLS, N_CELLS))}


def ground_truth(x_t, x_tm1, params):
    """TRUTH. Ring coupling + the second lag standing in for hidden adaptation."""
    return ar2_true(x_t, x_tm1, params)


ground_truth.DEFAULT_PARAMS = TRUE


def n_params(model):
    return sum(np.asarray(v).size for v in model.DEFAULT_PARAMS.values())


# ── parameter fitting ──


def _lag_pairs(X):
    """Stack all (x(t-1), x(t)) -> x(t+1) triples across recordings."""
    xt, xtm1, nxt = [], [], []
    for x in X:
        for t in range(1, x.shape[1] - 1):
            xtm1.append(x[:, t - 1])
            xt.append(x[:, t])
            nxt.append(x[:, t + 1])
    return np.array(xt), np.array(xtm1), np.array(nxt)


def fit(model, X):
    """One-step least-squares fit. Returns a params dict for `model`.

    Linear in the parameters for every model here, so this is closed-form. In
    EDGAR this job is split between parameter_estimator and gradient descent.
    """
    xt, xtm1, nxt = _lag_pairs(X)

    if model is persistence:
        return {}
    if model is ar1_leak:
        return {"alpha": float((xt * nxt).sum() / (xt * xt).sum())}
    if model is var1:
        return {"W1": np.linalg.lstsq(xt, nxt, rcond=None)[0].T}
    if model is var2:
        W = np.linalg.lstsq(np.hstack([xt, xtm1]), nxt, rcond=None)[0].T
        return {"W1": W[:, :N_CELLS], "W2": W[:, N_CELLS:]}
    if model is ground_truth:
        return (
            TRUE  # analytic; refitting these 6 is badly ill-conditioned (see journal)
        )
    raise ValueError(model)


# ── evaluation: teacher-forced restarts, H-step rollout ──


def rollout(step_fn, x, horizon):
    """From every valid start, roll `horizon` steps on the model's own predictions.

    step_fn(x_t, x_tm1) -> x_{t+1}.  x: (n_cells, T).
    Returns (preds, targets), both (n_starts, horizon, n_cells).
    """
    preds, targets = [], []
    for s in range(1, x.shape[1] - horizon):
        xm1, xt = x[:, s - 1], x[:, s]
        p = []
        for _ in range(horizon):
            nxt = step_fn(xt, xm1)
            p.append(nxt)
            xm1, xt = xt, nxt
        preds.append(np.stack(p))
        targets.append(x[:, s + 1 : s + 1 + horizon].T)
    return np.array(preds), np.array(targets)


def mse(step_fn, X, horizon):
    return float(
        np.mean(
            [
                ((p - t) ** 2).mean()
                for p, t in (rollout(step_fn, x, horizon) for x in X)
            ]
        )
    )


def bump_stats(x):
    """Circular mean of rectified activity: concentration and total drift."""
    theta = 2 * np.pi * np.arange(x.shape[0]) / x.shape[0]
    act = np.maximum(x, 0)
    z = (act * np.exp(1j * theta)[:, None]).sum(0) / (act.sum(0) + 1e-9)
    centre = np.unwrap(np.angle(z))
    return float(np.abs(z).mean()), float(centre[-1] - centre[0])


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"elimination max abs error (noise=0): {check_elimination_is_exact():.2e}")

    X_train = simulate(seed=0)
    X_test = simulate(n_recordings=8, seed=99)
    conc, drift = bump_stats(X_train[0])
    print(f"data: {X_train.shape} (recordings, cells, times)")
    print(f"activity std {X_train.std():.2f}  max {X_train.max():.2f}")
    print(
        f"bump concentration {conc:.2f}  drift {drift:.1f} rad over {X_train.shape[2]} steps"
    )

    models = [
        ("persistence", persistence, "seed"),
        ("AR(1) leak", ar1_leak, "seed"),
        ("VAR(1)", var1, "baseline"),
        ("VAR(2)", var2, "baseline"),
        ("ground truth", ground_truth, "truth"),
    ]
    fitted = {name: fit(model, X_train) for name, model, _ in models}
    step = lambda model, p: lambda xt, xm1: model(xt, xm1, p)  # noqa: E731

    print(
        f"\n{'model':<14}{'role':<10}{'n_params':>9}{'H=1':>10}{'H=3':>10}{'H=10':>10}"
    )
    for name, model, role in models:
        fn = step(model, fitted[name])
        print(
            f"{name:<14}{role:<10}{n_params(model):>9}"
            + "".join(f"{mse(fn, X_test, h):>10.5f}" for h in (1, 3, 10))
        )
    print(f"\nfitted AR(1) leak: alpha = {fitted['AR(1) leak']['alpha']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, r in zip(axes[:2], range(2)):
        ax.imshow(X_train[r], aspect="auto", cmap="magma", origin="lower")
        ax.set(xlabel="time", ylabel="cell", title=f"recording {r}")
    ax = axes[2]
    x = X_test[0]
    # rollout start s (s = 1..T-H) predicts x[:, s+1 : s+1+H]; preds[s-1, h] targets x[:, s+1+h]
    p_true, tgt = rollout(step(ground_truth, fitted["ground truth"]), x, 3)
    p_var2, _ = rollout(step(var2, fitted["VAR(2)"]), x, 3)
    ax.plot(tgt[:100, 2, 8], "k", lw=2, label="data (cell 8)")
    ax.plot(p_true[:100, 2, 8], "C0", label="ground truth, 3-step")
    ax.plot(p_var2[:100, 2, 8], "C3", alpha=0.8, label="VAR(2), 3-step")
    ax.set(xlabel="time", ylabel="activity", title="3-step-ahead prediction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("journal/2026-07-13_ar_synth.png", dpi=110)
    print("\nwrote journal/2026-07-13_ar_synth.png")
