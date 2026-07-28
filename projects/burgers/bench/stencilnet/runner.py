"""STENCIL-NET runner for the Burgers benchmark (Step 2).

Thin wrapper that trains the *vendored* STENCIL-NET (see stencilnet_src/, copied
verbatim from the reference repo) on our shared Burgers field and reports the
shared forecast metric. All the physics/architecture is theirs; this file only:

  1. loads the shared data bundle (data_loader.load_data),
  2. builds their MLPConv [fs,64,64,64,1] + forcing terms,
  3. trains with their forcing-aware forward+backward RK3 loss; the noisy case
     adds a learnable latent-noise field (net.noise) + its regularization, which
     the same error functions handle (noise-as-latent denoising),
  4. scores forecast MSE against the CLEAN field using teacher-forced RK3
     restarts (data_loader.teacher_forced_forecast) — the SAME protocol EDGAR is
     graded on (evaluate.py), not a single full-horizon free-run.

Training recipe mirrors ForcedBurgersSimulation.ipynb (clean) and
ForcedBurgersNoiseDecomposition.ipynb (noisy): Adam lr=1e-3, ExponentialLR(.9998)
after 15k epochs, decay 0.9, l_wd=1e-7, l_n=1e-5. Training and scoring share one
rollout horizon (config evaluate.rollout_steps), so the net is trained over the
same number of steps it is graded on (was a fixed m=4 in the reference).

CLI:
    python runner.py --noise 0.0 --epochs 30000
    python runner.py --noise 0.1 --epochs 30000   # uses noise-as-latent path
Outputs a per-run dict to results/stencilnet_noise<level>.json and a forecast
field to results/stencilnet_pred_noise<level>.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "stencilnet_src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "data_loader"))

from network import MLPConv  # noqa: E402  (vendored)
from timestepping import (  # noqa: E402  (vendored)
    forward_rk3_error, backward_rk3_error,
)
import load_data as ld  # noqa: E402
import burgers_sim as bs  # noqa: E402


# ── forcing terms on the coarse grid (ported from utils.forcing_terms) ──

def _forcing_terms(A, w, phi, l, L, Lxc, Ltc, dtc, N):
    """Coarse forcing at t, t±0.5dt, t±dt. Verbatim structure from utils.forcing_terms.

    Note the reference uses T (physical end time) only to build a linspace of length
    Ltc; the actual times used are index*dtc. We reproduce that exactly by passing
    T = (Ltc-1)*dtc so the grid matches the coarse sampling.
    """
    T = (Ltc - 1) * dtc
    x = np.linspace(0, L, Lxc)
    def field(shift):
        t = np.linspace(0 + shift, T + shift, Ltc)
        XX, TT = np.meshgrid(x, t); xx = XX.T; tt = TT.T
        F = np.zeros((Lxc, Ltc))
        for k in range(N):
            F = F + A[k] * np.sin(w[k] * tt + 2.0 * np.pi * l[k] * (xx / L) + phi[k])
        return F
    return field(0.0), field(0.5 * dtc), field(dtc), field(-0.5 * dtc), field(-dtc)


def run(noise_level=0.0, epochs=30000, seed=1, train_cols=1001, fs=7,
        neurons=64, device=None, out_dir=None, rollout_steps=None):
    out_dir = out_dir or os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    # One rollout horizon for both training and scoring (default: config).
    if rollout_steps is None:
        rollout_steps = ld.benchmark_rollout_steps()
    rollout_steps = int(rollout_steps)
    # device=None -> auto-detect (CUDA if present, else MPS, else CPU) so the same
    # runner is portable between this CPU box and a GPU machine.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    b = ld.load_bundle()
    L = b["L"]; dtc = b["dtc"]; N = b["N"]
    A, w, phi, l = b["A"], b["w"], b["phi"], b["l"]
    u_clean = b["u_coarse"]                       # (Lx, T)
    u_obs = ld.noise_field(b, noise_level)        # observed (noisy) field
    Lxc = u_clean.shape[0]

    # train on the first `train_cols` time columns (ref uses 1001)
    u_tr = u_obs[:, :train_cols]
    Ltc = u_tr.shape[1]
    u_train = torch.tensor(u_tr.T, requires_grad=True, dtype=torch.float, device=device)

    noisy = noise_level not in (0, 0.0)
    torch.manual_seed(seed); np.random.seed(seed)

    if noisy:
        # noise-as-latent: Tikhonov-smoothed init (ref noise_initialization)
        noise_init = bs.__dict__.get("noise_initialization")
        # noise_initialization lives in utils, not our sim module; reimplement inline
        noise0 = _tikhonov_noise_init(u_tr, lam=0.1)
        noise_t = torch.tensor(noise0.T, requires_grad=True, dtype=torch.float, device=device)
        net = MLPConv([fs, neurons, neurons, neurons, 1], seed=seed, fs=fs,
                      noise=noise_t, activation=nn.ELU()).to(device)
    else:
        net = MLPConv([fs, neurons, neurons, neurons, 1], seed=seed, fs=fs,
                      activation=nn.ELU()).to(device)

    # forcing terms (known input, supplied to the loss in the clean case)
    Fc, Fc_0p5, Fc_p1, Fc_0m5, Fc_m1 = _forcing_terms(A, w, phi, l, L, Lxc, Ltc, dtc, N)
    to_t = lambda F: torch.tensor(F.T, dtype=torch.float, device=device)
    fc, fc_0p5, fc_p1, fc_0m5, fc_m1 = map(to_t, (Fc, Fc_0p5, Fc_p1, Fc_0m5, Fc_m1))

    optimizer = Adam([{"params": net.parameters(), "lr": 1e-3}])
    scheduler = ExponentialLR(optimizer, 0.9998)
    wd = torch.tensor([0.9 ** j for j in range(rollout_steps + 1)], dtype=torch.float32, device=device)
    l_wd, l_n = 1e-7, 1e-5

    t0 = time.time()
    loss_hist = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forcing-aware forward/backward RK3 loss for BOTH clean and noisy. These
        # error fns read net.noise (the latent-noise field, set only in the noisy
        # case) and take the known forcing fc, so the net learns the UNFORCED
        # operator N(u) with forcing supplied separately -- matching the scoring
        # convention (rhs = net(u) + f). The vendored TVD variants are the
        # reference's *unforced* denoising path: with no fc they force the net to
        # absorb the forcing, which is then added again at score time (double-count).
        fwd = forward_rk3_error(net, u_train, dtc, rollout_steps, wd, fc=fc, fc_0p5=fc_0p5, fc_p1=fc_p1)
        bwd = backward_rk3_error(net, u_train, dtc, rollout_steps, wd, fc=fc, fc_0m5=fc_0m5, fc_m1=fc_m1)
        res_w = 0
        for layer in net.layer:
            W = layer.weight.view(layer.weight.shape[0] * layer.weight.shape[1], -1)
            res_w = res_w + (torch.norm(W, p=2, dim=0) ** 2)
        loss = fwd + bwd + l_wd * res_w
        if noisy:
            reg_n = torch.sum(net.noise.reshape(Lxc * Ltc, -1) ** 2)
            loss = loss + l_n * reg_n
        loss = loss.sum() if loss.ndim else loss
        loss.backward()
        optimizer.step()
        if epoch > 15000:
            scheduler.step()
        if epoch % max(1, epochs // 50) == 0:
            loss_hist.append((epoch, float(loss.item())))
    train_s = time.time() - t0

    # teacher-forced restart forecast, scored vs CLEAN on the same protocol as
    # EDGAR (projects/burgers/evaluate/evaluate.py): from every true state, RK3-roll
    # `rollout_steps` on the net's own predictions and score against the clean field.
    # The batched rhs is net(u) + known forcing at each restart's physical time.
    xg = np.linspace(0, L, Lxc)

    def rhs(state, t_arr):
        tens = torch.tensor(state, dtype=torch.float, device=device)
        net_out = net(tens).cpu().data.numpy()          # (n_starts, Lx)
        tt = np.asarray(t_arr)[:, None]                  # (n_starts, 1)
        F = np.zeros((state.shape[0], Lxc))
        for k in range(N):
            F = F + A[k] * np.sin(w[k] * tt + 2.0 * np.pi * l[k] * (xg[None, :] / L) + phi[k])
        return net_out + F

    preds, targets = ld.teacher_forced_forecast(rhs, u_clean, dtc, rollout_steps)
    heldout = np.arange(preds.shape[0]) >= train_cols   # restarts seeded beyond the train window
    mse_full = ld.forecast_mse(preds, targets)
    mse_heldout = ld.forecast_mse(preds[heldout], targets[heldout])

    result = {
        "method": "stencilnet", "noise_level": noise_level, "epochs": epochs,
        "seed": seed, "train_cols": train_cols, "fs": fs, "neurons": neurons,
        "rollout_steps": rollout_steps,
        "n_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
        "train_seconds": round(train_s, 1),
        "forecast_mse_full": mse_full, "forecast_mse_heldout": mse_heldout,
        "loss_hist": loss_hist,
    }
    tag = f"noise{noise_level}"
    with open(os.path.join(out_dir, f"stencilnet_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    # one-step-ahead teacher-forced field (preds[:, 0]) for visualisation
    np.savez_compressed(os.path.join(out_dir, f"stencilnet_pred_{tag}.npz"),
                        pred_one_step=preds[:, 0].T, u_clean=u_clean)
    return result


def _tikhonov_noise_init(u_coarse_noise, lam=0.1):
    """Verbatim from utils.noise_initialization: Tikhonov-smoothed noise estimate."""
    Lxc, Ltc = u_coarse_noise.shape
    m = Lxc
    DD = np.zeros((m, m))
    DD[0, :4] = [2, -5, 4, -1]
    DD[m - 1, m - 4:] = [-1, 4, -5, 2]
    for i in range(1, m - 1):
        DD[i, i] = -2; DD[i, i + 1] = 1; DD[i, i - 1] = 1
    DD = DD.dot(DD)
    u_smooth = np.zeros((Lxc, Ltc))
    A = np.eye(m) + lam * DD.T.dot(DD)
    for t in range(Ltc):
        u_smooth[:, t] = np.linalg.solve(A, u_coarse_noise[:, t])
    return u_coarse_noise - u_smooth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None,
                    help="cuda|mps|cpu; default auto-detects (CUDA > MPS > CPU)")
    args = ap.parse_args()
    r = run(noise_level=args.noise, epochs=args.epochs, seed=args.seed, device=args.device)
    print(json.dumps({k: v for k, v in r.items() if k != "loss_hist"}, indent=2))
