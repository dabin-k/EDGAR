"""STENCIL-NET runner for the Burgers benchmark (Step 2).

Thin wrapper that trains the *vendored* STENCIL-NET (see stencilnet_src/, copied
verbatim from the reference repo) on one sample of a shared benchmark dataset and
reports the shared forecast metric. All the physics/architecture is theirs; this
file only:

  1. loads one sample of a shared dataset (data_loader.load_dataset),
  2. builds their MLPConv [fs,64,64,64,1] (+ forcing terms, if the data is forced),
  3. trains with their forcing-aware forward+backward RK3 loss, summed over the
     TRAIN blocks of the shared leak-free split (data_loader.block_split) so no
     rollout window crosses into a test block; the noisy case adds a single
     learnable latent-noise field over the training columns (sliced per block,
     regularized once) which the same error functions handle,
  4. scores forecast MSE against the OBSERVED (noisy) field, seeded from observed
     states, using teacher-forced RK3 restarts
     (data_loader.teacher_forced_forecast), reporting forecast_mse_train
     and forecast_mse_test on the same split as SINDy — the SAME protocol EDGAR is
     graded on (evaluate.py), not a single full-horizon free-run — alongside the
     do-nothing (persistence) forecast on the identical restarts, so the MSE can
     be read against the trivial floor rather than in isolation.

Training recipe mirrors ForcedBurgersSimulation.ipynb (clean) and
ForcedBurgersNoiseDecomposition.ipynb (noisy): Adam lr=1e-3, ExponentialLR(.9998)
after 15k epochs, decay 0.9, l_wd=1e-7, l_n=1e-5. Training and scoring share one
rollout horizon (config evaluate.rollout_steps), so the net is trained over the
same number of steps it is graded on (was a fixed m=4 in the reference).

CLI:
    python runner.py --data-path .../ic_seed_0_nl_0.0.npz --sample-idx 1 --epochs 30000
    python runner.py --data-path .../ic_seed_0_nl_0.1.npz --sample-idx 1   # noise-as-latent path
Outputs a per-run dict to results/rollout<r>/stencilnet_ic<ic>_nl<level>_s<j>.json, a
forecast field to results/rollout<r>/stencilnet_pred_ic<ic>_nl<level>_s<j>.npz, and the
trained weights to <weights_dir>/ic<ic>_nl<level>_r<r>_s<j>.pt.
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
    forward_rk3_error,
    backward_rk3_error,
)
import load_data as ld  # noqa: E402

_DEFAULT_WEIGHTS_DIR = "/home/dabin/data/burgers_simulated/stencilnet_weights"


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
        XX, TT = np.meshgrid(x, t)
        xx = XX.T
        tt = TT.T
        F = np.zeros((Lxc, Ltc))
        for k in range(N):
            F = F + A[k] * np.sin(w[k] * tt + 2.0 * np.pi * l[k] * (xx / L) + phi[k])
        return F

    return field(0.0), field(0.5 * dtc), field(dtc), field(-0.5 * dtc), field(-dtc)


def run(
    data_path,
    sample_idx,
    epochs=30000,
    seed=1,
    block_len=200,
    fs=7,
    neurons=64,
    device=None,
    out_dir=None,
    rollout_steps=None,
    weights_dir=None,
):
    # One rollout horizon for both training and scoring (default: config).
    if rollout_steps is None:
        rollout_steps = ld.benchmark_rollout_steps()
    rollout_steps = int(rollout_steps)
    sample_idx = int(sample_idx)
    out_dir = out_dir or os.path.join(_HERE, "results", f"rollout{rollout_steps}")
    weights_dir = weights_dir or _DEFAULT_WEIGHTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    # device=None -> auto-detect (CUDA if present, else MPS, else CPU) so the same
    # runner is portable between this CPU box and a GPU machine.
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    device = torch.device(device)

    ds = ld.load_dataset(data_path)
    L = ds["L"]
    dtc = ds["dtc"]
    N = ds["N"]
    A, w, phi, l = ds["A"], ds["w"], ds["phi"], ds["l"]
    forced = bool(ds["forced"])
    noise_level = float(ds["noise_level"])
    ic_seed = int(ds["ic_seed"])
    D = float(ds["D"][sample_idx])
    u_clean = ds["u_clean"][sample_idx]  # (Lx, T)
    u_obs = ds["u_noisy"][sample_idx]  # observed field, noise baked in
    Lxc, T = u_clean.shape

    train_cols, test_cols = ld.block_split(T, block_len=block_len)
    train_runs = _col_runs(train_cols)  # per-block absolute column indices
    row_offsets = np.cumsum([0] + [r.size for r in train_runs])  # block -> noise rows
    n_train_cols = int(row_offsets[-1])

    noisy = noise_level > 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build the net without an internal noise Parameter; in the noisy case we hold
    # ONE full-field latent (net.noise) over all training columns and slice a view
    # of it per block, so a single denoising field is learned and regularized once
    # while each block's rollout only touches its own columns.
    net = MLPConv(
        [fs, neurons, neurons, neurons, 1], seed=seed, fs=fs, activation=nn.ELU()
    ).to(device)
    param_groups = [{"params": net.parameters(), "lr": 1e-3}]
    if noisy:
        # noise-as-latent: Tikhonov-smoothed init (ref noise_initialization),
        # column-wise so restarting it per block is exact.
        noise0 = _tikhonov_noise_init(
            u_obs[:, train_cols], lam=0.1
        )  # (Lxc, n_train_cols)
        noise_full = nn.Parameter(
            torch.tensor(noise0.T, dtype=torch.float, device=device)
        )  # (n_train_cols, Lx)
        param_groups.append({"params": [noise_full], "lr": 1e-3})
    else:
        noise_full = None
        net.noise = None

    # Forcing terms over the WHOLE field (known input, supplied to the loss); each
    # block slices its own columns below. On the unforced datasets there is nothing
    # to supply: leave every slice None so the vendored error fns substitute zeros
    # themselves rather than us materialising five zero (Lx, T) fields.
    if forced:
        Fc, Fc_0p5, Fc_p1, Fc_0m5, Fc_m1 = _forcing_terms(
            A, w, phi, l, L, Lxc, T, dtc, N
        )

        def to_t(F):
            return torch.tensor(F.T, dtype=torch.float, device=device)

        def slice_f(F, lo, hi):
            return F[lo:hi]

        fc, fc_0p5, fc_p1, fc_0m5, fc_m1 = map(to_t, (Fc, Fc_0p5, Fc_p1, Fc_0m5, Fc_m1))
    else:
        fc = fc_0p5 = fc_p1 = fc_0m5 = fc_m1 = None

        def slice_f(F, lo, hi):
            return None

    # per-block observed field tensors + forcing slices (contiguous column ranges)
    blocks = []
    for bi, cols in enumerate(train_runs):
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        blocks.append(
            {
                "u": torch.tensor(
                    u_obs[:, lo:hi].T,
                    requires_grad=True,
                    dtype=torch.float,
                    device=device,
                ),
                "fc": slice_f(fc, lo, hi),
                "fc_0p5": slice_f(fc_0p5, lo, hi),
                "fc_p1": slice_f(fc_p1, lo, hi),
                "fc_0m5": slice_f(fc_0m5, lo, hi),
                "fc_m1": slice_f(fc_m1, lo, hi),
                "n_lo": int(row_offsets[bi]),
                "n_hi": int(row_offsets[bi + 1]),
            }
        )

    optimizer = Adam(param_groups)
    scheduler = ExponentialLR(optimizer, 0.9998)
    wd = torch.tensor(
        [0.9**j for j in range(rollout_steps + 1)], dtype=torch.float32, device=device
    )
    l_wd, l_n = 1e-7, 1e-5

    t0 = time.time()
    loss_hist = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forcing-aware forward/backward RK3 loss, summed over train blocks. These
        # error fns read net.noise (the latent-noise field, set per block only in
        # the noisy case) and take the known forcing fc, so the net learns the
        # UNFORCED operator N(u) with forcing supplied separately -- matching the
        # scoring convention (rhs = net(u) + f). The vendored TVD variants are the
        # reference's *unforced* denoising path: with no fc they force the net to
        # absorb the forcing, which is then added again at score time (double-count).
        fwd_bwd = 0
        for blk in blocks:
            if noisy:
                net.noise = noise_full[blk["n_lo"] : blk["n_hi"]]  # differentiable view
            fwd = forward_rk3_error(
                net,
                blk["u"],
                dtc,
                rollout_steps,
                wd,
                fc=blk["fc"],
                fc_0p5=blk["fc_0p5"],
                fc_p1=blk["fc_p1"],
            )
            bwd = backward_rk3_error(
                net,
                blk["u"],
                dtc,
                rollout_steps,
                wd,
                fc=blk["fc"],
                fc_0m5=blk["fc_0m5"],
                fc_m1=blk["fc_m1"],
            )
            fwd_bwd = fwd_bwd + fwd + bwd
        res_w = 0
        for layer in net.layer:
            W = layer.weight.view(layer.weight.shape[0] * layer.weight.shape[1], -1)
            res_w = res_w + (torch.norm(W, p=2, dim=0) ** 2)
        loss = fwd_bwd + l_wd * res_w
        if noisy:
            reg_n = torch.sum(noise_full**2)  # regularize the whole latent once
            loss = loss + l_n * reg_n
        loss = loss.sum() if loss.ndim else loss
        loss.backward()
        optimizer.step()
        if epoch > 15000:
            scheduler.step()
        if epoch % max(1, epochs // 50) == 0:
            loss_hist.append((epoch, float(loss.item())))
    train_s = time.time() - t0

    # The batched rhs is net(u) + known forcing at each restart's physical time.
    xg = np.linspace(0, L, Lxc)

    def rhs(state, t_arr):
        tens = torch.tensor(state, dtype=torch.float, device=device)
        net_out = net(tens).cpu().data.numpy()  # (n_starts, Lx)
        if not forced:
            return net_out
        tt = np.asarray(t_arr)[:, None]  # (n_starts, 1)
        F = np.zeros((state.shape[0], Lxc))
        for k in range(N):
            F = F + A[k] * np.sin(
                w[k] * tt + 2.0 * np.pi * l[k] * (xg[None, :] / L) + phi[k]
            )
        return net_out + F

    preds, targets = ld.teacher_forced_forecast(rhs, u_obs, dtc, rollout_steps)
    # Score on the SAME leak-free split as SINDy: a restart is train/test only if
    # its whole rollout window lies in the train/test blocks (straddling windows
    # belong to neither), so no scored forecast crosses the split.
    train_mask, test_mask = ld.split_start_masks(
        train_cols, test_cols, T, rollout_steps
    )
    mse_train = ld.forecast_mse(preds[train_mask], targets[train_mask])
    mse_test = ld.forecast_mse(preds[test_mask], targets[test_mask])

    # Do-nothing (persistence) forecast on the IDENTICAL restarts and masks: rhs=0
    # holds the seed state for the whole window. The unforced field decays hard, so
    # a small absolute MSE can just be tracking the decay -- persistence decays the
    # same way, and is the floor any real model must clear.
    p_preds, p_targets = ld.teacher_forced_forecast(
        lambda state, t_arr: np.zeros_like(state), u_obs, dtc, rollout_steps
    )
    pers_train = ld.forecast_mse(p_preds[train_mask], p_targets[train_mask])
    pers_test = ld.forecast_mse(p_preds[test_mask], p_targets[test_mask])

    result = {
        "method": "stencilnet",
        "data_path": data_path,
        "ic_seed": ic_seed,
        "sample_idx": sample_idx,
        "D": D,
        "noise_level": noise_level,
        "forced": forced,
        "epochs": epochs,
        "seed": seed,
        "block_len": block_len,
        "fs": fs,
        "neurons": neurons,
        "rollout_steps": rollout_steps,
        "n_train_cols": n_train_cols,
        "n_train_starts": int(train_mask.sum()),
        "n_test_starts": int(test_mask.sum()),
        "n_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
        "train_seconds": round(train_s, 1),
        "forecast_mse_train": mse_train,
        "forecast_mse_test": mse_test,
        "persistence_mse_train": pers_train,
        "persistence_mse_test": pers_test,
        "skill_test": 1.0 - mse_test / pers_test,
        "loss_hist": loss_hist,
    }
    tag = f"ic{ic_seed}_nl{noise_level}_s{sample_idx}"
    with open(os.path.join(out_dir, f"stencilnet_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    # one-step-ahead teacher-forced field (preds[:, 0]) for visualisation
    np.savez_compressed(
        os.path.join(out_dir, f"stencilnet_pred_{tag}.npz"),
        pred_one_step=preds[:, 0].T,
        u_clean=u_clean,
    )

    # Trained weights, so predictions can be regenerated without retraining. The
    # latent noise lives OUTSIDE the module (net.noise only ever holds a transient
    # per-block view), so it is saved separately or it would be lost.
    net.noise = None
    ckpt_path = os.path.join(
        weights_dir, f"ic{ic_seed}_nl{noise_level}_r{rollout_steps}_s{sample_idx}.pt"
    )
    torch.save(
        {
            "state_dict": net.state_dict(),
            "noise_full": None if noise_full is None else noise_full.detach().cpu(),
            "sizes": [fs, neurons, neurons, neurons, 1],
            "fs": fs,
            "seed": seed,
            "block_len": block_len,
            "rollout_steps": rollout_steps,
            "data_path": data_path,
            "ic_seed": ic_seed,
            "sample_idx": sample_idx,
            "D": D,
            "noise_level": noise_level,
            "train_cols": train_cols,
        },
        ckpt_path,
    )
    result["weights_path"] = ckpt_path
    return result


def _col_runs(cols):
    """Split a sorted column-index array into its maximal runs of consecutive columns.

    `block_split` returns the train columns as one concatenated array; this recovers
    each contiguous time-block's absolute column indices so a rollout window never
    straddles a block boundary. Mirrors data_loader.contiguous_blocks but returns the
    indices (not the field slices), since the runner also needs them to slice forcing
    and the noise latent.
    """
    cols = np.asarray(cols, int)
    if cols.size == 0:
        return []
    breaks = np.where(np.diff(cols) != 1)[0] + 1
    return np.split(cols, breaks)


def _tikhonov_noise_init(u_coarse_noise, lam=0.1):
    """Verbatim from utils.noise_initialization: Tikhonov-smoothed noise estimate."""
    Lxc, Ltc = u_coarse_noise.shape
    m = Lxc
    DD = np.zeros((m, m))
    DD[0, :4] = [2, -5, 4, -1]
    DD[m - 1, m - 4 :] = [-1, 4, -5, 2]
    for i in range(1, m - 1):
        DD[i, i] = -2
        DD[i, i + 1] = 1
        DD[i, i - 1] = 1
    DD = DD.dot(DD)
    u_smooth = np.zeros((Lxc, Ltc))
    A = np.eye(m) + lam * DD.T.dot(DD)
    for t in range(Ltc):
        u_smooth[:, t] = np.linalg.solve(A, u_coarse_noise[:, t])
    return u_coarse_noise - u_smooth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-path",
        required=True,
        dest="data_path",
        help="ic_seed_*_nl_*.npz written by data_loader/generate_datasets.py",
    )
    ap.add_argument(
        "--sample-idx",
        type=int,
        required=True,
        dest="sample_idx",
        help="which sample (viscosity D) of the file to train on",
    )
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        dest="rollout_steps",
        help="training+scoring horizon; default reads config evaluate.rollout_steps",
    )
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--device",
        default=None,
        help="cuda|mps|cpu; default auto-detects (CUDA > MPS > CPU)",
    )
    ap.add_argument(
        "--weights-dir",
        default=None,
        dest="weights_dir",
        help=f"where to save the trained net; default {_DEFAULT_WEIGHTS_DIR}",
    )
    args = ap.parse_args()
    r = run(
        data_path=args.data_path,
        sample_idx=args.sample_idx,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        weights_dir=args.weights_dir,
    )
    print(json.dumps({k: v for k, v in r.items() if k != "loss_hist"}, indent=2))
