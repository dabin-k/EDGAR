"""Diagnostic plots for the Wilson-Cowan discovery task.

Overlays each program's one-step-ahead predicted (E, I) on the true observed
traces, for a few discover cells and one stimulus condition. This is the honest
picture of what the loss sees: teacher-forced one-step prediction, both
channels, per-channel-normalized MSE.

Signature matches the other projects' `plot_model_fits` so the EDGAR image-
feedback hook can call it unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import jax.numpy as jnp

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    max_show: int = 3,
    stim_index: int = 0,
    window: int = 0,
):
    """One-step predicted (E, I) vs true, for a few cells and one stim condition."""
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    # plot.py is loaded via exec(), so __file__ is unavailable — walk up from the
    # save_path to find the repo root and import the project's apply_model.
    save_p = Path(save_path).resolve()
    repo_root = save_p
    for _ in range(10):
        if (repo_root / "projects" / "wilson_cowan").is_dir():
            break
        if repo_root.parent == repo_root:
            raise RuntimeError(
                f"couldn't locate repo root walking up from {save_p}; "
                "expected projects/wilson_cowan/ somewhere above."
            )
        repo_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from projects.wilson_cowan.data_loader.load_data import apply_model  # noqa: E402

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    E = np.asarray(data["E"])                 # (n, n_stim, T)
    I = np.asarray(data["I"])
    n_samples, n_stim, T = E.shape
    si = int(min(max(stim_index, 0), n_stim - 1))
    T_show = T if window <= 0 else int(min(window, T))

    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)

    model_fns = []
    for p in programs:
        try:
            model_fns.append(p.compile_model())
        except Exception:
            model_fns.append(None)

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(n_show, 2, figsize=(12, 2.6 * n_show), squeeze=False)

    for row, s in enumerate(show_idx):
        for j, model_fn in enumerate(model_fns):
            if model_fn is None or params[j] is None:
                continue
            try:
                sample_data = {
                    "E": jnp.asarray(E[s:s + 1]),
                    "I": jnp.asarray(I[s:s + 1]),
                    "stim_E": jnp.asarray(np.asarray(data["stim_E"])[s:s + 1]),
                    "stim_I": jnp.asarray(np.asarray(data["stim_I"])[s:s + 1]),
                }
                sample_params = {
                    k: jnp.asarray(np.asarray(v)[s:s + 1]) for k, v in params[j].items()
                }
                out = np.asarray(apply_model(model_fn, sample_data, sample_params))
                # out: (1, n_stim, T-1, 2)  -> predicted (E, I) at t = 1..T-1
                for ci in range(2):
                    axes[row, ci].plot(
                        np.arange(1, T_show), out[0, si, :T_show - 1, ci],
                        color=colors[j % len(colors)], lw=0.9, label=program_names[j],
                    )
            except Exception:
                continue

        for ci, (chan, obs) in enumerate([("E", E), ("I", I)]):
            ax = axes[row, ci]
            ax.plot(np.arange(T_show), obs[s, si, :T_show],
                    "k-", lw=0.7, alpha=0.5, label="true")
            ax.set_ylabel(f"cell {s} — {chan}")
            if row == 0 and ci == 1:
                ax.legend(fontsize=7, loc="upper right")
            if row == n_show - 1:
                ax.set_xlabel("time bin")

    title_parts = []
    for j in range(len(programs)):
        loss_str = f"{losses[j]:.4f}" if losses[j] is not None else "n/a"
        title_parts.append(f"{program_names[j]}: loss={loss_str}")
    fig.suptitle(
        f"wilson_cowan one-step (E, I) vs true — stim condition {si}\n"
        + "  |  ".join(title_parts),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
