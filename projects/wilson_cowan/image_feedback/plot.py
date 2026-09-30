"""Full free-rollout prediction vs observed values. These images are fed as diagnostic input to the LLM.

The rollout shown is always ``pred_y_full_rollout`` from the project's ``apply_model`` — seeded
from the first observation (and the ``s0_`` hidden state) and then fed its own E/I predictions,
with only the stimulus supplied — whatever the training objective (A/B/FULL). It is the one view
that exposes the model's own dynamics (drift, missing rebounds, instability) that teacher-forced
or short-horizon predictions hide.
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
    rng: np.random.Generator | None = None,
    max_show: int = 6,
):
    """Data vs full free rollout for ``max_show`` randomly drawn (sample, stim condition) pairs.

    One row per pair; E panel (left) and I panel (right). x-axis: time from stim onset (ms).
    """
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    # plot.py is loaded via exec(), so __file__ is unavailable — walk up from the
    # save_path to find the repo root and import the project's apply_model (the same
    # apply_model_fn the scorer uses, TaskSpec.apply_model_fn).
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
    from projects.wilson_cowan.data_loader.load_data import (  # noqa: E402
        apply_model as apply_model_fn,
    )

    if rng is None:
        rng = np.random.default_rng(0)
    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    target_y = np.asarray(data["target_y"])   # (n, n_stim, T, 2), last axis (E, I)
    stim_E = np.asarray(data["stim_E"])       # (n, n_stim, T)
    stim_I = np.asarray(data["stim_I"])
    time_ms = np.asarray(data["time"])[0] * 1000.0   # (T,) shared grid, t=0 at stim onset
    n_samples, n_stim, T, _ = target_y.shape

    # Pick the best program that actually compiles and has params: lowest loss.
    candidates = []
    for j, p in enumerate(programs):
        if params[j] is None:
            continue
        try:
            fn = p.compile_model()
        except Exception:
            continue
        loss_j = losses[j] if (losses[j] is not None) else np.inf
        candidates.append((loss_j, j, fn))
    if not candidates:
        raise RuntimeError("no program could be compiled for the rollout plot")
    _, best_j, model_fn = min(candidates, key=lambda t: t[0])

    # Randomly draw distinct (sample, condition) pairs, then run apply_model on just those:
    # each pair becomes its own "sample" (its sample's params) with a single condition.
    n_show = int(min(max_show, n_samples * n_stim))
    flat = np.sort(rng.choice(n_samples * n_stim, n_show, replace=False))
    s_idx, c_idx = np.divmod(flat, n_stim)

    show_data = {
        "target_y": jnp.asarray(target_y[s_idx, c_idx][:, None]),   # (n_show, 1, T, 2)
        "stim_E": jnp.asarray(stim_E[s_idx, c_idx][:, None]),       # (n_show, 1, T)
        "stim_I": jnp.asarray(stim_I[s_idx, c_idx][:, None]),
    }
    show_params = {k: jnp.asarray(np.asarray(v)[s_idx]) for k, v in params[best_j].items()}
    out = apply_model_fn(model_fn, show_data, show_params)
    pred = np.asarray(out["pred_y_full_rollout"])[:, 0]           # (n_show, T-1, 2): t = 1..T-1
    obs = target_y[s_idx, c_idx]                                   # (n_show, T, 2)
    rollout_mse = np.mean((pred - obs[:, 1:]) ** 2, axis=(1, 2))   # (n_show,)

    loss_str = f"{losses[best_j]:.4f}" if losses[best_j] is not None else "n/a"
    model_name = f"{program_names[best_j]}: objective loss={loss_str}"

    def _runs(mask):
        """(start, end_exclusive) index pairs for each contiguous True run in mask."""
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[brk + 1]))
        ends = np.concatenate((idx[brk], [idx[-1]])) + 1
        return list(zip(starts.tolist(), ends.tolist()))

    dt_ms = float(time_ms[1] - time_ms[0]) if T > 1 else 1.0
    fig, axes = plt.subplots(
        n_show, 2, figsize=(11, 2.2 * n_show + 0.5), squeeze=False,
    )
    for row in range(n_show):
        s, c = int(s_idx[row]), int(c_idx[row])
        # Pulse windows for this condition — shaded in BOTH panels:
        # faint red where the E pulse is on, faint blue where the I pulse is on.
        e_spans = _runs(stim_E[s, c] > 0.5)
        i_spans = _runs(stim_I[s, c] > 0.5)
        for ci, (chan, mcolor) in enumerate([("E", "tab:red"), ("I", "tab:blue")]):
            ax = axes[row, ci]
            for a, b in e_spans:
                ax.axvspan(time_ms[a], time_ms[b - 1] + dt_ms, color="tab:red", alpha=0.12, lw=0)
            for a, b in i_spans:
                ax.axvspan(time_ms[a], time_ms[b - 1] + dt_ms, color="tab:blue", alpha=0.12, lw=0)
            ax.plot(time_ms, obs[row, :, ci], color="0.35", lw=0.9, label="data")
            ax.plot(time_ms[1:], pred[row, :, ci], color=mcolor, lw=0.9,
                    alpha=0.9, label="model (free rollout)")
            ax.set_title(
                f"sample {s}, cond {c} — {chan}   (rollout MSE={rollout_mse[row]:.4f})",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)
            y_max = float(np.max(obs[row, :, ci]))
            ax.set_ylim(-0.1, y_max * 1.1)
            if ci == 0:
                ax.set_ylabel("activity", fontsize=8)
            if row == n_show - 1:
                ax.set_xlabel("time from stim onset (ms)", fontsize=8)
            if row == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Data vs full free rollout  |  {model_name}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
