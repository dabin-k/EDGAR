"""Full free-rollout prediction vs observed values. These images are fed as diagnostic input to the LLM.

All parent programs are overlaid (best = red/blue, next = orange/green). The rollout shown is always ``pred_y_full_rollout`` from the project's ``apply_model`` — seeded
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
    max_show: int = 5,
):
    """Data vs full free rollout for ``max_show`` randomly drawn (sample, stim condition) pairs.

    One row per pair; E panel (left) and I panel (right). x-axis: time from stim onset (ms).
    On real single-trial data (``mask``/``cond_id`` present) only real trials are drawn, and the
    noisy trial (grey dots) is overlaid with the mean of that sample's trials of the same
    condition (black line) — a single 1 ms trial alone is unreadable.
    Every program that compiles and has params is overlaid in each panel, ranked by loss: the
    best (lowest-loss) model is drawn in red (E) / blue (I), the next in orange (E) / green (I).
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
    mask = np.asarray(data["mask"]) if "mask" in data else np.ones((n_samples, n_stim))
    cond_id = np.asarray(data["cond_id"]) if "cond_id" in data else None

    # Keep every program that compiles and has params, ranked best (lowest loss) first.
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
    candidates.sort(key=lambda t: t[0])
    # (E colour, I colour) per rank: strong red/blue for the best model, orange/green for the next.
    palette = [("tab:red", "tab:blue"), ("tab:orange", "tab:green"), ("tab:purple", "tab:brown")]

    # Randomly draw distinct (sample, condition) pairs, then run apply_model on just those:
    # each pair becomes its own "sample" (its sample's params) with a single condition.
    real_flat = np.flatnonzero(mask.reshape(-1) > 0)
    n_show = int(min(max_show, real_flat.size))
    flat = np.sort(rng.choice(real_flat, n_show, replace=False))
    s_idx, c_idx = np.divmod(flat, n_stim)

    show_data = {
        "target_y": jnp.asarray(target_y[s_idx, c_idx][:, None]),   # (n_show, 1, T, 2)
        "stim_E": jnp.asarray(stim_E[s_idx, c_idx][:, None]),       # (n_show, 1, T)
        "stim_I": jnp.asarray(stim_I[s_idx, c_idx][:, None]),
        "time": jnp.asarray(np.asarray(data["time"])[s_idx]),       # (n_show, T) seconds
    }
    obs = target_y[s_idx, c_idx]                                   # (n_show, T, 2)
    if cond_id is not None:
        # Mean over the same sample's real trials of the same condition, (n_show, T, 2).
        cond_mean = np.stack([
            target_y[s][(cond_id[s] == cond_id[s, c]) & (mask[s] > 0)].mean(axis=0)
            for s, c in zip(s_idx, c_idx)
        ])
    else:
        cond_mean = None

    # One free rollout per ranked program, each with its own per-sample params.
    models = []   # (name, loss_str, pred (n_show, T-1, 2), rollout_mse (n_show,), colours)
    for rank, (loss_j, j, fn) in enumerate(candidates):
        show_params = {k: jnp.asarray(np.asarray(v)[s_idx]) for k, v in params[j].items()}
        out = apply_model_fn(fn, show_data, show_params)
        pred = np.asarray(out["pred_y_full_rollout"])[:, 0]       # (n_show, T-1, 2): t = 1..T-1
        rollout_mse = np.mean((pred - obs[:, 1:]) ** 2, axis=(1, 2))
        loss_str = f"{loss_j:.4f}" if np.isfinite(loss_j) else "n/a"
        models.append((program_names[j], loss_str, pred, rollout_mse,
                       palette[min(rank, len(palette) - 1)]))

    def _runs(mask):
        """(start, end_exclusive) index pairs for each contiguous True run in mask."""
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[brk + 1]))
        ends = np.concatenate((idx[brk], [idx[-1]])) + 1
        return list(zip(starts.tolist(), ends.tolist()))

    dt_ms = float(time_ms[1] - time_ms[0]) if T > 1 else 10.0   # time_ms holds bin centres
    fig, axes = plt.subplots(
        n_show, 2, figsize=(14, 3.0 * n_show + 0.6), squeeze=False,
    )
    for row in range(n_show):
        s, c = int(s_idx[row]), int(c_idx[row])
        # Pulse windows for this condition — shaded in BOTH panels:
        # faint red where the E pulse is on, faint blue where the I pulse is on.
        e_spans = _runs(stim_E[s, c] > 0.5)
        i_spans = _runs(stim_I[s, c] > 0.5)
        for ci, chan in enumerate(("E", "I")):
            ax = axes[row, ci]
            for a, b in e_spans:
                ax.axvspan(time_ms[a] - dt_ms / 2, time_ms[b - 1] + dt_ms / 2, color="tab:red", alpha=0.12, lw=0)
            for a, b in i_spans:
                ax.axvspan(time_ms[a] - dt_ms / 2, time_ms[b - 1] + dt_ms / 2, color="tab:blue", alpha=0.12, lw=0)
            if cond_mean is None:
                ax.scatter(time_ms, obs[row, :, ci], color="k", alpha=0.5, linewidths=0, label="data")
            else:
                ax.scatter(time_ms, obs[row, :, ci], color="0.6", s=4, alpha=0.5, linewidths=0,
                           label="this trial")
                ax.plot(time_ms, cond_mean[row, :, ci], color="k", lw=1.2, label="condition mean")
            # Best model drawn last (on top); legend order stays best-first.
            for rank in reversed(range(len(models))):
                name, _, pred, rollout_mse, colours = models[rank]
                ax.plot(time_ms[1:], pred[row, :, ci], color=colours[ci], lw=1.1, alpha=0.9,
                        zorder=3 + len(models) - rank,
                        label=f"{name} (rollout MSE={rollout_mse[row]:.4f})")
            handles, labels = ax.get_legend_handles_labels()
            n_data = 1 if cond_mean is None else 2
            # data first, then models best -> worst
            order = list(range(n_data)) + list(range(len(handles) - 1, n_data - 1, -1))
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      fontsize=8, loc="upper left", framealpha=0.8)
            what = f"cond {c}" if cond_id is None else f"trial {c} (cond {cond_id[s, c]})"
            ax.set_title(f"sample {s}, {what} — {chan}", fontsize=10)
            ax.tick_params(labelsize=8)
            # Scale to the condition mean when present: single-trial spikes would flatten it.
            ref = obs[row, :, ci] if cond_mean is None else cond_mean[row, :, ci]
            y_max = max(float(np.max(ref)), 1e-6)
            ax.set_ylim(-0.1, y_max * 1.1)
            if ci == 0:
                ax.set_ylabel("activity", fontsize=9)
            if row == n_show - 1:
                ax.set_xlabel("time from stim onset (ms)", fontsize=9)

    model_summary = "   vs   ".join(f"{name}: objective loss={loss_str}"
                                   for name, loss_str, *_ in models)
    fig.suptitle(f"Data vs full free rollout  |  {model_summary}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
