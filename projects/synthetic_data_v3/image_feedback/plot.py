import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TRACE_CELLS = (4, 14, 24)  # three cells spread around the ring


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    *,
    evaluate_fn,
):
    """
    Plot observed population activity against each program's autoregressive prediction.

    Left column: heatmaps of one recording block, cells by time, all sharing one colour
    scale and one colourbar — the data on top, each program's rollout prediction below.
    Right column: for three example cells, the residual (prediction minus data) over
    time at the first and last rollout step, so a model that is right one step ahead but
    drifts over the rollout is visibly different from one that is biased from the start.

    Programs are titled model_1, model_2, ... in the order given.
    
    Args:
        data: X_disc_train dict with key 'x', shape (n_samples, n_blocks, n_cells, block_len).
        programs: list of Program objects with .params and .compile_model().
        save_path: file path to save the figure.
        losses: per-program loss, defaults to program.program_losses.discover.final.
        sample_losses: per-program array over samples, defaults to program.sample_losses.
        program_names: unused; programs are labelled by position.
        params: per-program fitted params, defaults to program.params.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if params is None:
        params = [p.params for p in programs]

    sample, block = 0, 0
    truth = np.asarray(data["x"])[sample, block]  # (n_cells, block_len)
    vmax = float(np.abs(truth).max())
    kw = dict(aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")

    # target time of prediction (start s, horizon h) is s + 1 + h
    n_rows = 1 + len(programs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows), squeeze=False)

    im = axes[0, 0].imshow(truth, **kw)
    axes[0, 0].set(title="data (one recording block)", ylabel="cell", xlabel="time")
    for c in TRACE_CELLS:
        axes[0, 0].axhline(c, color="k", lw=0.6, ls=":")
    fig.colorbar(im, ax=axes[0, 0], label="activity")

    ax = axes[0, 1]
    for i, c in enumerate(TRACE_CELLS):
        ax.plot(truth[c], color=f"C{i}", label=f"cell {c}")
    ax.set(title="activity of the three marked cells", xlabel="time", ylabel="activity")
    ax.legend(fontsize=7)

    for row, program in enumerate(programs, start=1):
        model_fn = program.compile_model()
        preds, targets = evaluate_fn(model_fn, data, params[row - 1])
        preds = np.asarray(preds)[sample, block]  # (n_starts, horizon, n_cells)
        targets = np.asarray(targets)[sample, block]
        residual = preds - targets
        t_first = np.arange(preds.shape[0])
        t_last = np.arange(preds.shape[0])
        rollout_steps = preds.shape[1]

        label = f"model_{row}"

        im = axes[row, 0].imshow(preds[:, rollout_steps - 1].T, **kw)
        axes[row, 0].set(
            title=f"{label}: prediction, {rollout_steps} steps ahead"
            f"  (loss {losses[row - 1]:.4g})",
            ylabel="cell",
            xlabel="time",
        )
        fig.colorbar(im, ax=axes[row, 0], label="activity")

        ax = axes[row, 1]
        ax.axhline(0, color="k", lw=0.6)
        for i, c in enumerate(TRACE_CELLS):
            ax.plot(
                t_first,
                residual[:, 0, c],
                color=f"C{i}",
                lw=1.0,
                label=f"cell {c}, 1 step",
            )
            ax.plot(
                t_last,
                residual[:, rollout_steps - 1, c],
                color=f"C{i}",
                lw=1.4,
                ls="--",
                label=f"cell {c}, {rollout_steps} steps",
            )
        ax.set(
            title=f"{label}: residual (prediction - data)",
            xlabel="time",
            ylabel="residual",
        )
        ax.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
