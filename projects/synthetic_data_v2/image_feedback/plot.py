import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
from evaluate import M, evaluate  # noqa: E402


N_TRACE_CELLS = 4


def _trace_cells(programs, n_samples):
    """Choose which cells to show: the same ones for every model of a generation, a
    different set each generation, and the same sequence on every rerun of the run."""
    try:
        seed = int(programs[0].birth.generation)
    except (AttributeError, IndexError, TypeError, ValueError):
        seed = 0
    seed = max(seed, 0)  # seed programs are born at generation -1
    n = min(N_TRACE_CELLS, n_samples)
    return np.sort(np.random.default_rng(seed).choice(n_samples, n, replace=False))


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
):
    """
    Plot each cell's observed activity against every program's one-step prediction.

    One row per cell. Left: the true trace over one block, with each program's
    prediction of the next step overlaid; the leading `M` timepoints have no window
    behind them and are never predicted, so they are shaded. Right: the residual
    (prediction minus data) for the same cell, which is where a mediocre and a good
    model actually differ — their traces look alike, their residuals do not.

    The cells shown are re-drawn each generation, so the models are compared on a fresh
    sample of the population rather than on four cells that might happen to be easy.

    Programs are titled model_1, model_2, ... in the order given.

    Args:
        data: X_disc_test dict with key 'x', shape (n_samples, n_blocks, block_len).
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

    x = np.asarray(data["x"])
    n_samples = x.shape[0]
    cells = _trace_cells(programs, n_samples)
    block = 0

    # every program's predictions, once
    preds = []
    for program, prm in zip(programs, params):
        p, _ = evaluate(program.compile_model(), data, prm)
        preds.append(np.asarray(p)[:, block])  # (n_samples, block_len - M)

    block_len = x.shape[2]
    t_pred = np.arange(M, block_len)

    fig, axes = plt.subplots(
        len(cells), 2, figsize=(13, 2.5 * len(cells)), squeeze=False
    )

    for row, cell in enumerate(cells):
        truth = x[cell, block]

        ax = axes[row, 0]
        ax.axvspan(0, M, color="0.9", zorder=0)
        ax.text(
            M / 2,
            1.0,
            "not predicted",
            transform=ax.get_xaxis_transform(),  # x in data coords, y in axes coords
            ha="center",
            va="bottom",
            fontsize=6,
            color="0.4",
        )
        ax.plot(np.arange(block_len), truth, color="k", lw=1.6, label="data", zorder=3)
        for i, pred in enumerate(preds):
            ax.plot(
                t_pred,
                pred[cell],
                color=f"C{i}",
                lw=1.0,
                alpha=0.9,
                label=f"model_{i + 1}",
            )
        ax.set(title=f"cell {cell}: activity", xlabel="time", ylabel="activity")
        ax.legend(fontsize=6, ncol=len(preds) + 1)

        ax = axes[row, 1]
        ax.axhline(0, color="k", lw=0.6)
        for i, pred in enumerate(preds):
            ax.plot(
                t_pred,
                pred[cell] - truth[M:],
                color=f"C{i}",
                lw=1.0,
                label=f"model_{i + 1}",
            )
        ax.set(
            title=f"cell {cell}: residual (prediction - data)",
            xlabel="time",
            ylabel="residual",
        )
        ax.legend(fontsize=6, ncol=len(preds))

    titles = "   ".join(
        f"model_{i + 1}: loss {loss:.4g}" for i, loss in enumerate(losses)
    )
    fig.suptitle(titles, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
