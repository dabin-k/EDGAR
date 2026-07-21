import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
from evaluate import MAX_LENGTH, ROLLOUT_STEPS, evaluate  # noqa: E402


TRACE_SENSORS = (10, 30, 50)  # three sensors spread along the row


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
    Plot observed sensor displacements against each program's autoregressive prediction.

    Left column: heatmaps of one block, sensors by time, all sharing one colour scale
    and one colourbar — the data on top, each program's rollout prediction below.
    Right column: for three example sensors, the residual (prediction minus data) over
    time at the first and last rollout step, so a model that is right one step ahead but
    drifts over the rollout is visibly different from one that is biased from the start.

    Programs are titled model_1, model_2, ... in the order given.

    Args:
        data: X_disc_train / X_eval dict with key 'x', shape
            (n_samples, n_blocks, n_sensors, block_len).
        programs: list of Program objects with .params and .compile_model().
        save_path: file path to save the figure.
        losses: per-program loss, defaults to program.program_losses.discover.final.
        params: per-program fitted params, defaults to program.params.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if params is None:
        params = [p.params for p in programs]

    sample, block = 0, 0
    truth = np.asarray(data["x"])[sample, block]  # (n_sensors, block_len)
    vmax = float(np.abs(truth).max())
    kw = dict(aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")

    # target time of prediction (start s, horizon h) is s + 1 + h
    starts = np.arange(MAX_LENGTH - 1, truth.shape[1] - ROLLOUT_STEPS)
    t_first = starts + 1
    t_last = starts + ROLLOUT_STEPS

    n_rows = 1 + len(programs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows), squeeze=False)

    im = axes[0, 0].imshow(truth, **kw)
    axes[0, 0].set(title="data (one block)", ylabel="sensor", xlabel="time")
    for c in TRACE_SENSORS:
        axes[0, 0].axhline(c, color="k", lw=0.6, ls=":")
    fig.colorbar(im, ax=axes[0, 0], label="displacement")

    ax = axes[0, 1]
    for i, c in enumerate(TRACE_SENSORS):
        ax.plot(truth[c], color=f"C{i}", label=f"sensor {c}")
    ax.set(title="displacement of the three marked sensors", xlabel="time",
           ylabel="displacement")
    ax.legend(fontsize=7)

    for row, program in enumerate(programs, start=1):
        model_fn = program.compile_model()
        preds, targets = evaluate(model_fn, data, params[row - 1])
        preds = np.asarray(preds)[sample, block]  # (n_starts, horizon, n_sensors)
        targets = np.asarray(targets)[sample, block]
        residual = preds - targets

        label = f"model_{row}"

        im = axes[row, 0].imshow(preds[:, ROLLOUT_STEPS - 1].T, **kw)
        axes[row, 0].set(
            title=f"{label}: prediction, {ROLLOUT_STEPS} steps ahead"
            f"  (loss {losses[row - 1]:.4g})",
            ylabel="sensor",
            xlabel="time",
        )
        fig.colorbar(im, ax=axes[row, 0], label="displacement")

        ax = axes[row, 1]
        ax.axhline(0, color="k", lw=0.6)
        for i, c in enumerate(TRACE_SENSORS):
            ax.plot(t_first, residual[:, 0, c], color=f"C{i}", lw=1.0,
                    label=f"sensor {c}, 1 step")
            ax.plot(t_last, residual[:, ROLLOUT_STEPS - 1, c], color=f"C{i}", lw=1.4,
                    ls="--", label=f"sensor {c}, {ROLLOUT_STEPS} steps")
        ax.set(title=f"{label}: residual (prediction - data)", xlabel="time",
               ylabel="residual")
        ax.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
