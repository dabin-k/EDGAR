import numpy as np
import matplotlib.pyplot as plt


TRACE_SENSORS = (10, 30, 50)  # three sensors spread along the row
BLOCKS_SHOWN = 2  # blocks laid side by side, separated by a band of NaN
GAP = 6  # width of that band, in columns


def _with_gap(panels):
    """Lay panels side by side, separated by `GAP` columns of NaN."""
    sep = np.full((panels[0].shape[0], GAP), np.nan)
    out = [panels[0]]
    for panel in panels[1:]:
        out += [sep, panel]
    return np.hstack(out)


def _mark_gaps(ax, offsets):
    """Draw the block boundary on a line axis, at the centre of each gap."""
    for off in offsets[1:]:
        ax.axvline(off - GAP / 2, color="k", lw=0.8)


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
    Plot observed sensor displacements against each program's autoregressive prediction.

    The first `BLOCKS_SHOWN` blocks are shown side by side, separated by a band of NaN.
    Blocks are not contiguous in time — a whole unshown block separates them — so the
    band is a reminder that nothing carries across it, not a short pause.

    Left column: heatmaps of sensors by time, all sharing one colour scale and one
    colourbar — the data on top, each program's rollout prediction below. Predictions are
    written into their true time columns of a full-width canvas, so every panel is
    directly comparable column by column; columns with no prediction (the lead-in, before
    the first start has enough history) stay NaN.
    Right column: for three example sensors, the residual (prediction minus data) over
    time at the first and last rollout step, so a model that is right one step ahead but
    drifts over the rollout is visibly different from one that is biased from the start.

    NaN renders black, which covers three cases at once: the separator band, the lead-in
    columns, and any prediction that diverged. Zero displacement is white, so a model that
    blows up and a model that collapses to predicting nothing look different.

    Programs are titled model_1, model_2, ... in the order given.

    Args:
        data: batched data dict with key 'x', shape
            (n_samples, n_blocks, n_sensors, block_len).
        programs: list of Program objects with .params and .compile_model().
        save_path: file path to save the figure.
        losses: per-program loss, defaults to program.program_losses.discover.final.
        params: per-program fitted params, defaults to program.params.
        evaluate_fn: harness evaluator; `preds, targets = evaluate_fn(model_fn, data, params)`
            runs the autoregressive rollout. The rollout length is read back from the
            output shape, so no rollout config is needed here.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if params is None:
        params = [p.params for p in programs]

    x = np.asarray(data["x"])
    sample = 0
    n_show = min(BLOCKS_SHOWN, x.shape[1])
    block_len = x.shape[-1]
    offsets = [b * (block_len + GAP) for b in range(n_show)]

    truth_blocks = [x[sample, b] for b in range(n_show)]
    truth = _with_gap(truth_blocks)
    vmax = float(np.nanmax(np.abs(truth)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("black")
    kw = dict(aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap=cmap)

    n_rows = 1 + len(programs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows), squeeze=False)

    im = axes[0, 0].imshow(truth, **kw)
    axes[0, 0].set(
        title=f"data ({n_show} blocks, NOT contiguous in time)",
        ylabel="sensor",
        xlabel="time",
    )
    for c in TRACE_SENSORS:
        axes[0, 0].axhline(c, color="k", lw=0.6, ls=":")
    fig.colorbar(im, ax=axes[0, 0], label="displacement")

    ax = axes[0, 1]
    for i, c in enumerate(TRACE_SENSORS):
        for b, off in enumerate(offsets):
            ax.plot(
                off + np.arange(block_len),
                truth_blocks[b][c],
                color=f"C{i}",
                label=f"sensor {c}" if b == 0 else None,
            )
    _mark_gaps(ax, offsets)
    ax.set(title="displacement of the three marked sensors", xlabel="time",
           ylabel="displacement")
    ax.legend(fontsize=7)

    for row, program in enumerate(programs, start=1):
        model_fn = program.compile_model()
        preds, targets = evaluate_fn(model_fn, data, params[row - 1])
        preds = np.asarray(preds)[sample]  # (n_blocks, n_starts, rollout_steps, n_sensors)
        targets = np.asarray(targets)[sample]
        residual = preds - targets

        # rollout geometry, read back from the evaluator's output shape: start s needs
        # history [s-m+1 .. s] and targets [s+1 .. s+h], so the last prediction of the
        # rollout lands on the block's final column and the first start is m-1.
        n_starts, rollout_steps = preds.shape[1], preds.shape[2]
        starts = (block_len - rollout_steps - n_starts) + np.arange(n_starts)

        label = f"model_{row}"

        panels = []
        for b in range(n_show):
            canvas = np.full((truth_blocks[b].shape[0], block_len), np.nan)
            canvas[:, starts + rollout_steps] = preds[b, :, rollout_steps - 1].T
            panels.append(canvas)

        im = axes[row, 0].imshow(_with_gap(panels), **kw)
        axes[row, 0].set(
            title=f"{label}: prediction, {rollout_steps} steps ahead"
            f"  (loss {losses[row - 1]:.4g})",
            ylabel="sensor",
            xlabel="time",
        )
        fig.colorbar(im, ax=axes[row, 0], label="displacement")

        ax = axes[row, 1]
        ax.axhline(0, color="k", lw=0.6)
        for i, c in enumerate(TRACE_SENSORS):
            for b, off in enumerate(offsets):
                first = b == 0
                ax.plot(off + starts + 1, residual[b, :, 0, c], color=f"C{i}", lw=1.0,
                        label=f"sensor {c}, 1 step" if first else None)
                ax.plot(off + starts + rollout_steps,
                        residual[b, :, rollout_steps - 1, c],
                        color=f"C{i}", lw=1.4, ls="--",
                        label=f"sensor {c}, {rollout_steps} steps" if first else None)
        _mark_gaps(ax, offsets)
        ax.set(title=f"{label}: residual (prediction - data)", xlabel="time",
               ylabel="residual")
        ax.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
