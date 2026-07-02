import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from edgar.llm.code_loading import load_function_from_source


def _load_model(program):
    """Loads a program's callable model for plotting.

    Prefers the numpy source (`code.model`), which is present from the moment a program
    has code — unlike `code.model_jax`, which only exists after JAX translation. The live
    feedback-image call (`generate_feedback_image`) happens before a parent is translated,
    so `program.compile_model()` (JAX) would raise there; the numpy model avoids that.
    """
    src = program.code.model or program.code.model_jax
    fn = load_function_from_source(src, "model")
    if fn is None:
        raise ValueError(f"could not load model for program #{program.idx}")
    return fn


def plot_model_fits(
    data,
    parent_programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
):
    """Retinotopic contour maps in cortical space for a few example recordings.

    The cortex->visual-field map is a smooth scalar field over the cortical sheet, so the
    most legible diagnostic is to plot it *in cortical coordinates*: the panel axes are
    cortical x and y, and the visual-field value (azimuth or elevation) is shown as a
    field over that sheet. Each recording gets two panels — azimuth and elevation. The
    measured field is drawn as a filled contour background (`tricontourf`) and each parent
    model's predicted field is overlaid as iso-value contour *lines* on shared levels. A
    model whose contour lines hug the boundaries between the measured colour bands has the
    right map shape; lines that cut across the gradient, are rotated, or are spaced wrong
    are the structured error the LLM should react to.

    Because the data are scattered subsampled pixels (not a dense grid), contouring is
    done on a Delaunay triangulation of the cortical points (`matplotlib.tri`).

    Two call sites exist (`edgar/io/plotting.py`): `generate_feedback_image` calls with
    just (data, parents, save_path) for live LLM feedback; `generate_program_fits`
    additionally passes `losses`/`sample_losses`/`program_names`/`params` (e.g. init vs.
    final fit of one program) for the dashboard. All four default to the corresponding
    `Program` attributes when not given.

    Args:
        data: X_disc_train dict of JAX arrays. data['cortical_pos'] shape
            (n_recordings, n_pixels, 2) = (x, y), data['visual_field'] shape
            (n_recordings, n_pixels, 2) = (azimuth, elevation).
        parent_programs: list of Program objects, each with a loadable model.
        save_path: file path (not directory) to save the figure.
        losses: optional list of scalar loss values, one per program; defaults to
            program.program_losses.discover.final.
        sample_losses: optional list of per-recording loss arrays (shape
            (n_recordings,) or None), one per program; defaults to program.sample_losses.
        program_names: optional list of display names; defaults to program.name.
        params: optional list of per-recording param dicts; defaults to program.params.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in parent_programs]
    if sample_losses is None:
        sample_losses = [p.sample_losses for p in parent_programs]
    if program_names is None:
        program_names = [p.name for p in parent_programs]
    if params is None:
        params = [p.params for p in parent_programs]

    cortical_pos = np.asarray(data["cortical_pos"])  # (n_rec, n_pix, 2)
    visual_field = np.asarray(data["visual_field"])  # (n_rec, n_pix, 2)
    n_rec = cortical_pos.shape[0]

    n_show = min(3, n_rec)
    rec_indices = np.random.choice(n_rec, size=n_show, replace=False)
    # line_colours = ["tab:red", "lime", "tab:orange", "tab:purple"]
    line_colours = ["k", "white", "tab:red"]
    coord_names = ["azimuth", "elevation"]
    n_levels = 16

    model_fns = [_load_model(program) for program in parent_programs]

    fig, axes = plt.subplots(n_show, 2, figsize=(11, 5 * n_show), squeeze=False)

    for i, s in enumerate(rec_indices):
        xy = cortical_pos[s]  # (n_pix, 2)
        sample_data = {"cortical_pos": xy, "visual_field": visual_field[s]}
        y_obs = visual_field[s]  # (n_pix, 2)

        # Delaunay triangulation of the scattered cortical points (shared across panels).
        tri = Triangulation(xy[:, 0], xy[:, 1])

        preds = []
        for j, model_fn in enumerate(model_fns):
            params_s = {k: np.asarray(v[s]) for k, v in params[j].items()}
            preds.append(np.asarray(model_fn(sample_data, params_s)))  # (n_pix, 2)

        for c in range(2):  # azimuth, elevation
            ax = axes[i, c]
            obs_c = y_obs[:, c]
            levels = np.linspace(float(obs_c.min()), float(obs_c.max()), n_levels)

            cf = ax.tricontourf(tri, obs_c, levels=levels, cmap="viridis", alpha=0.85)
            cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"observed {coord_names[c]} (deg)", fontsize=8)

            handles = []
            for j in range(len(model_fns)):
                colour = line_colours[j % len(line_colours)]
                ax.tricontour(
                    tri,
                    preds[j][:, c],
                    levels=levels,
                    colors=colour,
                    linewidths=1.2,
                )
                sl = sample_losses[j][s] if sample_losses[j] is not None else None
                label = (
                    f"{program_names[j]} (loss={sl:.3f})"
                    if sl is not None
                    else program_names[j]
                )
                handles.append(plt.Line2D([], [], color=colour, lw=1.2, label=label))

            ax.set_title(f"Recording {s} — {coord_names[c]}")
            ax.set_xlabel("cortical x")
            ax.set_ylabel("cortical y")
            ax.set_aspect("equal")
            ax.legend(handles=handles, fontsize=8, loc="best")

    title_parts = [
        f"{program_names[j]}: loss={losses[j]:.4f}"
        if losses[j] is not None
        else f"{program_names[j]}: loss=n/a"
        for j in range(len(parent_programs))
    ]
    plt.suptitle(
        "Filled = observed field, lines = predicted iso-contours  |  "
        + "  |  ".join(title_parts),
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)
