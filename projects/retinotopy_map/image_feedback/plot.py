import numpy as np
import matplotlib.pyplot as plt

from edgar.llm.code_loading import load_function_from_source

# Targets, in the model's output-column order: (azimuth, elevation, log-magnification).
TARGET_NAMES = ["azimuth (deg)", "elevation (deg)", "log area-magnification"]


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


def _scatter(ax, xy, vals, cmap, vmin, vmax):
    """Raw per-pixel scatter (no triangulation/smoothing) coloured by `vals`.

    Square markers on the true cortical pixel coordinates so the granular, salt-and-pepper
    structure of the mouse map is shown honestly rather than blurred by interpolation.
    NaNs (e.g. border pixels with no magnification) are dropped.
    """
    m = np.isfinite(vals)
    return ax.scatter(
        xy[m, 0], xy[m, 1], c=vals[m], s=7, marker="s", linewidths=0,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )


def _tidy(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("cortical x", fontsize=8)
    ax.set_ylabel("cortical y", fontsize=8)
    ax.tick_params(labelsize=7)


def _resid_limit(resid_cols):
    """Symmetric colour limit for residual panels: robust 99th pct of |residual|."""
    allv = (
        np.concatenate([r[np.isfinite(r)] for r in resid_cols])
        if resid_cols
        else np.array([])
    )
    v = float(np.percentile(np.abs(allv), 99)) if allv.size else 1.0
    return v if v > 0 else 1.0


def plot_model_fits(
    data,
    parent_programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
):
    """Per-pixel residual maps of a retinotopy fit, in cortical space.

    For each shown recording and each target (azimuth, elevation, log area-magnification),
    the observed field and each model's *signed residual* (pred − obs) are drawn as a raw
    per-pixel scatter.

    The residual is the diagnostic the LLM should react to: on a symmetric diverging scale
    (red = model over-predicts, blue = under), the *location*, *magnitude*, and *sign* of
    the error are the colour directly. Coherent red/blue patches are structured
    error a better map form can fix; random speckle is the irreducible local disorder a
    smooth parametric map cannot (and should not try to) capture. The log-magnification row
    is shown because it is a scored target that discriminates a structured (dipole) map from
    a flat-magnification (affine) null, yet is otherwise invisible in az/el.

    Two recordings are shown — the worst- and best-fit (by the first model's per-recording
    loss) — so both a failure case and a success case are in view.

    Two call sites exist (`edgar/io/plotting.py`): `generate_feedback_image` calls with just
    (data, parents, save_path) for live LLM feedback; `generate_program_fits` additionally
    passes `losses`/`sample_losses`/`program_names`/`params` (e.g. init vs. final fit of one
    program) for the dashboard. All four default to the corresponding `Program` attributes.

    Args:
        data: X-split dict of arrays. data['cortical_pos'] shape (n_rec, n_pix, 2) = (x, y),
            data['visual_field'] shape (n_rec, n_pix, 2) = (azimuth, elevation), optional
            data['log_mag'] shape (n_rec, n_pix, 1) = observed log area-magnification.
        parent_programs: list of Program objects, each with a loadable model.
        save_path: file path (not directory) to save the figure.
        losses: optional list of scalar loss values, one per program; defaults to
            program.program_losses.discover.final.
        sample_losses: optional list of per-recording loss arrays (shape (n_rec,) or None),
            one per program; defaults to program.sample_losses. The first drives which
            recordings are shown.
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
    log_mag = np.asarray(data["log_mag"])[..., 0] if "log_mag" in data else None
    n_rec = cortical_pos.shape[0]
    n_p = len(parent_programs)
    model_fns = [_load_model(p) for p in parent_programs]

    # Show the worst- and best-fit recordings (by the first model's per-recording loss),
    # so a failure and a success case are both in view.
    ref = sample_losses[0]
    if ref is not None and n_rec > 1:
        order = np.argsort(np.asarray(ref))
        rec_indices = sorted({int(order[-1]), int(order[0])})
    else:
        rec_indices = list(range(min(2, n_rec)))
    n_show = len(rec_indices)

    n_t = None  # common target width, resolved from the first recording.
    fig = axes = None

    for ri, s in enumerate(rec_indices):
        xy = cortical_pos[s]
        sample_data = {"cortical_pos": xy, "visual_field": visual_field[s]}
        if log_mag is not None:
            sample_data["log_mag"] = log_mag[s][:, None]

        preds = []
        for j, fn in enumerate(model_fns):
            ps = {k: np.asarray(v[s]) for k, v in params[j].items()}
            preds.append(np.asarray(fn(sample_data, ps)))

        obs_cols = [visual_field[s][:, 0], visual_field[s][:, 1]]
        if log_mag is not None:
            obs_cols.append(log_mag[s])

        if n_t is None:
            n_t = min(len(obs_cols), min(p.shape[1] for p in preds))
            fig, axes = plt.subplots(
                n_show * n_t,
                1 + n_p,
                figsize=(4.2 * (1 + n_p), 3.6 * n_show * n_t),
                squeeze=False,
            )

        obs = np.stack(obs_cols[:n_t], axis=-1)
        preds = [p[:, :n_t] for p in preds]

        for t in range(n_t):
            row = ri * n_t + t
            o = obs[:, t]
            finite = o[np.isfinite(o)]
            vmin, vmax = (
                (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            )

            ax = axes[row][0]
            sc = _scatter(ax, xy, o, "viridis", vmin, vmax)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"rec {s} · observed {TARGET_NAMES[t]}", fontsize=9)
            _tidy(ax)

            resid_cols = [preds[j][:, t] - o for j in range(n_p)]
            rlim = _resid_limit(resid_cols)
            for j in range(n_p):
                ax = axes[row][1 + j]
                sc = _scatter(ax, xy, resid_cols[j], "RdBu_r", -rlim, rlim)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                sl = sample_losses[j][s] if sample_losses[j] is not None else None
                lbl = f"  (loss={sl:.3f})" if sl is not None else ""
                ax.set_title(f"{program_names[j]}{lbl}\nresidual (pred − obs)", fontsize=8)
                _tidy(ax)

    suptitle = "  |  ".join(
        f"{program_names[j]}: loss={losses[j]:.3f}"
        if losses[j] is not None
        else f"{program_names[j]}: n/a"
        for j in range(n_p)
    )
    fig.suptitle(
        "Residual maps (pred − obs) — red = model over-predicts, blue = under.  " + suptitle,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)
