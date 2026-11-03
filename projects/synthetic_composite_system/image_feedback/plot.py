"""Model-fit feedback plots for the synthetic composite system (plant growth).

Shows parity plots (predicted vs observed growth) for two random plants, coloured
by light and by nitrogen, so that systematic misfit can be tied to the input that
drives it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# `plot.py` is exec'd from a source string, so relative imports don't work. `edgar
# run` is invoked from the repo root, so put it on the path and import the project's
# modules directly — predictions go through the same `apply_model` used in scoring.
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from projects.synthetic_composite_system.data_loader.load_data import (  # noqa: E402
    apply_model,
)

INPUTS = [("L", "Light (L)"), ("N", "Nitrogen (N)")]
MODEL_MARKERS = ["o", "^", "s", "D"]


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    n_show: int = 2,
    *,
    rng: np.random.Generator,
):
    """Plots predicted vs observed growth for `n_show` random plants.

    Args:
        data (dict): `X_disc_train` with "L", "N" and "growth_rate", each of shape
            (n_samples, n_trials).
        programs (list): `Program` objects providing `.compile_model()`, `.params`,
            `.sample_losses`, `.name` and `.program_losses.discover.final`.
        save_path (str): File path to save the figure to.
        losses (list | None): Overall loss per program; defaults to
            `program.program_losses.discover.final`.
        sample_losses (list | None): Per-sample losses per program, each of shape
            (n_samples,); defaults to `program.sample_losses`.
        program_names (list | None): Legend names; defaults to `program.name`.
        params (list | None): Per-sample parameter dicts per program; defaults to
            `program.params`.
        n_show (int): Number of plants to plot.
        rng (np.random.Generator): Random generator used to choose the plants.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if sample_losses is None:
        sample_losses = [p.sample_losses for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    n_samples = np.asarray(data["growth_rate"]).shape[0]
    show = np.sort(rng.choice(n_samples, size=min(n_show, n_samples), replace=False))
    data_show = {
        k: jnp.asarray(np.asarray(data[k])[show]) for k in ("L", "N", "growth_rate")
    }

    preds = {}
    for program, name, loss, prm in zip(programs, program_names, losses, params):
        prm_show = {k: jnp.asarray(np.asarray(v)[show]) for k, v in prm.items()}
        label = f"{name} (loss={loss:.3g})" if loss is not None else name
        preds[label] = np.asarray(
            apply_model(program.compile_model(), data_show, prm_show)
        )

    sample_labels = []
    for s in show:
        s_losses = [f"{sl[s]:.3g}" for sl in sample_losses if sl is not None]
        suffix = f" (losses: {', '.join(s_losses)})" if s_losses else ""
        sample_labels.append(f"Sample {s}{suffix}")

    plot_parity(
        np.asarray(data_show["L"]),
        np.asarray(data_show["N"]),
        np.asarray(data_show["growth_rate"]),
        preds,
        sample_labels,
        save_path,
        title="Predicted vs observed growth rate",
    )


def plot_parity(L, N, growth, preds, sample_labels, save_path, title=""):
    """Predicted vs observed growth, coloured by L (left) and by N (right).

    Points on the faint y = x line are fitted perfectly; colour shows which input
    values the off-line points share. Models are distinguished by marker shape.
    """
    inputs = {"L": L, "N": N}
    fig, axes = plt.subplots(
        len(sample_labels), 2, figsize=(11, 4.8 * len(sample_labels)), squeeze=False
    )
    for r in range(len(sample_labels)):
        all_vals = np.concatenate([growth[r], *(p[r] for p in preds.values())])
        pad = 0.05 * (all_vals.max() - all_vals.min())
        lims = (all_vals.min() - pad, all_vals.max() + pad)
        for c, (key, x_label) in enumerate(INPUTS):
            ax = axes[r, c]
            ax.plot(lims, lims, ls=":", color="0.5", lw=1, alpha=0.6, zorder=0)
            colour = inputs[key][r]
            norm = plt.Normalize(colour.min(), colour.max())
            for j, (name, pred) in enumerate(preds.items()):
                sc = ax.scatter(
                    growth[r],
                    pred[r],
                    c=colour,
                    cmap="viridis",
                    norm=norm,
                    s=30,
                    marker=MODEL_MARKERS[j % len(MODEL_MARKERS)],
                    edgecolors="k",
                    linewidths=0.3,
                    alpha=0.85,
                )
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.set_xlabel("Observed growth rate")
            ax.set_ylabel("Predicted growth rate")
            ax.set_title(f"{sample_labels[r]}\ncoloured by {x_label}", fontsize=10)
            fig.colorbar(sc, ax=ax, label=x_label)
    for j, name in enumerate(preds):
        axes[0, 0].scatter(
            [],
            [],
            marker=MODEL_MARKERS[j % len(MODEL_MARKERS)],
            c="0.6",
            edgecolors="k",
            linewidths=0.3,
            label=name,
        )
    axes[0, 0].legend(fontsize=9, loc="upper left")
    fig.suptitle(title or "Predicted vs observed growth rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
