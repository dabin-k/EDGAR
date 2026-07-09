import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    max_show: int = 4,
    window: int = 800,
):
    """Visualise windowed point-process fits for a few neurons.

    Under the windowed contract each sample is a strictly-past window mapped to
    the next-bin count, so there is no contiguous series to slide over. Instead
    we evaluate every model on the neuron's anchor windows and plot, per neuron:

        left  : predicted intensity mu at each anchor (line per model), with the
                observed next-bin spikes drawn as a rug — anchor index is the
                x-axis (a strided sample of SWS time).
        right : calibration — anchors bucketed by predicted mu (deciles), mean
                predicted mu vs mean observed spike-count per bucket (points on
                y=x indicate a well-calibrated intensity).

    Args mirror the EDGAR plot_fn contract used by generate_program_fits:
        data: a windowed split dict with keys 'history' (n_neurons, A, W) and
              'target_y' (n_neurons, A).
        programs: list of Program objects (.compile_model(), .params, ...).
    """
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    history = np.asarray(data["history"])  # (n_neurons, A, W)
    target_y = np.asarray(data["target_y"])  # (n_neurons, A)
    n_neurons, A, _ = history.shape
    n_show = min(max_show, n_neurons)
    show_idx = np.linspace(0, n_neurons - 1, n_show).astype(int)
    A_show = int(min(window, A))

    model_fns = []
    for p in programs:
        try:
            model_fns.append(p.compile_model())
        except Exception:
            model_fns.append(None)

    def _mu_for(model_fn, ps_s, hist_s):
        # mu at every anchor for one neuron: model sees only a (W,) window.
        return np.asarray(jax.vmap(lambda w: model_fn(w, ps_s))(jnp.asarray(hist_s))).ravel()

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(n_show, 2, figsize=(13, 2.6 * n_show), squeeze=False)

    for row, s in enumerate(show_idx):
        ys = target_y[s]
        ax = axes[row, 0]
        mu_max = 1e-6
        for j, model_fn in enumerate(model_fns):
            if model_fn is None or params[j] is None:
                continue
            try:
                ps = {k: jnp.asarray(np.asarray(v)[s]) for k, v in params[j].items()}
                mu = _mu_for(model_fn, ps, history[s])
                ax.plot(
                    np.arange(A_show), mu[:A_show],
                    color=colors[j % len(colors)], lw=1.0, label=program_names[j],
                )
                mu_max = max(mu_max, float(np.nanmax(mu[:A_show])))
                # calibration on the right, over ALL anchors
                axr = axes[row, 1]
                edges = np.quantile(mu, np.linspace(0, 1, 11))
                edges = np.unique(edges)
                if edges.size >= 3:
                    bucket = np.clip(np.digitize(mu, edges[1:-1]), 0, edges.size - 2)
                    pred = [mu[bucket == b].mean() for b in range(edges.size - 1)]
                    obs = [ys[bucket == b].mean() for b in range(edges.size - 1)]
                    axr.plot(pred, obs, "o-", ms=3, color=colors[j % len(colors)])
            except Exception:
                continue
        # observed spikes as a rug along the top so mu structure stays visible
        spk = np.nonzero(ys[:A_show] > 0)[0]
        top = mu_max * 1.15
        ax.vlines(spk, mu_max * 1.05, top, color="0.5", lw=0.5, label="spikes")
        ax.set_ylim(0, top * 1.05)
        ax.set_ylabel(f"neuron {s}\nmu (spikes/bin)")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"Predicted intensity mu at anchors (first {A_show})")
        if row == n_show - 1:
            ax.set_xlabel("anchor index")

        axr = axes[row, 1]
        lim = max(1e-3, float(np.nanmax(ys.mean()) * 3))
        axr.plot([0, lim], [0, lim], "k--", lw=0.6)
        axr.set_ylabel("mean observed count")
        if row == 0:
            axr.set_title("Calibration: predicted vs observed (deciles)")
        if row == n_show - 1:
            axr.set_xlabel("mean predicted mu")

    title = "  |  ".join(
        f"{program_names[j]}: loss={losses[j]:.3f}" if losses[j] is not None
        else f"{program_names[j]}: loss=n/a"
        for j in range(len(programs))
    )
    fig.suptitle("mPFC windowed point-process fits\n" + title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
