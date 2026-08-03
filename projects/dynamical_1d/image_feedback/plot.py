import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pred_at_anchors(model_fn, ps, history):
    return np.asarray(
        jax.vmap(lambda w: model_fn(w, ps))(jnp.asarray(history))
    ).ravel()


def _free_run(model_fn, ps, seed_window, steps):
    w = np.asarray(seed_window, dtype=float).copy()
    preds = []
    for _ in range(steps):
        pred = float(np.asarray(model_fn(jnp.asarray(w), ps)))
        if not np.isfinite(pred):
            pred = float(w[-1])
        preds.append(pred)
        w = np.concatenate([w[1:], np.atleast_1d(pred)])
    return np.asarray(preds, dtype=float)


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    max_show: int = 4,
    window: int = 400,
):
    """One-step predictions vs targets (left) and free-run rollout (right)."""
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    history = np.asarray(data["history"])
    target_y = np.asarray(data["target_y"])
    persistence = history[:, :, -1]
    mean_baseline = np.mean(history, axis=-1)
    pers_mse = np.asarray(data.get("_persistence_mse", []))
    rollout_seed = data.get("_rollout_seed")
    rollout_true = data.get("_rollout_true")

    n_samples, A, _ = history.shape
    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)
    A_show = int(min(window, A))

    model_fns = []
    for p in programs:
        try:
            model_fns.append(p.compile_model())
        except Exception:
            model_fns.append(None)

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(n_show, 2, figsize=(13, 2.8 * n_show), squeeze=False)

    for row, s in enumerate(show_idx):
        ys = target_y[s]
        pers = persistence[s]
        mean_pred = np.full_like(ys, mean_baseline[s].mean())
        ax = axes[row, 0]

        for j, model_fn in enumerate(model_fns):
            if model_fn is None or params[j] is None:
                continue
            try:
                ps = {k: jnp.asarray(np.asarray(v)[s]) for k, v in params[j].items()}
                pred = _pred_at_anchors(model_fn, ps, history[s])[:A_show]
                ax.plot(
                    np.arange(A_show),
                    pred,
                    color=colors[j % len(colors)],
                    lw=1.0,
                    label=program_names[j],
                )
            except Exception:
                continue

        ax.plot(np.arange(A_show), ys[:A_show], "k-", lw=0.8, alpha=0.5, label="true")
        ax.plot(np.arange(A_show), pers[:A_show], "--", color="0.5", lw=0.8, label="persist")
        ax.plot(np.arange(A_show), mean_pred[:A_show], ":", color="0.65", lw=0.8, label="mean")
        ax.set_ylabel(f"traj {s}")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"Next-value at anchors (first {A_show})")
        if row == n_show - 1:
            ax.set_xlabel("anchor index")

        axr = axes[row, 1]
        if rollout_seed is not None and rollout_true is not None:
            true_roll = np.asarray(rollout_true[s])
            H_roll = true_roll.shape[0]
            axr.plot(np.arange(H_roll), true_roll, "k-", lw=1.0, alpha=0.6, label="true")
            for j, model_fn in enumerate(model_fns):
                if model_fn is None or params[j] is None:
                    continue
                try:
                    ps = {k: jnp.asarray(np.asarray(v)[s]) for k, v in params[j].items()}
                    roll = _free_run(model_fn, ps, rollout_seed[s], H_roll)
                    roll = np.nan_to_num(roll, nan=np.nan, posinf=np.nan, neginf=np.nan)
                    axr.plot(
                        np.arange(H_roll),
                        roll,
                        color=colors[j % len(colors)],
                        lw=1.0,
                        label=program_names[j],
                    )
                except Exception:
                    continue
            if row == 0:
                axr.legend(fontsize=7, loc="upper right")
                axr.set_title(f"Free-run rollout ({H_roll} steps)")
            if row == n_show - 1:
                axr.set_xlabel("rollout step")

    title_parts = []
    for j in range(len(programs)):
        loss_str = f"{losses[j]:.6f}" if losses[j] is not None else "n/a"
        skill = ""
        if pers_mse.size and sample_losses is not None and sample_losses[j] is not None:
            m_mse = float(np.mean(sample_losses[j]))
            p_mse = float(np.mean(pers_mse))
            if p_mse > 0:
                skill = f" skill={m_mse / p_mse:.2f}"
        title_parts.append(f"{program_names[j]}: loss={loss_str}{skill}")
    fig.suptitle("dynamical_1d windowed fits\n" + "  |  ".join(title_parts), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
