import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    title_prefix: str | None = None,
):
    """
    Plot observed target responses and overlaid model predictions for 9 random
    stimulus angles, with binned per-angle MSE shown underneath each panel.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Expected keys:
        - 'stimulus': array of shape (n_samples, n_trials)
        - 'response': array of shape (n_samples, n_trials, n_cells)
        - 'signal': array of shape (n_samples, n_trials, n_cells) (tuning curves)
    programs : list
        List of Program objects.
    save_path : str
        Output path for the plot.
    losses : list[float], optional
        List of scalar losses for each program.
    sample_losses : list[np.ndarray], optional
        List of per-sample losses for each program.
    program_names : list[str], optional
        List of names for each program.
    params : list[dict], optional
        List of parameter dicts for each program.
    title_prefix : str, optional
        Prefix for the figure title.
    """
    if save_path == "":
        raise ValueError("plot_model_fits requires a non-empty save_path")

    # 1. Resolve arguments
    if program_names is None:
        program_names = [p.name for p in programs]
    if losses is None:
        losses = [p.program_losses.discover.final if hasattr(p, 'program_losses') else None for p in programs]
    if params is None:
        params = [p.params for p in programs]

    # 2. Handle the sample dimension (take the first sample for plotting)
    sample_idx = 0
    stims = np.asarray(data["stimulus"][sample_idx]).reshape(-1)
    actual_response = np.asarray(data["response"][sample_idx])
    sig = np.asarray(data["signal"][sample_idx]) # (n_trials, n_cells)
    
    # 3. Calculate Preferred Angles using Vector Concentration
    # Use nansum because data might be masked (NaNs)
    complex_sum = np.nansum(sig * np.exp(2j * stims)[:, np.newaxis], axis=0)
    pref_angles = (np.angle(complex_sum) / 2.0) % np.pi
    
    # Filter out cells that are entirely NaN in the response (masked out)
    valid_cells = np.where(~np.all(np.isnan(actual_response), axis=0))[0]
    if len(valid_cells) == 0:
        valid_cells = np.arange(actual_response.shape[1])
    
    # Sort only the valid cells
    pref_angles_valid = pref_angles[valid_cells]
    sort_idx = np.argsort(pref_angles_valid)
    
    final_cell_idx = valid_cells[sort_idx]
    sorted_pref_angles = pref_angles[final_cell_idx]
    sorted_actual = actual_response[:, final_cell_idx]

    # 4. Pick 9 random trials that actually have data
    valid_trials = np.where(~np.all(np.isnan(actual_response), axis=1))[0]
    if len(valid_trials) == 0:
        valid_trials = np.arange(len(stims))
    
    n_show = min(9, len(valid_trials))
    rng = np.random.default_rng(42)
    random_trials = rng.choice(valid_trials, size=n_show, replace=False)

    # 5. Compute predictions
    predictions = []
    binned_mse_losses = []
    
    n_bins = 60
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(sorted_pref_angles, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for i, program in enumerate(programs):
        model_fn = program.compile_model() if hasattr(program, 'compile_model') else program['model']
        p_dict = params[i]
        
        # Slice params for the first sample
        def _slice_leaf(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)) and x.ndim > 0 and x.shape[0] > sample_idx:
                 return x[sample_idx]
            return x
        plot_params = jax.tree_util.tree_map(_slice_leaf, p_dict)
        
        # Model expects data dict where arrays don't have the sample dim
        single_sample_data = {k: v[sample_idx] if hasattr(v, '__getitem__') and len(v) > sample_idx else v 
                              for k, v in data.items()}
        
        y_pred = np.asarray(model_fn(single_sample_data, plot_params))
        y_pred_sorted = y_pred[:, final_cell_idx] # Use the same filtered/sorted indices
        predictions.append(y_pred_sorted)

        # Binned MSE
        sq_err = (y_pred_sorted - sorted_actual) ** 2
        binned_mse = np.zeros((n_bins, sq_err.shape[0]))
        for t_idx in range(sq_err.shape[0]):
            for b_idx in range(n_bins):
                mask = bin_indices == b_idx
                if np.any(mask):
                    binned_mse[b_idx, t_idx] = np.nanmean(sq_err[t_idx, mask])
        binned_mse_losses.append(binned_mse)

    # 6. Plotting
    fig = plt.figure(figsize=(15, 15))
    outer_gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

    for i, trial_idx in enumerate(random_trials):
        row, col = divmod(i, 3)
        ax_slot = outer_gs[row, col]
        inner_gs = ax_slot.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        
        ax1 = fig.add_subplot(inner_gs[0])
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

        angle = stims[trial_idx]
        
        # Top: Population Response
        ax1.scatter(sorted_pref_angles, sorted_actual[trial_idx], color="tab:grey", alpha=0.3, s=10, label="Observed")
        for j, y_pred in enumerate(predictions):
            label = program_names[j] if j < len(program_names) else f"Model {j+1}"
            ax1.plot(sorted_pref_angles, y_pred[trial_idx], color=colors[j % len(colors)], label=label, linewidth=1.5)
        
        ax1.axvline(angle % np.pi, color="red", linestyle="--", alpha=0.5, label="Stimulus")
        ax1.set_title(f"Trial {trial_idx} (Angle: {angle:.2f} rad)")
        ax1.set_ylabel("Response")
        if i == 0: ax1.legend(fontsize=8)
        ax1.tick_params(labelbottom=False)

        # Bottom: Binned Error
        for j, binned_err in enumerate(binned_mse_losses):
            ax2.plot(bin_centers, binned_err[:, trial_idx], color=colors[j % len(colors)], linewidth=1.5)
        
        ax2.axvline(angle % np.pi, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("MSE", fontsize=8)
        ax2.set_xlabel("Pref. Orientation (rad)")

    # Construct Title
    summary_parts = []
    for i in range(len(programs)):
        name = program_names[i]
        loss = losses[i]
        if loss is not None:
            summary_parts.append(f"{name} loss: {loss:.4f}")
        else:
            summary_parts.append(f"{name} loss: n/a")
    
    title = " | ".join(summary_parts)
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    
    plt.suptitle(title, fontsize=16)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
