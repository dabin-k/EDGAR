# Use the EDGAR scoring logic to fit WC/WCS model to synthetic data
# Investigate different ways of normalizing the BZ015 data.
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score
from edgar.evolution.population import Population

# JAX GPU commands/flags
if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()


def parse_configs_arg(arg_str: str) -> list[tuple[int, int]]:
    """Parse rollout configurations from a string format.

    Args:
        arg_str: Comma-separated K:stride rollout configurations, e.g., "50:1,50:50"

    Returns:
        A list of (k, stride) tuples.
    """
    configs = []
    for pair in arg_str.split(","):
        if ":" in pair:
            k_str, stride_str = pair.split(":")
            configs.append((int(k_str), int(stride_str)))
    return configs


def generate_full_rollout(program, sample_idx: int, stim_E_design: np.ndarray, stim_I_design: np.ndarray, tmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a full model prediction rollout.

    Args:
        program: The fitted Program object.
        sample_idx: Index of the sample to plot.
        stim_E_design: Stimulus design for E channel (T,).
        stim_I_design: Stimulus design for I channel (T,).
        tmax: Length of simulation.

    Returns:
        tuple of (Et, It) arrays.
    """
    # Extract optimized parameter dict for this sample
    params = {k: float(np.asarray(v)[sample_idx]) for k, v in program.params.items()}
    
    E0, I0 = 1.0, 1.0
    
    # Initialize hidden carry states (S0 is steady-state value of I0)
    state = {}
    for k in program.params.keys():
        if k.startswith("s0_") and len(k) > 3:
            state_key = k.removeprefix("s0_")
            if state_key == 'S':
                state['S'] = I0
            else:
                state[state_key] = 0.0
                
    # Simulate step-by-step using program's model function
    model_fn = program.compile_model()
    
    Et = np.zeros(tmax)
    It = np.zeros(tmax)
    Et[0], It[0] = E0, I0
    
    # Convert state and params to JAX
    jax_state = {k: jnp.asarray(v) for k, v in state.items()}
    dyn_params_jax = {k: jnp.asarray(v) for k, v in params.items() if not k.startswith("s0_")}
    
    for t in range(1, tmax):
        y_prev = {
            'E_prev': jnp.asarray(Et[t-1]),
            'I_prev': jnp.asarray(It[t-1]),
            'stim_E_prev': jnp.asarray(stim_E_design[t-1]),
            'stim_I_prev': jnp.asarray(stim_I_design[t-1])
        }
        jax_state, mean = model_fn(jax_state, y_prev, dyn_params_jax)
        Et[t] = float(mean[0])
        It[t] = float(mean[1])
        
    return Et, It


def fit_population(configs_str: str, folder_name: str) -> None:
    """Run the parameter fitting loop for a list of rollout configurations.

    Saves the population and mapping config to:
    projects/wilson_cowan/sandbox/{folder_name}/
    """
    rollout_configs = parse_configs_arg(configs_str)
    if not rollout_configs:
        print("Error: No valid rollout configurations specified. Use e.g. --configs 50:1,50:50")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    config_path = repo_root / "projects" / "wilson_cowan" / "config.yaml"
    print(f"Loading config from: {config_path}")
    config = Config.from_yaml(config_path)

    # Set repeat cross-validation
    config.project_params["cv_type"] = "repeats"

    noise_levels = [0.05]
    data_base_path = str(repo_root / "projects" / "wilson_cowan" / "data_loader" / "synthetic")

    def load_datasets_for_config(noise_level: float, r_k: int, a_stride: int, objective: str):
        # Update config params explicitly
        config.project_params["objective"] = objective
        config.project_params["rollout_k"] = r_k
        config.project_params["anchor_stride"] = a_stride
        os.environ["EDGAR_WC_OBJECTIVE"] = objective
        os.environ["EDGAR_WC_ROLLOUT_K"] = str(r_k)
        os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(a_stride)

        # Load train (noisy or clean depending on noise_level)
        if noise_level == 0.0:
            config.io.data_path = f"{data_base_path}/noise_0.30/synthetic_data_clean.npz"
        else:
            config.io.data_path = f"{data_base_path}/noise_{noise_level:.2f}/synthetic_data_noisy.npz"
        
        spec = TaskSpec.from_config(config)
        X_discover, _ , _ = spec.load_data_fn(
            data_path=spec.io["data_path"], **spec.project_params
        )

        # Load eval (always clean)
        config.io.data_path = f"{data_base_path}/noise_0.30/synthetic_data_clean.npz"
        spec_eval = TaskSpec.from_config(config)
        _ , _ , X_eval = spec_eval.load_data_fn(
            data_path=spec_eval.io["data_path"], **spec_eval.project_params
        )

        return X_discover, X_eval, spec

    # Output directory
    save_dir = repo_root / "projects" / "wilson_cowan" / "sandbox" / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    pop_save_path = save_dir / "fitted_population.jsonl"
    config_save_path = save_dir / "config.json"

    population = Population()
    program_mapping = []
    program_idx = 0

    noise_levels.insert(0, 0.0) 

    # Sequential scoring of models
    for i, noise_level in enumerate(noise_levels):
        # 1. Fit Objective A (one-step)
        print(f"\n--- Fitting Objective A for noise level {noise_level:.2f} ---")
        X_discover, X_eval, spec = load_datasets_for_config(noise_level, 50, 50, "A")

        program = spec.seed_programs[0]
        program.code.model_jax = translate_to_jax(program.code.model)

        temp_pop = Population()
        temp_pop.add(program)
        score(
            temp_pop,
            X_discover,
            X_eval,
            spec.scoring,
            spec.loss_fn,
            split="discover",
            apply_model_fn=spec.apply_model_fn,
            rollout_fn=spec.rollout_fn,
        )
        population.add(program)
        program_mapping.append({
            "program_idx": program_idx,
            "noise_level": noise_level,
            "objective": "A"
        })
        program_idx += 1

        # 2. Fit Objective B for each rollout configuration
        for (k, stride) in rollout_configs:
            print(f"\n--- Fitting Objective B for noise level {noise_level:.2f} with K={k}, Stride={stride} ---")
            X_discover, X_eval, spec = load_datasets_for_config(noise_level, k, stride, "B")

            program = spec.seed_programs[0]
            program.code.model_jax = translate_to_jax(program.code.model)

            temp_pop = Population()
            temp_pop.add(program)
            score(
                temp_pop,
                X_discover,
                X_eval,
                spec.scoring,
                spec.loss_fn,
                split="discover",
                apply_model_fn=spec.apply_model_fn,
                rollout_fn=spec.rollout_fn,
            )
            population.add(program)
            program_mapping.append({
                "program_idx": program_idx,
                "noise_level": noise_level,
                "objective": "B",
                "k": k,
                "stride": stride
            })
            program_idx += 1

    print("\nScoring complete\n --------- \n ")

    # Serialize population
    population.save(str(pop_save_path))

    # Save mapping metadata
    metadata = {
        "rollout_configs": rollout_configs,
        "program_mapping": program_mapping
    }
    with open(config_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Population and config successfully saved to {save_dir}")


def plot_population(folder_path_str: str) -> None:
    """Load population and config from a folder, run evaluations, and generate comparison plots.

    Saves the plots inside the specified folder.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    folder_path = Path(folder_path_str).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Specified folder {folder_path} does not exist or is not a directory.")
        sys.exit(1)

    pop_file = folder_path / "fitted_population.jsonl"
    config_file = folder_path / "config.json"

    if not pop_file.exists():
        print(f"Error: {pop_file} not found in directory.")
        sys.exit(1)
    if not config_file.exists():
        print(f"Error: config.json not found in directory.")
        sys.exit(1)

    with open(config_file, "r") as f:
        metadata = json.load(f)
        rollout_configs = [tuple(cfg) for cfg in metadata.get("rollout_configs", [])]
        program_mapping = metadata.get("program_mapping", [])

    print(f"Loaded config metadata with rollout configurations: {rollout_configs}")

    # Set repeat cross-validation
    config_path = repo_root / "projects" / "wilson_cowan" / "config.yaml"
    config = Config.from_yaml(config_path)
    config.project_params["cv_type"] = "repeats"

    noise_levels = [0.05]
    data_base_path = str(repo_root / "projects" / "wilson_cowan" / "data_loader" / "synthetic")
    X_discovers = []

    # Load in data
    # Noisy data 
    for noise_level in noise_levels:
        config.io.data_path = f"{data_base_path}/noise_{noise_level:.2f}/synthetic_data_noisy.npz"
        spec = TaskSpec.from_config(config)
        X_discover, _ , _ = spec.load_data_fn(
            data_path=spec.io["data_path"], **spec.project_params
        )
        X_discovers.append(X_discover)
    
    # Clean data
    config.io.data_path = f"{data_base_path}/noise_0.30/synthetic_data_clean.npz"
    spec = TaskSpec.from_config(config)
    X_discover, _ , X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    X_discovers.insert(0, X_discover)
    noise_levels.insert(0, 0.0)

    # Load the population
    population = Population.load(str(pop_file))
    print(f"Loaded fitted population with {len(population)} programs from disk")

    sample_idx, stim_idx = 0, 0
    E_clean = X_discovers[0][0]['target_y'][sample_idx, stim_idx, :, 0]
    I_clean = X_discovers[0][0]['target_y'][sample_idx, stim_idx, :, 1]
    time_axis = np.asarray(X_discovers[0][0]["time"][sample_idx])

    colors = ["blue", "purple", "cyan", "magenta", "teal", "orange", "darkgreen"]

    # ─── Figure 1: Segment rollout plot (joined curves) ───
    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Wilson-Cowan Fitting vs Noise Level & Objectives (Sample {sample_idx}, Stim {stim_idx})", fontsize=16, y=0.98)

    for i, noise_level in enumerate(noise_levels):
        # Find Objective A program for this noise level
        mapping_a = next(m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "A")
        program_a = population[mapping_a["program_idx"]]
        E_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 0]
        I_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 1]

        # Build sample evaluation data
        sample_data = {
            "target_y": jnp.asarray(X_discovers[i][0]['target_y'][sample_idx:sample_idx+1]),
            "stim_E": jnp.asarray(X_discovers[i][0]['stim_E'][sample_idx:sample_idx+1]),
            "stim_I": jnp.asarray(X_discovers[i][0]['stim_I'][sample_idx:sample_idx+1]),
        }

        # Predict Objective A (one-step)
        params_a = {k: jnp.asarray(np.asarray(v)[sample_idx:sample_idx+1]) for k, v in program_a.params.items()}
        model_fn_a = program_a.compile_model()
        os.environ["EDGAR_WC_OBJECTIVE"] = "A"
        out_a = spec.apply_model_fn(model_fn_a, sample_data, params_a)
        pred_a = np.asarray(out_a["pred_y_1step"])[0, stim_idx]  # (T-1, 2)

        # Plot Excitatory Population (E)
        ax_E = axes[i, 0]
        ax_E.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_E.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6, label="Noisy Train Data" if i == 0 else "")
        ax_E.plot(time_axis[1:], pred_a[:, 0], color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")

        # Create E inset
        ax_E_inset = ax_E.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_E_inset.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5)
        ax_E_inset.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6)
        ax_E_inset.plot(time_axis[1:], pred_a[:, 0], color="red", linestyle="-", linewidth=1.5)

        # Plot Inhibitory Population (I)
        ax_I = axes[i, 1]
        ax_I.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_I.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6, label="Noisy Train Data" if i == 0 else "")
        ax_I.plot(time_axis[1:], pred_a[:, 1], color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")

        # Create I inset
        ax_I_inset = ax_I.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_I_inset.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5)
        ax_I_inset.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6)
        ax_I_inset.plot(time_axis[1:], pred_a[:, 1], color="red", linestyle="-", linewidth=1.5)

        mses_b_E = []
        mses_b_I = []

        # Predict and plot each rollout configuration
        for c_idx, (k, stride) in enumerate(rollout_configs):
            mapping_b = next(m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "B" and m["k"] == k and m["stride"] == stride)
            program_b = population[mapping_b["program_idx"]]

            # Configure env variables for this evaluation
            os.environ["EDGAR_WC_ROLLOUT_K"] = str(k)
            os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(stride)

            params_b = {k: jnp.asarray(np.asarray(v)[sample_idx:sample_idx+1]) for k, v in program_b.params.items()}
            model_fn_b = program_b.compile_model()
            os.environ["EDGAR_WC_OBJECTIVE"] = "B"
            out_b = spec.apply_model_fn(model_fn_b, sample_data, params_b)
            pred_b_rollout = np.asarray(out_b["pred_y_rollout"])[0, stim_idx]  # (A, K, 2)

            from projects.wilson_cowan.data_loader.load_data import _rollout_anchors
            anchor_starts, K = _rollout_anchors(len(time_axis))

            all_times_E = []
            all_vals_E = []
            all_times_I = []
            all_vals_I = []

            # Calculate robust index stride for non-overlapping rollouts
            idx_stride = int(np.ceil(K / stride))
            for idx in range(0, len(anchor_starts), idx_stride):
                a = anchor_starts[idx]
                seg_time = time_axis[a + 1 : a + K + 1]

                # Excitatory
                all_times_E.append(seg_time)
                all_vals_E.append(pred_b_rollout[idx, :, 0])

                # Inhibitory
                all_times_I.append(seg_time)
                all_vals_I.append(pred_b_rollout[idx, :, 1])

            color = colors[c_idx % len(colors)]
            label_b = f"Obj B (K={k}, S={stride})" if i == 0 else ""

            # E Channel segments
            mse_b_E = 0.0
            if all_times_E:
                flat_times_E = np.concatenate(all_times_E)
                flat_vals_E = np.concatenate(all_vals_E)
                mse_b_E = np.mean((flat_vals_E - E_clean[1 : len(flat_vals_E) + 1]) ** 2)
                ax_E.plot(flat_times_E, flat_vals_E, color=color, alpha=0.4, linewidth=1.0, label=label_b)
                ax_E_inset.plot(flat_times_E, flat_vals_E, color=color, alpha=0.4, linewidth=1.0)
            mses_b_E.append(mse_b_E)

            # I Channel segments
            mse_b_I = 0.0
            if all_times_I:
                flat_times_I = np.concatenate(all_times_I)
                flat_vals_I = np.concatenate(all_vals_I)
                mse_b_I = np.mean((flat_vals_I - I_clean[1 : len(flat_vals_I) + 1]) ** 2)
                ax_I.plot(flat_times_I, flat_vals_I, color=color, alpha=0.4, linewidth=1.0, label=label_b)
                ax_I_inset.plot(flat_times_I, flat_vals_I, color=color, alpha=0.4, linewidth=1.0)
            mses_b_I.append(mse_b_I)

        # Set limits and parameters for E inset
        ax_E_inset.set_xlim(400, 450)
        zoom_mask = (time_axis >= 400) & (time_axis <= 450)
        E_zoom_noisy = E_train[zoom_mask]
        if len(E_zoom_noisy) > 0:
            E_ymin, E_ymax = np.min(E_zoom_noisy), np.max(E_zoom_noisy)
            E_margin = (E_ymax - E_ymin) * 0.1 if E_ymax > E_ymin else 0.1
            ax_E_inset.set_ylim(E_ymin - E_margin, E_ymax + E_margin)
        ax_E_inset.tick_params(axis='both', which='major', labelsize=8)

        # Set limits and parameters for I inset
        ax_I_inset.set_xlim(400, 450)
        I_zoom_noisy = I_train[zoom_mask]
        if len(I_zoom_noisy) > 0:
            I_ymin, I_ymax = np.min(I_zoom_noisy), np.max(I_zoom_noisy)
            I_margin = (I_ymax - I_ymin) * 0.1 if I_ymax > I_ymin else 0.1
            ax_I_inset.set_ylim(I_ymin - I_margin, I_ymax + I_margin)
        ax_I_inset.tick_params(axis='both', which='major', labelsize=8)

        # Compute Objective A MSE
        mse_a_E = np.mean((pred_a[:, 0] - E_clean[1:]) ** 2)
        mse_a_I = np.mean((pred_a[:, 1] - I_clean[1:]) ** 2)

        # Annotate MSE values
        text_str_E = f"MSE A: {mse_a_E:.5f}\n"
        for (k, stride), mse in zip(rollout_configs, mses_b_E):
            text_str_E += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        ax_E.text(0.05, 0.95, text_str_E.strip(), transform=ax_E.transAxes, fontsize=8,
                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        text_str_I = f"MSE A: {mse_a_I:.5f}\n"
        for (k, stride), mse in zip(rollout_configs, mses_b_I):
            text_str_I += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        ax_I.text(0.05, 0.95, text_str_I.strip(), transform=ax_I.transAxes, fontsize=8,
                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax_E.set_ylabel(f"E Rate (Noise {noise_level:.2f})")
        ax_I.set_ylabel(f"I Rate (Noise {noise_level:.2f})")
        if i == 0:
            ax_E.legend(loc="upper right")
            ax_E.set_title("Excitatory (E) Channel")
            ax_I.legend(loc="upper right")
            ax_I.set_title("Inhibitory (I) Channel")

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    plt.tight_layout()

    plot_save_path = folder_path / "WC_fit_forced_comparison.png"
    plt.savefig(plot_save_path, dpi=150)
    print(f"Segment rollout plot successfully saved to {plot_save_path}")


    # ─── Figure 2: Full rollout plot (from steady state) ───
    fig2, axes2 = plt.subplots(len(noise_levels), 2, figsize=(14, 18), sharex=True)
    fig2.suptitle(f"Wilson-Cowan Full Rollout vs Noise Level & Objectives (Sample {sample_idx}, Stim {stim_idx})", fontsize=16, y=0.98)

    for i, noise_level in enumerate(noise_levels):
        # Find Objective A program
        mapping_a = next(m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "A")
        program_a = population[mapping_a["program_idx"]]
        E_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 0]
        I_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 1]

        stim_E_design = np.asarray(X_discovers[i][0]['stim_E'][sample_idx, stim_idx])
        stim_I_design = np.asarray(X_discovers[i][0]['stim_I'][sample_idx, stim_idx])
        tmax = len(time_axis)

        # Generate full rollout for Objective A
        Et_full_a, It_full_a = generate_full_rollout(program_a, sample_idx, stim_E_design, stim_I_design, tmax)

        # Plot Excitatory (E) Channel
        ax_E = axes2[i, 0]
        ax_E.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_E.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6, label="Noisy Train Data" if i == 0 else "")
        ax_E.plot(time_axis, Et_full_a, color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")

        # Create E Inset for Figure 2
        ax_E_inset = ax_E.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_E_inset.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5)
        ax_E_inset.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6)
        ax_E_inset.plot(time_axis, Et_full_a, color="red", linestyle="-", linewidth=1.5)

        # Plot Inhibitory (I) Channel
        ax_I = axes2[i, 1]
        ax_I.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_I.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6, label="Noisy Train Data" if i == 0 else "")
        ax_I.plot(time_axis, It_full_a, color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")

        # Create I Inset for Figure 2
        ax_I_inset = ax_I.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_I_inset.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5)
        ax_I_inset.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6)
        ax_I_inset.plot(time_axis, It_full_a, color="red", linestyle="-", linewidth=1.5)

        mses_b_E = []
        mses_b_I = []

        # Generate and plot full rollout for each Objective B configuration
        for c_idx, (k, stride) in enumerate(rollout_configs):
            mapping_b = next(m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "B" and m["k"] == k and m["stride"] == stride)
            program_b = population[mapping_b["program_idx"]]

            # Generate full rollout
            Et_full_b, It_full_b = generate_full_rollout(program_b, sample_idx, stim_E_design, stim_I_design, tmax)

            color = colors[c_idx % len(colors)]
            label_b = f"Obj B (K={k}, S={stride})" if i == 0 else ""

            # E Channel
            ax_E.plot(time_axis, Et_full_b, color=color, alpha=0.6, linewidth=1.0, label=label_b)
            ax_E_inset.plot(time_axis, Et_full_b, color=color, alpha=0.6, linewidth=1.0)
            mse_b_E = np.mean((Et_full_b - E_clean) ** 2)
            mses_b_E.append(mse_b_E)

            # I Channel
            ax_I.plot(time_axis, It_full_b, color=color, alpha=0.6, linewidth=1.0, label=label_b)
            ax_I_inset.plot(time_axis, It_full_b, color=color, alpha=0.6, linewidth=1.0)
            mse_b_I = np.mean((It_full_b - I_clean) ** 2)
            mses_b_I.append(mse_b_I)

        # Set limits and parameters for Figure 2 insets
        ax_E_inset.set_xlim(400, 450)
        zoom_mask = (time_axis >= 400) & (time_axis <= 450)
        E_zoom_noisy = E_train[zoom_mask]
        if len(E_zoom_noisy) > 0:
            E_ymin, E_ymax = np.min(E_zoom_noisy), np.max(E_zoom_noisy)
            E_margin = (E_ymax - E_ymin) * 0.1 if E_ymax > E_ymin else 0.1
            ax_E_inset.set_ylim(E_ymin - E_margin, E_ymax + E_margin)
        ax_E_inset.tick_params(axis='both', which='major', labelsize=8)

        ax_I_inset.set_xlim(400, 450)
        I_zoom_noisy = I_train[zoom_mask]
        if len(I_zoom_noisy) > 0:
            I_ymin, I_ymax = np.min(I_zoom_noisy), np.max(I_zoom_noisy)
            I_margin = (I_ymax - I_ymin) * 0.1 if I_ymax > I_ymin else 0.1
            ax_I_inset.set_ylim(I_ymin - I_margin, I_ymax + I_margin)
        ax_I_inset.tick_params(axis='both', which='major', labelsize=8)

        # Compute full rollout Objective A MSE
        mse_a_E = np.mean((Et_full_a - E_clean) ** 2)
        mse_a_I = np.mean((It_full_a - I_clean) ** 2)

        # Annotate MSE values
        text_str_E = f"MSE A: {mse_a_E:.5f}\n"
        for (k, stride), mse in zip(rollout_configs, mses_b_E):
            text_str_E += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        ax_E.text(0.05, 0.95, text_str_E.strip(), transform=ax_E.transAxes, fontsize=8,
                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        text_str_I = f"MSE A: {mse_a_I:.5f}\n"
        for (k, stride), mse in zip(rollout_configs, mses_b_I):
            text_str_I += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        ax_I.text(0.05, 0.95, text_str_I.strip(), transform=ax_I.transAxes, fontsize=8,
                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax_E.set_ylabel(f"E Rate (Noise {noise_level:.2f})")
        ax_I.set_ylabel(f"I Rate (Noise {noise_level:.2f})")
        if i == 0:
            ax_E.legend(loc="upper right")
            ax_E.set_title("Excitatory (E) Channel")
            ax_I.legend(loc="upper right")
            ax_I.set_title("Inhibitory (I) Channel")

    axes2[-1, 0].set_xlabel("Time")
    axes2[-1, 1].set_xlabel("Time")
    plt.tight_layout()

    plot_save_path2 = folder_path / "WC_fit_full_rollout_comparison.png"
    plt.savefig(plot_save_path2, dpi=150)
    print(f"Full rollout comparison plot successfully saved to {plot_save_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit or plot Wilson-Cowan synthetic data.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Fit subcommand
    fit_parser = subparsers.add_parser("fit", help="Fit WC/WCS models on synthetic data")
    fit_parser.add_argument("--configs", type=str, default="50:1,50:50",
                            help="Comma-separated K:stride rollout configurations, e.g., 50:1,50:50")
    fit_parser.add_argument("--folder", type=str, default="fitted_population_multi",
                            help="Output folder name in sandbox")

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot fitted rollouts and compare results")
    plot_parser.add_argument("folder", type=str, help="Path to the directory containing fitted_population.jsonl and config.json")

    args = parser.parse_args()

    if args.command == "fit":
        fit_population(configs_str=args.configs, folder_name=args.folder)
    elif args.command == "plot":
        plot_population(folder_path_str=args.folder)
    else:
        parser.print_help()
