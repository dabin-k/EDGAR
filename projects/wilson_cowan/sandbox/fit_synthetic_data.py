# Use the EDGAR scoring logic to fit WC/WCS model to synthetic data
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


def get_data_filename(noise_level: float, dataset_type: str) -> str:
    """Get the filename of the synthetic dataset.

    Args:
        noise_level: The noise level, e.g., 0.0, 0.05.
        dataset_type: The dataset type: "wc", "wcs", or "lin".

    Returns:
        The relative path of the npz file within synthetic/ directory.
    """
    suffix = f"_{dataset_type}" if dataset_type in ("wc", "wcs") else ""
    if noise_level == 0.0:
        return f"synthetic_data_clean{suffix}.npz"
    else:
        return f"noise_{noise_level:.2f}/synthetic_data_noisy{suffix}.npz"


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


def fit_population(
    configs_str: str,
    folder_name: str,
    objectives: list[str] = ["A", "B"],
    noise_levels: list[float] = [0.0, 0.05],
    dataset_type: str = "lin"
) -> None:
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

    data_base_path = str(repo_root / "projects" / "wilson_cowan" / "data_loader" / "synthetic")

    def load_datasets_for_config(noise_level: float, r_k: int, a_stride: int, objective: str):
        # Update config params explicitly
        config.project_params["objective"] = objective
        config.project_params["rollout_k"] = r_k
        config.project_params["anchor_stride"] = a_stride
        os.environ["EDGAR_WC_OBJECTIVE"] = objective
        os.environ["EDGAR_WC_ROLLOUT_K"] = str(r_k)
        os.environ["EDGAR_WC_ANCHOR_STRIDE"] = str(a_stride)

        # Load train dataset path
        train_filename = get_data_filename(noise_level, dataset_type)
        config.io.data_path = f"{data_base_path}/{train_filename}"
        
        spec = TaskSpec.from_config(config)
        X_discover, _ , _ = spec.load_data_fn(
            data_path=spec.io["data_path"], **spec.project_params
        )

        # Load eval dataset path (always clean)
        eval_filename = get_data_filename(0.0, dataset_type)
        config.io.data_path = f"{data_base_path}/{eval_filename}"
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

    # Sequential scoring of models
    for noise_level in noise_levels:
        for obj in objectives:
            obj = obj.upper()
            if obj in ("A", "FULL", "C", "D"):
                print(f"\n--- Fitting Objective {obj} for noise level {noise_level:.2f} ---")
                X_discover, X_eval, spec = load_datasets_for_config(noise_level, 50, 50, obj)

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
                    "objective": obj
                })
                program_idx += 1

            elif obj == "B":
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
        "noise_levels": noise_levels,
        "objectives": objectives,
        "dataset_type": dataset_type,
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
        noise_levels = metadata.get("noise_levels", [0.0, 0.05])
        dataset_type = metadata.get("dataset_type", "lin")

    print(f"Loaded config metadata with rollout configurations: {rollout_configs}")
    print(f"Loaded noise levels: {noise_levels}")
    print(f"Loaded dataset type: {dataset_type}")

    # Set repeat cross-validation
    config_path = repo_root / "projects" / "wilson_cowan" / "config.yaml"
    config = Config.from_yaml(config_path)
    config.project_params["cv_type"] = "repeats"

    data_base_path = str(repo_root / "projects" / "wilson_cowan" / "data_loader" / "synthetic")

    # Load ground-truth clean data
    clean_filename = get_data_filename(0.0, dataset_type)
    config.io.data_path = f"{data_base_path}/{clean_filename}"
    spec = TaskSpec.from_config(config)
    X_discover_clean, _ , _ = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    sample_idx, stim_idx = 0, 0
    E_clean = X_discover_clean[0]['target_y'][sample_idx, stim_idx, :, 0]
    I_clean = X_discover_clean[0]['target_y'][sample_idx, stim_idx, :, 1]
    time_axis = np.asarray(X_discover_clean[0]["time"][sample_idx])

    # Load in datasets for noise levels
    X_discovers = []
    for noise_level in noise_levels:
        train_filename = get_data_filename(noise_level, dataset_type)
        config.io.data_path = f"{data_base_path}/{train_filename}"
        spec_noise = TaskSpec.from_config(config)
        X_discover, _ , _ = spec_noise.load_data_fn(
            data_path=spec_noise.io["data_path"], **spec_noise.project_params
        )
        X_discovers.append(X_discover)

    # Load the population
    population = Population.load(str(pop_file))
    print(f"Loaded fitted population with {len(population)} programs from disk")

    colors = ["blue", "purple", "cyan", "magenta", "teal", "orange", "darkgreen"]

    # ─── Figure 2: Full rollout plot (from steady state) ───
    fig2, axes2 = plt.subplots(len(noise_levels), 2, figsize=(14, 4 * len(noise_levels) + 2), sharex=True)
    if len(noise_levels) == 1:
        axes2 = np.expand_dims(axes2, axis=0)
        
    fig2.suptitle(f"Wilson-Cowan Full Rollout vs Noise Level & Objectives (Sample {sample_idx}, Stim {stim_idx})", fontsize=16, y=0.98)

    for i, noise_level in enumerate(noise_levels):
        # Load training (possibly noisy) data for plotting reference points
        E_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 0]
        I_train = X_discovers[i][0]['target_y'][sample_idx, stim_idx, :, 1]

        stim_E_design = np.asarray(X_discovers[i][0]['stim_E'][sample_idx, stim_idx])
        stim_I_design = np.asarray(X_discovers[i][0]['stim_I'][sample_idx, stim_idx])
        tmax = len(time_axis)

        # Plot Excitatory (E) Channel
        ax_E = axes2[i, 0]
        ax_E.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_E.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6, label="Train Data" if i == 0 else "")

        # Create E Inset for Figure 2
        ax_E_inset = ax_E.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_E_inset.plot(time_axis, E_clean, color="black", linestyle="-", linewidth=1.5)
        ax_E_inset.scatter(time_axis, E_train, color="grey", alpha=0.5, s=6)

        # Plot Inhibitory (I) Channel
        ax_I = axes2[i, 1]
        ax_I.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5, label="Clean Data" if i == 0 else "")
        ax_I.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6, label="Train Data" if i == 0 else "")

        # Create I Inset for Figure 2
        ax_I_inset = ax_I.inset_axes([0.5, 0.5, 0.4, 0.4])
        ax_I_inset.plot(time_axis, I_clean, color="black", linestyle="-", linewidth=1.5)
        ax_I_inset.scatter(time_axis, I_train, color="grey", alpha=0.5, s=6)

        # 1. Plot Objective A if present
        mapping_a = next((m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "A"), None)
        mse_a_E = None
        mse_a_I = None
        if mapping_a is not None:
            program_a = population[mapping_a["program_idx"]]
            Et_full_a, It_full_a = generate_full_rollout(program_a, sample_idx, stim_E_design, stim_I_design, tmax)
            
            ax_E.plot(time_axis, Et_full_a, color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")
            ax_E_inset.plot(time_axis, Et_full_a, color="red", linestyle="-", linewidth=1.5)
            
            ax_I.plot(time_axis, It_full_a, color="red", linestyle="-", linewidth=1.5, label="Objective A (One-step)" if i == 0 else "")
            ax_I_inset.plot(time_axis, It_full_a, color="red", linestyle="-", linewidth=1.5)
            
            mse_a_E = np.mean((Et_full_a - E_clean) ** 2)
            mse_a_I = np.mean((It_full_a - I_clean) ** 2)

        # 2. Plot Objective FULL if present
        mapping_full = next((m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "FULL"), None)
        mse_full_E = None
        mse_full_I = None
        if mapping_full is not None:
            program_full = population[mapping_full["program_idx"]]
            Et_full_full, It_full_full = generate_full_rollout(program_full, sample_idx, stim_E_design, stim_I_design, tmax)
            
            ax_E.plot(time_axis, Et_full_full, color="green", linestyle="-", linewidth=1.5, label="Objective FULL (Full rollout)" if i == 0 else "")
            ax_E_inset.plot(time_axis, Et_full_full, color="green", linestyle="-", linewidth=1.5)
            
            ax_I.plot(time_axis, It_full_full, color="green", linestyle="-", linewidth=1.5, label="Objective FULL (Full rollout)" if i == 0 else "")
            ax_I_inset.plot(time_axis, It_full_full, color="green", linestyle="-", linewidth=1.5)
            
            mse_full_E = np.mean((Et_full_full - E_clean) ** 2)
            mse_full_I = np.mean((It_full_full - I_clean) ** 2)

        # 3. Plot Objective B configurations if present
        mses_b_E = []
        mses_b_I = []
        for c_idx, (k, stride) in enumerate(rollout_configs):
            mapping_b = next((m for m in program_mapping if m["noise_level"] == noise_level and m["objective"] == "B" and m.get("k") == k and m.get("stride") == stride), None)
            if mapping_b is None:
                continue
            program_b = population[mapping_b["program_idx"]]

            # Generate full rollout
            Et_full_b, It_full_b = generate_full_rollout(program_b, sample_idx, stim_E_design, stim_I_design, tmax)

            color = colors[c_idx % len(colors)]
            label_b = f"Obj B (K={k}, S={stride})" if i == 0 else ""

            # E Channel
            ax_E.plot(time_axis, Et_full_b, color=color, alpha=0.6, linewidth=1.0, label=label_b)
            ax_E_inset.plot(time_axis, Et_full_b, color=color, alpha=0.6, linewidth=1.0)
            mse_b_E = np.mean((Et_full_b - E_clean) ** 2)
            mses_b_E.append((k, stride, mse_b_E))

            # I Channel
            ax_I.plot(time_axis, It_full_b, color=color, alpha=0.6, linewidth=1.0, label=label_b)
            ax_I_inset.plot(time_axis, It_full_b, color=color, alpha=0.6, linewidth=1.0)
            mse_b_I = np.mean((It_full_b - I_clean) ** 2)
            mses_b_I.append((k, stride, mse_b_I))

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

        # Annotate MSE values
        text_str_E = ""
        if mse_a_E is not None:
            text_str_E += f"MSE A: {mse_a_E:.5f}\n"
        if mse_full_E is not None:
            text_str_E += f"MSE FULL: {mse_full_E:.5f}\n"
        for k, stride, mse in mses_b_E:
            text_str_E += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        if text_str_E:
            ax_E.text(0.05, 0.95, text_str_E.strip(), transform=ax_E.transAxes, fontsize=8,
                      verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        text_str_I = ""
        if mse_a_I is not None:
            text_str_I += f"MSE A: {mse_a_I:.5f}\n"
        if mse_full_I is not None:
            text_str_I += f"MSE FULL: {mse_full_I:.5f}\n"
        for k, stride, mse in mses_b_I:
            text_str_I += f"MSE B (K={k}, S={stride}): {mse:.5f}\n"
        if text_str_I:
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
    fit_parser.add_argument("--objectives", type=str, default="A,B",
                            help="Comma-separated list of objectives, e.g., A,B,FULL")
    fit_parser.add_argument("--noise-levels", type=str, default="0.0,0.05",
                            help="Comma-separated list of noise levels, e.g., 0.0,0.05")
    fit_parser.add_argument("--dataset-type", type=str, default="lin", choices=["lin", "wc", "wcs"],
                            help="Dataset type to fit: lin, wc, or wcs")

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot fitted rollouts and compare results")
    plot_parser.add_argument("folder", type=str, help="Path to the directory containing fitted_population.jsonl and config.json")

    args = parser.parse_args()

    if args.command == "fit":
        # Parse objectives
        objectives_list = [obj.strip().upper() for obj in args.objectives.split(",") if obj.strip()]
        # Parse noise levels as list of floats
        noise_levels_list = [float(nl.strip()) for nl in args.noise_levels.split(",") if nl.strip()]
        
        fit_population(
            configs_str=args.configs,
            folder_name=args.folder,
            objectives=objectives_list,
            noise_levels=noise_levels_list,
            dataset_type=args.dataset_type
        )
    elif args.command == "plot":
        plot_population(folder_path_str=args.folder)
    else:
        parser.print_help()
