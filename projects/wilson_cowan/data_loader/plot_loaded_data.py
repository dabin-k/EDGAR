import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure projects directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from projects.wilson_cowan.data_loader.load_data import load_data
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec

def test_loading(data_file: str, cv_type: str):
    print(f"\n--- Testing data loading for file: {data_file} with cv_type: {cv_type} ---")
    
    # Load config and spec
    config = Config.from_yaml("projects/wilson_cowan/config.yaml")
    spec = TaskSpec.from_config(config)
    
    # Override cv_type and project params
    project_params = spec.project_params.copy()
    project_params["cv_type"] = cv_type
    
    # Load the data splits
    (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval = load_data(
        data_path=data_file,
        **project_params
    )
    
    # Plot first sample, first condition E/I
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f"Wilson-Cowan Synthetic Split Verification\nFile: {data_file} | CV: {cv_type}", fontsize=14, fontweight="bold")
    
    splits = [
        ("Discover Train", X_disc_train, axes[0, 0]),
        ("Discover Test", X_disc_test, axes[0, 1]),
        ("Validate Train", X_val_train, axes[1, 0]),
        ("Validate Test", X_val_test, axes[1, 1])
    ]
    
    # Extract time axis in seconds
    time_axis = X_disc_train["time"][0] # (T,)
    
    for name, split_dict, ax in splits:
        target_y = split_dict["target_y"] # (n, C, T, 2)
        
        # Plot first sample (index 0), first condition (index 0)
        E_trace = target_y[0, 0, :, 0]
        I_trace = target_y[0, 0, :, 1]
        
        ax.plot(time_axis, E_trace, label="E(t)", color="C0", linewidth=2.0)
        ax.plot(time_axis, I_trace, label="I(t)", color="C1", linewidth=2.0)
        ax.set_title(f"{name} (Sample 0, Cond 0)")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Activity")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plot_name = f"synthetic_verification_{cv_type}_{Path(data_file).stem}.png"
    plt.savefig(plot_name, dpi=150)
    print(f"Saved plot: {plot_name}")
    plt.close()

if __name__ == "__main__":
    test_loading("synthetic/synthetic_data_clean.npz", "k_fold")
    test_loading("synthetic/synthetic_data_clean.npz", "exp_cond")
