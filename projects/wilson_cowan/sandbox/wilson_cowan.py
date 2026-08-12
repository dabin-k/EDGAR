import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np

repo_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(repo_root / "projects" / "wilson_cowan"))
from data_loader.load_data import load_data

if __name__ == "__main__":
    X_disc, X_val, X_eval = load_data(
        data_path="/home/rajah/datasets/wc_synthetic/wc_fold0.npz",
    )
    data_train = X_disc[0]
    print(f"data_train['E'] shape: {data_train['E'].shape}")
    print(f"data_train['I'] shape: {data_train['I'].shape}")
    print(f"data_train['stim_E'] shape: {data_train['stim_E'].shape}")
    print(f"data_train['stim_I'] shape: {data_train['stim_I'].shape}")

    # Extract data for the first sample (index 0)
    # Shape of E and I: (n_samples, n_stim, T) -> we take sample 0
    E = data_train['E'][0]       # (n_stim, T)
    I = data_train['I'][0]       # (n_stim, T)
    stim_E = data_train['stim_E'][0] # (n_stim, T)
    stim_I = data_train['stim_I'][0] # (n_stim, T)

    T = E.shape[1]
    time = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # ── TOP ROW: E response ──
    # Left subplot (LHS): Excitatory pulse (stim_idx = 0)
    axes[0, 0].plot(time, E[0], '.', label='E (Excitatory population)', color='tab:blue')
    # Plot active stimulus: stim_E in stim condition 0
    # Scaling to match/overlay nicely in arbitrary units (e.g., peak of stimulus shown scaled relative to max of E)
    stim_scale_E0 = 0.5 * np.max(E[0])
    axes[0, 0].fill_between(time, 0, stim_E[0] * stim_scale_E0, alpha=0.2, color='tab:orange', label='stim_E (scaled)')
    axes[0, 0].set_title("E Response: Excitatory Pulse")
    axes[0, 0].set_ylabel("E Activity")
    axes[0, 0].legend()

    # Right subplot (RHS): Inhibitory pulse (stim_idx = 1)
    axes[0, 1].plot(time, E[1], '.', label='E (Excitatory population)', color='tab:blue')
    # Plot active stimulus: stim_I in stim condition 1
    stim_scale_E1 = 0.5 * np.max(E[1])
    axes[0, 1].fill_between(time, 0, stim_I[1] * stim_scale_E1, alpha=0.2, color='tab:green', label='stim_I (scaled)')
    axes[0, 1].set_title("E Response: Inhibitory Pulse")
    axes[0, 1].legend()

    # ── BOTTOM ROW: I response ──
    # Left subplot (LHS): Excitatory pulse (stim_idx = 0)
    axes[1, 0].plot(time, I[0], '.', label='I (Inhibitory population)', color='tab:purple')
    # Plot active stimulus: stim_E in stim condition 0
    stim_scale_I0 = 0.5 * np.max(I[0])
    axes[1, 0].fill_between(time, 0, stim_E[0] * stim_scale_I0, alpha=0.2, color='tab:orange', label='stim_E (scaled)')
    axes[1, 0].set_title("I Response: Excitatory Pulse")
    axes[1, 0].set_xlabel("Time steps (t)")
    axes[1, 0].set_ylabel("I Activity")
    axes[1, 0].legend()

    # Right subplot (RHS): Inhibitory pulse (stim_idx = 1)
    axes[1, 1].plot(time, I[1], '.', label='I (Inhibitory population)', color='tab:purple')
    # Plot active stimulus: stim_I in stim condition 1
    stim_scale_I1 = 0.5 * np.max(I[1])
    axes[1, 1].fill_between(time, 0, stim_I[1] * stim_scale_I1, alpha=0.2, color='tab:green', label='stim_I (scaled)')
    axes[1, 1].set_title("I Response: Inhibitory Pulse")
    axes[1, 1].set_xlabel("Time steps (t)")
    axes[1, 1].legend()

    plt.tight_layout()
    output_img = repo_root / "sandbox" / "wilson_cowan_responses.png"
    plt.savefig(output_img, dpi=150)
    print(f"Saved response plot to {output_img}")