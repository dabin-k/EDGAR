from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import jax.numpy as jnp
from edgar.data.neural import filtering, normalization, signal, trials


def _to_jax(d: dict) -> dict:
    """Recursively convert numpy arrays in a dictionary to JAX arrays."""
    return {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}


def _load_raw_data(data_paths: tuple[tuple[str, str], ...]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load neural responses and stimulus angles from files."""
    responses, angles = [], []
    for d_file, m_file in data_paths:
        resp = np.load(d_file)  # (n_trials, n_cells)
        mat_data = sp.io.loadmat(m_file, simplify_cells=True)
        ang = np.array([entry["gratingOrient"] for entry in mat_data["block"]["paramsValues"]])

        # Filter out invalid stimulus values
        mask = ang != 1
        responses.append(resp[mask])
        angles.append(ang[mask])

    # Convert angles to radians and ensure they are within [0, 2*pi)
    for i in range(len(angles)):
        angles[i] = np.deg2rad(angles[i])
        angles[i][angles[i] >= 2 * np.pi] = 2 * np.pi - 1e-5

    return responses, angles


def _filter_cells(
    responses_list: list[np.ndarray], angles_list: list[np.ndarray], activity_thresh: float, conc_thresh: float
) -> list[np.ndarray]:
    """Filter cells based on global activity and vector concentration.
        TODO: maybe filter differently for discover vs validate sets, e.g some cells good in discover but not validate ,or vice versa
    """
    all_responses = np.vstack(responses_list)
    all_angles = np.concatenate(angles_list)

    conc = filtering.vector_concentration(all_responses.T, all_angles)
    activity = filtering.activity(all_responses.T)

    good_cells_mask = (conc > conc_thresh) & (activity > activity_thresh)
    return [r[:, good_cells_mask] for r in responses_list]


def _get_signal(
    resp_disc: np.ndarray, ang_disc: np.ndarray, resp_val: np.ndarray, ang_val: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the binned mean signal (tuning curves) for discovery and validation sets."""
    all_resp = np.vstack([resp_disc, resp_val])
    all_ang = np.concatenate([ang_disc, ang_val])

    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Global tuning curves
    averaged_response = signal.binned_mean(all_ang, all_resp.T, bin_centers)

    # Map global tuning curves back to individual trials
    _, bin_indices_disc = signal.binned_mean(ang_disc, resp_disc.T, bin_centers, return_indices=True)
    signal_disc = averaged_response[:, bin_indices_disc].T

    _, bin_indices_val = signal.binned_mean(ang_val, resp_val.T, bin_centers, return_indices=True)
    signal_val = averaged_response[:, bin_indices_val].T

    return signal_disc, signal_val, bin_centers, averaged_response


def _apply_corner_mask(resp: np.ndarray, sig: np.ndarray, ang: np.ndarray) -> tuple[dict, dict]:
    """Create train/test pairs by masking the bottom-right corner of the response matrix."""
    n_trials, n_cells = resp.shape
    trial_mid, cell_mid = n_trials // 2, n_cells // 2

    # Train: Mask out bottom right corner
    resp_train = resp.copy()
    resp_train[trial_mid:, cell_mid:] = np.nan
    sig_train = sig.copy()
    sig_train[trial_mid:, cell_mid:] = np.nan

    # Test: Mask out everything EXCEPT bottom right corner
    resp_test = np.full_like(resp, np.nan)
    resp_test[trial_mid:, cell_mid:] = resp[trial_mid:, cell_mid:]
    sig_test = np.full_like(sig, np.nan)
    sig_test[trial_mid:, cell_mid:] = sig[trial_mid:, cell_mid:]

    return (
        {"response": resp_train, "signal": sig_train, "stimulus": ang},
        {"response": resp_test, "signal": sig_test, "stimulus": ang},
    )


def load_data(
    data_path: str,
    activity_thresh: float = 0.4,
    conc_thresh: float = 0.55,
    n_bins: int = 256,
    show_plots: bool = False,
    shuffle_trials: bool = True,
    random_seed: int = 42,
):
    """
    Load and preprocess neural data for trial-to-trial variability modeling.

    Returns data in (n_samples, n_trials, n_cells) shape, where n_samples=1 for population modeling.
    """
    # 1. Load and Filter
    data_paths = (
        (
            data_path+"/BZ015_2025-07-03_2/BZ015_2025-07-03_2_dspikes.npy",
            data_path+"/BZ015_2025-07-03_2/2025-07-03_2_BZ015_Block.mat",
        ),
        (
            data_path+"/BZ015_2025-07-03_3/BZ015_2025-07-03_3_dspikes.npy",
            data_path+"/BZ015_2025-07-03_3/2025-07-03_3_BZ015_Block.mat",
        ),
        (
            data_path+"/BZ015_2025-07-03_5/BZ015_2025-07-03_5_dspikes.npy",
            data_path+"/BZ015_2025-07-03_5/2025-07-03_5_BZ015_Block.mat",
        ),
    )
    responses_raw, angles_raw = _load_raw_data(data_paths)
    responses_filtered = _filter_cells(responses_raw, angles_raw, activity_thresh, conc_thresh)

    # 2. Partition and Normalize
    resp_disc = np.vstack(responses_filtered[:2])
    ang_disc = np.concatenate(angles_raw[:2])
    resp_val = responses_filtered[2]
    ang_val = angles_raw[2]

    resp_disc = normalization.by_vector_norm(resp_disc, axis=0)
    resp_val = normalization.by_vector_norm(resp_val, axis=0)

    # 3. Calculate Signal (Tuning Curves)
    sig_disc, sig_val, bin_centers, avg_resp = _get_signal(resp_disc, ang_disc, resp_val, ang_val, n_bins)

    # 4. Optional Shuffling
    if shuffle_trials:
        resp_disc, sig_disc, ang_disc = trials.shuffle(resp_disc, sig_disc, ang_disc, axis=0)
        resp_val, sig_val, ang_val = trials.shuffle(resp_val, sig_val, ang_val, axis=0)

    # 5. Masking
    disc_train, disc_test = _apply_corner_mask(resp_disc, sig_disc, ang_disc)
    val_train, val_test = _apply_corner_mask(resp_val, sig_val, ang_val)


    # 6. Add Sample Dimension to ALL fields
    for d in [disc_train, disc_test, val_train, val_test]:
        for k in d:
            if isinstance(d[k], np.ndarray):
                d[k] = d[k][np.newaxis, ...]
    
    # 7. Fingerprinting (Evaluation Set)
    eval_data = {**disc_train, "_sample_indices": np.array([0])}

    if show_plots:
        _plot_partitions(disc_train, disc_test, val_train, val_test)
        _plot_tuning_verification(
            np.vstack([resp_disc, resp_val]), np.concatenate([ang_disc, ang_val]), bin_centers, avg_resp, random_seed
        )

    return (
        (_to_jax(disc_train), _to_jax(disc_test)),
        (_to_jax(val_train), _to_jax(val_test)),
        _to_jax(eval_data),
    )


def _plot_partitions(disc_train, disc_test, val_train, val_test):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    partitions = [
        ("Discovery Train", disc_train),
        ("Discovery Test", disc_test),
        ("Validation Train", val_train),
        ("Validation Test", val_test),
    ]
    for ax, (name, data) in zip(axes.flatten(), partitions):
        resp = data["response"][0]
        im = ax.imshow(resp, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(
            f"{name}\nShape: {resp.shape}\nMean: {np.nanmean(resp):.4e}, Std: {np.nanstd(resp):.4e}", fontsize=10
        )
        ax.set_xlabel("Cells")
        ax.set_ylabel("Trials")
        fig.colorbar(im, ax=ax, label="Normalized Response")

    plt.suptitle("BZ015 Data Partitions (Normalized & Masked)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("bz015_partitions_colormap.png")
    plt.close()


def _plot_tuning_verification(all_resp, all_ang, bin_centers, avg_resp, seed):
    rng = np.random.default_rng(seed)
    cell_idx = rng.integers(0, all_resp.shape[1])
    plt.figure(figsize=(10, 6))
    plt.scatter(all_ang, all_resp[:, cell_idx], alpha=0.3, s=10, label="Raw Responses", color="gray")
    plt.plot(bin_centers, avg_resp[cell_idx, :], color="red", linewidth=2, label="Binned Mean (Tuning Curve)")
    plt.title(f"Tuning Curve Verification - Cell {cell_idx}")
    plt.xlabel("Theta (radians)")
    plt.ylabel("Normalized Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tuning_curve_verification.png")
    plt.close()


def loss_fn(model_output, data):
    """
    Mean squared error loss, ignoring NaNs.
    Expects data['response'] of shape (n_samples, n_trials, n_cells).
    Returns (n_samples,) array of losses.
    """
    mask = ~jnp.isnan(data["response"])
    diff_sq = (data["response"] - model_output) ** 2
    safe_diff_sq = jnp.where(mask, diff_sq, jnp.nan)
    return jnp.nanmean(safe_diff_sq, axis=(-2, -1))