from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def _discover_recordings(data_path: str):
    """Finds the per-recording folders under `data_path`.

    A recording is any directory directly containing both `azimuth.npy` and
    `elevation.npy`. The folder name is taken as the animal label (e.g. `CR031`,
    `SP068`), so the discover/validate split can hold out a whole animal.

    Args:
        data_path: Root directory holding one folder per recording.

    Returns:
        list[tuple[str, Path]]: (animal_label, folder_path), sorted by label for
            deterministic ordering.
    """
    root = Path(data_path)
    recordings = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "azimuth.npy").exists() and (sub / "elevation.npy").exists():
            recordings.append((sub.name, sub))
    if not recordings:
        raise FileNotFoundError(
            f"No recordings (folders with azimuth.npy + elevation.npy) under {data_path!r}"
        )
    return recordings


def _checkerboard_block_id(rows: np.ndarray, cols: np.ndarray, block_size: int) -> np.ndarray:
    """Checkerboard block class (0 or 1) for each pixel.

    Tiles the cortical image into `block_size`x`block_size` blocks and 2-colours them:
    `(block_row + block_col) % 2`. Whole blocks (not individual pixels) alternate
    between the two classes, so train/test pixels sit in spatially separated patches —
    held-out (class-1) pixels are never immediate neighbours of class-0 training pixels
    except across block borders. This is what limits the leakage a per-pixel random
    split would suffer on this smooth, autocorrelated map.

    Args:
        rows: Pixel row indices, shape (n_pixels,).
        cols: Pixel column indices, shape (n_pixels,).
        block_size: Side length of a block in pixels.

    Returns:
        np.ndarray: 0/1 block class per pixel, shape (n_pixels,).
    """
    return ((rows // block_size) + (cols // block_size)) % 2


def _sample_pixels(coords_xy, vf, n_target, rng):
    """Subsamples a fixed number of pixels so every recording is the same width.

    Args:
        coords_xy: Cortical (col, row) coordinates, shape (n_avail, 2).
        vf: Matching (azimuth, elevation) values, shape (n_avail, 2).
        n_target: Number of pixels to keep.
        rng: NumPy Generator.

    Returns:
        tuple: (coords_xy[idx], vf[idx]), each shape (n_target, 2). Samples without
            replacement when enough pixels are available, otherwise with replacement
            (with a warning), so the leading-trial axis is uniform across recordings.
    """
    n_avail = coords_xy.shape[0]
    if n_avail == 0:
        raise ValueError("A recording has no valid pixels in one split block class.")
    replace = n_avail < n_target
    if replace:
        print(
            f"Warning: only {n_avail} valid pixels available, sampling {n_target} "
            f"with replacement."
        )
    idx = rng.choice(n_avail, size=n_target, replace=replace)
    return coords_xy[idx], vf[idx]


def load_data(
    data_path: str = "/home/dabin/data/retinotopy_map",
    seed: int = 0,
    n_pixels_train: int = 3000,
    n_pixels_test: int = 3000,
    block_size: int = 200,
    normalize_coords: bool = True,
    validate_animals: tuple[str, ...] | None = None,
    n_eval_samples: int = 3,
    **kwargs,
):
    """Loads dense retinotopic maps and splits them for cortex->visual-field discovery.

    A "sample" is one recording (one animal folder): a dense map assigning every valid
    cortical pixel an (azimuth, elevation) visual-field coordinate. Within a recording,
    every valid pixel is a "trial" — `model()` is called once per recording and must
    explain every pixel's visual-field coordinate with one shared parameter set (the
    retinotopic map). Those map parameters are fit independently per recording and are
    never shared across recordings; agreement across animals is what the discover->
    validate split probes, not something the loss enforces.

    The raw data per recording is two `(H, W, 1)` arrays (`azimuth.npy`, `elevation.npy`)
    in degrees, mostly NaN outside the responsive region. Valid pixels (finite in both)
    are the data points. To keep the per-sample fit tractable and the arrays rectangular,
    a fixed number of valid pixels is subsampled per recording rather than using all
    ~10^5–10^6 of them.

    Splits:
        - Train/test is *within* a recording, along the *pixel* axis, via a checkerboard
          of spatial blocks (`block_size`): class-0 blocks supply train pixels, class-1
          blocks supply test pixels. Both classes tile the whole field, so neither split
          is an extrapolation to an unseen visual-field range, while whole-block
          alternation keeps held-out pixels spatially separated from training ones
          (limiting the leakage a per-pixel random split would incur on this smooth map).
        - Discover/validate is *across* recordings: all recordings of the held-out
          animal(s) form the validate set; the rest form discover. This tests that the
          discovered map *form* transfers to an unseen animal.

    Args:
        data_path: Root directory with one folder per recording (folder name = animal).
        seed: RNG seed for pixel subsampling and the eval-subset choice.
        n_pixels_train: Train pixels kept per recording (leading trial axis of train).
        n_pixels_test: Test pixels kept per recording.
        block_size: Checkerboard block side length in pixels.
        normalize_coords: If True, divide cortical (col, row) by (W, H) so features lie
            in [0, 1]; otherwise keep raw pixel coordinates.
        validate_animals: Animal label(s) held out as the validate set. If None, the
            last animal in sorted order is held out (deterministic).
        n_eval_samples: Number of discover recordings used for the fingerprint subset.

    Returns:
        tuple: (X_discover, X_validate, X_eval).

        X_discover = (X_disc_train, X_disc_test), X_validate = (X_val_train, X_val_test).
        Each dict has keys:
            'cortical_pos':  shape (n_recordings_in_split, n_pixels, 2) — (x, y) features.
            'visual_field':  shape (n_recordings_in_split, n_pixels, 2) — (az, el) target.
        X_eval: same keys as X_disc_train (subset of its recordings), plus
            '_sample_indices' (positions within X_disc_train's recording axis).
    """
    recordings = _discover_recordings(data_path)
    animals = [a for a, _ in recordings]

    if validate_animals is None:
        validate_animals = (sorted(set(animals))[-1],)
    validate_animals = tuple(validate_animals)

    rng = np.random.default_rng(seed)

    train_list, test_list, rec_animals = [], [], []
    for animal, folder in recordings:
        az = np.squeeze(np.load(folder / "azimuth.npy"))  # (H, W)
        el = np.squeeze(np.load(folder / "elevation.npy"))
        H, W = az.shape

        valid = np.isfinite(az) & np.isfinite(el)
        rows, cols = np.nonzero(valid)
        vf = np.stack([az[rows, cols], el[rows, cols]], axis=-1)  # (n_valid, 2)

        coords = np.stack([cols, rows], axis=-1).astype(np.float64)  # (n_valid, 2) = (x, y)
        if normalize_coords:
            coords = coords / np.array([W, H], dtype=np.float64)

        block_id = _checkerboard_block_id(rows, cols, block_size)
        train_mask = block_id == 0
        test_mask = block_id == 1

        tr_xy, tr_vf = _sample_pixels(coords[train_mask], vf[train_mask], n_pixels_train, rng)
        te_xy, te_vf = _sample_pixels(coords[test_mask], vf[test_mask], n_pixels_test, rng)

        train_list.append({"cortical_pos": tr_xy, "visual_field": tr_vf})
        test_list.append({"cortical_pos": te_xy, "visual_field": te_vf})
        rec_animals.append(animal)

    def _stack(dicts, idx):
        return {
            k: np.stack([dicts[i][k] for i in idx], axis=0)
            for k in ("cortical_pos", "visual_field")
        }

    rec_animals = np.array(rec_animals)
    disc_idx = np.nonzero(~np.isin(rec_animals, validate_animals))[0]
    val_idx = np.nonzero(np.isin(rec_animals, validate_animals))[0]
    if len(disc_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Discover/validate split empty: {len(disc_idx)} discover, {len(val_idx)} "
            f"validate from animals {sorted(set(animals))} holding out {validate_animals}."
        )

    X_disc_train = _stack(train_list, disc_idx)
    X_disc_test = _stack(test_list, disc_idx)
    X_val_train = _stack(train_list, val_idx)
    X_val_test = _stack(test_list, val_idx)

    n_eval = min(n_eval_samples, len(disc_idx))
    eval_pos = np.sort(rng.choice(len(disc_idx), size=n_eval, replace=False))
    X_eval = {k: v[eval_pos] for k, v in X_disc_train.items()}
    X_eval["_sample_indices"] = eval_pos

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def loss_fn(model_output, data):
    """Mean squared error between predicted and measured (azimuth, elevation), per recording.

    Called on the already-`vmap`'d batch (leading axis = recordings). Both the pixel
    axis and the 2-coordinate axis are reduced here so only the recording axis survives;
    `edgar/scoring/scoring.py` then mean-reduces over recordings.

    Args:
        model_output: Predicted (az, el), shape (n_recordings, n_pixels, 2).
        data: Split dict; data['visual_field'] shape (n_recordings, n_pixels, 2).

    Returns:
        JAX array of per-recording losses, shape (n_recordings,).
    """
    err = (data["visual_field"] - model_output) ** 2
    return jnp.mean(err, axis=tuple(range(1, err.ndim)))
