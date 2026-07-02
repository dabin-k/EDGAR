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


def dense_log_magnification(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Empirical log area-magnification per pixel from dense (H, W) az/el maps.

    Because the stimulus screen is a fixed size at a fixed distance, azimuth and
    elevation are in ABSOLUTE degrees, so the map's local area-magnification is fully
    determined by the measured data: it is |det J|, where J = d(az, el)/d(x=col, y=row)
    is the Jacobian of the dense visual-field image w.r.t. cortical pixel coordinates
    (raw-pixel spacing = 1). Units are deg^2 of visual field per pixel^2 of cortex.
    This is supplied as a self-supervised 3rd target with NO new measurement; the
    model's 3rd output column (predicted log area-magnification) is scored against it.

    Verified to recover the analytic dipole Jacobian to a median relative error ~1e-7.

    Returns NaN wherever the derivative cannot be computed (map border, or any pixel
    whose 4-neighbourhood touches a NaN) — about 2% of valid pixels near the border.

    IMPORTANT: this MUST be computed on the DENSE (H, W) map, before pixel subsampling.
    A spatial derivative needs each pixel's neighbours, which the scattered list of
    valid pixels no longer has.

    Args:
        az: Dense azimuth map, shape (H, W), NaN outside the responsive region.
        el: Dense elevation map, shape (H, W), NaN outside the responsive region.

    Returns:
        np.ndarray: log area-magnification, shape (H, W), NaN where uncomputable.
    """
    daz_dy, daz_dx = np.gradient(az)  # axis 0 = row = y, axis 1 = col = x
    del_dy, del_dx = np.gradient(el)
    detJ = np.abs(daz_dx * del_dy - daz_dy * del_dx)
    with np.errstate(divide="ignore"):
        logm = np.log(detJ)
    logm[~np.isfinite(logm)] = np.nan
    return logm


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


def _sample_pixels(coords_xy, vf, logm, n_target, rng):
    """Subsamples a fixed number of pixels so every recording is the same width.

    Args:
        coords_xy: Cortical (col, row) coordinates, shape (n_avail, 2).
        vf: Matching (azimuth, elevation) values, shape (n_avail, 2).
        logm: Matching log area-magnification, shape (n_avail,).
        n_target: Number of pixels to keep.
        rng: NumPy Generator.

    Returns:
        tuple: (coords_xy[idx], vf[idx], logm[idx]) sharing one index draw, shapes
            (n_target, 2), (n_target, 2), (n_target,). Samples without replacement when
            enough pixels are available, otherwise with replacement (with a warning), so
            the leading-trial axis is uniform across recordings.
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
    return coords_xy[idx], vf[idx], logm[idx]


def load_data(
    data_path: str = "/home/dabin/data/retinotopy_map",
    seed: int = 0,
    n_pixels_train: int = 3000,
    n_pixels_test: int = 3000,
    block_size: int = 200,
    normalize_coords: bool = False,
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

    Each valid pixel additionally carries a self-supervised `log_mag` target: the log
    area-magnification |det J| of the map, computed from the dense az/el image itself
    (see `dense_log_magnification`). Because the screen geometry is fixed, this needs no
    extra measurement, and it is the map's *derivative* — the quantity that discriminates
    a structured (dipole) map from a flat-magnification (affine) null.

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
            in [0, 1]; otherwise keep raw pixel coordinates. DEFAULT IS NOW False:
            dividing (x, y) by (W, H) separately injects a spurious anisotropy of aspect
            ratio W/H, which corrupts the conformality test the model's `aniso` parameter
            is meant to make. If you must normalise, divide BOTH axes by the SAME constant
            (e.g. a shared um/px scale), never by (W, H) separately.
        validate_animals: Animal label(s) held out as the validate set. If None, the
            last animal in sorted order is held out (deterministic).
        n_eval_samples: Number of discover recordings used for the fingerprint subset.

    Returns:
        tuple: (X_discover, X_validate, X_eval).

        X_discover = (X_disc_train, X_disc_test), X_validate = (X_val_train, X_val_test).
        Each dict has keys:
            'cortical_pos':  shape (n_recordings_in_split, n_pixels, 2) — (x, y) features.
            'visual_field':  shape (n_recordings_in_split, n_pixels, 2) — (az, el) target.
            'log_mag':       shape (n_recordings_in_split, n_pixels, 1) — log |det J|.
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

        # Self-supervised magnification target, computed on the DENSE map (needs
        # neighbours), then sampled at the valid pixels. Border pixels whose
        # neighbourhood touches a NaN come back NaN (~2%) and are dropped here so
        # every downstream array is finite.
        logm_dense = dense_log_magnification(az, el)  # (H, W)
        logm = logm_dense[rows, cols]                 # (n_valid,)
        finite_mag = np.isfinite(logm)
        rows, cols = rows[finite_mag], cols[finite_mag]
        vf = vf[finite_mag]
        logm = logm[finite_mag]

        coords = np.stack([cols, rows], axis=-1).astype(np.float64)  # (n_valid, 2) = (x, y)
        if normalize_coords:
            coords = coords / np.array([W, H], dtype=np.float64)

        block_id = _checkerboard_block_id(rows, cols, block_size)
        train_mask = block_id == 0
        test_mask = block_id == 1

        tr_xy, tr_vf, tr_m = _sample_pixels(
            coords[train_mask], vf[train_mask], logm[train_mask], n_pixels_train, rng
        )
        te_xy, te_vf, te_m = _sample_pixels(
            coords[test_mask], vf[test_mask], logm[test_mask], n_pixels_test, rng
        )

        train_list.append(
            {"cortical_pos": tr_xy, "visual_field": tr_vf, "log_mag": tr_m[:, None]}
        )
        test_list.append(
            {"cortical_pos": te_xy, "visual_field": te_vf, "log_mag": te_m[:, None]}
        )
        rec_animals.append(animal)

    def _stack(dicts, idx):
        return {
            k: np.stack([dicts[i][k] for i in idx], axis=0)
            for k in ("cortical_pos", "visual_field", "log_mag")
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


def loss_fn(model_output, data, lambda_mag: float = 0.3, reliability_weight: bool = True):
    """Position MSE + weighted magnification MSE between prediction and data, per recording.

    Called on the already-`vmap`'d batch (leading axis = recordings). The pixel axis and
    coordinate axes are reduced here so only the recording axis survives;
    `edgar/scoring/scoring.py` then mean-reduces over recordings.

    The model now outputs three columns: (azimuth, elevation, log area-magnification).
    The first two are scored against the measured visual field as before; the third is
    scored against `log_mag`, the self-supervised magnification target from the data.

    Args:
        model_output: Predicted (az, el, log_mag), shape (n_recordings, n_pixels, 3).
        data: Split dict; data['visual_field'] shape (n_recordings, n_pixels, 2),
            data['log_mag'] shape (n_recordings, n_pixels, 1).
        lambda_mag: Weight on the magnification term. The magnification is the map's
            DERIVATIVE, so it is what discriminates the affine null (flat magnification)
            from the dipole (structured) — but it is also noisier, hence a tunable weight
            rather than 1.0. Start ~0.3; raise if the search collapses to a good-centre /
            wrong-gradient map, lower if position accuracy suffers.
        reliability_weight: If True, down-weight pixels of extreme magnification (tiny or
            huge det J at the map fold / far periphery) where the empirical Jacobian is
            least reliable, via a robust 1/(1+z^2) on the median-centred log_mag.

    Returns:
        JAX array of per-recording losses, shape (n_recordings,).
    """
    pos_err = (data["visual_field"] - model_output[..., :2]) ** 2
    pos = jnp.mean(pos_err, axis=(1, 2))

    m_true = data["log_mag"][..., 0]        # (n_rec, n_pixels)
    m_pred = model_output[..., 2]
    m_err = (m_true - m_pred) ** 2
    if reliability_weight:
        med = jnp.median(m_true, axis=1, keepdims=True)
        mad = jnp.median(jnp.abs(m_true - med), axis=1, keepdims=True) + 1e-6
        zt = (m_true - med) / (1.4826 * mad)
        w = 1.0 / (1.0 + zt ** 2)
        mag = jnp.sum(w * m_err, axis=1) / jnp.sum(w, axis=1)
    else:
        mag = jnp.mean(m_err, axis=1)

    return pos + lambda_mag * mag
