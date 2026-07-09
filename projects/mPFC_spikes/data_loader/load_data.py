from __future__ import annotations

import glob
import os

import numpy as np
import jax
import jax.numpy as jnp
import scipy.io as sio


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


# ── data location ──


def _resolve_paths(data_path: str):
    """Find the SAll + WSRestrictedIntervals .mat files.

    `data_path` may be a directory containing the Buzsaki-lab session files, or a
    session prefix like ``.../Dino_061914_mPFC``.
    """
    if os.path.isdir(data_path):
        base = data_path
    else:
        base = os.path.dirname(data_path)
    sall = sorted(glob.glob(os.path.join(base, "*_SAll.mat")))
    ws = sorted(glob.glob(os.path.join(base, "*_WSRestrictedIntervals.mat")))
    if not sall:
        raise FileNotFoundError(f"No *_SAll.mat under {base}")
    if not ws:
        raise FileNotFoundError(f"No *_WSRestrictedIntervals.mat under {base}")
    return sall[0], ws[0]


def _load_spike_trains(sall_path: str):
    """Per-neuron spike times (seconds) from S_CellFormat."""
    m = sio.loadmat(sall_path)
    cell = m["S_CellFormat"]
    return [np.asarray(c).ravel().astype(float) for c in cell[0]]


def _sws_segments(ws_path: str, sws_mode: str):
    """Return SWS [start, stop] second-pairs. `longest` keeps only the single
    longest contiguous episode (avoids history-corrupting joins); `concat`
    keeps all episodes."""
    m = sio.loadmat(ws_path, struct_as_record=False, squeeze_me=True)
    key = None
    for pref in ("SWSEpisodeTimePairFormat", "SWSPacketTimePairFormat"):
        if pref in m:
            key = pref
            break
    if key is None:
        for k in m:
            if not k.startswith("__") and "SWS" in k and "TimePairFormat" in k:
                key = k
                break
    if key is None:
        raise KeyError(f"No SWS TimePair field in {ws_path}; keys={list(m.keys())}")
    seg = np.atleast_2d(np.asarray(m[key], dtype=float))
    if seg.shape[1] != 2 and seg.shape[0] == 2:
        seg = seg.T
    if sws_mode == "longest":
        lengths = seg[:, 1] - seg[:, 0]
        seg = seg[[int(np.argmax(lengths))]]
    elif sws_mode != "concat":
        raise ValueError(f"sws_mode must be 'longest' or 'concat', got {sws_mode!r}")
    return seg


def _bin_counts(spikes, segments, dt):
    """Bin each neuron's spikes into per-segment `dt`-second bins, concatenated.

    Returns counts of shape (n_neurons, T) and the number of bins per segment
    (so callers know where segment joins fall). Binning is done per segment and
    concatenated so no bin ever straddles a wake/REM gap.
    """
    n = len(spikes)
    per_seg = []
    seg_bins = []
    for lo, hi in segments:
        n_bins = int((hi - lo) / dt)
        if n_bins <= 0:
            continue
        edges = lo + np.arange(n_bins + 1) * dt
        block = np.zeros((n, n_bins), dtype=np.float32)
        for i, s in enumerate(spikes):
            sel = s[(s >= lo) & (s < hi)]
            if sel.size:
                block[i] = np.histogram(sel, bins=edges)[0]
        per_seg.append(block)
        seg_bins.append(n_bins)
    counts = np.concatenate(per_seg, axis=1) if per_seg else np.zeros((n, 0), np.float32)
    return counts, seg_bins


def _pick_anchors(lo: int, hi: int, n: int) -> np.ndarray:
    """Evenly spaced anchor bin-indices in [lo, hi). Caps at the number available."""
    span = hi - lo
    if span <= 0:
        raise ValueError(f"empty anchor region [{lo}, {hi}); widen T or shrink W")
    n = int(min(n, span))
    return (lo + np.unique(np.linspace(0, span - 1, n).astype(np.int64)))


def load_data(
    data_path: str,
    bin_ms: float = 5.0,
    state: str = "sws",
    sws_mode: str = "longest",
    window_W: int = 300,
    anchors_per_neuron: int = 2000,
    min_region_spikes: int = 100,
    n_eval: int = 1,
    eval_anchors: int = 200,
    random_seed: int = 42,
):
    """Load mPFC spike trains as a *windowed* causal next-bin point-process task.

    The recording is restricted to slow-wave sleep (SWS) — the most stationary
    state — binned at ``bin_ms`` into per-neuron spike counts ``x`` (N, T), then
    reframed so leakage is impossible **by construction**: each training example
    is a strictly-past window mapped to the next bin.

        sample axis 0 = neuron   (one shared param set per neuron)
        data["history"][n, a] = x[n, t_a - W : t_a]   # (W,) strictly-past window
        data["target_y"][n, a] = x[n, t_a]            # next-bin count (loss only)

    The model is called as ``model(window, params) -> scalar`` on a single ``(W,)``
    window and NEVER sees ``target_y`` (see ``apply_model``), so it cannot peek at
    the current bin. Window orientation: ``window[-1]`` is the most recent past
    bin ``x[t-1]``; ``window[0]`` is the oldest ``x[t-W]``.

    Splits (standard EDGAR contract):
        X_discover = (train, test)  discover-half neurons
        X_validate = (train, test)  (LLM-unseen) validate-half neurons
        X_eval                      a few discover neurons for fingerprinting
    Train/test is a contiguous *time* split with a single ``W`` guard band:
    train anchors live in ``[W, half)`` (windows subset ``[0, half)``), test
    anchors in ``[half + W, T)`` (windows subset ``[half, T)``) — so no bin is
    ever shared between train and test, and a test target is never in any train
    window.
    """
    if state != "sws":
        raise ValueError(f"only state='sws' is supported, got {state!r}")

    W = int(window_W)
    dt = bin_ms / 1000.0
    sall_path, ws_path = _resolve_paths(data_path)
    spikes = _load_spike_trains(sall_path)
    segments = _sws_segments(ws_path, sws_mode)
    counts, _ = _bin_counts(spikes, segments, dt)  # (n_neurons, T)

    T = counts.shape[1]
    half = T // 2
    if half <= W:
        raise ValueError(
            f"half-series ({half} bins) <= window_W ({W}); "
            "recording too short for this window — shrink window_W."
        )

    # Anchor bin-indices, shared across neurons (so the sample arrays are
    # rectangular for the nested vmap). Guard band = W keeps every train window
    # inside [0, half) and every test window inside [half, T).
    train_anchors = _pick_anchors(W, half, anchors_per_neuron)
    test_anchors = _pick_anchors(half + W, T, anchors_per_neuron)
    assert train_anchors.min() >= W and test_anchors.min() >= W
    assert train_anchors.max() < half and test_anchors.max() < T

    # Keep neurons with enough spikes in each REGION (mPFC units are sparse, so an
    # anchor-subset floor would drop nearly everything). The per-spike NLL
    # denominator is instead protected by a floor in loss_fn, so a neuron that
    # happens to be quiet at its sampled anchors still can't explode the loss.
    tr = counts[:, :half].sum(axis=1)
    te = counts[:, half:].sum(axis=1)
    keep = (tr >= min_region_spikes) & (te >= min_region_spikes)
    counts = counts[keep]
    n_neurons = counts.shape[0]
    if n_neurons < 2:
        raise ValueError(
            f"Only {n_neurons} neuron(s) pass min_region_spikes={min_region_spikes}; "
            "lower the threshold."
        )
    print(
        f"[load_data] SWS {sws_mode}: {len(spikes)} neurons -> {n_neurons} kept "
        f"(>= {min_region_spikes} spk/region); T={T} bins @ {bin_ms} ms; W={W}; "
        f"anchors train/test={len(train_anchors)}/{len(test_anchors)}; "
        f"median spikes/neuron={np.median(counts.sum(axis=1)):.0f}"
    )

    x = counts.astype(np.float32)  # (n_neurons, T)

    def _windows(x_sub: np.ndarray, anchors: np.ndarray) -> dict:
        # idx (A, W): row a = [t_a - W, ..., t_a - 1]  -> window[-1] == x[t_a - 1]
        idx = anchors[:, None] + np.arange(-W, 0)[None, :]
        history = x_sub[:, idx]  # (n_sub, A, W)
        target_y = x_sub[:, anchors]  # (n_sub, A)
        # Orientation guard: newest element of each window is the bin before t_a.
        assert np.array_equal(history[:, :, -1], x_sub[:, anchors - 1])
        return {"history": history.astype(np.float32), "target_y": target_y.astype(np.float32)}

    # ── split neurons 50/50 into discover / validate ──
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_neurons)
    disc_idx = np.sort(perm[: n_neurons // 2])
    val_idx = np.sort(perm[n_neurons // 2 :])
    x_disc = x[disc_idx]
    x_val = x[val_idx]

    X_disc_train = _windows(x_disc, train_anchors)
    X_disc_test = _windows(x_disc, test_anchors)
    X_val_train = _windows(x_val, train_anchors)
    X_val_test = _windows(x_val, test_anchors)

    # ── X_eval: a few discover neurons (train-region windows) for fingerprinting.
    # _sample_indices are positions INTO the discover axis (to match `params`).
    n_eval = int(min(max(1, n_eval), len(disc_idx)))
    eval_pos = np.sort(rng.choice(len(disc_idx), n_eval, replace=False))
    eval_anchor_sub = _pick_anchors(W, half, eval_anchors)
    X_eval = _windows(x_disc[eval_pos], eval_anchor_sub)
    X_eval["_sample_indices"] = eval_pos

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def apply_model(model_fn, data, params):
    """Windowed model application (mPFC override of ``apply_model_plain``).

    Hands each program's ``model_fn`` a single strictly-past window ``(W,)`` and
    NOTHING else — the target never enters the model's scope, so current-bin
    leakage is structurally impossible. Parameters are shared per neuron:

        outer vmap over neurons  -> one param set `p` per neuron
        inner vmap over that neuron's A windows -> shares `p`, isolates windows

    Returns ``mu`` of shape ``(n_neurons, A)`` (one scalar intensity per window).
    """
    H = data["history"]  # (n_neurons, A, W)

    def per_neuron(h, p):  # h: (A, W)   p: this neuron's params
        return jax.vmap(lambda w: model_fn(w, p))(h)  # (A,)

    return jax.vmap(per_neuron, in_axes=(0, 0))(H, params)  # (n_neurons, A)


def loss_fn(model_output, data):
    """Per-spike-normalised discrete Poisson negative log-likelihood.

    For predicted intensities ``mu[a]`` and observed next-bin counts
    ``y[a] = target_y[a]`` over the A anchors of a neuron:

        NLL = sum_a( mu[a] - y[a] * log(mu[a]) ) / sum_a y[a]

    (discrete-time point-process log-likelihood, integral(lam) - sum log lam,
    normalised per observed spike so the value is O(1)). Lower is better.
    Reduces over the anchor axis -> one loss per neuron.
    """
    eps = 1e-8
    y = data["target_y"]
    mu = jnp.clip(model_output, eps, None)
    nll = jnp.sum(mu - y * jnp.log(mu), axis=-1)
    # Floor the per-spike denominator at 1 spike: mPFC units are sparse, so a
    # neuron with very few spikes at its sampled anchors must not blow the loss
    # up to ~1/eps. Dense neurons are unaffected (sum(y) >> 1).
    return nll / jnp.maximum(jnp.sum(y, axis=-1), 1.0)
