"""Exploratory clustering utilities for EDGAR population data.

First-pass, rough-and-ready helpers to ask: does a naive clustering of a recorded
population carve out meaningful sub-populations (that could later be handed to EDGAR
separately)? Nothing here filters cells — we keep the whole population on purpose, so
that any "junk"/untuned cluster is itself informative.

Three clustering targets are supported:
  * cells by orientation tuning curve  (functional similarity)
  * trials by population response vector  (internal-state / trial-regime structure)
  * cells by spontaneous activity  (Stringer only; tests whether spontaneous
    co-activation structure predicts functional tuning)

See ``clustering_exploration.ipynb`` for the driver analysis.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

# ──────────────────────────────────────────────────────────────────────────────
# Loaders — return a common dict; NO cell filtering / preprocessing.
# ──────────────────────────────────────────────────────────────────────────────


def load_stringer(path: str) -> dict:
    """Load a Stringer drifting-grating ``.npy`` recording.

    Args:
        path: Path to the ``gratings_drifting_GT*.npy`` file (a pickled dict).

    Returns:
        Dict with ``response`` (n_cells, n_trials), ``angles`` (n_trials,) in
        radians, and ``spont_u`` (n_cells, n_spont_pcs) — the per-cell loadings on
        the spontaneous-activity PCs, usable as a ready-made spontaneous feature.
    """
    d = np.load(path, allow_pickle=True).item()
    response = np.asarray(d["sresp"], dtype=np.float64)  # n_pcs=0 → raw stimulus response
    angles = np.asarray(d["istim"], dtype=np.float64)
    spont_u = np.asarray(d["u_spont"], dtype=np.float64)
    return {"response": response, "angles": angles, "spont_u": spont_u}


def load_jacob(session_dir: str) -> dict:
    """Load a parsed Jacob drifting-grating session.

    The parsed format stores ``*_dspikes.npy`` as (n_trials, n_cells) row-aligned to
    ``*_stimulus_timings.csv``. Responses are already collapsed to one value per
    (trial, cell), so there is no continuous spontaneous timeseries to recover.

    Args:
        session_dir: Path to a parsed session folder (e.g. ``.../BZ015_2025-07-03_2``).

    Returns:
        Dict with ``response`` (n_cells, n_trials) and ``angles`` (n_trials,) in
        radians. No ``spont_u`` key (spontaneous clustering is Stringer-only).
    """
    dspikes_path = glob.glob(os.path.join(session_dir, "*dspikes.npy"))[0]
    csv_path = glob.glob(os.path.join(session_dir, "*stimulus_timings.csv"))[0]

    dspikes = np.load(dspikes_path)  # (n_trials, n_cells)
    stim = pd.read_csv(csv_path)
    angles = np.deg2rad(stim["orientation"].to_numpy(dtype=np.float64) % 360.0)

    response = dspikes.T  # (n_cells, n_trials)
    assert response.shape[1] == angles.shape[0], (
        f"dspikes trials {response.shape[1]} != csv rows {angles.shape[0]}"
    )
    return {"response": response, "angles": angles}


# ──────────────────────────────────────────────────────────────────────────────
# Feature construction
# ──────────────────────────────────────────────────────────────────────────────


def build_tuning_curves(
    response: np.ndarray,
    angles: np.ndarray,
    n_bins: int = 180,
    normalize: str | None = "l2",
) -> np.ndarray:
    """Bin per-cell responses by stimulus orientation into a tuning-curve feature.

    Empty bins (no trials) are filled with the cell's mean response so the feature
    stays finite; with dense orientation sampling this rarely triggers.

    Args:
        response: (n_cells, n_trials) response matrix.
        angles: (n_trials,) stimulus orientation in radians, in [0, 2π).
        n_bins: Number of equal-width orientation bins over [0, 2π).
        normalize: Per-cell normalization of the curve — ``"l2"`` (unit norm, so we
            cluster on *shape* not amplitude), ``"zscore"``, or ``None`` (raw mean).

    Returns:
        (n_cells, n_bins) tuning-curve feature matrix.
    """
    n_cells = response.shape[0]
    bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(angles, bin_edges) - 1, 0, n_bins - 1)

    curves = np.zeros((n_cells, n_bins), dtype=np.float64)
    cell_means = response.mean(axis=1)
    for b in range(n_bins):
        cols = np.where(bin_idx == b)[0]
        if cols.size:
            curves[:, b] = response[:, cols].mean(axis=1)
        else:
            curves[:, b] = cell_means

    if normalize == "l2":
        norms = np.linalg.norm(curves, axis=1, keepdims=True)
        curves = curves / np.where(norms == 0, 1.0, norms)
    elif normalize == "zscore":
        mu = curves.mean(axis=1, keepdims=True)
        sd = curves.std(axis=1, keepdims=True)
        curves = (curves - mu) / np.where(sd == 0, 1.0, sd)
    elif normalize is not None:
        raise ValueError(f"unknown normalize={normalize!r}")

    return curves


def trial_population_matrix(
    response: np.ndarray,
    angles: np.ndarray,
    remove_orientation_mean: bool = False,
    n_bins: int = 180,
) -> np.ndarray:
    """Build a (n_trials, n_cells) matrix for clustering trials.

    Args:
        response: (n_cells, n_trials) response matrix.
        angles: (n_trials,) stimulus orientation in radians.
        remove_orientation_mean: If True, subtract the mean population response for
            each trial's orientation bin, leaving the residual (internal-state)
            structure *beyond* the stimulus. This is the real test of whether trial
            clusters reflect anything other than the presented orientation.
        n_bins: Orientation bins used when ``remove_orientation_mean`` is True.

    Returns:
        (n_trials, n_cells) matrix.
    """
    mat = response.T.copy()  # (n_trials, n_cells)
    if remove_orientation_mean:
        bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
        bin_idx = np.clip(np.digitize(angles, bin_edges) - 1, 0, n_bins - 1)
        for b in np.unique(bin_idx):
            rows = np.where(bin_idx == b)[0]
            mat[rows] -= mat[rows].mean(axis=0, keepdims=True)
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# Reduction / embedding
# ──────────────────────────────────────────────────────────────────────────────


def pca_reduce(X: np.ndarray, n_components: int = 20, random_state: int = 0) -> np.ndarray:
    """PCA-reduce a feature matrix (denoise + speed up clustering)."""
    n_components = min(n_components, min(X.shape) - 1)
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def umap_embed(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """2-D UMAP embedding, for visualisation only (not the clustering decision)."""
    import umap  # local import; heavy

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────────


def cluster_kmeans(X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """KMeans labels."""
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)


def cluster_gmm(X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """Gaussian-mixture (soft) labels, returned as hard assignments."""
    return GaussianMixture(n_components=k, random_state=random_state).fit_predict(X)


def cluster_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    """Ward agglomerative labels."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


_CLUSTERERS = {
    "kmeans": cluster_kmeans,
    "gmm": cluster_gmm,
    "agglomerative": cluster_agglomerative,
}


def _fit(method: str, X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    fn = _CLUSTERERS[method]
    if method == "agglomerative":
        return fn(X, k)
    return fn(X, k, random_state=random_state)


# ──────────────────────────────────────────────────────────────────────────────
# Model selection / sanity checks
# ──────────────────────────────────────────────────────────────────────────────


def sweep_k(
    X: np.ndarray,
    ks=range(2, 11),
    method: str = "kmeans",
    sample_size: int | None = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Silhouette score vs number of clusters, to pick k without hand-tuning.

    Args:
        X: (n_samples, n_features) feature matrix.
        ks: Candidate cluster counts.
        method: One of ``"kmeans"``, ``"gmm"``, ``"agglomerative"``.
        sample_size: Cap for the silhouette computation (it is O(n²)); ``None`` uses
            all points.
        random_state: Seed for silhouette subsampling.

    Returns:
        DataFrame indexed by k with a ``silhouette`` column.
    """
    rows = []
    for k in ks:
        labels = _fit(method, X, k, random_state=random_state)
        sil = silhouette_score(
            X, labels, sample_size=sample_size, random_state=random_state
        )
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows).set_index("k")


def cluster_stability(
    X: np.ndarray,
    method: str,
    k: int,
    n_subsamples: int = 10,
    frac: float = 0.8,
    random_state: int = 0,
) -> float:
    """Mean pairwise ARI across clusterings of random subsamples.

    A sanity check that structure is real, not imposed: cluster ``n_subsamples``
    random ``frac``-sized subsets, then average the adjusted Rand index between every
    pair of runs over the samples they share. Values near 1 mean stable assignments;
    near 0 mean the partition is essentially arbitrary.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    m = int(frac * n)
    label_maps = []
    for i in range(n_subsamples):
        idx = rng.choice(n, size=m, replace=False)
        labels = _fit(method, X[idx], k, random_state=random_state + i)
        label_maps.append(dict(zip(idx.tolist(), labels.tolist())))

    aris = []
    for i in range(n_subsamples):
        for j in range(i + 1, n_subsamples):
            shared = np.array(sorted(set(label_maps[i]) & set(label_maps[j])))
            if shared.size < 2:
                continue
            li = np.array([label_maps[i][s] for s in shared])
            lj = np.array([label_maps[j][s] for s in shared])
            aris.append(adjusted_rand_score(li, lj))
    return float(np.mean(aris)) if aris else float("nan")


def compare_labelings(a: np.ndarray, b: np.ndarray) -> dict:
    """Adjusted Rand + normalized MI between two labelings of the same items."""
    return {
        "adjusted_rand": float(adjusted_rand_score(a, b)),
        "normalized_mi": float(normalized_mutual_info_score(a, b)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_embedding(emb: np.ndarray, labels: np.ndarray, ax=None, title: str = "", c=None):
    """Scatter a 2-D embedding coloured by cluster label (or by ``c`` if given)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    colour = labels if c is None else c
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=colour, s=4, cmap="tab10", alpha=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def plot_cluster_tuning_curves(curves: np.ndarray, labels: np.ndarray, ax=None, title: str = ""):
    """Mean ± SEM tuning curve per cluster."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 360, curves.shape[1])
    for lab in np.unique(labels):
        c = curves[labels == lab]
        mu = c.mean(axis=0)
        sem = c.std(axis=0) / np.sqrt(max(c.shape[0], 1))
        line, = ax.plot(x, mu, label=f"cl {lab} (n={c.shape[0]})")
        ax.fill_between(x, mu - sem, mu + sem, color=line.get_color(), alpha=0.2)
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel("normalized response")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return ax


def plot_cluster_sizes(labels: np.ndarray, ax=None, title: str = ""):
    """Bar chart of cluster sizes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    vals, counts = np.unique(labels, return_counts=True)
    ax.bar([str(v) for v in vals], counts)
    ax.set_xlabel("cluster")
    ax.set_ylabel("n")
    ax.set_title(title)
    return ax


def plot_crosstab(labels_a: np.ndarray, labels_b: np.ndarray, ax=None, title: str = "",
                  xlabel: str = "labels_b", ylabel: str = "labels_a"):
    """Heatmap of the contingency table between two labelings."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ct = pd.crosstab(labels_a, labels_b)
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ct
