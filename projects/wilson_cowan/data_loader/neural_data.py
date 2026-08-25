"""Phase 0 loader for the REAL opto E/I population-rate data.

Reads the ``population_rates_<animal_id>_s1.npz`` files documented in ``DATA.md`` and
turns them into batched trajectory samples for the neural-dynamics training-objective
benchmark (``edgar_neural_dynamics_training_objectives_plan.md``, Phase 0 / §17).

This is a **standalone** loader for that separate training pipeline (Phases 1-2 train
latent E/I dynamical models; EDGAR only enters at Phase 3-4). It deliberately does NOT
implement the EDGAR engine's ``load_data -> (X_discover, X_validate, X_eval)`` contract;
that is the sibling ``load_data.py``, which loads the *synthetic* ``wc_fold*.npz`` files.

One sample = one ``(animal x experiment_type x condition x fold)`` per-fold trial-averaged
E/I trace (§4). Folds with no trials are all-NaN (``DATA.md``) and are dropped. Every trace
lives on the same -0.5..+1.5 s, 1 ms grid, so samples are stacked into batched arrays with a
parallel metadata list. ``sample_weight`` is 1.0 for every fold (equal weighting, §4).
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


DEFAULT_RESULTS_DIR = "/home/dabin/code/ichun_opto/results"
DEFAULT_GLOB = "population_rates_*_s1_trimmed.npz"

# All experiment types that may be present; a type is present iff its keys exist (§17.2).
EXPERIMENT_TYPES = (
    "single_E", "single_I", "paired_EE", "paired_II", "paired_EI", "paired_IE",
)

# Pre-stimulus baseline-normalisation window used by the generator (DATA.md): rates are
# divided by the mean over -0.5..-0.1 s, so baseline ~ 1 there.
BASELINE_WINDOW = (-0.5, -0.1)

# keeping bins with ``chop_pre_s <= time < chop_post_s``. Defaults −50 ms .. +400 ms.
CHOP_PRE_S = -0.05
CHOP_POST_S = 0.40

_ANIMAL_RE = re.compile(r"population_rates_(.+)_s1_trimmed\.npz$")


@dataclass
class NeuralDataset:
    """Batched real-data trajectories plus a parallel metadata list.

    ``target_y``/``u`` are ``[N, T, 2]`` (last axis = (E, I) for the response, (u_E, u_I)
    for the stimulus). ``time`` is the shared ``[T]`` grid in seconds; ``dt`` is the scalar
    bin width. ``sample_weight`` is ``[N]`` (all 1.0). ``meta`` is a length-``N`` list of
    dicts (no arrays) aligned to axis 0 of the arrays.
    """
    target_y: jnp.ndarray      # [N, T, 2] float32
    u: jnp.ndarray             # [N, T, 2] float32
    time: jnp.ndarray          # [T] float32 (seconds)
    dt: float
    sample_weight: jnp.ndarray  # [N] float32
    meta: list[dict]

    @property
    def n_samples(self) -> int:
        return int(self.target_y.shape[0])


def make_stimulus(time_axis: np.ndarray, condition: dict) -> np.ndarray:
    """Build the two exogenous pulse channels ``(u_E, u_I)`` for one condition (§3).

    Pulse 1 drives ``first_pop`` at onset 0 s; for paired conditions (``ipi_ms > 0``) pulse 2
    drives ``second_pop`` at ``ipi_ms/1000``. Durations come from ``dur_ms`` (scalar -> both
    pulses; 2-element list -> per pulse). Channel 0 = E, channel 1 = I. Returns ``[T, 2]``.
    """
    u = np.zeros((len(time_axis), 2), dtype=np.float32)

    durations = condition["dur_ms"]
    if np.isscalar(durations):
        durations = [durations, durations]

    pulses = [(0.0, condition.get("first_pop"), durations[0])]
    if condition["ipi_ms"] > 0:
        pulses.append(
            (condition["ipi_ms"] / 1000.0, condition.get("second_pop"), durations[1])
        )

    for onset, pop, dur_ms in pulses:
        if pop not in ("E", "I"):
            continue
        channel = 0 if pop == "E" else 1
        offset = onset + dur_ms / 1000.0
        active = (time_axis >= onset) & (time_axis < offset)
        u[active, channel] = 1.0

    return u


def _animal_id(path: str) -> str:
    m = _ANIMAL_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse animal_id from {path!r}")
    return m.group(1)


def load_session(path: str) -> list[dict]:
    """Load one npz -> a list of per-(type, condition, fold) sample dicts.

    Enumerates the experiment types actually present, parses the condition JSON, and emits
    one sample per (condition, fold) with a nonzero trial count (empty folds are all-NaN).
    """
    animal_id = _animal_id(path)
    d = np.load(path, allow_pickle=True)

    samples: list[dict] = []
    for etype in EXPERIMENT_TYPES:
        rkey = f"{etype}__responses"
        if rkey not in d.files:
            continue
        resp = np.asarray(d[rkey])                       # (n_cond, n_folds, 2, n_bins)
        time_axis = np.asarray(d[f"{etype}__time_axis"], dtype=np.float64)
        conditions = json.loads(str(d[f"{etype}__conditions"]))

        n_cond, n_folds, _, _ = resp.shape
        for c in range(n_cond):
            cond = conditions[c]
            u = make_stimulus(time_axis, cond)           # (n_bins, 2), same for all folds
            for f in range(n_folds):
                if cond["n_trials_per_fold"][f] <= 0:
                    continue                             # empty fold -> all-NaN, drop
                target_y = np.ascontiguousarray(resp[c, f].T, dtype=np.float32)  # (n_bins, 2)
                samples.append({
                    "target_y": target_y,
                    "u": u,
                    "time": time_axis,
                    "sample_weight": 1.0,
                    "animal_id": animal_id,
                    "experiment_type": etype,
                    "condition_index": c,
                    "fold": f,
                    "ipi_ms": cond["ipi_ms"],
                    "dur_ms": cond["dur_ms"],
                    "first_pop": cond["first_pop"],
                    "second_pop": cond["second_pop"],
                    "n_trials": cond["n_trials_per_fold"][f],
                })
    return samples


def load_neural_dataset(
    results_dir: str = DEFAULT_RESULTS_DIR,
    glob: str = DEFAULT_GLOB,
) -> NeuralDataset:
    """Load every matching session and stack into one batched ``NeuralDataset``.

    All traces are asserted to share one time grid before stacking (they should: -0.5..+1.5 s
    at 1 ms per ``DATA.md``); a mismatch raises with the offending session/type named.
    """
    paths = sorted(_glob.glob(os.path.join(results_dir, glob)))
    if not paths:
        raise FileNotFoundError(f"no files matching {glob!r} in {results_dir!r}")

    all_samples: list[dict] = []
    per_session: dict[str, list[dict]] = {}
    ref_time: np.ndarray | None = None
    for path in paths:
        samples = load_session(path)
        per_session[_animal_id(path)] = samples
        for s in samples:
            if ref_time is None:
                ref_time = s["time"]
            elif s["time"].shape != ref_time.shape or not np.allclose(
                s["time"], ref_time, atol=1e-9
            ):
                raise ValueError(
                    f"time grid mismatch in {s['animal_id']}/{s['experiment_type']} "
                    f"(cond {s['condition_index']}): expected shape {ref_time.shape}, "
                    f"got {s['time'].shape}"
                )
        all_samples.extend(samples)

    if not all_samples:
        raise ValueError(f"no non-empty folds found across {len(paths)} session(s)")

    assert ref_time is not None
    time = ref_time.astype(np.float32)
    diffs = np.diff(ref_time)
    if not np.allclose(diffs, diffs[0], atol=1e-9):
        raise ValueError("time axis is not uniformly spaced")
    dt = float(np.median(diffs))

    target_y = jnp.asarray(np.stack([s["target_y"] for s in all_samples]), dtype=jnp.float32)
    u = jnp.asarray(np.stack([s["u"] for s in all_samples]), dtype=jnp.float32)
    sample_weight = jnp.asarray(
        np.array([s["sample_weight"] for s in all_samples], dtype=np.float32)
    )
    meta = [{k: v for k, v in s.items() if k not in ("target_y", "u", "time")}
            for s in all_samples]

    for animal, samples in per_session.items():
        types = sorted({s["experiment_type"] for s in samples})
        print(f"[neural_data] {animal}: {len(samples)} samples, types={types}")
    print(f"[neural_data] total N={len(all_samples)}, T={len(time)}, dt={dt:.6f} s "
          f"from {len(paths)} session(s)")

    return NeuralDataset(
        target_y=target_y,
        u=u,
        time=jnp.asarray(time),
        dt=dt,
        sample_weight=sample_weight,
        meta=meta,
    )


@dataclass
class CVSplit:
    """One (target, stimulus, metadata) set of conditions on a shared chopped time grid.

    ``target_y``/``stim`` are ``[N, T, 2]`` (last axis (E, I) for the response, (u_E, u_I) for the
    stimulus); ``meta`` is a length-``N`` list of per-condition dicts aligned to axis 0.
    """
    target_y: np.ndarray   # [N, T, 2]
    stim: np.ndarray       # [N, T, 2]
    meta: list[dict]

    @property
    def n(self) -> int:
        return int(self.target_y.shape[0])


@dataclass
class CVSamples:
    """A ``(train, test)`` cross-validation split of one session's conditions.

    ``train``/``test`` are ``CVSplit``s and ``time`` is the shared chopped ``[T]`` grid (seconds,
    t=0 at onset). ``cv_type`` records how the split was formed:

    * ``"k_fold"``   — SAME conditions in train and test, trials split by fold: ``train`` is the
      trial-weighted mean over the training folds, ``test`` the held-out fold. Tests robustness to
      trial noise (§5 primary CV). ``train.stim`` and ``test.stim`` are identical.
    * ``"exp_cond"`` — DISJOINT condition sets in train vs test (each an all-trials mean): fit
      params on one set of stimulus conditions, evaluate on held-out conditions. Tests
      generalization across perturbations (§5 secondary test). ``train.n`` and ``test.n`` differ.
    """
    train: CVSplit
    test: CVSplit
    time: np.ndarray       # [T]
    cv_type: str


def _iter_conditions(d, chop: tuple[float, float]):
    """Yield ``(etype, cond_idx, cond, resp_c, mask, time_axis)`` for every present (type, condition).

    ``resp_c`` is ``(n_folds, 2, n_bins)`` for condition ``c``; ``mask`` selects the chopped bins.
    """
    lo, hi = chop
    for etype in EXPERIMENT_TYPES:
        rkey = f"{etype}__responses"
        if rkey not in d.files:
            print(f"[neural_data] skipping {etype}: no {rkey} in npz")
            continue
        resp = np.asarray(d[rkey])                       # (n_cond, n_folds, 2, n_bins)
        time_axis = np.asarray(d[f"{etype}__time_axis"], dtype=np.float64)
        conditions = json.loads(str(d[f"{etype}__conditions"]))
        mask = (time_axis >= lo) & (time_axis < hi)
        if not mask.any():
            raise ValueError(f"chop window {chop} keeps no bins of {etype} time axis")
        for cond_idx, cond in enumerate(conditions):
            yield etype, cond_idx, cond, resp[cond_idx], mask, time_axis


def _cond_meta(animal_id: str, etype: str, c: int, cond: dict, **extra) -> dict:
    """Per-condition metadata dict (aligned to a ``CVSplit`` axis-0 entry)."""
    return {
        "animal_id": animal_id,
        "experiment_type": etype,
        "condition_index": c,
        "ipi_ms": cond["ipi_ms"],
        "dur_ms": cond["dur_ms"],
        "first_pop": cond["first_pop"],
        "second_pop": cond["second_pop"],
        "n_trials": cond["n_trials"],
        **extra,
    }


def _subsample_idx(n: int, max_n: int | None, seed: int) -> np.ndarray:
    """Indices kept after an optional deterministic cap of ``n`` items to ``max_n``."""
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, int(max_n), replace=False))


def build_cv_samples(
    path: str,
    cv_type: str = "k_fold",
    *,
    held_out_fold: int = 0,
    train_types: list[str] | None = None,
    test_types: list[str] | None = None,
    chop: tuple[float, float] = (CHOP_PRE_S, CHOP_POST_S),
    max_conditions: int | None = None,
    subsample_seed: int = 0,
) -> CVSamples:
    """Build one ``(train, test)`` CV split from a ``population_rates_*_s1_trimmed.npz`` session.

    Dispatches on ``cv_type`` to one of two train/test conventions (see ``CVSamples``):

    * ``"k_fold"``   → ``build_k_fold_cv_samples`` (uses ``held_out_fold``); same conditions in
      train and test, trials split by fold.
    * ``"exp_cond"`` → ``build_exp_cond_cv_samples`` (uses ``train_types``/``test_types``);
      disjoint condition sets in train vs test.

    ``chop``, ``max_conditions`` and ``subsample_seed`` are forwarded. Raises ``ValueError`` on an
    unknown ``cv_type`` or missing type lists for ``"exp_cond"``.
    """
    if cv_type == "k_fold":
        return build_k_fold_cv_samples(
            path, held_out_fold=held_out_fold, chop=chop,
            max_conditions=max_conditions, subsample_seed=subsample_seed,
        )
    if cv_type == "exp_cond":
        if train_types is None or test_types is None:
            raise ValueError("exp_cond CV requires train_types and test_types")
        return build_exp_cond_cv_samples(
            path, train_types=train_types, test_types=test_types, chop=chop,
            max_conditions=max_conditions, subsample_seed=subsample_seed,
        )
    raise ValueError(f"unknown cv_type {cv_type!r}; expected 'k_fold' or 'exp_cond'")


def build_k_fold_cv_samples(
    path: str,
    held_out_fold: int = 0,
    chop: tuple[float, float] = (CHOP_PRE_S, CHOP_POST_S),
    max_conditions: int | None = None,
    subsample_seed: int = 0,
) -> CVSamples:
    """k-fold **trial** CV: same conditions in train and test, trials split by fold.

    The saved responses are ``(n_conditions, n_folds, 2, n_bins)``: each condition's trials were
    deterministically partitioned into ``n_folds`` folds and trial-averaged per fold. For the
    chosen ``held_out_fold``, every kept condition contributes ``test`` = that fold's mean and
    ``train`` = the trial-count-weighted mean over the other folds. Same stimulus
    conditions in train and test, disjoint trials → measures robustness to trial noise. 
    Rotate ``held_out_fold`` over ``range(n_folds)`` for full k-fold CV.

    Conditions with an empty fold (a 0 in ``n_trials_per_fold`` → all-NaN, DATA.md)
    are dropped. ``max_conditions`` caps the shared condition set (deterministic subsample).

    Returns a ``CVSamples`` (``cv_type="k_fold"``) whose ``train`` and ``test`` share the same
    ``N`` conditions and the same stimulus; ``meta`` carries ``held_out_fold``.

    Raises:
        ValueError: if ``held_out_fold`` is out of range, the chop window keeps no bins, or no
            condition has all folds non-empty.
    """
    animal_id = _animal_id(path)
    d = np.load(path, allow_pickle=True)

    train_y: list[np.ndarray] = []
    test_y: list[np.ndarray] = []
    stim: list[np.ndarray] = []
    meta: list[dict] = []
    time_chopped: np.ndarray | None = None

    for etype, c, cond, resp_c, mask, time_axis in _iter_conditions(d, chop):
        n_folds = resp_c.shape[0]
        if not 0 <= held_out_fold < n_folds:
            raise ValueError(f"held_out_fold={held_out_fold} out of range for n_folds={n_folds}")
        w = np.asarray(cond["n_trials_per_fold"], dtype=np.float64)
        if np.any(w <= 0):
            continue                                     # empty fold -> all-NaN; skip condition
        train_folds = [f for f in range(n_folds) if f != held_out_fold]
        wt = w[train_folds]
        tr = np.tensordot(wt, resp_c[train_folds], axes=(0, 0)) / wt.sum()  # (2, n_bins)
        te = resp_c[held_out_fold]                       # (2, n_bins)
        if np.isnan(tr).any() or np.isnan(te).any():
            continue

        u = make_stimulus(time_axis, cond)               # (n_bins, 2)
        train_y.append(tr[:, mask].T.astype(np.float32))       # (T, 2) = (E, I)
        test_y.append(te[:, mask].T.astype(np.float32))
        stim.append(u[mask].astype(np.float32))                # (T, 2) = (u_E, u_I)
        if time_chopped is None:
            time_chopped = time_axis[mask].astype(np.float32)
        meta.append(_cond_meta(animal_id, etype, c, cond, held_out_fold=held_out_fold))

    if not train_y:
        raise ValueError(f"no valid (all-folds-nonempty) conditions in {path}")

    idx = _subsample_idx(len(train_y), max_conditions, subsample_seed)
    stim_arr = np.stack([stim[i] for i in idx])
    kept_meta = [meta[i] for i in idx]
    assert time_chopped is not None
    return CVSamples(
        train=CVSplit(np.stack([train_y[i] for i in idx]), stim_arr, kept_meta),
        test=CVSplit(np.stack([test_y[i] for i in idx]), stim_arr, kept_meta),
        time=time_chopped,
        cv_type="k_fold",
    )


def build_exp_cond_cv_samples(
    path: str,
    train_types: list[str],
    test_types: list[str],
    chop: tuple[float, float] = (CHOP_PRE_S, CHOP_POST_S),
    max_conditions: int | None = None,
    subsample_seed: int = 0,
) -> CVSamples:
    """Held-out **condition** CV: fit on some stimulus conditions, test on held-out ones.

    Conditions are partitioned by ``experiment_type``: types in ``train_types`` form the ``train``
    split, types in ``test_types`` the ``test`` split (types in neither are dropped). Each kept
    condition contributes its ALL-trials mean (trial-count-weighted mean over its non-empty folds)
    — folds are not used as a CV axis here. Fitting params on ``train`` and scoring on ``test``
    then measures generalization to unseen perturbation conditions, potentially a stronger
    probe than the trial-noise ``k_fold`` split.

    ``max_conditions`` caps each split independently (deterministic subsample). ``train`` and ``test`` 
    are disjoint condition sets and generally differ in ``N``.

    Raises:
        ValueError: if the type lists overlap or name an unknown type, the chop window keeps no
            bins, or either split ends up empty.
    """
    train_set, test_set = set(train_types), set(test_types)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"train_types and test_types overlap: {sorted(overlap)}")
    unknown = (train_set | test_set) - set(EXPERIMENT_TYPES)
    if unknown:
        raise ValueError(
            f"unknown experiment type(s) {sorted(unknown)}; valid types are {list(EXPERIMENT_TYPES)}"
        )

    animal_id = _animal_id(path)
    d = np.load(path, allow_pickle=True)
    buckets: dict[str, tuple[list, list, list]] = {"train": ([], [], []), "test": ([], [], [])}
    time_chopped: np.ndarray | None = None

    for etype, c, cond, resp_c, mask, time_axis in _iter_conditions(d, chop):
        which = "train" if etype in train_set else "test" if etype in test_set else None
        if which is None:
            continue
        w = np.asarray(cond["n_trials_per_fold"], dtype=np.float64)
        valid = w > 0
        if not valid.any():
            continue                                     # all folds empty; skip condition
        mean = np.tensordot(w[valid], resp_c[valid], axes=(0, 0)) / w[valid].sum()  # (2, n_bins)
        if np.isnan(mean).any():
            continue

        u = make_stimulus(time_axis, cond)               # (n_bins, 2)
        ys, ss, ms = buckets[which]
        ys.append(mean[:, mask].T.astype(np.float32))          # (T, 2) = (E, I)
        ss.append(u[mask].astype(np.float32))                  # (T, 2) = (u_E, u_I)
        ms.append(_cond_meta(animal_id, etype, c, cond, split=which))
        if time_chopped is None:
            time_chopped = time_axis[mask].astype(np.float32)

    if not buckets["train"][0] or not buckets["test"][0]:
        raise ValueError(
            f"exp_cond split leaves a side empty (train={len(buckets['train'][0])}, "
            f"test={len(buckets['test'][0])}); check train_types/test_types vs present types"
        )

    def _finish(which: str, seed: int) -> CVSplit:
        ys, ss, ms = buckets[which]
        idx = _subsample_idx(len(ys), max_conditions, seed)
        return CVSplit(
            np.stack([ys[i] for i in idx]),
            np.stack([ss[i] for i in idx]),
            [ms[i] for i in idx],
        )

    assert time_chopped is not None
    return CVSamples(
        train=_finish("train", subsample_seed),
        test=_finish("test", subsample_seed + 1),
        time=time_chopped,
        cv_type="exp_cond",
    )


def verify_dataset(ds: NeuralDataset, baseline_tol: float = 0.15) -> None:
    """Sanity-check a loaded dataset (§17 step 6). Hard-fails on structural problems;
    warns on baseline outliers. Raises ``AssertionError`` on failure."""
    y = np.asarray(ds.target_y)
    u = np.asarray(ds.u)
    time = np.asarray(ds.time)
    N, T, C = y.shape

    assert C == 2 and u.shape == (N, T, 2), f"bad shapes: y={y.shape}, u={u.shape}"
    assert len(ds.meta) == N, f"meta length {len(ds.meta)} != N {N}"
    assert ds.sample_weight.shape == (N,)
    assert not np.isnan(y).any(), "NaNs remain in target_y (empty folds should be dropped)"

    # dt matches a 1 ms grid.
    assert abs(ds.dt - 0.001) < 1e-6, f"dt={ds.dt} not ~0.001 s"

    # Baseline ~ 1 over the pre-stimulus normalisation window, per channel.
    lo, hi = BASELINE_WINDOW
    base_mask = (time >= lo) & (time < hi)
    assert base_mask.any(), f"no bins in baseline window {BASELINE_WINDOW}"
    base_mean = y[:, base_mask, :].mean(axis=(0, 1))   # (2,)
    for ci, name in enumerate("EI"):
        if abs(base_mean[ci] - 1.0) > baseline_tol:
            print(f"[verify] WARNING: {name} baseline mean {base_mean[ci]:.3f} "
                  f"deviates from 1 by > {baseline_tol}")

    # Pulse timing matches ipi_ms on a sampled subset: the first bin at/after each onset
    # (centres straddle 0, so the boxcar starts at the first bin with time >= onset).
    def _first_active(onset: float) -> int:
        return int(np.argmax(time >= onset))

    rng = np.random.default_rng(0)
    check = rng.choice(N, size=min(N, 64), replace=False)
    for i in check:
        m = ds.meta[i]
        ch1 = 0 if m["first_pop"] == "E" else 1
        assert u[i, _first_active(0.0), ch1] == 1.0, (
            f"sample {i} ({m['experiment_type']}): pulse-1 not on at t=0")
        if m["ipi_ms"] > 0:
            ch2 = 0 if m["second_pop"] == "E" else 1
            assert u[i, _first_active(m["ipi_ms"] / 1000.0), ch2] == 1.0, (
                f"sample {i} ({m['experiment_type']}): pulse-2 not on at "
                f"ipi={m['ipi_ms']}ms")

    print(f"[verify] OK: N={N}, T={T}, dt={ds.dt:.6f} s, "
          f"baseline E={base_mean[0]:.3f} I={base_mean[1]:.3f}")


if __name__ == "__main__":
    ds = load_neural_dataset()
    verify_dataset(ds)
    print(ds.target_y.shape, ds.u.shape, ds.time.shape, ds.dt, ds.n_samples)
    print(ds.meta[0])
