"""Loader for the REAL opto E/I single-trial spike-count data.

Reads the ``trial_counts_b<N>_<animal_id>_s<session>.npz`` files written by
``ichun_opto/regenerate_population_rates.py`` (format: ``ichun_opto/DATA.md``) and turns one
session into a ``(train, test)`` cross-validation split of **individual trials**. No trial
averaging happens here: every row of a ``CVSplit`` is one trial.

The files hold raw E/I population spike counts per trial ``(n_trials, 2, n_bins)`` (1 ms bins,
-0.5..+1.5 s around the first pulse) plus a condition table linked by ``cond_idx``. The
EDGAR-side transforms (``DATA.md`` next to this file) are, per session:

1. keep only conditions with screen contrast ``contrast`` (0 = the paper's blank-screen paradigm);
2. optionally rebin to ``bin_ms`` (exact count sums; default 1 ms = no rebinning);
3. split trials into train / test (``k_fold``: condition-stratified trial folds; ``exp_cond``:
   by experiment type);
4. counts -> rate, divided per population by the mean baseline rate (-0.5..-0.1 s) of the
   **train** trials only, so the test trials never inform the normaliser;
5. chop to the analysis window and build each trial's (u_E, u_I) pulse stimulus.

``load_data.py`` stacks these per-mouse splits into EDGAR's dense ``(n, C, T, 2)`` dicts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


DEFAULT_RESULTS_DIR = "/home/dabin/code/ichun_opto/results"
DEFAULT_GLOB = "trial_counts_b30_*_s1.npz"
FILE_PREFIX = "trial_counts_"

EXPERIMENT_TYPES = (
    "single_E", "single_I", "paired_EE", "paired_II", "paired_EI", "paired_IE",
)

# Paper's normalisation window (ichun_opto/trial_analysis.BASELINE_S).
BASELINE_WINDOW = (-0.5, -0.1)

# Keep bins with ``chop_pre_s <= time < chop_post_s``.
CHOP_PRE_S = -0.05
CHOP_POST_S = 0.40

_ANIMAL_RE = re.compile(r"trial_counts_b(\d+)_(.+)_s(\d+)\.npz$")


def _animal_id(path: str) -> str:
    m = _ANIMAL_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse animal_id from {path!r}")
    return m.group(2)


# ── Copies of ichun_opto/trial_analysis.py helpers ──
# Copied rather than imported: ichun_opto is not a package, and GCP runs upload only the data dir.

def _select(d: dict, **criteria) -> np.ndarray:
    """Trial mask for conditions matching every ``cond_<key> == value`` criterion."""
    rows = np.ones(d["cond_exp_type"].size, bool)
    for col, val in criteria.items():
        rows &= d[f"cond_{col}"] == val
    return np.isin(d["cond_idx"], np.flatnonzero(rows))


def _rebin(counts: np.ndarray, time_axis: np.ndarray,
           factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``factor`` adjacent bins (exact for counts); a trailing remainder is dropped."""
    n = counts.shape[-1] // factor * factor
    c = counts[..., :n].reshape(*counts.shape[:-1], -1, factor).sum(-1)
    t = time_axis[:n].reshape(-1, factor).mean(-1)
    return c, t


def _fold_labels(cond_idx: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per trial: each condition's trials randomly split as evenly as possible."""
    rng = np.random.default_rng(seed)
    fold = np.empty(cond_idx.size, int)
    for k in np.unique(cond_idx):
        idx = np.flatnonzero(cond_idx == k)
        for f, part in enumerate(np.array_split(rng.permutation(idx), n_folds)):
            fold[part] = f
    return fold


def make_stimulus(time_axis: np.ndarray, first_pop: str, second_pop: str,
                  onset2_s: float, dur1_ms: float, dur2_ms: float) -> np.ndarray:
    """The two exogenous pulse channels ``(u_E, u_I)`` for one condition, ``[T, 2]``.

    Pulse 1 drives ``first_pop`` from 0 s for ``dur1_ms``; pulse 2 (paired only, i.e.
    ``second_pop`` non-empty) drives ``second_pop`` from ``onset2_s`` — the condition's
    *onset-to-onset* interval ``cond_onset_ipi_ms`` (= intT + dur1, ichun_opto/DATA.md), not
    ``cond_ipi_ms`` — for ``dur2_ms``. Channel 0 = E, 1 = I.

    ``time_axis`` holds bin CENTRES; a bin is set to 1 if a pulse overlaps any part of it, so
    pulses shorter than a bin are not lost.
    """
    u = np.zeros((len(time_axis), 2), dtype=np.float32)
    dt = float(time_axis[1] - time_axis[0])
    bin_lo = time_axis - dt / 2
    bin_hi = time_axis + dt / 2

    pulses = [(0.0, first_pop, dur1_ms)]
    if second_pop:
        pulses.append((onset2_s, second_pop, dur2_ms))
    for onset, pop, dur_ms in pulses:
        channel = 0 if pop == "E" else 1
        offset = onset + dur_ms / 1000.0
        # The 1 us tolerance stops float error at an exact bin edge from also switching on the
        # neighbouring bin.
        active = (bin_lo < offset - 1e-6) & (bin_hi > onset + 1e-6)
        u[active, channel] = 1.0
    return u


@dataclass
class CVSplit:
    """One side (train or test) of a split: individual trials on a shared chopped time grid.

    ``target_y``/``stim`` are ``[N, T, 2]`` (last axis (E, I) for the response, (u_E, u_I) for
    the stimulus); ``cond_idx`` ``[N]`` is each trial's row in the session's condition table;
    ``meta`` is a length-``N`` list of per-trial dicts aligned to axis 0.
    """
    target_y: np.ndarray   # [N, T, 2] float32, baseline-normalised rate
    stim: np.ndarray       # [N, T, 2] float32
    cond_idx: np.ndarray   # [N] int
    meta: list[dict]

    @property
    def n(self) -> int:
        return int(self.target_y.shape[0])


@dataclass
class CVSamples:
    """A ``(train, test)`` split of one session's trials.

    ``time`` is the shared chopped ``[T]`` grid (s, t=0 at the first pulse onset). ``baseline``
    is the ``(E, I)`` train-trial baseline rate (Hz) both sides were divided by. ``cv_type``:

    * ``"k_fold"``   — trials grouped into ``n_folds`` condition-stratified folds (no averaging
      within a fold); ``test`` = the trials of the held-out fold, ``train`` = the trials of all
      other folds. Same conditions on both sides.
    * ``"exp_cond"`` — ``train`` = trials of ``train_types`` conditions, ``test`` = trials of the
      disjoint ``test_types`` conditions.
    """
    train: CVSplit
    test: CVSplit
    time: np.ndarray       # [T]
    baseline: np.ndarray   # [2] Hz
    cv_type: str


def _load_session(path: str, bin_ms: float, contrast: int):
    """Load one file, keep ``contrast`` conditions, rebin. Returns ``(d, counts, time_axis, keep)``.

    ``counts`` ``(n_kept, 2, n_bins)`` is already restricted to the kept trials ``keep``.
    """
    d = dict(np.load(path))
    file_bin_s = float(d["bin_samples"] / d["sampling_freq_hz"])
    factor_f = (bin_ms / 1000.0) / file_bin_s
    factor = int(round(factor_f))
    if factor < 1 or abs(factor_f - factor) > 1e-9:
        raise ValueError(
            f"bin_ms={bin_ms} is not an integer multiple of the file's {file_bin_s * 1e3:g} ms bins")
    pre_bins = float(d["pre_s"]) / file_bin_s
    if abs(pre_bins / factor - round(pre_bins / factor)) > 1e-9:
        raise ValueError(f"bin_ms={bin_ms} does not keep t=0 on a bin edge (pre_s={d['pre_s']})")

    keep = _select(d, contrast=contrast)
    counts = d["counts"][keep].astype(np.float64)
    time_axis = np.asarray(d["time_axis"], dtype=np.float64)
    if factor > 1:
        counts, time_axis = _rebin(counts, time_axis, factor)
    return d, counts, time_axis, keep


def _make_split(d: dict, animal_id: str, rate: np.ndarray, cond_idx: np.ndarray,
                trial_ids: np.ndarray, time_axis: np.ndarray, mask: np.ndarray,
                **extra) -> CVSplit:
    """Chop normalised rates ``rate (N, 2, n_bins)`` and attach per-trial stimulus + metadata."""
    stim_by_cond = {
        int(k): make_stimulus(
            time_axis, str(d["cond_first_pop"][k]), str(d["cond_second_pop"][k]),
            float(d["cond_onset_ipi_ms"][k]) / 1000.0,
            float(d["cond_dur1_ms"][k]), float(d["cond_dur2_ms"][k]),
        )[mask]
        for k in np.unique(cond_idx)
    }
    meta = [
        {
            "animal_id": animal_id,
            "trial": int(t),
            "condition_index": int(k),
            "experiment_type": str(d["cond_exp_type"][k]),
            "ipi_ms": int(d["cond_ipi_ms"][k]),
            "onset_ipi_ms": float(d["cond_onset_ipi_ms"][k]),
            "dur1_ms": float(d["cond_dur1_ms"][k]),
            "dur2_ms": float(d["cond_dur2_ms"][k]),
            "first_pop": str(d["cond_first_pop"][k]),
            "second_pop": str(d["cond_second_pop"][k]),
            **extra,
        }
        for t, k in zip(trial_ids, cond_idx)
    ]
    return CVSplit(
        target_y=np.ascontiguousarray(rate[:, :, mask].transpose(0, 2, 1), dtype=np.float32),
        stim=np.stack([stim_by_cond[int(k)] for k in cond_idx]).astype(np.float32),
        cond_idx=cond_idx.astype(int),
        meta=meta,
    )


def _split_by_trials(path: str, cv_type: str, train_sel, test_sel, chop, bin_ms, contrast,
                     **extra) -> CVSamples:
    """Shared body: ``train_sel``/``test_sel`` map the kept trials' ``cond_idx`` to boolean masks."""
    animal_id = _animal_id(path)
    d, counts, time_axis, keep = _load_session(path, bin_ms, contrast)
    cond_idx = d["cond_idx"][keep]
    trial_ids = np.flatnonzero(keep)
    tr, te = train_sel(cond_idx), test_sel(cond_idx)
    if not tr.any() or not te.any():
        raise ValueError(f"{cv_type} split of {path} leaves a side empty "
                         f"(train={int(tr.sum())}, test={int(te.sum())})")

    bin_s = float(time_axis[1] - time_axis[0])
    rate = counts / bin_s                                            # (N, 2, n_bins) Hz
    bl = (time_axis >= BASELINE_WINDOW[0]) & (time_axis < BASELINE_WINDOW[1])
    # Ratio of means over train trials only: single-trial baselines hold few spikes, and using
    # test trials here would leak them into the normaliser.
    baseline = rate[tr][:, :, bl].mean(axis=(0, 2))                  # (2,)
    if np.any(baseline <= 0):
        raise ValueError(f"non-positive train baseline rate {baseline} in {path}")
    rate = rate / baseline[None, :, None]

    lo, hi = chop
    mask = (time_axis >= lo) & (time_axis < hi)
    if not mask.any():
        raise ValueError(f"chop window {chop} keeps no bins")

    return CVSamples(
        train=_make_split(d, animal_id, rate[tr], cond_idx[tr], trial_ids[tr], time_axis, mask,
                          split="train", **extra),
        test=_make_split(d, animal_id, rate[te], cond_idx[te], trial_ids[te], time_axis, mask,
                         split="test", **extra),
        time=time_axis[mask].astype(np.float32),
        baseline=baseline,
        cv_type=cv_type,
    )


def build_cv_samples(
    path: str,
    cv_type: str = "k_fold",
    *,
    held_out_fold: int = 0,
    n_folds: int = 3,
    fold_seed: int = 0,
    train_types: list[str] | None = None,
    test_types: list[str] | None = None,
    chop: tuple[float, float] = (CHOP_PRE_S, CHOP_POST_S),
    bin_ms: float = 1.0,
    contrast: int = 0,
) -> CVSamples:
    """Build one ``(train, test)`` split of one session's individual trials (see ``CVSamples``).

    * ``"k_fold"``   — trials grouped into ``n_folds`` condition-stratified folds seeded by
      ``fold_seed``; ``held_out_fold``'s trials are ``test``, the rest ``train``. Rotate
      ``held_out_fold`` over ``range(n_folds)`` for full k-fold CV.
    * ``"exp_cond"`` — ``train_types`` trials vs ``test_types`` trials (disjoint).

    Raises ``ValueError`` on an unknown ``cv_type``, bad fold / type arguments, a ``bin_ms`` that
    is not a multiple of the file's bins, or a split with an empty side.
    """
    common = dict(chop=chop, bin_ms=bin_ms, contrast=contrast)
    if cv_type == "k_fold":
        if not 0 <= held_out_fold < n_folds:
            raise ValueError(f"held_out_fold={held_out_fold} out of range for n_folds={n_folds}")
        fold_of = lambda c: _fold_labels(c, n_folds, fold_seed)    # noqa: E731
        return _split_by_trials(
            path, cv_type,
            train_sel=lambda c: fold_of(c) != held_out_fold,
            test_sel=lambda c: fold_of(c) == held_out_fold,
            held_out_fold=held_out_fold, **common,
        )
    if cv_type == "exp_cond":
        if train_types is None or test_types is None:
            raise ValueError("exp_cond CV requires train_types and test_types")
        train_set, test_set = set(train_types), set(test_types)
        if train_set & test_set:
            raise ValueError(f"train_types and test_types overlap: {sorted(train_set & test_set)}")
        unknown = (train_set | test_set) - set(EXPERIMENT_TYPES)
        if unknown:
            raise ValueError(f"unknown experiment type(s) {sorted(unknown)}; "
                             f"valid types are {list(EXPERIMENT_TYPES)}")
        exp_type = np.asarray(np.load(path)["cond_exp_type"])
        return _split_by_trials(
            path, cv_type,
            train_sel=lambda c: np.isin(exp_type[c], list(train_set)),
            test_sel=lambda c: np.isin(exp_type[c], list(test_set)),
            **common,
        )
    raise ValueError(f"unknown cv_type {cv_type!r}; expected 'k_fold' or 'exp_cond'")


def verify_cv_samples(cv: CVSamples, baseline_tol: float = 1e-3) -> None:
    """Sanity-check a split. Raises ``AssertionError`` on structural problems.

    Train baseline is ~1 by construction only when the chop window contains the baseline
    window, so it is reported rather than asserted in general; pulse onsets are always checked.
    """
    time = np.asarray(cv.time)
    dt = float(time[1] - time[0])
    for name, sp in (("train", cv.train), ("test", cv.test)):
        N, T, C = sp.target_y.shape
        assert C == 2 and sp.stim.shape == (N, T, 2) and T == time.size, (
            f"{name}: bad shapes y={sp.target_y.shape} stim={sp.stim.shape} T={time.size}")
        assert len(sp.meta) == N and sp.cond_idx.shape == (N,)
        assert np.isfinite(sp.target_y).all(), f"{name}: non-finite target_y"

        def _first_active(onset: float) -> int:
            return int(np.argmax(time + dt / 2 > onset + 1e-6))     # same edge rule as make_stimulus

        for i in np.random.default_rng(0).choice(N, size=min(N, 64), replace=False):
            m = sp.meta[i]
            if not (time[0] - dt / 2 <= 0.0 < time[-1] + dt / 2):
                break                                                 # onset chopped away
            ch1 = 0 if m["first_pop"] == "E" else 1
            assert sp.stim[i, _first_active(0.0), ch1] == 1.0, f"{name} trial {i}: pulse 1 not on"
            onset2 = m["onset_ipi_ms"] / 1000.0
            if m["second_pop"] and onset2 < time[-1]:
                ch2 = 0 if m["second_pop"] == "E" else 1
                assert sp.stim[i, _first_active(onset2), ch2] == 1.0, (
                    f"{name} trial {i}: pulse 2 not on at {m['onset_ipi_ms']} ms")

    lo, hi = BASELINE_WINDOW
    bl = (time >= lo) & (time < hi)
    if time[0] - dt / 2 <= lo + 1e-9 and time[-1] + dt / 2 >= hi - 1e-9:   # chop holds the window
        b = cv.train.target_y[:, bl, :].mean(axis=(0, 1))
        assert np.all(np.abs(b - 1.0) < baseline_tol), f"train baseline {b} != 1"
    print(f"[verify] OK: {cv.cv_type}, train N={cv.train.n}, test N={cv.test.n}, T={time.size}, "
          f"dt={dt * 1e3:g} ms, baseline (Hz) E={cv.baseline[0]:.1f} I={cv.baseline[1]:.1f}")


if __name__ == "__main__":
    import glob
    for p in sorted(glob.glob(os.path.join(DEFAULT_RESULTS_DIR, DEFAULT_GLOB))):
        print(os.path.basename(p))
        verify_cv_samples(build_cv_samples(p, "k_fold"))
