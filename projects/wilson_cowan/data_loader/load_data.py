"""EDGAR entry points for the Wilson-Cowan (WC) neural-dynamics training-objective benchmark.

The evolved program is the WC transition (see ``seed_programs/wilson_cowan.py``):
    ``model(state, y_prev, params) -> (new_state, mean)``
    * ``y_prev`` is a **dict** ``{"E_prev","I_prev","stim_E_prev","stim_I_prev"}`` — the previous
      observation bundled with the previous stimulus.
    * ``new_state`` is the hidden carry (an empty dict for the stateless base model; a model
      with a latent ``S`` carries ``{"S": ...}``).
    * ``mean`` is ``(E, I)`` — the predicted next observation.
    * ``params`` is a dict of learnable WC params. Hidden-state initial values are declared with
      the ``s0_`` prefix; the per-objective ``_split_params_non_kalman`` / ``_split_params_kalman``
      helpers strip them into the scan carry and hand the rest to ``model``.

This module drives that single transition in the modes required by the four training
objectives — teacher-forced one-step, autonomous rollout from anchors, and a full autonomous rollout — 
and assembles``model_output`` dict that the loss functions consume. 

The four objectives live in ``losses/`` (one file each). ``loss_fn`` dispatches to one of them by
the objective set in the project ``config.yaml`` (``project_params.objective``, ``A``/``B``/``C``/
``D``) so a benchmark run selects its objective without editing code. The rollout horizon/anchors
are likewise config-driven (``project_params.rollout_k`` / ``anchor_stride``); see the
configuration note below for how they reach ``apply_model`` / ``loss_fn``. The time-axis dt is
inferred from each session's stored ``time_axis`` on the real opto path; ``dt_seconds`` is only a
bin->seconds convention for the synthetic files, which store no time axis.

Cross-validation is over repeats: ``simulate_data.save_kfold_splits`` writes ``wc*_fold{f}.npz``
files, each a repeat-averaged ``train_data`` / ``test_data`` pair (shape ``(n_samples, 2, T, 2)``);
``load_data`` reads one and splits the *samples* 50/50 into EDGAR's discover / validate sets.
"""
from __future__ import annotations

import glob as _glob
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# ``uv run edgar`` console-script entry point. Put the repo root on the path so all invocations work.
_this = globals().get("__file__")
_repo_root = (
    os.path.abspath(os.path.join(os.path.dirname(_this), "..", "..", ".."))  # <root>/projects/wilson_cowan/data_loader/
    if _this else os.getcwd()  # exec'd: EDGAR is always launched from the repo root (relative config path)
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from projects.wilson_cowan.data_loader.losses import (
    loss_A_one_step_tf,
    loss_B_rollout,
    loss_C_latent_consistency,
    loss_D_dynamics_aware,
    loss_E_kalman,
    loss_F_rollout_nll,
)
from projects.wilson_cowan.data_loader.neural_data import (
    DEFAULT_GLOB,
    EXPERIMENT_TYPES,
    _animal_id,
    build_cv_samples,
)

# ``loss_fn`` is cloudpickled to the scoring subprocess, which runs under the ``spawn`` start
# method (edgar.scoring.scoring) and does NOT inherit the launcher's sys.path — so the default
# pickle-by-reference of these ``projects.*`` loss modules could fail to re-import there. Force
# them to serialize by value so the subprocess needs no ``projects`` import.
try:
    import importlib as _importlib
    import cloudpickle as _cloudpickle

    for _modname in (
        "projects.wilson_cowan.data_loader.losses.loss_common",
        "projects.wilson_cowan.data_loader.losses.loss_a_one_step",
        "projects.wilson_cowan.data_loader.losses.loss_b_rollout",
        "projects.wilson_cowan.data_loader.losses.loss_c_latent_consistency",
        "projects.wilson_cowan.data_loader.losses.loss_d_dynamics_aware",
        "projects.wilson_cowan.data_loader.losses.loss_e_kalman",
        "projects.wilson_cowan.data_loader.losses.loss_f_rollout_nll",
    ):
        _cloudpickle.register_pickle_by_value(_importlib.import_module(_modname))
except Exception:
    pass


# ── Objective + rollout configuration ──
# Precedence: explicit config kwarg > env var > built-in default (below).
DEFAULT_OBJECTIVE = "A"       # A one-step MSE | B rollout MSE | C +latent-consistency | D +signatures | E Kalman NLL | F rollout NLL (deterministic SSM)
DEFAULT_ROLLOUT_K = 3        # autonomous-rollout horizon (bins)
DEFAULT_ANCHOR_STRIDE = 1   # bins between rollout anchors
DEFAULT_DT_SECONDS = 0.001    # SYNTHETIC-only fallback bin width; real data infers dt from time_axis
DEFAULT_WARMUP_BINS = 0       # burn-in bins excluded from the loss (plan §6); 0 = no burn-in

_OBJECTIVES = {
    "A": loss_A_one_step_tf,
    "B": loss_B_rollout,
    "C": loss_C_latent_consistency,
    "D": loss_D_dynamics_aware,
    "E": loss_E_kalman,
    "F": loss_F_rollout_nll,
}

# ── Objective-E (Kalman) noise/init hyperparameters ──
# The EKF adds noise-model + initial-covariance params (``kf_log_*``, stored as LOG-VARIANCES for
# positivity) on top of the WCS dynamics params. Their per-sample initial values come from
# ``param_est`` (data-driven; see param_est2), and their fixed fallback lives WITH THE MODEL as
# ``model.KF_DEFAULT_PARAMS`` — used when a key is absent from ``params`` (an evolved variant whose
# param_est omits them, or a fixed-noise run). ``_kf_hyper`` resolves each key params -> model
# default -> error. The initial E/I latent mean is seeded from the first observation; the initial
# S mean reuses the existing ``s0_S`` dynamics param.


def _rollout_anchors(T: int) -> tuple[np.ndarray, int]:
    """Pick the start times for the autonomous-rollout training windows (objectives B/C/D).

    Objectives B/C/D don't just roll the model out once from the start of the trajectory — they
    roll it out from many starting points *along* the trajectory and average the error. Each such
    starting time is an "anchor". From anchor ``a`` the model is seeded with the state inferred at
    time ``a`` and then run autonomously (feeding its own predictions back) to predict the next
    ``K`` steps, ``a+1 .. a+K``; those predictions are scored against the real data there.

    This returns the anchor start indices and the horizon ``K`` (both driven by config
    ``project_params.rollout_k`` / ``anchor_stride`` via env, see the top-of-module note):
      * ``K``  = rollout length in bins = ``min(rollout_k, T-2)``.
      * anchors = an evenly spaced grid ``w, w+anchor_stride, w+2*anchor_stride, ...`` up to the
        last start for which a full ``K``-step window still fits (``T-1-K``), where ``w`` is the
        warmup offset ``EDGAR_WC_WARMUP_BINS``. Starting the grid at ``w`` keeps every
        scored rollout window out of the burn-in region, so the burn-in is excluded from B/C/D
        loss without any per-window masking (Objective A masks its own one-step window).

    Example: ``T=8000``, ``rollout_k=50``, ``anchor_stride=200``, ``warmup_bins=0`` → ``K=50`` and
    anchors ``[0, 200, 400, ..., 7800]`` (40 windows). The number of anchors ``A`` becomes the
    third axis of ``pred_y_rollout`` ``[n, n_stim, A, K, 2]``. The grid is data-agnostic (depends
    only on ``T`` and the env settings), so ``load_data`` and ``apply_model`` compute the identical
    set independently.
    """
    rollout_k = int(os.environ.get("EDGAR_WC_ROLLOUT_K", DEFAULT_ROLLOUT_K))
    anchor_stride = int(os.environ.get("EDGAR_WC_ANCHOR_STRIDE", DEFAULT_ANCHOR_STRIDE))
    warmup_bins = int(os.environ.get("EDGAR_WC_WARMUP_BINS", DEFAULT_WARMUP_BINS))
    K = int(min(rollout_k, max(1, T - 2)))
    last_start = T - 1 - K
    if last_start < 0:
        return np.array([0], dtype=int), K
    first_start = int(min(max(0, warmup_bins), last_start))   # clamp burn-in offset into range
    starts = np.arange(first_start, last_start + 1, anchor_stride, dtype=int)
    if starts.size == 0:
        starts = np.array([first_start], dtype=int)
    return starts, K


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    sample_split_seed: int = 42,
    T_eval: int = 200,
    n_eval: int = 4,
    objective: str | None = None,
    rollout_k: int | None = None,
    anchor_stride: int | None = None,
    dt_seconds: float | None = None,
    warmup_steps_ms: float = 0.0,
    chop_pre_ms: float = -150.0,
    chop_post_ms: float = 400.0,
    max_conditions: int | None = None,
    cv_type: str = "k_fold",
    train_types: list[str] | None = None,
    test_types: list[str] | None = None,
):
    """Load one k-fold file and return EDGAR's ``(discover, validate, X_eval)`` split.

    ``data_path`` points at a ``wc*_fold{f}.npz`` from ``simulate_data.save_kfold_splits``. The
    ``n_samples`` samples are split 50/50 into discover / validate; params are fit per sample on
    ``train_data`` and cross-validated on the held-out (repeat-averaged) ``test_data``.

    Every top-level ``data`` value carries axis 0 = n_samples (matched to per-sample params);
    ``"target_y"`` is first so the engine's ``next(iter(data.values())).shape[0]`` reads n_samples.
    The dict carries ``target_y`` (both observed channels, last axis = (E, I)), the stimulus
    (``stim_E``, ``stim_I``), the rollout targets ``target_y_future``, and ``time`` — all with a
    leading n-axis so the param estimator's per-sample ``v[i]`` indexing stays valid. ``apply_model``
    reads E/I back off ``target_y`` and assembles the per-step ``y_prev`` dict internally.
    """
    if not data_path:
        raise ValueError(
            "wilson_cowan load_data requires data_path pointing to a wc*_fold*.npz file "
            "(generate with simulate_data.save_kfold_splits)."
        )

    # Republish config-provided settings to env so apply_model / loss_fn (separate exec namespaces)
    # and spawned scoring workers pick them up. Config wins when given; otherwise env/default stand.
    for _val, _env_key in (
        (objective, "EDGAR_WC_OBJECTIVE"),
        (rollout_k, "EDGAR_WC_ROLLOUT_K"),
        (anchor_stride, "EDGAR_WC_ANCHOR_STRIDE"),
        (dt_seconds, "EDGAR_WC_DT"),
    ):
        if _val is not None:
            os.environ[_env_key] = str(_val)

    # Real opto sessions (Phase 4): data_path is a dir, glob, or single
    real_paths = _resolve_mouse_paths(data_path)
    if real_paths:
        return _load_real(
            paths=real_paths,
            sample_split_seed=sample_split_seed,
            T_eval=T_eval,
            n_eval=n_eval,
            warmup_steps_ms=warmup_steps_ms,
            chop=(chop_pre_ms / 1000.0, chop_post_ms / 1000.0),
            max_conditions=max_conditions,
            cv_type=cv_type,
            train_types=train_types,
            test_types=test_types,
        )

    raw = np.load(data_path, allow_pickle=True)
    train_data = np.asarray(raw["train_data"])  # (n_samples, 2, T, 2)  last axis = (E, I)
    test_data = np.asarray(raw["test_data"])
    stimuli = np.asarray(raw["stimuli"])         # (2, T, 2)  [stim_cond, T, (stim_E, stim_I)]

    n_samples, n_stim, T, _ = train_data.shape

    # Synthetic wc*_fold*.npz files store NO time axis (unitless bins, H=1 in the generator), so
    # here dt is a config-supplied bin->seconds convention (`dt_seconds`, default 1 ms), not
    # inferred. The real opto path (`_load_real`) instead infers dt from each session's stored
    # time_axis — see the dt handling there.
    dt = float(os.environ.get("EDGAR_WC_DT", DEFAULT_DT_SECONDS))
    # Warmup offset (bins) → env, so _rollout_anchors (here + in apply_model) skips the burn-in.
    warmup_bins = int(round((warmup_steps_ms / 1000.0) / dt)) if warmup_steps_ms else 0
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(warmup_bins)

    anchor_starts, K = _rollout_anchors(T)
    A = len(anchor_starts)

    # Time axis in seconds, t=0 at stimulus onset (first bin where any stim channel is on). Used
    # only by Objective D's response-feature windows; onset is data-derived here so it need not be
    # recomputed under jit inside apply_model.
    stim_on = np.nonzero(np.abs(stimuli).sum(axis=(0, 2)) > 0)[0]
    onset_idx = int(stim_on[0]) if stim_on.size else 0
    time_axis = ((np.arange(T) - onset_idx) * dt).astype(np.float32)

    # 50/50 sample split → discover / validate (params are fit per sample).
    perm = np.random.default_rng(sample_split_seed).permutation(n_samples)
    disc_idx = np.sort(perm[: n_samples // 2])
    val_idx = np.sort(perm[n_samples // 2:])

    # The stimulus is shared across samples; broadcast to (n, n_stim, T) per channel so every
    # array carries the n_samples axis (axis 0) that vmap / params map over.
    def _stim_arrays(n: int):
        sE = np.broadcast_to(stimuli[None, :, :, 0], (n, n_stim, T))
        sI = np.broadcast_to(stimuli[None, :, :, 1], (n, n_stim, T))
        return jnp.asarray(sE), jnp.asarray(sI)

    def _build(split_data: np.ndarray, idx: np.ndarray) -> dict:
        n = len(idx)
        sE, sI = _stim_arrays(n)
        target_y = jnp.asarray(split_data[idx])     # (n, n_stim, T, 2)
        # Autonomous-rollout targets: for anchor a, the future window data[a+1 : a+1+K].
        target_y_future = jnp.stack(
            [target_y[:, :, a + 1: a + 1 + K, :] for a in anchor_starts], axis=2
        )                                           # (n, n_stim, A, K, 2)
        time = jnp.broadcast_to(jnp.asarray(time_axis)[None, :], (n, T))
        return {
            # target_y first: engine reads n_samples from next(iter(data.values())).shape[0].
            "target_y": target_y,
            "stim_E": sE,
            "stim_I": sI,
            "target_y_future": target_y_future,
            "time": time,
        }

    X_disc_train = _build(train_data, disc_idx)
    X_disc_test = _build(test_data, disc_idx)
    X_val_train = _build(train_data, val_idx)
    X_val_test = _build(test_data, val_idx)

    # X_eval: a small, short subset of the discover cells for fingerprint dedup.
    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    T_eval_actual = int(min(T_eval, T))
    eval_pos = np.sort(
        np.random.default_rng(sample_split_seed + 1).choice(
            len(disc_idx), n_eval_actual, replace=False
        )
    )
    disc_train = train_data[disc_idx]              # (n_disc, n_stim, T, 2)
    sE_eval, sI_eval = _stim_arrays(len(disc_idx))
    X_eval = {
        "target_y": jnp.asarray(disc_train[eval_pos, :, :T_eval_actual, :]),
        "stim_E": sE_eval[eval_pos, :, :T_eval_actual],
        "stim_I": sI_eval[eval_pos, :, :T_eval_actual],
        "_sample_indices": eval_pos,
        "_eval_fingerprint_key_name": "pred_y_1step",
    }

    print(
        f"[wilson_cowan] {data_path}: n_samples={n_samples}, n_stim={n_stim}, T={T}; "
        f"objective={os.environ.get('EDGAR_WC_OBJECTIVE', DEFAULT_OBJECTIVE).upper()}, "
        f"K={K}, anchors={A}, warmup_bins={warmup_bins}; "
        f"discover/validate={len(disc_idx)}/{len(val_idx)} "
        f"(disc={disc_idx.tolist()}, val={val_idx.tolist()}); "
        f"X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _resolve_mouse_paths(data_path: str) -> list[str]:
    """Resolve ``data_path`` to a sorted list of real ``population_rates_*_s1_trimmed.npz`` sessions.

    ``data_path`` may be a directory (globbed for the default pattern), a glob, or a single file.
    Only files whose basename starts with ``population_rates_`` are treated as real sessions, so a
    synthetic ``wc*_fold*.npz`` path resolves to ``[]`` and falls through to the synthetic loader.
    """
    if os.path.isdir(data_path):
        matches = _glob.glob(os.path.join(data_path, DEFAULT_GLOB))
    else:
        matches = _glob.glob(data_path)
    return sorted(m for m in matches if os.path.basename(m).startswith("population_rates_"))


def _read_n_folds(path: str) -> int:
    """Number of trial-CV folds stored in a session npz (from ``n_folds`` or the responses shape)."""
    d = np.load(path, allow_pickle=True)
    if "n_folds" in d.files:
        return int(d["n_folds"])
    rkey = next(k for k in d.files if k.endswith("__responses"))
    return int(np.asarray(d[rkey]).shape[1])


def _load_real(
    paths: list[str],
    sample_split_seed: int,
    T_eval: int,
    n_eval: int,
    warmup_steps_ms: float,
    chop: tuple[float, float],
    max_conditions: int | None,
    cv_type: str = "k_fold",
    train_types: list[str] | None = None,
    test_types: list[str] | None = None,
):
    """Real opto path for ``load_data``: multiple mice → EDGAR's ``(discover, validate, X_eval)``.

    **Mouse-level parameters.** One parameter set is fit per EDGAR sample, and a sample is one
    mouse (× CV rotation): the ``n_stim`` axis holds all of that mouse's conditions, so the fitted
    parameters must reproduce every stimulus condition at once with a single ``F_theta``. Only the
    equation form is shared across samples.

    **Discover/validate splits by mouse.** The sessions are split 50/50 into discover vs validate
    mice (seeded by ``sample_split_seed``); the discovered equation is thus validated on held-out
    *animals*. Because every dict is a dense ``(n_samples, n_stim, …)`` array, **all sessions must
    share the same condition count** — a mismatch raises (curate the sessions to one protocol).

    **Train/test within a sample** is set by ``cv_type`` (see ``build_cv_samples``):
      * ``"k_fold"``   — one sample per ``(mouse, held-out fold)``; ``train`` = other folds'
        trial-weighted mean, ``test`` = held-out fold (trial-noise CV, all condition types).
      * ``"exp_cond"`` — one sample per mouse; ``train`` = ``train_types`` conditions, ``test`` =
        disjoint ``test_types`` conditions (held-out-perturbation CV). ``train``/``test`` then have
        different ``n_stim``.

    Args:
        paths: Real session files (from ``_resolve_mouse_paths``); ≥2 for the mouse split.
        sample_split_seed: Seeds the mouse discover/validate split, the ``build_cv_samples``
            subsample, and the ``X_eval`` subset.
        T_eval, n_eval: Length / number of discover samples for the ``X_eval`` fingerprint.
        chop: ``(pre_s, post_s)`` peri-stimulus window kept, in seconds.
        max_conditions: Cap on conditions per split (deterministic subsample); ``None`` = all.
        cv_type: ``"k_fold"`` or ``"exp_cond"``.
        train_types, test_types: Experiment-type train/test partition, required for ``exp_cond``.

    Returns:
        ``((X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval)``. Each ``X_*`` dict
        carries ``target_y`` ``[n, C, T, 2]``, ``stim_E``/``stim_I`` ``[n, C, T]``,
        ``target_y_future`` ``[n, C, A, K, 2]`` and ``time`` ``[n, T]``; ``n`` = number of
        (mouse × fold) samples on that side, ``C`` = that split's condition count.
    """
    if len(paths) < 2:
        raise ValueError(
            f"mouse-level discover/validate needs >=2 sessions; got {len(paths)}: "
            f"{[os.path.basename(p) for p in paths]}"
        )

    # Split mice 50/50 into discover / validate (held-out animals).
    perm = np.random.default_rng(sample_split_seed).permutation(len(paths))
    if len(paths) < 2:
        raise ValueError(
            f"Fewer than 2 samples; got {len(paths)}: "
            f"{[os.path.basename(p) for p in paths]}"
        )
    n_val = max(1, len(paths) // 2)
    val_paths = [paths[i] for i in sorted(perm[n_val:])]
    disc_paths = [paths[i] for i in sorted(perm[:n_val])]

    def _mouse_units(path: str) -> list:
        """CV sample-units for one mouse: n_folds of them for k_fold, one for exp_cond."""
        if cv_type == "k_fold":
            return [
                build_cv_samples(
                    path, cv_type="k_fold", held_out_fold=f, chop=chop,
                    max_conditions=max_conditions, subsample_seed=sample_split_seed,
                )
                for f in range(_read_n_folds(path))
            ]
        return [
            build_cv_samples(
                path, cv_type="exp_cond", train_types=train_types, test_types=test_types,
                chop=chop, max_conditions=max_conditions, subsample_seed=sample_split_seed,
            )
        ]

    disc_units = [u for p in disc_paths for u in _mouse_units(p)]
    val_units = [u for p in val_paths for u in _mouse_units(p)]
    all_units = disc_units + val_units

    # Enforce a shared condition count across sessions (dense arrays can't be ragged), and a
    # shared time grid. Report per-mouse counts on failure.
    per_mouse = {u.train.meta[0]["animal_id"]: (u.train.n, u.test.n) for u in all_units}
    if len({n for n, _ in per_mouse.values()}) != 1 or len({n for _, n in per_mouse.values()}) != 1:
        raise ValueError(
            "all sessions must have the same n_stim (condition count); per-mouse "
            f"(train_n_stim, test_n_stim): {per_mouse}. Curate sessions to one protocol."
        )
    time_axis = np.asarray(all_units[0].time).astype(np.float32)   # (T,) seconds, t=0 at onset
    for u in all_units:
        if u.time.shape != time_axis.shape or not np.allclose(u.time, time_axis, atol=1e-9):
            raise ValueError("sessions do not share a common time grid after chop")
    T = time_axis.shape[0]

    dt_s = float(time_axis[1] - time_axis[0]) if T > 1 else DEFAULT_DT_SECONDS
    warmup_bins = int(round((warmup_steps_ms / 1000.0) / dt_s)) if warmup_steps_ms else 0
    os.environ["EDGAR_WC_WARMUP_BINS"] = str(warmup_bins)

    anchor_starts, K = _rollout_anchors(T)
    A = len(anchor_starts)

    def _build(units: list, split: str) -> dict:
        # Sample axis (axis 0) = mouse × CV rotation; n_stim axis (axis 1) = that mouse's conditions.
        target_y = jnp.asarray(
            np.stack([getattr(u, split).target_y for u in units], axis=0)
        )                                                            # (n, C, T, 2)
        stim = np.stack([getattr(u, split).stim for u in units], axis=0)  # (n, C, T, 2)
        sE = jnp.asarray(stim[..., 0])                              # (n, C, T)
        sI = jnp.asarray(stim[..., 1])
        target_y_future = jnp.stack(
            [target_y[:, :, a + 1: a + 1 + K, :] for a in anchor_starts], axis=2
        )                                                            # (n, C, A, K, 2)
        time = jnp.broadcast_to(jnp.asarray(time_axis)[None, :], (len(units), T))
        return {
            "target_y": target_y,
            "stim_E": sE,
            "stim_I": sI,
            "target_y_future": target_y_future,
            "time": time,
        }

    X_disc_train = _build(disc_units, "train")
    X_disc_test = _build(disc_units, "test")
    X_val_train = _build(val_units, "train")
    X_val_test = _build(val_units, "test")

    # X_eval: a small, short subset of the discover samples for fingerprint dedup.
    n_disc_samples = len(disc_units)
    n_eval_actual = int(min(max(1, n_eval), n_disc_samples))
    T_eval_actual = int(min(T_eval, T))
    eval_pos = np.sort(
        np.random.default_rng(sample_split_seed + 1).choice(
            n_disc_samples, n_eval_actual, replace=False
        )
    )
    disc_train = np.asarray(X_disc_train["target_y"])          # (n, C, T, 2)
    disc_sE = np.asarray(X_disc_train["stim_E"])               # (n, C, T)
    disc_sI = np.asarray(X_disc_train["stim_I"])
    X_eval = {
        "target_y": jnp.asarray(disc_train[eval_pos, :, :T_eval_actual, :]),
        "stim_E": jnp.asarray(disc_sE[eval_pos, :, :T_eval_actual]),
        "stim_I": jnp.asarray(disc_sI[eval_pos, :, :T_eval_actual]),
        "_sample_indices": eval_pos,
        "_eval_fingerprint_key_name": "pred_y_1step",
    }

    disc_ids = [os.path.basename(p) for p in disc_paths]
    val_ids = [os.path.basename(p) for p in val_paths]
    print(
        f"[wilson_cowan/real] {len(paths)} sessions, cv_type={cv_type}, T={T}, chop={chop} s; "
        f"objective={os.environ.get('EDGAR_WC_OBJECTIVE', DEFAULT_OBJECTIVE).upper()}, "
        f"K={K}, anchors={A}, warmup_bins={warmup_bins} (first anchor @ {anchor_starts[0]}); "
        f"mouse-level params; "
        f"discover {len(disc_paths)} mice={disc_ids} -> {len(disc_units)} samples (mouse x fold) / "
        f"validate {len(val_paths)} mice={val_ids} -> {len(val_units)} samples (mouse x fold); "
        f"n_stim train/test={all_units[0].train.n}/{all_units[0].test.n}; "
        f"X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _split_params_non_kalman(params: dict) -> tuple[dict, dict]:
    """Split params for objectives A–D → ``(init_state, dyn_params)``.

    ``s0_<name>`` keys (non-empty ``<name>``) are stripped into the initial hidden-state dict,
    routing them into the scan carry instead of the model's ``params``. ``kf_*`` keys (objective-E
    filter noise/init) are inert for A–D and dropped, so ``model_fn`` receives only dynamics params.
    """
    init_state = {}
    dyn_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        elif k.startswith("kf_"):
            continue
        else:
            dyn_params[k] = v
    return init_state, dyn_params


def _split_params_kalman(params: dict) -> tuple[dict, dict, dict]:
    """Split params for objective E → ``(init_state, dyn_params, kf_params)``.

    Same ``s0_``/dynamics split as ``_split_params_non_kalman``, but the ``kf_*`` filter
    noise/init params are collected into their own dict (consumed by ``_kf_hyper``) rather than
    dropped — so the EKF gets its hyperparameters while ``model_fn`` still sees only dynamics params.
    """
    init_state = {}
    dyn_params = {}
    kf_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        elif k.startswith("kf_"):
            kf_params[k] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params, kf_params


def _hidden_vec(state: dict, hkeys: list[str]):
    """Flatten a hidden-carry dict into a vector in fixed ``hkeys`` order (``[0]`` if stateless)."""
    if hkeys:
        return jnp.stack([state[k] for k in hkeys])
    return jnp.zeros((0,))


def apply_model(model_fn, data, params):
    """Drive the evolved transition in every mode the objectives need → the §8 ``model_output``.

    vmaps over samples (axis 0, matched to per-sample ``params``) and, inside, over the two stim
    conditions (params shared across conditions — same cell). The latent state is
    ``z = [E, I, *sorted(hidden_carry)]`` (``z_dim = 2`` for the stateless base model, ``3`` for
    the slow-``S`` variant). Returned dict, shapes ``[n, n_stim, ...]``:

    * ``pred_y_1step``        ``[…, T-1, 2]`` — teacher-forced one-step prediction (data E/I fed in).
    * ``z_inferred``          ``[…, T, z]``   — latent inferred along the teacher-forced trajectory.
    * ``pred_y_rollout``      ``[…, A, K, 2]``— autonomous rollout from A anchors (own E/I fed back).
    * ``z_rollout``           ``[…, A, K, z]``— latent along those autonomous rollouts.
    * ``z_target_future``     ``[…, A, K, z]``— inferred latent at the same absolute times.
    * ``pred_y_full_rollout`` ``[…, T, 2]``   — one full-length autonomous rollout (Objective D only;
      a placeholder equal to the observed trajectory otherwise, to avoid a long autonomous
      backprop scan the other objectives never read).

    Always returns this dict (including for ``X_eval``); the engine's ``_eval_fingerprint`` reduces
    it to the dedup fingerprint array using the field named by ``X_eval["_eval_fingerprint_key_name"]``
    (``"pred_y_1step"`` here).

    Objective E takes a **separate** state-space path (``_apply_model_kalman``): the transition is
    driven from the filter's latent estimate rather than the data, and the returned dict carries
    EKF innovations/covariances instead of the teacher-forced/rollout tensors. Objectives A–D are
    unchanged.
    """
    # Objective F uses the same apply_model as others but is not teacher forced - the loss function handles the rollout and nll computation.
    if os.environ.get("EDGAR_WC_OBJECTIVE", DEFAULT_OBJECTIVE).upper() == "E":
        return _apply_model_kalman(model_fn, data, params)

    target_y = data["target_y"]   # (n, n_stim, T, 2), last axis = (E, I)
    E = target_y[..., 0]          # (n, n_stim, T)
    I = target_y[..., 1]
    sE = data["stim_E"]
    sI = data["stim_I"]

    T = E.shape[-1]
    anchor_starts, K = _rollout_anchors(T)
    want_full = os.environ.get("EDGAR_WC_OBJECTIVE", DEFAULT_OBJECTIVE).upper() == "D"

    def per_sample(E_s, I_s, sE_s, sI_s, p):
        init_state, dyn_params = _split_params_non_kalman(p)
        hkeys = sorted(init_state.keys())

        def per_stim(E_c, I_c, sE_c, sI_c):
            # ── Teacher-forced one-step pass (feed the data E/I into y_prev) ──
            xs = (E_c[:-1], I_c[:-1], sE_c[:-1], sI_c[:-1])

            def tf_step(state, inp):
                E_p, I_p, sE_p, sI_p = inp
                y_prev = {
                    "E_prev": E_p, "I_prev": I_p,
                    "stim_E_prev": sE_p, "stim_I_prev": sI_p,
                }
                new_state, mean = model_fn(state, y_prev, dyn_params)
                E_n, I_n = mean
                return new_state, (jnp.stack([E_n, I_n]), _hidden_vec(new_state, hkeys))

            _, (means, hid_seq) = jax.lax.scan(tf_step, init_state, xs)  # (T-1,2), (T-1,nh)

            obs = jnp.stack([E_c, I_c], axis=-1)                        # (T,2)
            hid_full = jnp.concatenate(
                [_hidden_vec(init_state, hkeys)[None, :], hid_seq], axis=0
            )                                                          # (T,nh)
            z_inferred = jnp.concatenate([obs, hid_full], axis=-1)     # (T,z)

            # ── Autonomous rollout from each anchor (feed own E/I back; §7 free-running) ──
            def rollout(a):
                E0 = jax.lax.dynamic_slice_in_dim(E_c, a, 1, 0)[0]
                I0 = jax.lax.dynamic_slice_in_dim(I_c, a, 1, 0)[0]
                hid0 = jax.lax.dynamic_slice_in_dim(hid_full, a, 1, 0)[0]  # (nh,)
                state0 = {k: hid0[i] for i, k in enumerate(hkeys)}
                sE_win = jax.lax.dynamic_slice_in_dim(sE_c, a, K, 0)   # (K,)
                sI_win = jax.lax.dynamic_slice_in_dim(sI_c, a, K, 0)

                def r_step(carry, inp):
                    state, E_p, I_p = carry
                    sE_p, sI_p = inp
                    y_prev = {
                        "E_prev": E_p, "I_prev": I_p,
                        "stim_E_prev": sE_p, "stim_I_prev": sI_p,
                    }
                    new_state, mean = model_fn(state, y_prev, dyn_params)
                    E_n, I_n = mean
                    y = jnp.stack([E_n, I_n])
                    z = jnp.concatenate([y, _hidden_vec(new_state, hkeys)])
                    return (new_state, E_n, I_n), (y, z)

                _, (pred, zr) = jax.lax.scan(r_step, (state0, E0, I0), (sE_win, sI_win))
                return pred, zr                                        # (K,2), (K,z)

            pred_rollout, z_rollout = jax.vmap(rollout)(anchor_starts)  # (A,K,2), (A,K,z)

            def gather_future(a):
                return jax.lax.dynamic_slice_in_dim(z_inferred, a + 1, K, 0)  # (K,z)

            z_target_future = jax.vmap(gather_future)(anchor_starts)   # (A,K,z)

            # ── Full-length autonomous rollout (Objective D only) ──
            if want_full:
                def f_step(carry, inp):
                    state, E_p, I_p = carry
                    sE_p, sI_p = inp
                    y_prev = {
                        "E_prev": E_p, "I_prev": I_p,
                        "stim_E_prev": sE_p, "stim_I_prev": sI_p,
                    }
                    new_state, mean = model_fn(state, y_prev, dyn_params)
                    E_n, I_n = mean
                    return (new_state, E_n, I_n), jnp.stack([E_n, I_n])

                _, pred_full = jax.lax.scan(
                    f_step, (init_state, E_c[0], I_c[0]), (sE_c[:-1], sI_c[:-1])
                )                                                     # (T-1,2)
                pred_full = jnp.concatenate([obs[:1], pred_full], axis=0)  # (T,2)
            else:
                pred_full = obs                                        # unused placeholder

            return {
                "pred_y_1step": means,
                "z_inferred": z_inferred,
                "pred_y_rollout": pred_rollout,
                "z_rollout": z_rollout,
                "z_target_future": z_target_future,
                "pred_y_full_rollout": pred_full,
            }

        return jax.vmap(per_stim)(E_s, I_s, sE_s, sI_s)

    return jax.vmap(per_sample, in_axes=(0, 0, 0, 0, 0))(E, I, sE, sI, params)


def _kf_hyper(kf_params: dict, kf_defaults: dict):
    """Build the EKF noise/init matrices from the ``kf_*`` params (log-variances), with a fallback.

    ``kf_params`` is the filter-param dict from ``_split_params_kalman``. Each ``kf_log_*`` value is
    resolved ``kf_params`` -> ``kf_defaults`` (the model's ``KF_DEFAULT_PARAMS``) -> ``KeyError``.
    So the noise params are learnable when ``param_est`` supplies them, fall back to the model's
    fixed defaults otherwise, and fail loudly if neither the estimator nor the model declares them.
    Returns ``(Q, Sigma, P0_diag)`` — Q is 3x3 (E,I,S process noise), Sigma is 2x2 (E,I observation
    noise), P0_diag the length-3 initial-state variance vector.
    """
    def var(key):
        if key in kf_params:
            v = kf_params[key]
        elif key in kf_defaults:
            v = kf_defaults[key]
        else:
            raise KeyError(
                f"objective E needs '{key}': supply it from param_est or declare it in the "
                "model's KF_DEFAULT_PARAMS."
            )
        return jnp.exp(v)
    Q = jnp.diag(jnp.stack([var("kf_log_q_E"), var("kf_log_q_I"), var("kf_log_q_S")]))
    Sigma = jnp.diag(jnp.stack([var("kf_log_sig_E"), var("kf_log_sig_I")]))
    P0_diag = jnp.stack([var("kf_log_p0_E"), var("kf_log_p0_I"), var("kf_log_p0_S")])
    return Q, Sigma, P0_diag


def _apply_model_kalman(model_fn, data, params):
    """Objective-E state-space path: extended Kalman filter over ``z = [E, I, S]``.

    The transition ``F_theta`` is the SAME evolved ``model_fn``, but fed the filter's latent mean
    (never the observed E/I) — so the data enters only through the innovation, and persistence is
    not an available solution. vmaps over samples then stim conditions like ``apply_model``.

    Model / observation:
        z_t   = F_theta(z_{t-1}, stim_{t-1}) + N(0, Q)     z = [E, I, S], S latent
        y_obs = H z_t + N(0, Sigma)                        H = [[1,0,0],[0,1,0]]

    Per (sample, condition) the EKF runs a predict/update recursion for t = 1..T-1 (matching the
    ``pred_y_1step`` alignment of the other objectives: prediction of ``y[t]`` from step ``t-1``).
    Initial mean seeds E/I from ``y_obs[0]`` and S from ``s0_S``; initial covariance is ``P0``.

    Returned dict (shapes ``[n, n_stim, ...]``):
        * ``innovations``     ``[…, T-1, 2]``     — r_t = y_obs[t] - H m_t^-  (loss reads this).
        * ``innovation_cov``  ``[…, T-1, 2, 2]``  — S_t = H P_t^- H^T + Sigma (loss reads this).
        * ``pred_y_1step``    ``[…, T-1, 2]``     — one-step predicted observation H m_t^- (BEFORE
          seeing y_t): the one-step forecast, used for plotting + the eval fingerprint for now.
        * ``filtered_z``      ``[…, T, z]``       — posterior latent mean m_t (denoised trajectory).
    """
    E = data["target_y"][..., 0]   # (n, n_stim, T)
    I = data["target_y"][..., 1]
    sE = data["stim_E"]
    sI = data["stim_I"]

    H = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # (2, 3) - no obs noise in S because S is not observed
    I3 = jnp.eye(3)
    kf_defaults = getattr(model_fn, "KF_DEFAULT_PARAMS", {})   # model-declared fixed-noise fallback

    def per_sample(E_s, I_s, sE_s, sI_s, p):
        init_state, dyn_params, kf_params = _split_params_kalman(p)   # init_state["S"] = s0_S
        S0 = init_state["S"]
        Q, Sigma, P0_diag = _kf_hyper(kf_params, kf_defaults)

        def f_dyn(z, sE_p, sI_p):
            """Latent transition mean: z=[E,I,S] -> z_next, wrapping the evolved model_fn."""
            state = {"S": z[2]}
            y_prev = {"E_prev": z[0], "I_prev": z[1],
                      "stim_E_prev": sE_p, "stim_I_prev": sI_p}
            new_state, mean = model_fn(state, y_prev, dyn_params)
            E_n, I_n = mean
            return jnp.stack([E_n, I_n, new_state["S"]])

        def per_stim(E_c, I_c, sE_c, sI_c):
            m0 = jnp.stack([E_c[0], I_c[0], S0])        # seed E/I from first obs, S from s0_S
            P0 = jnp.diag(P0_diag)

            def ekf_step(carry, inp):
                m, P = carry
                E_obs, I_obs, sE_p, sI_p = inp
                # predict
                m_pred = f_dyn(m, sE_p, sI_p)
                A = jax.jacfwd(f_dyn)(m, sE_p, sI_p)    # (3, 3)
                P_pred = A @ P @ A.T + Q
                # update
                y = jnp.stack([E_obs, I_obs])
                r = y - H @ m_pred                      # innovation (2,)
                S = H @ P_pred @ H.T + Sigma            # innovation cov (2, 2)
                K = P_pred @ H.T @ jnp.linalg.inv(S)    # gain (3, 2)
                m_new = m_pred + K @ r
                P_new = (I3 - K @ H) @ P_pred
                return (m_new, P_new), (r, S, H @ m_pred, m_new)

            xs = (E_c[1:], I_c[1:], sE_c[:-1], sI_c[:-1])   # y_obs[t], stim[t-1] for t=1..T-1
            _, (r_seq, S_seq, pred_seq, m_seq) = jax.lax.scan(ekf_step, (m0, P0), xs)
            filtered_z = jnp.concatenate([m0[None, :], m_seq], axis=0)   # (T, 3)
            return {
                "innovations": r_seq,          # (T-1, 2)
                "innovation_cov": S_seq,       # (T-1, 2, 2)
                "pred_y_1step": pred_seq,      # (T-1, 2)
                "filtered_z": filtered_z,      # (T, 3)
            }

        return jax.vmap(per_stim)(E_s, I_s, sE_s, sI_s)

    return jax.vmap(per_sample, in_axes=(0, 0, 0, 0, 0))(E, I, sE, sI, params)


def loss_fn(model_output, data):
    """Dispatch to the objective selected in config.yaml (``project_params.objective``, A/B/C/D/E/F).

    Returns ``(n,)``. The engine wraps this in ``jnp.mean(loss_fn(...))``; each objective returns
    per-sample losses (see ``losses/``). A–D are MSE-based; E is the EKF marginal NLL; F is the
    free-running rollout Gaussian NLL of a deterministic state-space model. The selection is read
    from the ``EDGAR_WC_OBJECTIVE`` env var that ``load_data`` republishes from config.
    """
    objective = os.environ.get("EDGAR_WC_OBJECTIVE", DEFAULT_OBJECTIVE).upper()
    return _OBJECTIVES[objective](model_output, data)
