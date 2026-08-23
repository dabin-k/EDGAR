"""EDGAR entry points for the Wilson-Cowan (WC) neural-dynamics training-objective benchmark.

The evolved program is the WC transition (see ``seed_programs/wilson_cowan.py``):
    ``model(state, y_prev, params) -> (new_state, mean)``
    * ``y_prev`` is a **dict** ``{"E_prev","I_prev","stim_E_prev","stim_I_prev"}`` — the previous
      observation bundled with the previous stimulus.
    * ``new_state`` is the hidden carry (an empty dict for the stateless base model; a model
      with a latent ``S`` carries ``{"S": ...}``).
    * ``mean`` is ``(E, I)`` — the predicted next observation.
    * ``params`` is a dict of learnable WC params. Hidden-state initial values are declared with
      the ``s0_`` prefix; ``_split_params_s0`` strips them
      into the scan carry and hands the rest to ``model``.

This module drives that single transition in the modes required by the four training
objectives — teacher-forced one-step, autonomous rollout from anchors, and a full autonomous rollout — 
and assembles``model_output`` dict that the loss functions consume. 

The four objectives live in ``losses/`` (one file each). ``loss_fn`` dispatches to one of them by
the objective set in the project ``config.yaml`` (``project_params.objective``, ``A``/``B``/``C``/
``D``) so a benchmark run selects its objective without editing code. The rollout horizon/anchors
and time-axis dt are likewise config-driven (``project_params.rollout_k`` / ``anchor_stride`` /
``dt_seconds``); see the configuration note below for how they reach ``apply_model`` / ``loss_fn``.

Cross-validation is over repeats: ``simulate_data.save_kfold_splits`` writes ``wc*_fold{f}.npz``
files, each a repeat-averaged ``train_data`` / ``test_data`` pair (shape ``(n_samples, 2, T, 2)``);
``load_data`` reads one and splits the *samples* 50/50 into EDGAR's discover / validate sets.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from projects.wilson_cowan.data_loader.losses import (
    loss_A_one_step_tf,
    loss_B_rollout,
    loss_C_latent_consistency,
    loss_D_dynamics_aware,
)
from projects.wilson_cowan.data_loader.neural_data import (
    EXPERIMENT_TYPES,
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
    ):
        _cloudpickle.register_pickle_by_value(_importlib.import_module(_modname))
except Exception:
    pass


# ── Objective + rollout configuration ──
# Precedence: explicit config kwarg > env var > built-in default (below).
DEFAULT_OBJECTIVE = "A"       # A one-step MSE | B rollout MSE | C +latent-consistency | D +signatures
DEFAULT_ROLLOUT_K = 50        # autonomous-rollout horizon (bins)
DEFAULT_ANCHOR_STRIDE = 200   # bins between rollout anchors
DEFAULT_DT_SECONDS = 0.001    # bin width for the time axis

_OBJECTIVES = {
    "A": loss_A_one_step_tf,
    "B": loss_B_rollout,
    "C": loss_C_latent_consistency,
    "D": loss_D_dynamics_aware,
}


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
      * anchors = an evenly spaced grid ``0, anchor_stride, 2*anchor_stride, ...`` up to the last
        start for which a full ``K``-step window still fits (``T-1-K``).

    Example: ``T=8000``, ``rollout_k=50``, ``anchor_stride=200`` → ``K=50`` and anchors
    ``[0, 200, 400, ..., 7800]`` (40 windows). The number of anchors ``A`` becomes the third axis
    of ``pred_y_rollout`` ``[n, n_stim, A, K, 2]``. The grid is data-agnostic (depends only on the
    length ``T``), so ``load_data`` and ``apply_model`` compute the identical set independently.
    """
    rollout_k = int(os.environ.get("EDGAR_WC_ROLLOUT_K", DEFAULT_ROLLOUT_K))
    anchor_stride = int(os.environ.get("EDGAR_WC_ANCHOR_STRIDE", DEFAULT_ANCHOR_STRIDE))
    K = int(min(rollout_k, max(1, T - 2)))
    last_start = T - 1 - K
    if last_start < 0:
        return np.array([0], dtype=int), K
    starts = np.arange(0, last_start + 1, anchor_stride, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
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
    chop_pre_ms: float = -50.0,
    chop_post_ms: float = 400.0,
    max_conditions: int | None = None,
    disc_types: list[str] | None = None,
    val_types: list[str] | None = None,
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

    raw = np.load(data_path, allow_pickle=True)

    # Real opto session (Phase 4) vs synthetic wc*_fold*.npz
    if any(k.endswith("__responses") for k in raw.files):
        return _load_real(
            data_path=data_path,
            sample_split_seed=sample_split_seed,
            T_eval=T_eval,
            n_eval=n_eval,
            chop=(chop_pre_ms / 1000.0, chop_post_ms / 1000.0),
            max_conditions=max_conditions,
            disc_types=disc_types,
            val_types=val_types,
        )

    train_data = np.asarray(raw["train_data"])  # (n_samples, 2, T, 2)  last axis = (E, I)
    test_data = np.asarray(raw["test_data"])
    stimuli = np.asarray(raw["stimuli"])         # (2, T, 2)  [stim_cond, T, (stim_E, stim_I)]

    n_samples, n_stim, T, _ = train_data.shape
    anchor_starts, K = _rollout_anchors(T)
    A = len(anchor_starts)

    # Time axis in seconds, t=0 at stimulus onset (first bin where any stim channel is on). Used
    # only by Objective D's response-feature windows; onset is data-derived here so it need not be
    # recomputed under jit inside apply_model.
    stim_on = np.nonzero(np.abs(stimuli).sum(axis=(0, 2)) > 0)[0]
    onset_idx = int(stim_on[0]) if stim_on.size else 0
    dt = float(os.environ.get("EDGAR_WC_DT", DEFAULT_DT_SECONDS))
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
        f"K={K}, anchors={A}; "
        f"discover/validate={len(disc_idx)}/{len(val_idx)} "
        f"(disc={disc_idx.tolist()}, val={val_idx.tolist()}); "
        f"X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _load_real(
    data_path: str,
    sample_split_seed: int,
    T_eval: int,
    n_eval: int,
    chop: tuple[float, float],
    max_conditions: int | None,
    disc_types: list[str] | None = None,
    val_types: list[str] | None = None,
):
    """Real opto-session path for ``load_data``: chop + full trial CV → EDGAR's contract.

    **Mouse-level parameters.** One parameter set is fit per EDGAR sample, and here a sample is
    one CV rotation of the mouse, NOT a single condition: the ``n_stim`` axis holds all of the
    mouse's conditions (on that discover/validate side), so the fitted parameters must reproduce
    every stimulus condition at once with a single ``F_theta``. Only the equation form is shared
    across samples; the numeric parameters are refit per CV fold.

    Layout: ``build_cv_samples`` is called once per held-out fold and the folds become the sample
    axis (axis 0, ``n = n_folds``). For fold ``f`` the sample's ``*_train`` is the trial-weighted
    mean of the other folds and ``*_test`` is fold ``f`` — the 3-fold trial CV, one param set per
    rotation. Conditions sit on axis 1 (``n_stim``), each with its own stimulus.

    Args:
        data_path: One ``population_rates_<animal>_s1.npz`` session file.
        sample_split_seed: Seeds the random condition split and the ``X_eval`` subset, and is
            passed to ``build_cv_samples`` as its subsample seed.
        T_eval: Max trajectory length of the (short) ``X_eval`` fingerprint traces.
        n_eval: Number of discover samples (CV folds) used for the ``X_eval`` fingerprint.
        chop: ``(pre_s, post_s)`` peri-stimulus window kept, in seconds.
        max_conditions: Cap on conditions kept (deterministic subsample); ``None`` = all.
        disc_types, val_types: If both given, the mouse's conditions are partitioned by
            ``experiment_type`` into the discover vs validate ``n_stim`` sets (types in neither
            list are dropped) — the §5 held-out-perturbation-class test (params fit on discover
            conditions must, after a refit, also explain the held-out condition types). If either
            is ``None``, conditions are split 50/50 at random.

    Returns:
        ``((X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval)`` — the EDGAR
        ``(discover, validate, X_eval)`` contract. Each ``X_*`` dict carries ``target_y``
        ``[n_folds, C, T, 2]``, ``stim_E``/``stim_I`` ``[n_folds, C, T]``, ``target_y_future``
        ``[n_folds, C, A, K, 2]`` and ``time`` ``[n_folds, T]``, where ``C`` = number of
        conditions on that discover/validate side.
    """
    d = np.load(data_path, allow_pickle=True)
    if "n_folds" in d.files:
        n_folds = int(d["n_folds"])
    else:
        rkey = next(k for k in d.files if k.endswith("__responses"))
        n_folds = int(np.asarray(d[rkey]).shape[1])

    # One CVSamples per held-out fold; the shared subsample_seed makes the kept-condition set and
    # its order identical across folds, so a condition sits at the same index in every fold.
    cvs = [
        build_cv_samples(
            data_path, held_out_fold=f, chop=chop, max_conditions=max_conditions,
            subsample_seed=sample_split_seed,
        )
        for f in range(n_folds)
    ]
    n_conditions = len(cvs[0].meta)
    time_axis = np.asarray(cvs[0].time).astype(np.float32)     # (T,) seconds, t=0 at onset
    T = time_axis.shape[0]
    anchor_starts, K = _rollout_anchors(T)
    A = len(anchor_starts)

    # Discover/validate split over CONDITIONS (indices shared across folds).
    if disc_types is not None and val_types is not None:
        disc_set, val_set = set(disc_types), set(val_types)
        overlap = disc_set & val_set
        if overlap:
            raise ValueError(f"disc_types and val_types overlap: {sorted(overlap)}")
        unknown = (disc_set | val_set) - set(EXPERIMENT_TYPES)
        if unknown:
            raise ValueError(
                f"unknown experiment type(s) {sorted(unknown)}; "
                f"valid types are {list(EXPERIMENT_TYPES)}"
            )
        types = [m["experiment_type"] for m in cvs[0].meta]
        disc_cond = np.array([i for i, t in enumerate(types) if t in disc_set], dtype=int)
        val_cond = np.array([i for i, t in enumerate(types) if t in val_set], dtype=int)
        dropped = sorted(set(types) - disc_set - val_set)
        if dropped:
            print(f"[wilson_cowan/real] dropping conditions of unassigned type(s): {dropped}")
        if disc_cond.size == 0 or val_cond.size == 0:
            raise ValueError(
                f"type split leaves an empty side (discover={disc_cond.size}, "
                f"validate={val_cond.size}); present types: {sorted(set(types))}"
            )
    else:
        perm = np.random.default_rng(sample_split_seed).permutation(n_conditions)
        disc_cond = np.sort(perm[: n_conditions // 2])
        val_cond = np.sort(perm[n_conditions // 2:])

    def _build(kind: str, cond_idx: np.ndarray) -> dict:
        # Sample axis (axis 0) = CV folds; n_stim axis (axis 1) = conditions (shared param set).
        target_y = jnp.asarray(
            np.stack([getattr(cv, kind)[cond_idx] for cv in cvs], axis=0)
        )                                                            # (n_folds, C, T, 2)
        stim = cvs[0].stim[cond_idx]                                 # (C, T, 2); same for all folds
        stim = np.broadcast_to(stim[None], (n_folds,) + stim.shape)  # (n_folds, C, T, 2)
        sE = jnp.asarray(stim[..., 0])                              # (n_folds, C, T)
        sI = jnp.asarray(stim[..., 1])
        target_y_future = jnp.stack(
            [target_y[:, :, a + 1: a + 1 + K, :] for a in anchor_starts], axis=2
        )                                                            # (n_folds, C, A, K, 2)
        time = jnp.broadcast_to(jnp.asarray(time_axis)[None, :], (n_folds, T))
        return {
            "target_y": target_y,
            "stim_E": sE,
            "stim_I": sI,
            "target_y_future": target_y_future,
            "time": time,
        }

    X_disc_train = _build("target_y_train", disc_cond)
    X_disc_test = _build("target_y_test", disc_cond)
    X_val_train = _build("target_y_train", val_cond)
    X_val_test = _build("target_y_test", val_cond)

    # X_eval: a small, short subset of the discover samples (CV folds) for fingerprint dedup.
    n_eval_actual = int(min(max(1, n_eval), n_folds))
    T_eval_actual = int(min(T_eval, T))
    eval_pos = np.sort(
        np.random.default_rng(sample_split_seed + 1).choice(
            n_folds, n_eval_actual, replace=False
        )
    )
    disc_train = np.asarray(X_disc_train["target_y"])          # (n_folds, C, T, 2)
    disc_sE = np.asarray(X_disc_train["stim_E"])               # (n_folds, C, T)
    disc_sI = np.asarray(X_disc_train["stim_I"])
    X_eval = {
        "target_y": jnp.asarray(disc_train[eval_pos, :, :T_eval_actual, :]),
        "stim_E": jnp.asarray(disc_sE[eval_pos, :, :T_eval_actual]),
        "stim_I": jnp.asarray(disc_sI[eval_pos, :, :T_eval_actual]),
        "_sample_indices": eval_pos,
        "_eval_fingerprint_key_name": "pred_y_1step",
    }

    split_by = "type" if (disc_types is not None and val_types is not None) else "random"
    print(
        f"[wilson_cowan/real] {data_path}: n_conditions={n_conditions}, n_folds={n_folds}, "
        f"T={T}, chop={chop} s; "
        f"objective={os.environ.get('EDGAR_WC_OBJECTIVE', DEFAULT_OBJECTIVE).upper()}, "
        f"K={K}, anchors={A}; "
        f"mouse-level params; samples=n_folds={n_folds} per side; "
        f"discover/validate conditions (n_stim)={len(disc_cond)}/{len(val_cond)} "
        f"({split_by} split); X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Strip ``s0_``-prefixed keys → initial hidden-state dict; rest is dyn params.

    Only strips keys of the form ``s0_<name>`` with ``<name>`` non-empty. The initial values are
    ordinary GD parameters (per sample); stripping them here routes them into the scan carry
    instead of the model's ``params``.
    """
    init_state = {}
    dyn_params = {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params


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
    """
    target_y = data["target_y"]   # (n, n_stim, T, 2), last axis = (E, I)
    E = target_y[..., 0]          # (n, n_stim, T)
    I = target_y[..., 1]
    sE = data["stim_E"]
    sI = data["stim_I"]

    T = E.shape[-1]
    anchor_starts, K = _rollout_anchors(T)
    want_full = os.environ.get("EDGAR_WC_OBJECTIVE", DEFAULT_OBJECTIVE).upper() == "D"

    def per_sample(E_s, I_s, sE_s, sI_s, p):
        init_state, dyn_params = _split_params_s0(p)
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


def loss_fn(model_output, data):
    """Dispatch to the objective selected in config.yaml (``project_params.objective``, A/B/C/D).

    Returns ``(n,)``. The engine wraps this in ``jnp.mean(loss_fn(...))``; each objective returns
    per-sample losses (see ``losses/``). All are MSE-based — no NLL / observation-noise term. The
    selection is read from the ``EDGAR_WC_OBJECTIVE`` env var that ``load_data`` republishes from
    config.
    """
    objective = os.environ.get("EDGAR_WC_OBJECTIVE", DEFAULT_OBJECTIVE).upper()
    return _OBJECTIVES[objective](model_output, data)
