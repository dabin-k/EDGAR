"""EDGAR entry points for the Wilson-Cowan (WC) discovery task.

The underlying system (see ``simulate_data.py`` — the source of truth) is the base
Wilson-Cowan model: two **fully observed** populations, excitatory ``E`` and
inhibitory ``I``, driven by a per-timestep external stimulus (an excitatory pulse
or an inhibitory pulse). There is no hidden variable.

Contract for the LLM's program (see ``seed_programs/wilson_cowan.py``):
    ``model(state, y_prev, params) -> (new_state, mean)``
    * ``y_prev`` is a **dict** ``{"E_prev","I_prev","stim_E_prev","stim_I_prev"}`` —
      the previous observation bundled with the previous stimulus.
    * ``new_state`` is an (empty) dict carry — the base model needs no hidden state.
    * ``mean`` is ``(E, I)`` — the predicted next observation.
    * ``params`` is a dict of the learnable WC parameters.

Prediction is one-step-ahead and teacher-forced: the prediction of ``y[t]`` is paired
with everything at ``t-1`` (``E[t-1], I[t-1], stim_E[t-1], stim_I[t-1]``). The scan
inputs are the ``[:-1]`` slice; the targets are ``[1:]``.

Loss is per-channel-normalized MSE, averaged over both stim conditions and time, so E
and I (which live on different scales) contribute comparably. There is no observation-
noise parameter and no ``s0_*`` initial-state parameters.

Cross-validation is over the 12 repeats: ``simulate_data.save_kfold_splits`` writes
``wc_fold{f}.npz`` files, each holding a repeat-averaged ``train_data`` / ``test_data``
pair (shape ``(n_samples, 2, T, 2)``). ``load_data`` reads one such file and additionally
splits the *samples* 50/50 into EDGAR's discover / validate sets.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── EDGAR entry points ──


def load_data(
    data_path: str = "",
    sample_split_seed: int = 42,
    T_eval: int = 200,
    n_eval: int = 4,
):
    """Load one k-fold file and return EDGAR's ``(discover, validate, X_eval)`` split.

    ``data_path`` must point at a ``wc_fold{f}.npz`` produced by
    ``simulate_data.save_kfold_splits``. The 8 samples are split 50/50 into discover
    and validate; parameters are fit per sample on ``train_data`` and cross-validated
    on the held-out (repeat-averaged) ``test_data``.

    Every top-level dict value is a plain array with axis 0 = n_samples (the per-sample
    axis matched to per-sample params). ``"E"`` is the first key so the engine's
    ``next(iter(data.values())).shape[0]`` sees n_samples. The per-step dict ``y_prev``
    is assembled inside ``apply_model`` (a ``jax.lax.scan`` over a pytree), so no core
    EDGAR change is needed.
    """
    if not data_path:
        raise ValueError(
            "wilson_cowan load_data requires data_path pointing to a wc_fold*.npz file "
            "(generate with simulate_data.save_kfold_splits)."
        )
    raw = np.load(data_path)
    train_data = np.asarray(raw["train_data"])  # (n_samples, 2, T, 2)  last axis = (E, I)
    test_data = np.asarray(raw["test_data"])
    stimuli = np.asarray(raw["stimuli"])         # (2, T, 2)  [stim_cond, T, (stim_E, stim_I)]

    n_samples, n_stim, T, _ = train_data.shape

    # 50/50 sample split → discover / validate (params are fit per sample).
    perm = np.random.default_rng(sample_split_seed).permutation(n_samples)
    disc_idx = np.sort(perm[: n_samples // 2])
    val_idx = np.sort(perm[n_samples // 2:])

    # The stimulus is shared across samples; broadcast to (n, n_stim, T) per channel so
    # every array carries the n_samples axis (axis 0) that vmap/params map over.
    def _stim_arrays(n: int):
        sE = np.broadcast_to(stimuli[None, :, :, 0], (n, n_stim, T))
        sI = np.broadcast_to(stimuli[None, :, :, 1], (n, n_stim, T))
        return jnp.asarray(sE), jnp.asarray(sI)

    # Per-(sample, stim) channel scale = std over time on the TRAIN window, reused for
    # both train and test so the normalisation is identical across the split.
    train_E_scale = np.maximum(train_data[..., 0].std(axis=-1), 1e-6)  # (n_samples, n_stim)
    train_I_scale = np.maximum(train_data[..., 1].std(axis=-1), 1e-6)

    def _build(split_data: np.ndarray, idx: np.ndarray) -> dict:
        n = len(idx)
        sE, sI = _stim_arrays(n)
        return {
            # "E" first: engine reads n_samples from next(iter(data.values())).shape[0].
            "E": jnp.asarray(split_data[idx, :, :, 0]),      # (n, n_stim, T)
            "I": jnp.asarray(split_data[idx, :, :, 1]),
            "stim_E": sE,
            "stim_I": sI,
            "E_scale": jnp.asarray(train_E_scale[idx]),      # (n, n_stim)
            "I_scale": jnp.asarray(train_I_scale[idx]),
        }

    X_disc_train = _build(train_data, disc_idx)
    X_disc_test = _build(test_data, disc_idx)
    X_val_train = _build(train_data, val_idx)
    X_val_test = _build(test_data, val_idx)

    # X_eval: a small, short subset of the discover cells for fingerprint dedup.
    # _sample_indices index positions WITHIN the discover set (the scorer does
    # params[_sample_indices] against the per-discover-sample params).
    n_eval_actual = int(min(max(1, n_eval), len(disc_idx)))
    T_eval_actual = int(min(T_eval, T))
    eval_pos = np.sort(
        np.random.default_rng(sample_split_seed + 1).choice(
            len(disc_idx), n_eval_actual, replace=False
        )
    )
    disc_train_E = train_data[disc_idx, :, :, 0]
    disc_train_I = train_data[disc_idx, :, :, 1]
    sE_eval, sI_eval = _stim_arrays(len(disc_idx))
    X_eval = {
        "E": jnp.asarray(disc_train_E[eval_pos, :, :T_eval_actual]),
        "I": jnp.asarray(disc_train_I[eval_pos, :, :T_eval_actual]),
        "stim_E": sE_eval[eval_pos, :, :T_eval_actual],
        "stim_I": sI_eval[eval_pos, :, :T_eval_actual],
        "_sample_indices": eval_pos,
    }

    print(
        f"[wilson_cowan] {data_path}: n_samples={n_samples}, n_stim={n_stim}, T={T}; "
        f"discover/validate={len(disc_idx)}/{len(val_idx)} "
        f"(disc={disc_idx.tolist()}, val={val_idx.tolist()}); "
        f"X_eval n={n_eval_actual}, T={T_eval_actual}"
    )

    return (
        (X_disc_train, X_disc_test),
        (X_val_train, X_val_test),
        X_eval,
    )


def apply_model(model_fn, data, params):
    """Teacher-forced one-step-ahead scan of ``model_fn`` over every (sample, stim).

    Builds the per-step dict ``y_prev`` and scans over it (``jax.lax.scan`` handles the
    pytree natively). vmaps over samples (axis 0, matched to per-sample ``params``) and,
    inside, over the two stim conditions (params are shared across conditions — same
    cell). Returns ``(n_samples, n_stim, T-1, 2)``: predicted (E, I) at each step.
    """
    E = data["E"]        # (n, n_stim, T)
    I = data["I"]
    sE = data["stim_E"]
    sI = data["stim_I"]

    def per_sample(E_s, I_s, sE_s, sI_s, p):
        def per_stim(E_c, I_c, sE_c, sI_c):
            xs = {
                "E_prev": E_c[:-1],
                "I_prev": I_c[:-1],
                "stim_E_prev": sE_c[:-1],
                "stim_I_prev": sI_c[:-1],
            }

            def step(state, y_prev):
                new_state, mean = model_fn(state, y_prev, p)
                E_next, I_next = mean
                return new_state, jnp.stack([E_next, I_next])

            _, means = jax.lax.scan(step, {}, xs)  # (T-1, 2)
            return means

        return jax.vmap(per_stim)(E_s, I_s, sE_s, sI_s)  # (n_stim, T-1, 2)

    return jax.vmap(per_sample, in_axes=(0, 0, 0, 0, 0))(E, I, sE, sI, params)


def loss_fn(model_output, data):
    """Per-channel-normalized MSE, averaged over stim conditions and time.

    ``model_output`` is ``(n, n_stim, T-1, 2)``. Targets are the ``[1:]`` slice of the
    observed E/I. Each channel's squared error is divided by that channel's scale
    (per (sample, stim), from the train window) so E and I contribute comparably.
    Returns one loss per sample ``(n,)``.
    """
    E_hat = model_output[..., 0]      # (n, n_stim, T-1)
    I_hat = model_output[..., 1]
    E_tgt = data["E"][:, :, 1:]
    I_tgt = data["I"][:, :, 1:]
    E_scale = data["E_scale"][:, :, None]  # (n, n_stim, 1)
    I_scale = data["I_scale"][:, :, None]

    e2 = ((E_hat - E_tgt) / E_scale) ** 2
    i2 = ((I_hat - I_tgt) / I_scale) ** 2
    return jnp.mean(e2 + i2, axis=(1, 2))  # (n,)
