"""Objective FULL — full-trajectory autonomous rollout MSE.

Scientific question: does scoring the model on one free-running trajectory over the whole
trial (seeded only from the first observation and the ``s0_`` initial hidden state) recover
the governing equation better than local (A) or short-horizon (B) prediction? Only the stimulus
is fed in; E/I are the model's own predictions throughout.

As in Objective A, the first ``EDGAR_WC_WARMUP_BINS`` predictions are dropped from the loss
(the rollout still runs through the warm-up window).
"""
from __future__ import annotations

import os

from .loss_common import per_sample_mse


def loss_full_rollout(model_output, data):
    """Full-trajectory rollout MSE, per sample ``(n,)``.

    ``pred_y_full_rollout`` is ``[n, n_stim, T-1, 2]`` (prediction of ``y[t]`` for ``t = 1..T-1``),
    aligned to the ``[1:]`` slice of ``target_y`` — same indexing as ``pred_y_1step``.
    """
    w = max(0, int(os.environ.get("EDGAR_WC_WARMUP_BINS", "0")))
    pred = model_output["pred_y_full_rollout"][:, :, w:, :]   # [n, n_stim, T-1-w, 2]
    target = data["target_y"][:, :, 1 + w:, :]                # [n, n_stim, T-1-w, 2]
    return per_sample_mse(pred, target)
