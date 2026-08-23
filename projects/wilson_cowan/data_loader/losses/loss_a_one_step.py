"""Objective A — teacher-forced one-step MSE (plan §10-A).

Scientific question: is accurate *local* prediction enough for EDGAR to recover the correct
dynamical equation? This is the baseline: at every step predict ``z_{t+1}``, score the
predicted E/I against the real E/I, then replace predicted E/I with the real E/I before the
next transition (the teacher forcing is done inside ``apply_model``).
"""
from __future__ import annotations

from .loss_common import per_sample_mse


def loss_A_one_step_tf(model_output, data):
    """One-step teacher-forced MSE, per sample ``(n,)``.

    ``pred_y_1step`` is ``[n, n_stim, T-1, 2]`` (prediction of ``y[t]`` from step ``t-1``);
    it is aligned to the ``[1:]`` slice of the observed trajectory ``target_y``.
    """
    pred = model_output["pred_y_1step"]        # [n, n_stim, T-1, 2]
    target = data["target_y"][:, :, 1:, :]     # [n, n_stim, T-1, 2]
    return per_sample_mse(pred, target)
