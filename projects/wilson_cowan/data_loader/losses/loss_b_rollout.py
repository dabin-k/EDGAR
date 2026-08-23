"""Objective B — autonomous multi-step rollout MSE (plan §10-B).

Scientific question: does explicitly training *autonomous* trajectories improve recovery of
the governing equation? Observed data is used only to synchronise the latent state at each
anchor; the model is then rolled forward without clamping E/I (§7 free-running), and the free
prediction is scored against the future data window.
"""
from __future__ import annotations

from .loss_common import per_sample_mse


def loss_B_rollout(model_output, data):
    """Autonomous-rollout MSE, per sample ``(n,)``.

    ``pred_y_rollout`` and ``target_y_future`` are both ``[n, n_stim, A, K, 2]`` (A anchors,
    K-step horizon).
    """
    pred = model_output["pred_y_rollout"]       # [n, n_stim, A, K, 2]
    target = data["target_y_future"]            # [n, n_stim, A, K, 2]
    return per_sample_mse(pred, target)
