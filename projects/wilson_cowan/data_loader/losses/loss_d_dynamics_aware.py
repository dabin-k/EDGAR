"""Objective D — rollout + response featureloss (plan §10-D).

Scientific question: does training directly on the macroscopic response features that
distinguish Wilson-Cowan-like mechanisms improve equation discovery? We keep the autonomous
rollout MSE (never a signature-only objective) and add a differentiable feature loss over a
full post-stimulus autonomous rollout.
"""
from __future__ import annotations

from .loss_common import per_sample_mse, response_features


def loss_D_dynamics_aware(model_output, data, lambda_dyn=1.0):
    """Rollout MSE + ``lambda_dyn`` × response-feature MSE, per sample ``(n,)``.

    Features are computed on one full autonomous rollout per sample
    (``pred_y_full_rollout`` ``[n, n_stim, T, 2]``) against the observed ``target_y``, on the
    shared time grid ``data["time"][0]``.
    """
    rollout_loss = per_sample_mse(
        model_output["pred_y_rollout"], data["target_y_future"]
    )

    time_axis = data["time"][0]                       # shared [T] grid
    pred_feat = response_features(model_output["pred_y_full_rollout"], time_axis)
    target_feat = response_features(data["target_y"], time_axis)
    feature_loss = per_sample_mse(pred_feat, target_feat)

    return rollout_loss + lambda_dyn * feature_loss
