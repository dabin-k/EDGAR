import os
from .loss_common import per_sample_mse

def loss_full_rollout(model_output, data):
    """Full rollout MSE
    """
    pred = model_output["pred_y_full_rollout"] # [n_samples, n_stim, T, 2]
    target = data["target_y"] # [n_samples, n_stim, T, 2]
    return per_sample_mse(pred, target)