"""Objective C — rollout + latent-dynamics consistency (plan §10-C).

Scientific question: does making the hidden state obey a consistent autonomous flow improve
symbolic discoverability? On top of the autonomous-rollout MSE we penalise disagreement
between the autonomously-rolled latent ``z_rollout`` and the latent reached by running the
model along the observed trajectory ``z_target_future``.

Caveat (§10-C): only the E, I coordinates are directly identifiable; hidden coordinates may be
transformed without changing observations, so this term shapes representation, not a claimed
biological mechanism.
"""
from __future__ import annotations

from .loss_common import per_sample_mse


def loss_C_latent_consistency(model_output, data, lambda_z=1.0):
    """Rollout MSE + ``lambda_z`` × latent-consistency MSE, per sample ``(n,)``.

    ``z_rollout`` / ``z_target_future`` are ``[n, n_stim, A, K, z]``; for the stateless base
    WC model ``z = [E, I]`` (z_dim=2) so this term reduces to an E/I autonomous-vs-inferred
    penalty, still well-defined.
    """
    obs_loss = per_sample_mse(
        model_output["pred_y_rollout"], data["target_y_future"]
    )
    latent_loss = per_sample_mse(
        model_output["z_rollout"], model_output["z_target_future"]
    )
    return obs_loss + lambda_z * latent_loss
