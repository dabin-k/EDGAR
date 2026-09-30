"""Training objectives for the WC neural-dynamics benchmark (plan §10).

Each objective is one file; all consume the §8 ``model_output`` dict from ``apply_model`` and
return per-sample losses ``(n,)``. Import the one you want as the engine ``loss_fn`` (see
``load_data.py``'s ``EDGAR_WC_OBJECTIVE`` selector).
"""
from .loss_a_one_step import loss_A_one_step_tf
from .loss_b_rollout import loss_B_rollout
from .loss_c_full import loss_C_full_rollout

__all__ = [
    "loss_A_one_step_tf",
    "loss_B_rollout",
    "loss_C_full_rollout",
]
