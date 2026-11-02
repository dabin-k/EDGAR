from __future__ import annotations

import numpy as np
import jax.numpy as jnp

def load_data(
    data_path: str,
    n_eval_samples: int = 10,
    **kwargs,  # Additional params from project_params in config.yaml
):
    """Loads and preprocesses data, then splits it into discovery, validation, and evaluation sets.

    This function is responsible for loading raw experimental data, performing any
    necessary preprocessing, and then dividing the data into distinct sets:
    `discover` (used during the evolutionary algorithm's search phase), `validate`
    (used for final model selection), and `eval` (a small subset for model
    fingerprinting).

    Args:
        data_path (str): The path to the raw data file.
        n_eval_samples (int): The number of samples to use for generating the
            evaluation fingerprint, which is used for deduplication.
        **kwargs: Additional parameters passed from the `project_params`
            section of `config.yaml`.

    Returns:
        tuple: A tuple containing (X_discover, X_validate, X_eval).

            - **X_discover** (tuple): Contains `X_disc_train` and `X_disc_test`.
                - `X_disc_train` (dict): A dictionary of numpy arrays, typically
                  with shape `(n_samples_discover_train, n_trials)`. This data
                  is exposed to the LLM during the model discovery phase.
                - `X_disc_test` (dict): A dictionary of numpy arrays, typically
                  with shape `(n_samples_discover_test, n_trials)`. This is a
                  held-out test set used within the discovery phase for scoring
                  and feedback.
            - **X_validate** (tuple): Contains `X_val_train` and `X_val_test`.
                - `X_val_train` (dict): A dictionary of numpy arrays, typically
                  with shape `(n_samples_validate_train, n_trials)`. This data
                  is never seen during the discovery phase and is reserved for
                  final model validation.
                - `X_val_test` (dict): A dictionary of numpy arrays, typically
                  with shape `(n_samples_validate_test, n_trials)`. This is a
                  final held-out evaluation set for robust performance assessment.
            - **X_eval** (dict): A small subset of data from `X_disc_train` used
              for generating model fingerprints for deduplication. It contains numpy
              arrays plus `_sample_indices` (a NumPy integer array indicating
              positions within `disc_idx`).
    """
    raise NotImplementedError


def loss_fn(model_output, data):
    """Computes the per-sample loss between model predictions and actual data.

    This function calculates the loss for each individual sample, which can then
    be aggregated to determine the overall fitness of a model. The loss should be
    designed such that lower values indicate a better fit.

    Args:
        model_output: A JAX array of model predictions. Its shape can be
                      `(n_trials,)` for a single sample or `(n_samples, n_trials)`
                      if batched.
        data (dict): A dictionary of JAX arrays containing the ground truth data
                     for the current split, e.g., `data['response']`.

    Returns:
        JAX array: An array of per-sample losses, typically with shape
        `(n_samples,)`.
    """
    raise NotImplementedError
