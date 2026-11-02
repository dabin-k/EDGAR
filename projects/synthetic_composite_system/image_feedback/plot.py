import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_model_fits(data, parent_programs, save_path=""):
    """Optional. Generates plots comparing model predictions to data for feedback.

    This function is used to create visual feedback for the LLM during the
    evolutionary process and for visualizing program performance in the dashboard.
    It should generate plots that help in understanding how well the parent programs
    fit the provided data.

    Args:
        data (dict): A dictionary of JAX arrays, typically the `X_disc_train`
                     data with shape `(n_samples, n_trials)`.
        parent_programs (list): A list of `Program` objects. Each `Program`
                                object provides access to its compiled model,
                                optimized parameters, sample-wise losses,
                                and overall scalar loss. Key attributes include:
                                - `.compile()`: Returns a tuple `(model_fn, param_est_fn)`.
                                - `.params`: A dictionary of per-sample parameters,
                                             where each value has shape `(n_samples, ...)`.
                                - `.sample_losses`: Per-sample losses, with shape
                                                    `(n_samples,)`, or `None`.
                                - `.program_losses.discover.final`: The scalar
                                                                    overall loss.
        save_path (str): The file path (not a directory) where the generated
                         figure should be saved.
    """
    pass
