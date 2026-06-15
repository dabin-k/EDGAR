import sys

sys.path.append("/home/rajah/repos/EDGAR/projects/trial_variability")
import numpy as np
from edgar.llm.code_loading import load_function_from_source
from pathlib import Path
from data_loader.load_data import load_data, loss_fn, _to_jax
import jax.numpy as jnp
from edgar.scoring.scoring import _eval_loss

X_discover, X_validate, X_eval = load_data(show_plots=True)
print("X_discover_train shape: ", X_discover[0]["response"].shape)
print("X_discover_test shape: ", X_discover[1]["response"].shape)
print("X_validate_train shape: ", X_validate[0]["response"].shape)
print("X_validate_test shape: ", X_validate[1]["response"].shape)
print("X_eval shape: ", X_eval["response"].shape)

model_path = Path(
    "/home/rajah/repos/EDGAR/projects/trial_variability/seed_programs/model2.py"
)
model_code = model_path.read_text()
model_fn = load_function_from_source(model_code, "model")
model_params = {
    "multiplicative_gain": np.ones(
        (1, X_discover[0]["response"].shape[1])
    ),  # (n_samples, n_trials)
    "additive_offset": np.ones(
        (1, X_discover[0]["response"].shape[1])
    ),  # (n_samples, n_trials)
    "coupling_factor": np.ones(
        (1, X_discover[0]["response"].shape[2])
    ),  # (n_samples, n_cells)
}

# For direct model_fn call, we need to pass a single sample (unmapped)
sample_data = {k: v[0] for k, v in X_discover[0].items() if isinstance(v, jnp.ndarray)}
sample_params = {k: v[0] for k, v in model_params.items()}


def model_jax(data, params):
    signal = data["signal"]
    gain = params["multiplicative_gain"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]
    return gain[:, jnp.newaxis] * signal + jnp.outer(offset, coupling)


output = model_fn(sample_data, sample_params)
jax_output = jnp.array(output)
print("Model output shape: ", output.shape)
loss = loss_fn(jax_output, X_discover[0])
print("Loss:", loss)
eval_loss = _eval_loss(model_jax, loss_fn, _to_jax(model_params), X_discover[0])
print("Evaluation loss:", eval_loss)
