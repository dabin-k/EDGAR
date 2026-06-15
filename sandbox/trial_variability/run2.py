from edgar.io.task_spec import TaskSpec
from edgar.io.config import Config
from pathlib import Path
import jax.numpy as jnp
from edgar.scoring.scoring import _eval_loss, _get_params
import sys

sys.path.append("/home/rajah/repos/EDGAR/projects/trial_variability")
from data_loader.load_data import load_data, loss_fn

X_discover, X_validate, X_eval = load_data(
    data_path="/home/rajah/datasets/BZ015", show_plots=True
)
print("X_discover_train shape: ", X_discover[0]["response"].shape)
print("X_discover_test shape: ", X_discover[1]["response"].shape)
print("X_validate_train shape: ", X_validate[0]["response"].shape)
print("X_validate_test shape: ", X_validate[1]["response"].shape)
print("X_eval shape: ", X_eval["response"].shape)


def model1_jax(data, params):
    signal = data["signal"]
    gain = params["multiplicative_gain"]
    return gain[:, jnp.newaxis] * signal


def model2_jax(data, params):
    signal = data["signal"]
    gain = params["multiplicative_gain"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]
    return gain[:, jnp.newaxis] * signal + jnp.outer(offset, coupling)


def param_est_fn(data):
    raise RuntimeError("No parameter estimation function defined for this model")


models_jax = [model1_jax, model2_jax]

config_path = "/home/rajah/repos/EDGAR/projects/trial_variability/config.yaml"
path = Path(config_path)
config = Config.from_yaml(path)
spec = TaskSpec.from_config(config)
sample_data = {k: v[0] for k, v in X_discover[0].items()}  # remove sample dimension
for i, seed in enumerate(spec.seed_programs):
    print(f"Seed {i}")
    print(f"Model code:\n{seed.code.model}\n")
    print(f"Model default_params:\n{seed.default_params}\n")
    print(f"Model n_params: {seed.n_params}\n")
    seed.code.model_jax = seed.code.model  # dummy this for now
    model_fn = seed.compile_model()
    # Single unbatched evaluation of model
    output = model_fn(sample_data, seed.default_params)
    print("Model output shape: ", output.shape)
    # Now for batched evaluation
    batched_params = _get_params(param_est_fn, seed.default_params, X_discover[0])
    eval_loss = _eval_loss(models_jax[i], loss_fn, batched_params, X_discover[0])
    print("Evaluation loss:", eval_loss)
