import asyncio
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _get_params, _eval_loss, _optimize
from pathlib import Path
import os
import numpy as np
import jax.numpy as jnp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

SAMPLE_SPLIT_SEED = 42
FOLD_DEFAULT = "/home/rajah/datasets/wc_synthetic/wc_fold0.npz"
# Column order of the npz `params` array == PARAM_MEDIAN.keys() in simulate_data.
PARAM_KEYS = [
    "tau_E", "tau_I", "W_EE", "W_IE", "W_EI", "W_II",
    "E_max", "I_max", "C_E", "C_I", "XE", "XI",
]

def _discover_idx(n_samples: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n_samples)
    return np.sort(perm[: n_samples // 2])

def _params_dict(mat: np.ndarray) -> dict:
    """(n, 12) rows in PARAM order -> {key: (n,) jnp array}."""
    return {k: jnp.asarray(mat[:, j]) for j, k in enumerate(PARAM_KEYS)}

async def main():
    # Use absolute path for robustness in sandbox
    project_root = Path(__file__).parent.parent
    path = project_root / "projects" / "wilson_cowan" / "config.yaml"

    print(f"Loading config from: {path}")
    config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)

    # Load data
    print("Loading data...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], sample_split_seed = SAMPLE_SPLIT_SEED
    )

    seed = spec.seed_programs[0]

    # Translate model code to JAX
    model_numpy = seed.code.model
    model_jax = model_numpy.replace(
        "import numpy as np", "import jax.numpy as jnp"
    ).replace("np.", "jnp.")
    seed.code.model_jax = model_jax

    # Compile model & param estimator
    model_fn = seed.compile_model()
    param_est_fn = seed.compile_param_est()

    #Get actual parameters
    raw = np.load(FOLD_DEFAULT)
    params_all = np.asarray(raw["params"])            # (8, 12), PARAM order
    n_samples = params_all.shape[0]
    disc_idx = _discover_idx(n_samples, SAMPLE_SPLIT_SEED)
    true_disc = params_all[disc_idx]
    true_pd = _params_dict(true_disc)      

    # Get initial parameters
    params_init = _get_params(param_est_fn, seed.default_params, X_discover[0])

    print("Initial Parameters:")
    for k, v in params_init.items():
        print(f"  {k}: {v}")

    print("True Parameters:")
    for k, v in true_pd.items():
        print(f"  {k}: {v}")

    loss = _eval_loss(model_fn, spec.loss_fn, params_init, X_discover[0], spec.apply_model_fn)
    print(f"\nInitial loss, param est, train split: {loss:.4f}")
    loss = _eval_loss(model_fn, spec.loss_fn, true_pd, X_discover[0], spec.apply_model_fn)
    print(f"\nInitial loss, true params, train split: {loss:.4f}")
    loss = _eval_loss(model_fn, spec.loss_fn, params_init, X_discover[1], spec.apply_model_fn)
    print(f"\nInitial loss, param est, test split: {loss:.4f}")
    loss = _eval_loss(model_fn, spec.loss_fn, true_pd, X_discover[1], spec.apply_model_fn)
    print(f"\nInitial loss, true params, test split: {loss:.4f}")

    #     # Optimize on train split (X_discover[0]) using loss_fn_train
    #     print("\nOptimizing parameters...")
    #     params = _optimize(
    #         model_fn,
    #         loss_fn_train,
    #         params_init,
    #         X_discover[0],
    #         spec.scoring["gradient_descent"],
    #     )

    #     print("\nOptimized Parameters:")
    #     for k, v in params.items():
    #         print(f"  {k}: shape {v.shape if hasattr(v, 'shape') else type(v)}")

    #     # Evaluate final loss on test split (X_discover[1]) using loss_fn_test
    #     final_loss = _eval_loss(model_fn, loss_fn_test, params, X_discover[1])
    #     print(f"\nFinal loss: {final_loss:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
