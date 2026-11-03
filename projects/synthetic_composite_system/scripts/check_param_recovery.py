"""Checks that gradient descent recovers the true parameters of the true model.

If the LLM discovers the right model, scoring should also recover each plant's
true (K_L, K_N, mu_max). This fits `ground_truth_model` to the discovery training
data with the same optimiser and settings EDGAR uses in scoring
(`edgar.scoring.scoring._optimize` with `scoring.gradient_descent` from
config.yaml), from several initialisations, and compares the fitted parameters
with the ones used to simulate the data.

A parameter is only identifiable if its sub-component actually limits growth at
some time points, so the fraction of training trials that are light- vs
nitrogen-limited is reported for each plant.

Run from the repo root:
    python projects/synthetic_composite_system/scripts/check_param_recovery.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "projects" / "synthetic_composite_system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edgar.io.config import Config  # noqa: E402
from edgar.io.task_spec import TaskSpec  # noqa: E402
from edgar.scoring.scoring import _optimize  # noqa: E402
from edgar.scoring.utils import evaluate_sample_losses  # noqa: E402
from projects.synthetic_composite_system.data_loader.load_data import (  # noqa: E402
    ground_truth_model,
    simulate_growth,
)
from projects.synthetic_composite_system.data_loader.sub_components import (  # noqa: E402
    g_light,
    g_nitrogen,
)

logger = logging.getLogger(__name__)

PARAM_NAMES = ("K_L", "K_N", "mu_max")
DEFAULT_INIT = {"K_L": 1.0, "K_N": 1.0, "mu_max": 1.0}
# Log-uniform ranges for random initialisations, deliberately wider than the
# simulation's sampling ranges.
RANDOM_INIT_RANGES = {"K_L": (1e-3, 1.0), "K_N": (0.1, 100.0), "mu_max": (0.1, 5.0)}


def _match_true_params(
    X_train: dict[str, np.ndarray], X_sim: dict[str, np.ndarray], true_params: dict
) -> dict[str, np.ndarray]:
    """Finds each training plant's simulated row and returns its true parameters.

    Matching on the light values avoids re-implementing load_data's split logic.

    Args:
        X_train: Discovery training split, arrays of shape (n_disc, n_train_trials).
        X_sim: Full simulated data, arrays of shape (n_plants, n_trials).
        true_params: True parameters from `simulate_growth`, each (n_plants,).

    Returns:
        True parameters for the training plants, each of shape (n_disc,).

    Raises:
        RuntimeError: If a training plant does not match exactly one simulated plant.
    """
    rows = []
    for L_row in X_train["L"]:
        matches = [
            s for s, L_sim in enumerate(X_sim["L"]) if np.isin(L_row, L_sim).all()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one matching simulated plant, got {matches}")
        rows.append(matches[0])
    return {k: v[rows] for k, v in true_params.items()}


def _light_limited_fraction(X: dict[str, np.ndarray], params: dict) -> np.ndarray:
    """Fraction of trials per plant where light (not nitrogen) limits growth."""
    light = g_light(X["L"], params["K_L"][:, None])
    nitrogen = g_nitrogen(X["N"], params["K_N"][:, None])
    return np.mean(light < nitrogen, axis=1)


def _make_inits(
    X_train: dict[str, np.ndarray],
    true_params: dict,
    n_plants: int,
    n_random: int,
    seed: int,
) -> dict[str, dict]:
    """Builds the named initialisations, each a dict of (n_plants,) float32 arrays.

    "estimate" mimics a cheap parameter estimator, as EDGAR would use in a run:
    half-saturation constants at the median input, mu_max at the maximum growth.
    """
    rng = np.random.default_rng(seed)
    inits = {
        "true": {k: np.asarray(true_params[k], np.float32) for k in PARAM_NAMES},
        "default": {
            k: np.full(n_plants, v, np.float32) for k, v in DEFAULT_INIT.items()
        },
        "estimate": {
            "K_L": np.median(X_train["L"], axis=1).astype(np.float32),
            "K_N": np.median(X_train["N"], axis=1).astype(np.float32),
            "mu_max": np.max(X_train["growth_rate"], axis=1).astype(np.float32),
        },
    }
    for i in range(n_random):
        inits[f"random_{i}"] = {
            k: np.exp(rng.uniform(np.log(lo), np.log(hi), n_plants)).astype(np.float32)
            for k, (lo, hi) in RANDOM_INIT_RANGES.items()
        }
    return {
        name: {k: jnp.asarray(v) for k, v in p.items()} for name, p in inits.items()
    }


def main() -> int:
    """Runs the recovery check and returns 0 if every initialisation recovers the truth."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-random", type=int, default=5, help="Random initialisations."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random inits.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="Max relative error to count as recovered.",
    )
    parser.add_argument(
        "--max-iter", type=int, help="Override gradient_descent.max_iter."
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Override gradient_descent.learning_rate."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = TaskSpec.from_config(Config.from_yaml(PROJECT_DIR / "config.yaml"))
    gd_config = dict(spec.scoring["gradient_descent"])
    if args.max_iter is not None:
        gd_config["max_iter"] = args.max_iter
    if args.learning_rate is not None:
        gd_config["learning_rate"] = args.learning_rate
    loss_fn = spec.loss_fn[0] if isinstance(spec.loss_fn, tuple) else spec.loss_fn

    (X_train, X_test), _, _ = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    X_sim, sim_params = simulate_growth(
        **{
            k: v
            for k, v in spec.project_params.items()
            if k in ("seed", "n_plants", "n_trials")
        }
    )
    true_params = _match_true_params(X_train, X_sim, sim_params)
    n_plants = len(true_params["K_L"])

    data_train = {k: jnp.asarray(v) for k, v in X_train.items()}
    data_test = {k: jnp.asarray(v) for k, v in X_test.items()}
    inits = _make_inits(X_train, true_params, n_plants, args.n_random, args.seed)

    logger.info(
        "Fitting ground_truth_model to %d plants x %d trials, gradient_descent=%s",
        n_plants,
        X_train["L"].shape[1],
        gd_config,
    )
    light_frac = _light_limited_fraction(X_train, true_params)
    logger.info(
        "Light-limited fraction of training trials per plant: %s",
        np.round(light_frac, 2),
    )

    fitted_list, _ = _optimize(
        ground_truth_model,
        loss_fn,
        list(inits.values()),
        data_train,
        gd_config,
        spec.apply_model_fn,
    )

    all_ok = True
    logger.info(
        "\n%-10s %10s %10s %8s %8s %8s  %s",
        "init",
        "train",
        "test",
        "K_L",
        "K_N",
        "mu_max",
        "recovered",
    )
    for (name, init), fitted in zip(inits.items(), fitted_list):
        fitted = {k: np.asarray(v) for k, v in fitted.items()}
        rel_err = {k: np.abs(fitted[k] / true_params[k] - 1) for k in PARAM_NAMES}
        train = np.asarray(
            evaluate_sample_losses(
                ground_truth_model, loss_fn, fitted, data_train, spec.apply_model_fn
            )
        ).mean()
        test = np.asarray(
            evaluate_sample_losses(
                ground_truth_model, loss_fn, fitted, data_test, spec.apply_model_fn
            )
        ).mean()
        ok = all((rel_err[k] < args.rtol).all() for k in PARAM_NAMES)
        all_ok &= ok
        logger.info(
            "%-10s %10.2e %10.2e %7.1f%% %7.1f%% %7.1f%%  %s",
            name,
            train,
            test,
            *(100 * rel_err[k].max() for k in PARAM_NAMES),
            "yes" if ok else "NO",
        )
        if not ok:
            for p in range(n_plants):
                logger.info(
                    "    plant %d: %s",
                    p,
                    ", ".join(
                        f"{k} true={true_params[k][p]:.4g} init={float(init[k][p]):.4g} "
                        f"fit={fitted[k][p]:.4g}"
                        for k in PARAM_NAMES
                    ),
                )
    logger.info("\n(K_L, K_N, mu_max columns: max relative error across plants)")
    logger.info("Result: %s", "PASS" if all_ok else f"FAIL (rtol={args.rtol})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
