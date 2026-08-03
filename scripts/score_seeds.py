#!/usr/bin/env python
"""Score seed programs for a project without LLM calls (step 7b)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _score_one_model


def _numpy_to_jax_source(src: str) -> str:
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def main(config_path: str) -> int:
    config = Config.from_yaml(Path(config_path))
    spec = TaskSpec.from_config(config)
    X_discover, _, X_eval = spec.load_data_fn(
        spec.io["data_path"], **spec.project_params
    )
    data = X_discover
    scoring_cfg = spec.scoring

    for prog in spec.seed_programs:
        prog.code.model_jax = _numpy_to_jax_source(prog.code.model)
        final, init, *_ = _score_one_model(
            prog,
            data,
            spec.loss_fn,
            scoring_cfg,
            X_eval=X_eval,
            apply_model_fn=spec.apply_model_fn,
        )
        # Prefer MSE baseline if present (windowed projects); fall back to
        # NLL baseline for state-space projects; else print without skill.
        test_dict = X_discover[1]
        if "_persistence_mse" in test_dict:
            base = float(test_dict["_persistence_mse"].mean())
            base_key = "persistence_mse"
        elif "_persistence_nll" in test_dict:
            base = float(test_dict["_persistence_nll"].mean())
            base_key = "persistence_nll"
        else:
            base = None
            base_key = None
        if base is not None:
            skill_str = f" skill={final / base:.3f}" if base > 0 else ""
            print(
                f"{prog.name}: init={init:.6f} final={final:.6f} "
                f"{base_key}={base:.6f}{skill_str}"
            )
        else:
            print(f"{prog.name}: init={init:.6f} final={final:.6f}")
        if not (final < float("inf")):
            return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/score_seeds.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
