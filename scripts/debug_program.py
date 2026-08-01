#!/usr/bin/env python
"""Interactive triage: run one state-space program's scan step-by-step.

Runs the LLM's ``model(state, y_prev, params)`` in a plain Python for-loop
(not ``jax.lax.scan``) for T=50 steps, printing state + prediction at every
step. Useful for figuring out WHY a program produces NaNs or inf losses
during regular scoring.

Not integrated into the scoring pipeline — the pipeline is jit-wrapped, and
Python ``print`` inside a traced function fires only once at trace time.

Usage:
    EDGAR_SCAN_DEBUG=1 python scripts/debug_program.py \\
        projects/oscillator_ss/config.yaml \\
        projects/oscillator_ss/seed_programs/model1.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.code_loading import load_function_from_source


def _numpy_to_jax_source(src: str) -> str:
    """Trivial numpy → jax rewrite (mirrors score_seeds.py)."""
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def _extract_default_params(source: str) -> dict:
    """Read ``model.DEFAULT_PARAMS`` out of a program source string."""
    ns = {}
    exec(source, ns)
    model = ns.get("model")
    if model is None or not hasattr(model, "DEFAULT_PARAMS"):
        raise ValueError("program must define model.DEFAULT_PARAMS")
    return dict(model.DEFAULT_PARAMS)


def debug_run(config_path: Path, program_path: Path, max_steps: int = 50) -> int:
    if not os.environ.get("EDGAR_SCAN_DEBUG"):
        print("[debug_program] set EDGAR_SCAN_DEBUG=1 to enable step-by-step prints",
              file=sys.stderr)
        return 2

    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)
    (X_train, _), _, _ = spec.load_data_fn(spec.io["data_path"], **spec.project_params)
    y_traj = X_train["y"][0]                    # first trajectory

    src = program_path.read_text()
    model_fn = load_function_from_source(_numpy_to_jax_source(src), "model")
    if model_fn is None:
        print(f"[debug_program] couldn't load model() from {program_path}", file=sys.stderr)
        return 1

    default_params = _extract_default_params(src)

    # Validate before running — fail fast with a clear error.
    from projects.oscillator_ss.data_loader.load_data import validate_step, _split_params_s0
    validate_step(model_fn, default_params, program_code=src)

    init_state, dyn_params = _split_params_s0(default_params)
    init_state = jax.tree_util.tree_map(jnp.asarray, init_state)
    dyn_params = jax.tree_util.tree_map(jnp.asarray, dyn_params)

    print(f"=== {program_path.name} on {config_path.parent.name} ===")
    print(f"init state:  {dict(init_state)}")
    print(f"dyn params:  {dict(dyn_params)}")
    print(f"y[:5]:       {y_traj[:5]}")
    print(f"---\nstepping {max_steps} iterations (Python for-loop, outside jit):")

    state = init_state
    for i in range(min(max_steps, y_traj.shape[0] - 1)):
        y_prev = y_traj[i]
        new_state, mean = model_fn(state, y_prev, dyn_params)
        print(
            f"  t={i:3d}  y_prev={float(y_prev):+.4f}  mean={float(mean):+.4f}  "
            f"state={ {k: float(v) for k, v in new_state.items()} }"
        )
        if not bool(jnp.isfinite(mean)):
            print(f"[debug_program] non-finite mean at step {i}, stopping.")
            return 1
        state = new_state
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("program", type=Path)
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args()
    return debug_run(args.config, args.program, max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
