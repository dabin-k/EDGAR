"""Score synthetic_data_v2's seed programs through edgar's real scoring path.

Builds the TaskSpec from the project config (which loads evaluate/evaluate.py), then
calls edgar.scoring.score on a Population of the two seeds. No LLM: the seeds' JAX
code is the numpy source, which is already valid JAX here.

Checks that scoring uses the project's evaluate_fn (losses match the standalone
harness, ~0.043) rather than default_evaluate (which would crash: the model expects a
window, not a (n_blocks, n_cells, block_len) sample).

The __main__ guard is required: scoring spawns a subprocess per program, and `spawn`
re-imports this module in each child.
"""

from pathlib import Path

from edgar.evolution.population import Population
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import score


def main():
    spec = TaskSpec.from_config(
        Config.from_yaml(Path("projects/synthetic_data_v2/config.yaml"))
    )
    print(f"evaluate_fn loaded from project: {spec.evaluate_fn is not None}")
    assert spec.evaluate_fn is not None

    X_disc, _, X_eval = spec.load_data_fn(spec.io["data_path"], **spec.project_params)

    population = Population()
    for program in spec.seed_programs:
        program.code.model_jax = program.code.model.replace("numpy", "jax.numpy")
        population.add(program)

    score(
        population,
        X_disc,
        X_eval,
        spec.scoring,
        spec.loss_fn,
        split="discover",
        evaluate_fn=spec.evaluate_fn,
    )

    print(f"\n{'program':<16}{'n_params':>9}{'init':>10}{'final':>10}")
    for i in range(len(population)):
        p = population[i]
        losses = p.program_losses.discover
        print(f"{p.name:<16}{p.n_params:>9}{losses.init:>10.5f}{losses.final:>10.5f}")
        assert losses.final < 0.06, "seed loss far from the standalone harness's ~0.043"
        assert p.eval_fingerprint is not None, "fingerprint did not go via evaluate_fn"

    decay = population[1].params["decay"]
    print(f"\nfitted decay per recording: {[round(float(d), 4) for d in decay]}")
    print(f"fingerprint shape: {population[0].eval_fingerprint.shape}")

    # plot_fn imports the project's evaluate.py via __file__, so it only loads if
    # task_spec binds the source path into the namespace.
    # run.py passes X_discover[1] -- the *test* blocks -- with params fitted on train,
    # so the LLM is shown generalisation error, not training fit. Mirror that here.
    assert spec.plot_fn is not None, "plot_fn failed to load"
    out = "journal/2026-07-13_synthetic_data_v2_feedback.png"
    spec.plot_fn(
        X_disc[1], [population[i] for i in range(len(population))], save_path=out
    )
    print(f"plot_fn rendered {out}  (test blocks, train-fitted params)")

    print("\nscoring integration OK")


if __name__ == "__main__":
    main()
