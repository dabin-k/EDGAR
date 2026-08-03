# EDGAR: Equation Discovery with Graphical AI Reasoning

[![Tests](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml/badge.svg)](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml)

EDGAR is an evolutionary framework for discovering scientific equations using LLM-generated programs and parameter estimators.

Each run evolves a population of candidate models across multiple islands. The LLM generates numpy model code and parameter estimators; JAX-translated versions are then optimised via gradient descent. Programs are selected, pruned, and migrated between islands over many generations.

---

## Prerequisites

The recommended way to manage dependencies and environments is [uv](https://docs.astral.sh/uv/).

To install `uv` on Linux or macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quickstart

### 1. Install
#### uv (recommended)
Run the command
```bash
uv sync
```
from the repo root which will automatically setup the environment there.
Now any commands will be run in this environment when using the prefix `uv run`, e.g
```bash
uv run edgar test projects/synthetic_data/config.yaml
```

#### Conda + pip
```bash
conda create -n edgar python=3.13 -y
conda activate edgar
pip install -e .
```
This installs the `edgar` package in editable mode.
`import edgar` then works from any cwd / IDE cell without `sys.path` hacks.

To verify your environment is setup correctly run the script
```bash
bash scripts/check_env.sh
```
### 2. Set API key

Add your Gemini API key to `.env` in the project root. The code reads `GOOGLE_API_KEY` (not `GEMINI_API_KEY`):

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

The key is loaded automatically at runtime via `python-dotenv`. You can also export it directly:

```bash
export GOOGLE_API_KEY=your_key_here
```
You can do the same for an `ANTHROPIC_API_KEY` if using an anthropic model.

Verify your API key is configured correctly by running
```bash
bash scripts/check_api_keys.sh
```

### 3. Run an experiment

```bash
edgar run projects/orientation_tuning/config.yaml
```

Control logging verbosity (default: `compact`):

```bash
edgar run projects/orientation_tuning/config.yaml --log-level code
edgar run projects/orientation_tuning/config.yaml --log-level prompts
```

Override config values on the command line:

```bash
edgar run projects/orientation_tuning/config.yaml --evolution.n_generations=5
edgar projects/orientation_tuning/config.yaml --llms.model_llm=gemini-2.5-pro
```

Run a quick test with reduced settings (1 generation, 2 islands, batch size 2) to verify the pipeline is wired correctly

```bash
edgar test projects/orientation_tuning/config.yaml
```

Reproduce a previous run from its saved `task_spec.yaml`:

```bash
edgar run program_databases/05-06/14-32-10/task_spec.yaml
```

To launch the dashboard to view in progress and finished experiments:
```bash
edgar dashboard
```
By default this allows access to data saved in `program_databases`. If the data you require is saved elsewhere, do
```bash
edgar dashboard {data_directory}
```

A run which failed can be resume via, for example:
```bash
edgar resume program_databases/mm-dd/hh-mm-ss
```

---

## Run output

By default, each run writes to `program_databases/MM-DD/HH-MM-SS/`:

```text
program_databases/
└── MM-DD/
    └── HH-MM-SS/
        ├── task_spec.yaml          # Full config + git SHA + prompt schemas + seed code. Read-only.
        ├── population.jsonl        # All Programs — code, losses, params, lineage. Main scientific output.
        ├── island_census.jsonl     # Island membership at the end of each generation.
        ├── metrics.jsonl           # Timing, token and retry statistics for the various parts of the algorithm.
        ├── status.json             # Overall status of the run, read by the dashboard.
        ├── run.log                 # Human-readable execution trace.
        └── image_feedback/         # Only present if plot_fn is defined, image_feedback prompt shown to LLM.
            └── gen_000/
                └── island_000/
                    └── batch_000/
                        └── image.png
        └── image_fits/             # Only present if plot_fn is defined, plots each program before and after parameter optimization.
            └── P0000.png 
```

---

## Running on a remote server (long runs over SSH)

A full run can take a few hours. SSH connections drop. Use `tmux` so the run survives disconnects, then port-forward the dashboard back to your laptop.

The examples below use the Janelia workstation `ruttenv-ws1.hhmi.org` to keep things concrete; substitute your own server, conda root, and repo path.

### 1. Start a tmux session on the server and launch the run

```bash
ssh ruttenv-ws1.hhmi.org
tmux new -s edgar          # creates a session named "edgar"

# Inside tmux, in the first window:
source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh && conda activate edgar
cd /groups/ahrens/home/ruttenv/python_packages/EDGAR

edgar run projects/orientation_tuning/config.yaml \
  --io.data_path=data/gratings_drifting_GT1_2019_04_12_1.npy \
  --llms.model_llm=claude-sonnet-4-6 \
  --llms.param_est_llm=claude-sonnet-4-6 \
  --llms.jax_model_translator_llm=claude-haiku-4-5
```

### 2. Open a second tmux window for the dashboard

Press `Ctrl-b c` to open a new window inside tmux, then:

```bash
source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh && conda activate edgar
cd /groups/ahrens/home/ruttenv/python_packages/EDGAR
edgar dashboard --no-open
```

Defaults bind to `127.0.0.1:8765`. Use `Ctrl-b 0` / `Ctrl-b 1` to flip between windows.

If you want the dashboard reachable from anywhere on the network (no SSH tunnel) — less secure — pass `--host 0.0.0.0`:

```bash
edgar dashboard --host 0.0.0.0 --no-open
```

### 3. Access the dashboard from your laptop

In a local terminal on your Mac, set up port forwarding:

```bash
ssh -L 8765:localhost:8765 ruttenv-ws1.hhmi.org
```

Leave that terminal open. Then in your browser: `http://localhost:8765`.

### 4. Detach, drop SSH, come back later

- `Ctrl-b d` — detach from tmux. The session keeps running on the server; you can now safely drop SSH.
- `tmux attach -t edgar` — reattach in a later SSH session, even from a different machine.
- `tmux ls` — list sessions.

### 5. If the run dies anyway (OOM, reboot, killed tmux)

```bash
edgar resume program_databases/MM-DD/HH-MM-SS/
```

Loads the saved population and continues from the next unfinished generation. See `tutorials/how_to_run.md` for the full resume semantics.

---

## Building a new project

A project is a directory under `projects/<task>/` that tells EDGAR three things: where the data comes from (`data_loader/load_data.py`), what a candidate model looks like (`seed_programs/`), and how to prompt the LLM (`prompts.yaml`). This section covers the shared scaffolding, then splits into the two model contracts EDGAR supports.

### Choose a model contract

EDGAR ships two model contracts. They are not competing defaults — they solve different problem classes, and you should pick based on the shape of your data.

- **Classical contract** — the model is a function `model(data, params) -> mean` mapping features to predictions. Use this for static regression (orientation tuning, place-field fitting, any stimulus-response curve) where the trial dimension is not a time series and there is no temporal causality to worry about. Reference project: `projects/orientation_tuning/`.

- **State-space (SSM) contract** — the model is a one-step causal update `model(state, y_prev, params) -> (new_state, mean)` that the framework scans over the trajectory. Use this whenever the data is a time series and the task is one-step prediction. The scan structurally removes `y[≥t]` from the model's scope at step `t`, which eliminates the temporal-leakage failure mode the classical contract has on time-series data (see `docs/state_space_dsl_leakage_fix.md`). Reference projects: `projects/fhn_excitable/` and `projects/vdp_relaxation/`.

If your data is a time series, use SSM. If it isn't, use classical. The rest of this section is arranged the same way: shared scaffolding first, then the parts of the contract that differ.

### Shared scaffolding

Scaffold a new project with:

```bash
edgar init-project my_task
```

This creates:

```text
projects/my_task/
├── config.yaml
├── seed_programs/
│   ├── model1.py
│   ├── model2.py
│   ├── param_est1.py
│   └── param_est2.py
├── data_loader/
│   └── load_data.py
└── image_feedback/
    └── plot.py
```

**If you're building an SSM project, `init-project`'s stubs are classical-shaped and will point you in the wrong direction.** Copy a canonical SSM project instead:

```bash
cp -r projects/fhn_excitable projects/my_task
```

Then edit `data_loader/load_data.py` for your synthesis, `config.yaml` for your `project_params`, and `prompts.yaml` for your domain hints. The classical `init-project` template is fine as-is for regression / non-time-series problems.

Regardless of which contract you use, the following are the same:

**`config.yaml`.** Override only what differs from `projects/config_default.yaml`. Minimum:

```yaml
io:
  data_path: /path/to/data.npy
```

Common overrides:

```yaml
project_params:
  my_threshold: 0.5   # kwargs passed to load_data()

evolution:
  n_generations: 20

llms:
  model_llm: gemini-2.5-pro
```

Contract-specific `config.yaml` knobs (SSM needs `gradient_clip_norm` and a lower `param_penalty_weight`) are covered below.

**`prompts.yaml`.** Overrides `projects/prompt_defaults.yaml`. The two files are deep-merged, so include only the fields you want to change — everything else is inherited.

String fields (`base`, `explore`, `code_guidelines`, etc.) are replaced entirely when specified. **List fields (`config_vars`, `parent_vars`) are also replaced entirely** — if you add a new variable, you must re-list all of them.

`explore` and `exploit` can be set to `null` (or omitted entirely) if you don't need a mode-specific section — the JAX translator and parameter estimator prompts typically leave both as `null`.

Example — override only the `base` and `code_guidelines` for model generation:

```yaml
model:
  base: |
    You are an AI scientist modelling orientation tuning in visual cortex.
    Below are {k_max} neuron models sorted from worst to best.
    Create a new model with lower loss than all of them.
  code_guidelines: |
    * Model signature: def model(data, params):
    * data has keys "stimulus" (radians) and "response".
    * Clip free parameters to biologically plausible ranges at the top of the function.
```

All other `model` fields (explore, exploit, docstring_guidelines, parent_detail_template, config_vars, parent_vars) are inherited from the defaults unchanged.

For SSM projects, the overrides also need to teach the LLM the state-space contract — the reference `projects/fhn_excitable/prompts.yaml` overrides `model.base`, `model.code_guidelines`, `parameter_estimator.code_guidelines`, and `jax_translator_model.code_guidelines`.

**`image_feedback/plot.py` (optional).** Must define:

```python
plot_model_fits(data, parent_programs, save_path="")
```

- `data`: `X_disc_train` dictionary of JAX arrays.
- `parent_programs`: list of `Program` objects to visualize.
- `save_path`: file path to save the generated figure.

If this file is left as a `pass` stub, no images are generated or provided as LLM feedback.

**Data-split contract for `load_data`.** Both contracts share the top-level five-way split. `load_data(data_path, **project_params)` returns `(X_discover, X_validate, X_eval)`:

- `X_discover = (X_disc_train, X_disc_test)` — seen by the LLM discovery loop.
- `X_validate = (X_val_train, X_val_test)` — never seen during discovery.
- `X_eval` — small subset of `X_disc_train` used for generating model fingerprints. Must be a dict containing feature/response JAX arrays (same keys as the other splits) plus `_sample_indices`: a NumPy array of integer indices indicating which samples from `X_disc_train` are included.

Data-array shape convention: `(n_samples, n_trials)` per key. For SSM projects the trial dimension is time (`(n_samples, T)`) and there are extra flags in `X_eval` — see below.

**Validate, then run.**

```bash
edgar validate my_task
edgar run projects/my_task/config.yaml
```

The parts that differ between the two contracts — the model function signature, the extra callables in `load_data.py`, and the config knobs that matter — follow.

### Classical contract

Reference project: `projects/orientation_tuning/`.

**Model signature.** `model*.py` defines:

```python
def model(data, params):
    ...
```

- `data`: dict of JAX arrays for one sample, e.g. `data["stimulus"]` of shape `(n_trials,)`.
- `params`: dict of named scalars / arrays.
- Returns predictions of shape `(n_trials,)`.
- Must set `model.DEFAULT_PARAMS = {"param_name": initial_value, ...}`.

`param_est*.py` defines:

```python
def parameter_estimator(data):
    ...
```

Returns a parameter dict with the same keys as `model.DEFAULT_PARAMS`. Keep it simple — no `scipy.optimize` or `curve_fit`.

**`load_data.py` callables.** Two required:

- `load_data(data_path, **kwargs) -> (X_discover, X_validate, X_eval)` — as above.
- `loss_fn(model_output, data) -> jnp.ndarray` of shape `(n_samples,)` — per-sample loss between predictions and data.

That is the full contract. The framework applies the model per-sample via `vmap` internally; you don't need to write `apply_model`.

### State-space (SSM) contract

Reference projects: `projects/fhn_excitable/` (FitzHugh-Nagumo excitable neuron with a hidden recovery variable) and `projects/vdp_relaxation/` (Van der Pol relaxation oscillator). Copy either as a starting point.

The rest of this section is what the LLM needs to know and what you need to wire up in `load_data.py`. For the design argument — why the DSL exists, what leakage looked like under the classical contract, and what the DSL forbids by construction — see `docs/state_space_dsl_leakage_fix.md`.

#### Why this contract exists, in one paragraph

EDGAR's classical scoring path lets the LLM see the whole time series and predict every point of it. On time-series data this rewards LLM programs that peek at the answer — off-by-one bugs, `cumsum` tricks, deliberate "use the observation to correct the mean" reasoning. Post-optimization losses drop below the noise floor and the winners fail to generalize. The SSM contract makes causal leakage a Python scope error rather than a discipline problem: the LLM writes a one-step update, the framework scans it over the trajectory, and `y[t]` is never in the function's arguments at step `t`. Every model class that is scientifically a predictive dynamical system (ODE/SDE discretizations, Kalman filters, HMMs, RNN cells, Hawkes processes, AR/MA) is expressible; non-causal smoothers and global spectral operators are not.

#### The model contract

The LLM writes `model(state, y_prev, params) -> (new_state, mean)`:

- `state` — dict of scalars carrying the model's current belief, built entirely from `y[<t-1]`.
- `y_prev` — scalar, the single most recent observation `y[t-1]`.
- `params` — dict of learnable scalars. Must include `log_sigma_obs`. Any key prefixed `s0_` is stripped and used as the initial-state value for the matching state key (so `s0_V: -1.0` initializes `state["V"]`).
- Returns `(new_state, mean)`. `new_state` must have the same keys as `state` (the scan carry is invariant). `mean` is the predicted mean of the next observation.

Minimal persistence baseline (`projects/fhn_excitable/seed_programs/model1.py`):

```python
def model(state, y_prev, params):
    new_state = {"y_last": y_prev}
    mean = new_state["y_last"]
    return new_state, mean

model.DEFAULT_PARAMS = {
    "log_sigma_obs": 0.0,
    "s0_y_last": 0.0,
}
```

A meatier example — FitzHugh-Nagumo with Kalman-style innovation on a hidden recovery variable `w`:

```python
def model(state, y_prev, params):
    V, w = state["V"], state["w"]
    dt, I0, eps = params["dt"], params["I0"], params["eps"]
    a, b = params["a"], params["b"]
    k_V, k_w = params["k_V"], params["k_w"]

    # Kalman-style correction from the previous observation.
    innov = y_prev - V
    V_c = V + k_V * innov
    w_c = w + k_w * innov

    # FHN dynamics: cubic-threshold voltage + slow recovery.
    dV = V_c - V_c**3 / 3.0 - w_c + I0
    dw = eps * (V_c + a - b * w_c)
    V_new = V_c + dt * dV
    w_new = w_c + dt * dw

    new_state = {"V": V_new, "w": w_new}
    return new_state, V_new

model.DEFAULT_PARAMS = {
    "dt": 0.05, "I0": 0.5, "eps": 0.08, "a": 0.7, "b": 0.8,
    "k_V": 0.4, "k_w": 0.05,
    "log_sigma_obs": -1.5,
    "s0_V": -1.0, "s0_w": -0.5,
}
```

Correcting the state from `y_prev` is not leakage — `y[t-1]` is legitimate causal information, exactly what a Kalman filter uses. The scoring loss is the Gaussian NLL of the predicted `mean` against the true `y[t]`, averaged over the trajectory after a warmup skip.

For more seeds (moving-average blend, damped oscillator with frequency tracking) see `projects/fhn_excitable/seed_programs/model{2,3,4}.py`.

#### Framework hooks in `load_data.py`

An SSM `load_data.py` must define four callables and one module-level constant:

- `WARMUP_STEPS: int` — module constant (typically 50-100). Baked into `loss_fn`'s closure at import time; changing it means editing this line. It must be a Python int, not a config value, because the loss runs inside `jit`.
- `load_data(data_path, **project_params) -> (X_discover, X_validate, X_eval)` — the standard EDGAR split; each `X_*` dict must contain key `"y"` of shape `(n_samples, T)`. `X_eval` also carries `_sample_indices` and `_fingerprint_only: True` (short traces + narrower feature set — see `docs/state_space_dsl_leakage_fix.md` §5.7 for why raw-trace cosine similarity fails on long trajectories).
- `apply_model(model_fn, data, params)` — scans `model_fn` over `data["y"]` with `jax.lax.scan` and `vmap`. Returns `(n_samples, T-1, 2)` where column 0 is means and column 1 is the broadcast `log_sigma`.
- `loss_fn(model_output, data)` — Gaussian NLL over the post-warmup horizon, returned per-sample.
- `validate_step(model_fn, default_params, source)` — eager sanity check called by `leakage_check.py`. Asserts the function returns `(state_pytree, scalar)` with the same pytree structure as the initial state, contains `log_sigma_obs`, and produces finite output.

Skeleton `load_data.py` (fill in `_synth` and the `X_eval` block for the task):

```python
import jax
import jax.numpy as jnp
import numpy as np

WARMUP_STEPS: int = 100


def load_data(data_path="", T=2400, n_trajectories=32, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    ys = np.stack([_synth_one_trajectory(rng, T, **kwargs)
                   for _ in range(n_trajectories)])
    split = n_trajectories // 2
    y_disc, y_val = jnp.asarray(ys[:split]), jnp.asarray(ys[split:])
    X_disc_train = {"y": y_disc}
    X_disc_test  = {"y": y_disc}
    X_val_train  = {"y": y_val}
    X_val_test   = {"y": y_val}
    X_eval = {
        "y": jnp.asarray(ys[:4, :200]),          # short traces for dedup
        "_sample_indices": np.arange(4),
        "_fingerprint_only": True,
    }
    return (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval


def _split_params_s0(params):
    init_state, dyn_params = {}, {}
    for k, v in params.items():
        if k.startswith("s0_") and len(k) > 3:
            init_state[k.removeprefix("s0_")] = v
        else:
            dyn_params[k] = v
    return init_state, dyn_params


def apply_model(model_fn, data, params):
    y = data["y"]
    fingerprint_only = bool(data.get("_fingerprint_only", False))

    def per_sample(y_traj, p):
        init_state, dyn_params = _split_params_s0(p)

        def scan_step(state, y_prev):
            return model_fn(state, y_prev, dyn_params)

        _, means = jax.lax.scan(scan_step, init_state, y_traj[:-1])
        if fingerprint_only:
            return y_traj[1:] - means
        log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
        return jnp.stack([means, log_sigma], axis=-1)

    return jax.vmap(per_sample, in_axes=(0, 0))(y, params)


def loss_fn(model_output, data):
    y = data["y"][:, 1:]
    means      = model_output[:, WARMUP_STEPS:, 0]
    log_sigmas = model_output[:, WARMUP_STEPS:, 1]
    tgt        = y[:, WARMUP_STEPS:]
    nll = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    return jnp.mean(nll, axis=-1)


def validate_step(model_fn, default_params, source=""):
    assert "log_sigma_obs" in default_params
    init_state, dyn_params = _split_params_s0(default_params)
    init_state_j = jax.tree_util.tree_map(jnp.asarray, init_state)
    dyn_params_j = jax.tree_util.tree_map(jnp.asarray, dyn_params)
    new_state, mean = model_fn(init_state_j, jnp.asarray(0.5), dyn_params_j)
    assert jax.tree_util.tree_structure(init_state_j) == jax.tree_util.tree_structure(new_state)
    assert jnp.asarray(mean).shape == () and bool(jnp.isfinite(mean))
```

See `projects/fhn_excitable/data_loader/load_data.py` for the full production version including sidecar keys (`_persistence_nll`, `_w_true`, `_V_true`) used by the oracle-NLL script.

#### `config.yaml`: two SSM-specific knobs

Minimum:

```yaml
io:
  data_path: ""                     # unused for synthetic testbeds

evolution:
  n_generations: 4
  n_islands: 8
  batch_size: 4

llms:
  model_llm: claude-sonnet-4-6

scoring:
  param_penalty_weight: 0.001       # SSM adds s0_* scalars; drop the penalty ~10x
  timeout_s: 600.0
  gradient_descent:
    max_iter: 500
    learning_rate: 0.005
    gradient_clip_norm: 5.0         # required — see below

project_params:
  T: 2400
  n_trajectories: 32
  # ... task-specific kwargs passed to load_data()
```

- `gradient_descent.gradient_clip_norm` — without it, ~15% of programs diverge to `inf` loss on the first Adam steps because back-propagating through T=2400 scan steps compounds gradients. `5.0` is a safe default.
- `scoring.param_penalty_weight` should be an order of magnitude lower than in non-SSM projects (typical `0.001` vs `0.01`), because the `s0_*` initial-state values inflate `n_params` and would otherwise unfairly punish state-space programs.

#### Recommended extras

- `leakage_check.py` — standalone self-test that exercises the isolation invariant and validates every seed. Run this before evolution.
- `scripts/` — oracle-NLL floor computation and post-run analysis.

#### Running an SSM project

Three commands, in order of what a new user should run first:

```bash
# 1. Structural leakage self-test — asserts y[>=t] cannot affect mean[<t],
#    validates every seed, and confirms the seed floor is above the oracle.
#    Runs in <1 minute on CPU. Failure here means the DSL contract is broken.
python projects/fhn_excitable/leakage_check.py

# 2. Oracle NLL floor — the best possible one-step Gaussian NLL given the
#    true dynamics and true hidden state. Sets the lower bound any evolved
#    program is measured against. ~30 seconds.
python projects/fhn_excitable/scripts/fhn_oracle_nll.py

# 3. Full evolution run. Config defaults are smoke-test values; override
#    on the CLI. Wall clock: ~2 hours for 4 generations on a single A4000
#    (Claude Sonnet 4.x as the model LLM). See docs/cost_breakdown_1d_run.md
#    for the full profile.
edgar run projects/fhn_excitable/config.yaml \
  --evolution.n_generations=4 \
  --evolution.n_islands=8 \
  --evolution.batch_size=4
```

For a 4-generation, 8-island, batch-4 FHN run, expect roughly 32-44 programs, ~80% of wall clock in scoring (JIT compile dominates over gradient descent), ~16% in parallelized LLM calls.

#### Interpreting an SSM run

After a run finishes, generate the post-run analysis:

```bash
python projects/fhn_excitable/scripts/post_run_analysis.py program_databases/MM-DD/HH-MM-SS/
```

This writes `post_analysis.md` and a `figures/top4_fits.png` panel. A representative result from the 4-generation FHN run at `program_databases/08-01/17-51-50/` (44 programs, 41 with finite loss):

| benchmark              | NLL (nat/bin) |
|------------------------|--------------:|
| oracle floor           | **-2.3134**   |
| best evolved program   | -2.1992       |
| best hand-written seed | -2.1703       |
| persistence baseline   | -2.1355       |

Discovery budget from best seed to oracle: 0.143 nat. The top evolved program closes 20% of it on held-out validate cells. More telling than the point number:

- **80% of finite-loss programs (33/41) beat the best hand-written seed floor.**
- **90% (37/41) introduced a state variable outside the seeds' vocabulary** — evidence that evolution converged on the hidden recovery variable the task requires, not on parametric tweaks of the 1-D seeds.
- The top 10 programs all carry `state = {"V", "w"}` or extensions (`{"V", "u", "w"}`).

Top-3 excerpt:

| rank | gen | disc NLL | state keys       | name                                                    |
|-----:|----:|--------:|------------------|---------------------------------------------------------|
|    1 |   0 | -2.1992 | `[V, w]`         | FitzHugh-Nagumo Excitable Filter with Innovation-Coupled Recovery |
|    2 |   3 | -2.1985 | `[V, w]`         | Semi-Implicit FitzHugh-Nagumo Filter with Trapezoidal Recovery    |
|    3 |   2 | -2.1978 | `[V, w]`         | FitzHugh-Nagumo RK4 Excitable Filter with Affine Observation      |

The scoring signal is doing what it's supposed to: no program beat the oracle floor (would indicate residual leakage), and the seed-to-evolved gap is closed by state expansion, not indexing tricks.

For interactive exploration and per-program trajectories, launch the dashboard:

```bash
edgar dashboard program_databases/
```

---

## Further reading

- `docs/state_space_dsl_leakage_fix.md` — full design report on the SSM contract: what was tried before, why the DSL beats prompt engineering and windowed scoring, the two-line engine change, alternatives ruled out, generality analysis.
- `docs/cost_breakdown_1d_run.md` — wall-clock and dollar-cost profile of a real 4-generation FHN run.
- `docs/plans/multivariate_dsl.md` — the follow-on spec for d-dimensional observations.
