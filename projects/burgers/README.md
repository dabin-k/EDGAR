# Burgers benchmark (STENCIL-NET vs SINDy vs EDGAR)

Benchmark for quantifying EDGAR on near-equilibrium, low-but-nonzero-noise systems.
Full plan: `journal/burgers_benchmark_plan.md`.

Layout:
- `data_loader/` — forced-Burgers generator (physics vendored verbatim from
  mosaic-group/STENCIL-NET), shared data-access layer, and the shared forecast-MSE
  metric. `burgers_clean.npz` is **git-ignored**; regenerate it per machine.
- `bench/stencilnet/` — Step 2. STENCIL-NET in isolation. `stencilnet_src/` holds
  their `MLPConv` + RK3 time-stepping copied unmodified (see its PROVENANCE.md);
  `runner.py`/`sweep.py` are the only original code and just wire their model to our
  data. Device auto-detects (CUDA > MPS > CPU).

## Setup on a new machine

```bash
# 1. Python deps (GPU box: install the CUDA build of torch for your CUDA version)
pip install numpy scipy matplotlib pysindy
pip install torch            # or the cu121/cu118 wheel from pytorch.org for your GPU

# 2. Regenerate the data bundle (~40 s; writes data_loader/burgers_clean.npz)
python projects/burgers/data_loader/regenerate.py
```

The STENCIL-NET model/physics are **vendored** into this repo, so you do NOT need to
clone github.com/mosaic-group/STENCIL-NET to run the benchmark. Clone it only if you
want their original notebooks / pretrained `.pth` models for reference:

```bash
git clone https://github.com/mosaic-group/STENCIL-NET.git
```

## Run Step 2 (STENCIL-NET)

```bash
cd projects/burgers/bench/stencilnet
# single noise level
PYTHONPATH=. python runner.py --noise 0.1 --epochs 30000
# full noise sweep (writes results/stencilnet_sweep.json)
PYTHONPATH=. python sweep.py --epochs 30000 --levels 0.0 0.01 0.05 0.1 0.3
```

Reference recipe (ForcedBurgers* notebooks): Adam lr=1e-3, 30k epochs, m=4 rollout,
decay 0.9, l_wd=1e-7, l_n=1e-5 (noisy only), train on first 1001 coarse time steps.
Clean levels use forcing-aware RK3 loss; noisy levels use TVD-RK3 + noise-as-latent.

## EDGAR control task (Step 5)

This project is also a full EDGAR task. It asks EDGAR to discover a **discrete
autoregressive map**

    x(t) = f( x(t-1), ..., x(t-m) )            [m = evaluate.MAX_LENGTH, currently 2]

for the population of sensors — the temporal-lag **control** from the plan (no
spatial-stencil structure imposed; that comes in Step 6). Two deliberate choices:

- **Unforced field.** The map is fit to the *autonomous* (`forcing_seed=None`)
  coarse field, so the dynamics are a function of recent state alone and the AR
  form is well-posed. (The Steps 1-4 benchmark field is forced, and SINDy /
  STENCIL-NET were *given* the forcing; an AR model has no exogenous input, so it
  would be mis-specified on the forced field.)
- **No exact ground-truth equation.** The field is a continuous PDE integrated at
  fine `dt` then coarsened x20 in time and x4 in space. So (a) the exact one-step
  map is the time-`dt` flow map, which has no finite closed form, and (b)
  space-coarsening makes the coarse field non-Markovian in the observed variable —
  a second lag genuinely carries information about the unresolved sub-grid state
  (the same partial-observation problem the project targets). Success is therefore
  **forecast MSE only**, graded against three reference posts.

**Reference scoreboard** (5-step rollout MSE on the unforced coarse field,
discover-train split; measured in `journal/2026-07-21_edgar_burgers_setup.md`):

| reference                          | 5-step rollout MSE | meaning                                   |
|------------------------------------|--------------------|-------------------------------------------|
| do-nothing (persistence)           | 1.5e-4             | trivial floor; models must beat this      |
| naive one-step surrogate           | 6.8e-6             | discretised clean operator, one Euler step|
| best-achievable AR map (LS floor)  | 2.7e-6             | empirical lower bound for smooth f(x(t-1),x(t)) |

The seed programs bracket this range out of the box: `model1` (persistence) sits at
the floor (1.5e-4) and `model2` (neighbour-smoothing + velocity, closed-form fitted)
reaches ~2.5e-6, so the LLM starts with a real worst-to-best gradient to climb.

### Project contract (files EDGAR loads)

- `config.yaml` — evolution knobs, calibrated `param_penalty_weight=1e-7`
  (loss scale is ~1e-4 down to ~1e-6, params counted elementwise), and
  `project_params` (coarsening factors, block length).
- `data_loader/load_data.py` — `load_data()` builds the sample tensors
  `x: (n_samples=1, n_blocks, n_sensors, block_len)` by dealing time-blocks
  round-robin into four leak-free splits; `loss_fn()` is per-sample MSE. The
  unforced field is cached to `burgers_unforced_coarse.npz` (git-ignored, ~70 s
  to regenerate).
- `evaluate/evaluate.py` — the autoregressive rollout wrapper (`MAX_LENGTH=2`,
  `ROLLOUT_STEPS=5`); teacher-forced restarts, windows never cross a block
  boundary. This is the single point through which a model is called.
- `prompts.yaml` — **neutral** framing: "motion of a population of particles
  measured at a row of evenly-spaced, periodic sensors". No mention of the
  generating physics.
- `image_feedback/plot.py` — sensor x time heatmaps + per-sensor residual traces.
- `seed_programs/` — `model1` persistence, `model2` neighbour-smoothing+velocity,
  each with a closed-form `param_est`.

### Run the EDGAR control (Python >=3.13 + jax, on the EDGAR box)

```bash
# first call regenerates burgers_unforced_coarse.npz (~70 s), then caches it
edgar test projects/burgers/config.yaml          # smoke-test the scaffold
edgar run  projects/burgers/config.yaml          # full evolutionary run
```

`MAX_LENGTH` (lags m) and `ROLLOUT_STEPS` live in `evaluate/evaluate.py`; the
coarsening factors and block length are `project_params` in `config.yaml`.
