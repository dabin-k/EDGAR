# EDGAR Burgers control — project setup (plan Step 5)

Wired `projects/burgers/` into a full EDGAR task: discover a discrete
autoregressive map `x(t) = f(x(t-1), ..., x(t-m))` (m = MAX_LENGTH = 2) for the
population of sensors. This is the **temporal-lag control** — no spatial-stencil
structure imposed (that is Step 6).

## Key decisions

- **Unforced field.** Fit to the autonomous (`forcing_seed=None`) coarse field.
  The AR form has no exogenous input, so it is only well-posed on the autonomous
  field; the forced benchmark field would require leaking the forcing as a data
  channel, which breaks the neutral framing and EDGAR's generality.
- **No exact ground-truth map.** The data is a continuous PDE integrated at fine
  `dt` then coarsened x20 (time) and x4 (space). The exact one-step map is the
  time-`dt` flow map (no finite closed form), and space-coarsening makes the coarse
  field non-Markovian in the observed variable — a second lag carries information
  about the unresolved sub-grid state. This is deliberately the same
  partial-observation problem as `synthetic_data_v3` and as neural recordings.
  Scoring is **forecast MSE only**.

## Reference scoreboard (5-step rollout MSE, unforced coarse field, discover-train)

| reference                          | 5-step rollout MSE |
|------------------------------------|--------------------|
| do-nothing (persistence)           | 1.49e-4            |
| naive one-step surrogate (Euler)   | 6.79e-6            |
| best-achievable AR map (LS floor)  | 2.75e-6            |

Best-achievable floor is the LS fit of a rich 9-feature set over the 2-lag window,
rolled out 5 steps. Its fitted weights land on local averaging (u, u_left, u_right
each ~0.42) plus a momentum/lag term (u - u(t-1) ~ +0.69), with the physically
"correct" u*u_x and u_xx coefficients ~0 — concrete evidence that the best discrete
coarse-grid map is a smoothing + extrapolation kernel, not the discretised PDE.

## Calibration → penalty

Loss scale spans ~1e-4 (do-nothing) down to ~1e-6 (best AR). Parameters counted
elementwise. `param_penalty_weight = 1e-7`: a 5-param model costs 5e-7 — breaks
ties, stays well below the ~1e-5 gaps between good and mediocre models. (Same
reasoning as synthetic_data_v3, rescaled.)

## Seed programs (scaffold; final seeds TBD)

- `model1` persistence + `param_est1` (empty). Validated 5-step rollout = 1.490e-4
  (= do-nothing floor, as expected).
- `model2` neighbour-smoothing + velocity, params {blend, velocity} + `param_est2`
  (closed-form joint LS). Validated: default = 1.006e-4, fitted = 2.499e-6
  (estimator recovers blend~0.09, velocity~1.0 — near the best-achievable floor).

So the seeds bracket the whole range and give the LLM a real gradient.

## Files (all under projects/burgers/)

- `config.yaml` — evolution knobs, `param_penalty_weight=1e-7`, `project_params`.
- `data_loader/load_data.py` — appended EDGAR `load_data()` + `loss_fn()`;
  round-robin 4-way block split; unforced field cached to
  `burgers_unforced_coarse.npz` (git-ignored, ~70 s regen).
- `evaluate/evaluate.py` — AR rollout wrapper (MAX_LENGTH=2, ROLLOUT_STEPS=5),
  teacher-forced restarts via jax.lax.scan + dynamic_slice.
- `prompts.yaml` — neutral framing (particle motion at periodic sensors); no
  mention of the generating physics.
- `image_feedback/plot.py` — sensor x time heatmaps + per-sensor residual traces.
- `seed_programs/{model1,param_est1,model2,param_est2}.py`.

## Validation status

numpy paths (loader layout, seeds, estimators, plot) run and give the numbers
above. jax files (`evaluate.py`, the `to_jax` tail of `load_data`) are
py_compile-checked only — jax is NOT installed on this machine by request; run
`edgar test projects/burgers/config.yaml` on the EDGAR box to smoke-test the jax
path before a full run.
