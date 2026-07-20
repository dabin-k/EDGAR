# Benchmark plan: STENCIL-NET vs SINDy vs EDGAR on noisy Burgers

**Goal.** Build a reproducible benchmark that quantifies how well EDGAR discovers
autoregressive/PDE models for systems in near-equilibrium with low-but-nonzero noise,
using STENCIL-NET and SINDy as the two reference points. The scientific hypothesis under
test: *SINDy degrades sharply as observation noise rises (it differentiates the noisy
signal to build its library), while STENCIL-NET stays bounded (it fits state-to-state
through a time integrator), and EDGAR — proposing symbolic equations but scoring them with
a propagator-style rollout loss — inherits STENCIL-NET's noise tolerance while keeping
SINDy's interpretable output.*

**Non-goals.** Reproducing either paper perfectly. STENCIL-NET's coarse-grid
generalization study and SINDy's coordinate-discovery (autoencoder) are out of scope for
v1. We copy reference code where it is cheap and faithful, and we spend effort only where
it informs EDGAR.

---

## Key decisions (locked)

1. **SINDy = PDE-FIND via PySINDy, not the attached autoencoder paper.** A Burgers field
   `u(x,t)` is already in the coordinates where the dynamics are sparse, so the Champion
   autoencoder's coordinate-discovery adds nothing to the *noise* question — and its TF1
   codebase is costly to run in 2026. The variant that actually competes with STENCIL-NET
   on Burgers/KdV, and that breaks under noise for the reason we care about (finite-
   differenced `u_x, u_xx` from a noisy signal), is PDE-FIND (Rudy et al. 2017). PySINDy is
   the maintained Brunton/Kutz descendant, pip-installable, and ships PDE-FIND *and* the
   weak/integral formulation. **This narrows the project's earlier "SINDy = autoencoder"
   ruling for this benchmark only.** The autoencoder is a documented v2 milestone (§7).

2. **The hybrid is mostly already built.** EDGAR core already implements "propose an
   equation, score it with a rollout loss":
   - `projects/synthetic_data_v3/evaluate/evaluate.py` rolls the model forward
     `ROLLOUT_STEPS` on its own predictions via `jax.lax.scan` — that is the discrete
     propagator `T` applied H times; the equation `f` is `model_fn`.
   - `edgar/scoring/scoring.py::_optimize` runs Adam through `evaluate_fn ∘ loss_fn` with
     `jax.value_and_grad` — i.e. it already backprops through the integrator, so the
     equation's parameters are fit by gradient descent through the propagator.
   - Cross-validation already exists: disjoint discover/validate samples + alternating
     train/test blocks with windows that never cross a block boundary (leak-free).

   Consequence: the benchmark lives almost entirely in **project space**
   (`data_loader/`, `evaluate/`, `prompts.yaml`, `config.yaml`), mirroring how `v2` and
   `v3` already differ. **EDGAR core needs one small, general change** (Step 6): make the
   loss/penalty pluggable so v2's L1-sparsity term drops in without touching scoring. The
   temporal-lag control (Step 5) needs *no* core change at all — it is the current
   `v2`/`v3` machinery pointed at Burgers data.

3. **Noise is the load-bearing axis.** Every method is compared on one shared metric:
   **forecast MSE against the clean ground-truth field on held-out data, swept over
   observation-noise level σ.** STENCIL-NET emits a forecast directly; SINDy emits an
   equation we integrate to a forecast; EDGAR already computes forecast error. Coarse-grid
   generalization is not measured in v1.

4. **Design v1 so "evolve the dictionary" (the real next step) is a drop-in.** Write the
   EDGAR model as `f(x) = Σ_j params[j]·θ_j(x)`. Then `params` *is* the SINDy coefficient
   vector Ξ and `{θ_j}` is the library. v1: the LLM evolves the whole `f`. v2: the LLM
   evolves the library `{θ_j}`, Ξ is fit by the same GD + an L1 term. Both run through the
   identical propagator. Keep the loss modular now so only "what the LLM writes" and
   "whether the loss has an L1 term" change later.

---

## Shared artifacts (produced once, consumed by all three methods)

- **Data generator** — forced viscous Burgers, `u_t + u·u_x = ν·u_xx + forcing`, matching
  STENCIL-NET's setup (ν, domain, spectral/WENO reference solver, RK time stepping).
  Emits a clean field `U_clean[t, x]` and a noise knob σ producing
  `U_obs = U_clean + σ·N(0, ·)`. Saved as `burgers_clean.npz` + a noisy variant per σ.
- **Noise sweep** — σ ∈ {0, 0.01, 0.05, 0.1} (0.1 matches their KdV noise level).
- **Shared metric + harness** — `forecast_mse(U_pred, U_clean_heldout)`; a single
  `evaluate_forecast()` used identically by all three methods so numbers are comparable.

---

## Steps

### Step 1 — Burgers data generator + noise + visualization
Port STENCIL-NET's Burgers generator (their repo, PyTorch/NumPy) as faithfully as cheap.
Deliverables: `projects/burgers/data_loader/load_data.py` (clean + noisy fields, train/test
split leak-free by block), `burgers_clean.npz`, and a figure
`burgers_data.png` (space-time heatmap of clean vs σ=0.1 noisy, plus a few time-slice
line plots). Sanity check: shock forms and viscous width matches ν.
**Runs locally (CPU).**

### Step 2 — STENCIL-NET in isolation
Copy STENCIL-NET's PyTorch model verbatim (mlpconv stencil operator + RK integrator +
their loss with the noise-as-latent term for the 1D noisy case). Train on clean Burgers,
then on noisy Burgers. Deliverables: `bench/stencilnet/` (copied code + thin runner),
`stencilnet_forecast.png` (predicted vs clean field, forecast MSE vs σ), and the trained
model's forecast MSE table. Their net is ~2–3k params; **CPU is sufficient** at this grid
size (GPU in the paper was for speed, not necessity). Verify: clean-data forecast MSE
reproduces their order of magnitude.

### Step 3 — SINDy (PDE-FIND) in isolation
Use PySINDy: `PDELibrary` (spatial derivatives `{1, u, u_x, u_xx, u·u_x, …}`) + `STLSQ`
sparse regression. Fit on clean Burgers → confirm it recovers `u_t = -u·u_x + ν·u_xx` with
correct coefficients. Then fit on noisy Burgers at each σ. Deliverables:
`bench/sindy/` (runner + the recovered-equation printout per σ), `sindy_recovery.png`
(recovered coefficients vs σ — the picture of the equation falling apart as noise rises),
and forecast MSE (integrate the recovered PDE forward, compare to clean field).
**Stretch:** add `WeakPDELibrary` (integral formulation) as a noise-robust SINDy variant —
this is the exact "integral SINDy" remedy the Champion paper cites, and previews the
lesson EDGAR borrows. **Runs locally (CPU).**

### Step 4 — Head-to-head: STENCIL-NET vs SINDy over the noise sweep
One figure, one table, the core scientific result. `benchmark_noise_sweep.png`: forecast
MSE vs σ for both methods on shared axes (expected: SINDy's curve explodes, STENCIL-NET's
stays bounded; weak-SINDy sits between). `benchmark_results.csv`: method × σ × forecast
MSE, plus SINDy's recovered-coefficient error. Written up in
`benchmark_findings.md`.

### Step 5 — EDGAR as-is, temporal-lag control (no spatial stencil)
Run EDGAR in its **current** form as a control, *before* adding the spatial-stencil
inductive bias, so we can attribute any later gain to the stencil rather than to "EDGAR +
rollout" in general. The model form is purely temporal autoregression at a fixed spatial
location:

    u(x, t) = f( u(x, t-1), ..., u(x, t-m) )        # each grid point = one "cell"

This is exactly the `synthetic_data_v2/v3` setup pointed at Burgers data: every spatial
point `x` is treated as an independent series with its own temporal-lag window, the model
never sees its spatial neighbors, and parameters are fit per point (or shared — a config
knob). Deliverable: `projects/burgers_temporal/` (data_loader reshaping the field to
per-point series + the existing temporal-lag `evaluate` reused unchanged), and the same
forecast-MSE-vs-σ numbers on the shared metric.

**Expected outcome and why it is the right control.** Burgers dynamics are inherently
spatial (`u·u_x`, `u_xx`): the evolution at `x` is governed by its neighbors, not by its
own past alone. A fixed-Eulerian-point temporal model can only see the wave passing
*through* `x`, so it should track advection weakly and miss diffusion coupling — i.e. do
noticeably worse than STENCIL-NET even on clean data. If it does *not* do worse, that is
itself an important finding (the temporal history at a point secretly encodes enough
spatial structure), and it would change the case for Step 6. Either way it sets the
baseline the stencil version must beat. **Harness validated + short dry run here; full
sweep launched by you** (same split as Step 6 below).

### Step 6 — Wire the Burgers project into EDGAR (the hybrid, spatial stencil)
Create `projects/burgers/` as a sibling of `synthetic_data_v2/3`:
- `data_loader/load_data.py` — reuse Step 1; layout the field as **one global sample**
  (all grid points share one equation, params fit once), window = spatial stencil.
- `evaluate/evaluate.py` — spatial-stencil rollout propagator: hand the model a stencil of
  neighbors, roll forward H steps with `lax.scan` (adapt `v3`'s temporal rollout to
  space+time). This is the STENCIL-NET fitting strategy with a symbolic `f`.
- `seed_programs/` — model1 = linear diffusion `ν·u_xx`; model2 = `-u·u_x + ν·u_xx`
  written in the `Σ params[j]·θ_j` form (so §Decision-4's v2 is a drop-in); matching
  `param_est`.
- `prompts.yaml` — adapt the AR prompts to the PDE/stencil framing.
- `config.yaml` — small run for harness validation.

**EDGAR core change (single, general, minimal):** make the complexity penalty / loss
composition pluggable via config (e.g. an optional L1-on-params term and configurable
penalty), so v2's dictionary-sparsity need not touch `scoring.py`. Nothing Burgers-specific
enters core. Deliverable: `evaluate`/scoring harness validated end-to-end on the seed
programs + a short dry run (1–2 generations) confirming losses are finite and the rollout
is differentiable. **The full 12-gen × 8-island sweep is launched by you on normal EDGAR
infra (LLM API + compute); I build and validate the harness, not run the full evolution.**

### Step 7 — Full comparison (all methods + the EDGAR control)
Add both EDGAR variants — the temporal-lag control (Step 5) and the spatial-stencil hybrid
(Step 6) — to the Step-4 figure/table → `full_comparison.png` + updated
`benchmark_results.csv`. Four curves of forecast MSE vs σ: SINDy, STENCIL-NET, EDGAR-temporal,
EDGAR-stencil. Report: (a) does EDGAR-stencil match STENCIL-NET's noise robustness while
returning an interpretable equation like SINDy does on clean data? (b) how much of that is
the spatial stencil, read off as the gap between EDGAR-temporal and EDGAR-stencil?
Write-up in `benchmark_findings.md`.

---

## v2 milestones (explicitly deferred, documented so v1 doesn't foreclose them)
- **Evolve the dictionary, not the equation** (§Decision-4): LLM edits `{θ_j}`, Ξ fit by
  GD + L1. Enabled by the pluggable-loss change in Step 5.
- **SINDy autoencoder** (the attached Champion paper): coordinate discovery for cases where
  the observed variables are *not* already the right coordinates (the neural-recording
  case — partial, high-dim observation of a latent system). Out of scope for Burgers.

## Ownership / compute
- Steps 1–4: fully built and run here (CPU).
- Steps 5–6 (both EDGAR variants): project built + harness validated + short dry run here;
  full evolutionary sweeps launched by you on normal EDGAR infra (LLM API + compute).
- Step 7: assembled here from the Step-5/6 outputs you provide (or from the dry runs if
  that is all that is available at the time).

## Reference code sources
- STENCIL-NET: github.com/mosaic-group/STENCIL-NET (PyTorch — copy verbatim).
- SINDy/PDE-FIND: PySINDy (`pip install pysindy`) — maintained Brunton/Kutz library.
- Attached papers: Champion et al. PNAS 116(45):22445 (2019); Maddu et al. Sci. Rep.
  13:12787 (2023).
