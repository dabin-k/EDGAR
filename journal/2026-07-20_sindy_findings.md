# 2026-07-20 — Step 3 (SINDy / PDE-FIND) findings

Set up SINDy in isolation for the Burgers benchmark using PySINDy's PDE-FIND
(`PDELibrary` of {1, u, u², u_x, u u_x, u_xx, u² u_x} + `STLSQ`). The headline is a
clean, somewhat surprising result: **on the benchmark grid SINDy fails at zero noise**,
and the reason is coarse-graining, not noise.

## Two fairness/setup issues found before anything worked

1. **Forcing must be subtracted.** The benchmark field is *forced* Burgers,
   `u_t + (u²/2)_x = D u_xx + f(x,t)`. SINDy's library only has functions of the state
   `u`, so it cannot represent the explicit `f(x,t)` and the forcing corrupts the whole
   fit (naive fit gave `-0.25 u u_x`, no `u_xx`). This is also the fair setup: STENCIL-NET
   is *given* the forcing (it enters its loss as the `fc` terms), so SINDy should get it
   too. Fix: subtract the analytic `f(x,t)` from the estimated `u_t`, fit the homogeneous
   operator `u_t − f = N(u, u_x, u_xx, …)`. Our reconstructed forcing is bit-identical to
   the simulator's (same seeded draw).

2. **The config itself is correct** — verified by isolating each effect (see
   `bench/sindy/coarsening_diagnostic.py`).

## The validation ladder (all at zero noise unless noted)

| test | grid | recovered equation | verdict |
|---|---|---|---|
| unforced, well-resolved | Nx=256 | `0.021 u_xx − 0.995 u u_x` | ✓ |
| forced, forcing subtracted | Nx=256 | `0.020 u_xx − 0.997 u u_x` (identical at dt=0.002/0.01/0.04) | ✓ |
| coarsen ×2 | Nx=128 | `0.019 u_xx − 0.887 u u_x` + spurious | degrading |
| **coarsen ×4 (benchmark)** | **Nx=64** | `−0.261 u u_x`, **no u_xx**, +4 spurious | **✗** |
| coarsen ×8 | Nx=32 | `−0.235 u u_x`, no u_xx, spurious | ✗ |

Time-subsampling is *not* the culprit: on the fine grid the recovery is identical from
dt=0.002 up to dt=0.04. The failure is purely spatial coarse-graining.

## Interpretation

The coarse-grained field does not obey the fine-grid PDE — the true `u u_x` and `u_xx`
operators, evaluated with finite differences on 64 points where shocks span ~1–2 cells,
no longer close. This is a **closure / coarse-graining failure**, and it is *exactly the
gap STENCIL-NET is built to fill*: STENCIL-NET does not try to recover the fine-grid PDE;
it learns an effective discrete propagator directly on the coarse grid. So the benchmark
is set up to show the intended contrast — SINDy needs the fine, well-resolved field;
STENCIL-NET (and the EDGAR hybrid) target the coarse regime.

Noise then compounds it. On the coarse grid, σ=0→0.01 changes the recovered equation not
at all (coarse-graining dominates); σ≥0.05 visibly degrades `u u_x` further and inflates
the model from 4→6 active terms. Figure: `sindy_recovery.png`.

## What's committed

- `bench/sindy/runner.py` — PDE-FIND runner (forcing-subtracted; `--noise`, `--sweep`,
  `--weak` for the WeakPDELibrary/integral-SINDy stretch). Writes `results/*.json`.
- `bench/sindy/coarsening_diagnostic.py` — regenerates the fine field and runs the ladder.
- `sindy_recovery.png` — coefficient recovery vs grid coarseness and noise.

## UPDATE 2026-07-20 — see 2026-07-20_weak_sindy.md

- **WeakPDELibrary (integral SINDy) stretch is now DONE** (`bench/sindy/weak_vs_strong.py`,
  figure `weak_sindy_comparison.png`). Result: the integral form fixes the **noise**
  problem decisively (holds −1.0 u u_x to σ=0.1 where strong collapses by σ=0.01) but does
  **NOT** rescue coarse-graining (a closure failure, not a derivative-noise failure).
- **Correction to the coarse-graining claim below:** the catastrophic Nx=64 failure is
  specific to the *continuously forced* benchmark field, not coarse-graining in general —
  a *decaying* field coarsens gracefully to Nx=64 (breaks only at Nx=32). Mechanism: the
  residual ‖(u_t − f) − N(u)‖ climbs 0.17→0.71 over Nx 256→64 because persistent forcing
  sustains under-resolved shocks the coarse stencils can't represent. See the other note.

## Open / next

- Forecast-MSE (integrate the recovered PDE forward, compare to clean field) not yet
  added — needed for the Step-4 head-to-head vs STENCIL-NET on the shared metric.
- Weak SINDy on the *forced* benchmark field needs a weak-form forcing term (open item).
