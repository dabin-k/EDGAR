# 2026-07-20 — Integral (weak-form) SINDy: what it fixes, what it doesn't

Ran the Step-3 stretch: weak/integral SINDy (`WeakPDELibrary`) vs strong/derivative
SINDy (`PDELibrary`), the noise remedy the Champion paper cites. Tested on two axes
independently. Script: `bench/sindy/weak_vs_strong.py`; figure: `weak_sindy_comparison.png`.

Tested on **unforced** Burgers. Reason: `WeakPDELibrary` builds its own weak-form `u_t`
(integrated against test functions) and ignores the `x_dot` argument, so the
forcing-subtraction trick used for strong SINDy on the forced benchmark field does not
transfer. Unforced fields let the weak form apply cleanly and isolate the two questions.
`runner.py` now warns if `weak=True` is called with a forcing (it would be corrupted).

## Result 1 — NOISE (well-resolved grid, Nx=256): weak wins decisively

target: `u_t = 0.02 u_xx − 1.0 u u_x`; table shows recovered `u u_x` coefficient.

| σ | 0.0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|
| strong | −1.00 | −0.14 | −0.14 | −0.13 | −0.10 | −0.04 | −0.03 |
| weak   | −1.00 | −1.00 | −1.00 | −1.01 | −1.01 | −0.99 | −0.79 |

Strong SINDy collapses by σ=0.01 (also loses `u_xx`). Weak holds `~0.02 u_xx − 1.0 u u_x`
to σ=0.1, softening only at σ=0.2. This is the derivative-noise-amplification story:
strong differentiates noisy data; weak integrates it against smooth test functions.

## Result 2 — COARSE-GRAINING (zero noise): weak does NOT rescue it

multiscale decaying field, recovered `u u_x` coefficient:

| Nx | 256 | 128 | 64 | 32 |
|---|---|---|---|---|
| strong | −1.01 | −1.01 | −0.94 | −0.81 |
| weak   | −1.01 | −1.02 | −1.06 | −1.23 |

Strong and weak track each other. Coarse-graining is a **closure** failure (the coarse
operators don't reproduce the fine dynamics), not a derivative-noise failure, so
integrating more carefully buys nothing — it can't recover information the grid discarded.

## Correction to the original Step-3 note (2026-07-20_sindy_findings.md)

That note attributed the catastrophic Nx=64 benchmark failure to coarse-graining alone.
More careful testing shows the claim needs a qualifier:

- Coarsening a **decaying/unforced** field is *forgiving* — recovery stays near −1.0 down
  to Nx=64 (both strong and weak); both only break at Nx=32.
- The catastrophic Nx=64 failure (`−0.26 u u_x`, no `u_xx`) is specific to the
  **continuously forced** benchmark field. Mechanism confirmed: relative residual
  `‖(u_t − f) − N(u)‖` on the coarse grid climbs 0.17 (Nx=256) → 0.45 (128) → 0.71 (64)
  → 0.87 (32). The forcing modes are low (`l_k ∈ {2,3,4}`, ~16 pts/wavelength at Nx=64),
  so it is **not** forcing under-resolution — it is that persistent forcing sustains
  under-resolved shock structure everywhere, and the coarse `u u_x`/`u_xx` finite-difference
  stencils fail on sustained shocks.

So the Step-3 direction stands (a resolution/closure effect worsened by noise, not a pure
noise effect), but the precise statement is: *SINDy fails on the coarse **forced**
benchmark field; coarse-graining a decaying field alone is far more forgiving.*

## Consequence for the benchmark (Steps 4+)

- Carry **weak SINDy as the noise-robust SINDy baseline** in the Step-4 head-to-head — it
  is the fair strong baseline against STENCIL-NET in the low-noise, well-resolved regime.
- The benchmark cleanly isolates the one thing SINDy fundamentally can't do that
  STENCIL-NET/EDGAR can: operate directly on the coarse grid. That is the lesson EDGAR
  borrows (learn an effective coarse propagator rather than recover the fine-grid PDE).
- **Open item:** to run weak SINDy on the *forced* benchmark field, need a weak-form
  forcing term (add `f(x,t)` as a known control column to the weak library, or integrate
  it against the same test functions). Not needed for the unforced isolation; needed if we
  want weak SINDy's forecast on the actual benchmark data.
