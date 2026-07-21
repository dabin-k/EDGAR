# SINDy vs STENCIL-NET head-to-head (Step 4)

Script: `projects/burgers/bench/compare_sindy_stencilnet.py`
Figure: `results/sindy_vs_stencilnet.png` (two panels)

## Setup
Both methods scored on the **shared metric**: forecast MSE against the CLEAN
field (`load_data.forecast_mse`), swept over observation noise sigma. Coarse
benchmark grid (Nx=64, s_factor=4), forcing known to both methods.

- **STENCIL-NET**: learns a discrete propagator, free-runs it with RK3 + known
  forcing. Numbers come from its own runner (GPU box). Guaranteed finite output.
- **SINDy**: (both weak and strong variants) recovers a continuous PDE u_t = N(u; Ξ) + f. To score it on the shared forecast metric, we integrate the  
  recovered PDE forward with the same time stepper (SSP-RK3) and known forcing used for STENCIL-NET. Each library term is discretized by its type: advective / first-derivative terms c·u^p·u_x (where p is the power of u outside the derivative — p=1 gives the Burgers term u·u_x) are written in conservative form ∂ₓ[c/(p+1)·u^(p+1)] and integrated with a shock-capturing Lax-Friedrichs flux; second-derivative diffusion terms use central differences; zeroth-order reaction terms are evaluated pointwise. The Lax-Friedrichs flux is the stabilizing ingredient: its built-in numerical dissipation is what keeps even the true equation stable on the coarse, shock-forming grid, where a naive central-difference of the advective term diverges.
- **Oracle**: free-run of the true 0.02 u_xx - 1.0 u u_x with the same integrator
  = forecast-MSE floor (~0.13 on this grid). Shows how much error is the coarse
  integrator itself vs the recovered dynamics.

## Headline finding: coefficient accuracy != forecast stability
On the coarse forced grid, the two SINDy variants trade places depending on the
metric:

Panel A (coefficient recovery):
- **weak** nails the leading coefficients: ~0.023 u_xx, ~-0.99 u u_x, essentially
  flat from sigma=0 to sigma=0.3 (noise-robust, as designed).
- **strong** is badly closure-corrupted: u u_x coefficient only -0.26, plus
  spurious bias/u/u^2 terms; degrades further with noise.

Panel B (forecast MSE, shared metric):
- **strong** is numerically STABLE and sits right at the oracle floor (~0.13) up
  to sigma=0.1; only blows up at sigma=0.3.
- **weak** BLOWS UP at every noise level (red x, capped at 1e3): its otherwise-clean
  fit carries a spurious +0.04 u^2 reaction term (and u^2 u_xx), which is
  anti-diffusive / super-linear and diverges in finite time when integrated.

So the fit that is "better" by coefficient error is "worse" by forecast, and vice
versa. This is exactly the kind of instability the benchmark set out to probe:
SINDy optimises an equation-space residual, not forecast stability, so a
low-residual fit can still be a useless propagator. STENCIL-NET has no such failure
mode (it optimises the propagator directly).

## Caveats
- STENCIL-NET converged (30k-epoch) numbers are pending the user's GPU run; the
  comparison script skips un-converged (<30k) smoke results and marks STENCIL-NET
  "pending". Re-run the script after the GPU results land to fill panel B.
- SINDy STLSQ threshold defaults to 0.01 here (0.05 thresholds out the true
  0.02 u_xx term -- the threshold artifact documented in 2026-07-20_sindy_findings.md).
- Forecast MSE capped at 1e3 for blown-up integrations so the plot stays readable;
  `stable=False` in compare_results.json flags them.
