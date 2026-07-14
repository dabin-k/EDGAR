# 2026-07-14

From a conversation with edgar-analyzer on 'program_databases/synthetic_data_v2/synthetic-seed-0/2026-07-14/10-59-30/'

I think the way we fit the parameters at the moment might bad? But there are a few things that I don't understand. 
- It sounds like the parameter_estimator, when configured correctly, pushes us to optimise for the wrong thing - we minimise the loss for the t+1 step only. But claude also said that gradient descent allows us to optimise for the full t+3 rollout. Why is it the case that gradient descent does the correct thing when the parameter_estimator is bad, but having a broken parameter estimator allows us to optimise using the correct loss function? 

## Initial todo 
- [x] Confirm eactly how the parameters are fit. What is the data? What is the loss function? How is the model evaluated when estimating the parameters? Does it really only care about t+1 step, or does it incorporate the losses for all 3 step rollout? 
- [ ] Think about whether the current implementation of parameter_estimator makes sense. Think about how we had to update the way the model is evaluated by writing a wrapper function (2026-07-13.md). Do we need to do something similar for parameter_estimator? Or, another way to think about this is, shouldn't the parameter_estimator receive the same data as the model? 

## Findings after conversation with claude 

1. Mechanism 
_optimize (edgar/scoring/scoring.py:114) builds total_loss from evaluate_fn + loss_fn — that's the 3-step rollout. And both losses you see in the database are the same objective: initial_loss is the rollout loss evaluated at params_init (line 293), final_loss the rollout loss at the optimized params (line 304), both on the test blocks. So gradient descent is on the correct 3-step objective always, and init vs final is apples-to-apples.

- So the original worry ("a broken estimator lets us optimise the right loss") is not a coherent mechanism. GD is *always* on the 3-step loss. The estimator only chooses the **starting point**.
- The only channel by which a starting point can matter is **non-convexity** — and this loss is very non-convex (model composed 3x => prediction is cubic in the weights, squared loss is degree 6).

2. The concern, restated
- Estimator writes the exact one-step OLS/MLE solution for a linear AR.
- Observed in the run: correct-estimator programs have `init == final` — **GD moves them nowhere**.
- Hypothesis: the one-step OLS point is also a *local minimum* of the 3-step loss, so GD is trapped there and contributes nothing.
- Competing hypothesis: GD is just too weak (lr / max_iter) to escape. Needed to distinguish these.

3. Test (scratchpad `grad_at_ols.py`; program idx 8, model held **fixed**, only the init varied — so no model confound)
- Measured `||grad||/||params||` of the 3-step train loss at each start, then ran the real Adam optimizer (lr 3e-3, 1000 steps) from each.

| start | ‖grad‖/‖params‖ | train after | **test after** | param move |
|---|---|---|---|---|
| A. estimator OLS init | **2.0e-04** | 0.000586 | 0.000689 → **0.000691** | **0.026** |
| B. DEFAULT_PARAMS | 9.1e-02 | 0.00119 | 0.0618 → **0.000636** | 3.11 |
| C. OLS + N(0, 0.05) | 6.8e-01 | 0.000632 | → 0.000714 | 0.25 |

(reproduces the database numbers for idx 8 exactly; test losses include the 6e-5 param penalty)

4. Takeaways
- **CONFIRMED: the OLS point is a genuine stationary point of the 3-step loss.** Relative gradient ~450x smaller than at defaults; Adam travels 0.026 vs 3.11. No optimizer bug, and "lr too small" is dead — Adam is scale-free.
- **=> The GD stage is vestigial for correct-estimator programs.** Their recorded `final` loss is just the estimator's one-step answer wearing a 3-step label. The multi-step objective — the whole reason for the rollout (see loader docstring) — is **silently bypassed for exactly the programs that get the estimator right**. GD only does real work when the estimator fails.
- **This corrupts selection.** In the run, 16 programs with the *byte-identical* equation spanned 330x in loss, driven entirely by estimator quality. Evolution is selecting on estimator-coding luck, not on the science. The equation itself was already solved at gen 0 (linear ring-AR(2), rediscovered independently on 7 islands).
- Softer result (**needs replication**): same model, starting from DEFAULT_PARAMS reaches a *better test* loss (0.000636 vs 0.000691) despite a *2x worse train* loss. So idx 124's win was not purely its equation — the init really does buy something. But n=1 model, one split, 8 samples, ~9% gap: could be noise.
- Landscape is **pathologically sharp**: perturbing the OLS params by 0.05 blows the train loss up 1000x (0.000588 → 0.690). Marginal stability (coeffs sum to ~1.0) means small weight changes push eigenvalues off the unit circle and error compounds over the 3 steps. Good solutions sit on a knife edge.
- Run B was **still descending** when the 1000 steps ran out (moved 3.11) — `max_iter` is plausibly binding.

## Done
- **Rewrote the `parameter_estimator` prompt** for synthetic_data_v2 (`projects/synthetic_data_v2/prompts.yaml`, new `parameter_estimator:` section). Root cause of the axis bugs: the project overrode only `model:`, so the estimator fell back to the default prompt, which claims `data` is "the single-sample data dict used by `model`". False here — the model sees a `(n_cells, max_length)` window from `evaluate/evaluate.py`, the estimator sees `(n_blocks, n_cells, block_len)` from `scoring._get_params`. The prompt also shows the LLM the new model's code, so copying the model's indexing was the *natural* mistake => ~80% of estimators got the axes wrong.
  - New prompt states the 3-D contract axis-by-axis, warns against rolling axis 0, and gives a worked joint-OLS design-matrix snippet (univariate `sum(y*f)/sum(f*f)` fits were the other big failure mode).
  - Also fixed the misleading generic line in `projects/prompt_defaults.yaml` — now says `data` is one sample's *training* data and may not match the model's shape.
  - Verified: `scripts/print_prompts.py synthetic_data_v2` renders correctly; `uv run pytest` 190 passed / 8 skipped.

## Open / next (by claude)
- [ ] Bigger question (todo #2 above): should the estimator be evolvable at all here, given it can *solve* the one-step problem and thereby pin GD? Or should it receive the same data as the model?
- [ ] **Replicate the crossed-init test** across the other correct-estimator programs (61, 118, 119, 108) + a second data seed. Does "DEFAULT_PARAMS beats OLS on test" hold, or is it noise?
- [ ] Check whether more Adam steps from the default init keeps improving (B was under-converged).
- [ ] If it replicates: contained fix in `_worker` — optimize from **both** `params_init` and `default_params`, keep the better. Costs one extra optimize call per program, removes the trap.

## Dabin's interpretation
This is probalby the most interesting point : "The GD stage is vestigial for correct-estimator programs.** Their recorded `final` loss is just the estimator's one-step answer wearing a 3-step label. The multi-step objective — the whole reason for the rollout (see loader docstring) — is **silently bypassed for exactly the programs that get the estimator right**. GD only does real work when the estimator fails." 
I wonder if these two hypotheses are true : 
- 1. Given the way we've set up the data input for parameter estimator, the parameter estimator simply tries to optimise the 1-step answer, in spite of the loss function including the residuals of all 3steps. 
- 2. Optimising for 1step answer always / is likely to land you in a local minimum from which gradient descent is difficult. 

If 2 is true, either we have to implement the parameter estimator in a different way that better mimicks the way the model is evaluated on data OR we have to do away with the parameter estimator set up altogether. 
What might be mathematical basis for either 1 or 2 being true? 

## Update: the correct-model thought experiment (discussion with claude)

Asked: suppose the LLM *had* discovered the true model — what would estimator + GD do then?

- Eliminating the hidden `a` gives the truth in observable form:
  `x(t+1) = (A+B)x(t) − (AB+g·dt²)x(t−1) + dt·Kφ(x(t)) − B·dt·Kφ(x(t−1)) + η(t)`, with `η(t) = ε(t) − B·ε(t−1)` (MA(1) noise). MAX_LENGTH=2 is exactly sufficient; the discovered linear ring-AR(2) is this equation with φ linearized away.
- For the correct model the pathology **disappears**: it is nonlinear in its params (σ inside the Gaussian kernel; B multiplies the shared kernel term) → no closed-form fit, and one-step LS is *biased* under the MA(1) noise (regressor x(t) contains ε(t−1), which is also in η(t)) → the estimator init is not a stationary point of the rollout loss → GD does real work. Estimator finds the basin (knife-edge landscape), GD corrects the one-step bias on the multi-step objective. Neither stage vestigial.
- So the architecture is right **at the endgame**; the trap is confined to the linear-in-params region of model space, where a closed form exists and coincides with a rollout stationary point.
- **Real risk, sharpened:** when the correct model first appears, its first-draft estimator will be rough, so its score rests entirely on GD — which is under-converged at max_iter=1000 (run B). Incumbents are linear models sitting at their analytic optimum. The right answer could be pruned on arrival.
- **Criterion for any fix:** does the true model with a mediocre estimator survive scoring against a polished linear incumbent? Keep the estimator, keep the rollout; strengthen GD (convergence-based stopping, not fixed max_iter).

New diagnostic for the todo list:
- [ ] Take the true model + degraded ("first-draft quality") params; check whether 1000-step GD recovers enough to beat idx 8's polished linear score. Directly measures whether the pipeline would recognize the right answer if it appeared.