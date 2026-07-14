# Autoregressive Models for EDGAR

## Q: What is the data? What is the function?

Suppose that data has shape (n_cells, n_times). We want to discover equations that predict the activity of a cell at time t given data (n_cells, t-1).

For $X \in \mathbb{R}^{N_c \times N_t}$, find $f(X_{t-1}; \beta)$ $\to$ $\underline{x}_t$ where $\underline{x}_t$ $\in$ $\mathbb{R}^N_c$ and $X_{t-1}$ $\in$ $\mathbb{R}^{N_c \times (t-1)}$.

We can also allow the model to specify the max_length as a parameter, which determines how many time windows it the model gets access to. 

## Current problem 
The input to the model being discovered is static - i.e. we are looking for a model function $f$ which always takes the whole data matrix $X$ as input. 
This makes it very easy for the LLM to "cheat" and just spit out x(t) when given $X_{t-1}$. 

This is an engineering problem rather than a problem with the evolutionary algorithm. I think we can get around this problem by introducing a wrapper function g which controls the input to the function
and letting the LLM know that the function will be evluated inside the wrapper function.

> **Retrospective (2026-07-14): "engineering, not evolution" turned out to be wrong.** Why:
> - EDGAR evolves a `(model, parameter_estimator)` **pair**. `g` was applied to the model only: the model
>   gets a `(n_cells, MAX_LENGTH)` window, the estimator still gets the whole `(n_blocks, n_cells,
>   block_len)` block from `scoring._get_params`.
> - Seeing the full history, the estimator can **solve the one-step problem in closed form** (OLS) for any
>   model that is linear in its parameters.
> - That OLS point is a **stationary point of the 3-step rollout loss** (`‖grad‖/‖params‖ ≈ 2e-4` there vs
>   `9e-2` at defaults; Adam moves 0.026 vs 3.11). GD does nothing.
> - So the rollout objective — the whole reason for `g` — is **bypassed for exactly the programs whose
>   estimator is correct**. Their recorded loss is a one-step answer wearing a 3-step label.
> - Result: 16 programs with the **byte-identical equation** spanned 330× in loss, ranked purely by
>   estimator quality. Evolution selected on estimator-coding luck, not on the science.
>
> How you feed a model *is* part of the objective, and the objective is the algorithm. See
> `journal/2026-07-14.md`.

## How should we cross-validate?

We can cut up the data into N chunks. ~~Train and test sets should maintain their n_cells x n_times shape while having the other dataset masked to nan (~~ this is actually not a good idea. If we define the loss to be 0 when $x_t$ = nan, which we would, this would be equivalent to vmapping over the N chunks but with a downside that we waste time running the model on weird input data that contains a mixture of nan’s and floats).

By calculating n_times // N, we can establish a hardcoded MAX_LENGTH to ensure there is no leakage. 

In order to ensure that the LLM doesn’t cheat, we will evaluate the function under a wrapper function g. 
First pass. Writing this without vmap, you get :

```python
MAX_LENGTH_CEILING = X.shape[1] // n_chunks

def g(X, f, beta):
    """X: (n_cells, T). Returns preds (n_cells, T - max_length),
    where preds[:, i] predicts X[:, i + max_length]."""
    max_length = min(beta['max_length'], MAX_LENGTH_CEILING)
    preds = []
    # start at max_length-1 so the first window is full; stop at T-2 so a target exists
    for t in range(max_length - 1, X.shape[1] - 1):
        window = X[:, t - max_length + 1 : t + 1]  # ends at t INCLUSIVE
        preds.append(f(window, beta))              # predicts X[:, t+1]
    return jnp.stack(preds, axis=1)

def loss_fn(preds, X, max_length):
    targets = X[:, max_length:]  # the t+1 column of each window
    return jnp.mean((preds - targets)**2)
```

Two easy mistakes, both of which I made in the original draft of this doc:
- The window must **end at `t` inclusive** and be scored against `t+1`. Slicing `X[:, start:t+1]` and then
  comparing `preds` to `X` unshifted hands the model its own answer — the exact cheat `g` exists to close.
- The loop must **start at `max_length - 1`**, not 0, or the first windows are short.

`complexity_penalty` is **not** a project function. `scoring` already applies
`param_penalty_weight * n_params`, and `n_params` counts array *elements*, so an N×N coupling correctly
costs 1024. (The 0.01 default is catastrophic here and must be retuned per project — see
`journal/2026-07-13.md`.)

I realised along the way that trying to parameterise the max_length was not very well thought through. The model changes discontinuous when the max_length changes. The options are either to 

- Do a discrete search  : for max_length in candidate_lengths: …
- If the evaluation at each t was somehow linear, we could have used a memory-decay parameter by using a differentiable weight according to lag: $w_k = exp(-k / \tau)$ where $\tau$ is learnt. But I think we can’t guarantee this - it’s possible that f simply takes the mean of all input data.
- There are some other alternatives, such as evaluating at all/sampled max_length and interpolating if there is a pattern.

I’m really not sure which method is best, so I’m going to fix MAX_LENGTH_CEILING = 2 and leave this for v2, and focus on the rest for now. 

Going back to vmapping, I think it’ll look something like :

**Two changes from the sketch above, both load-bearing.** (a) The loss is an **H-step rollout**, not
one-step-ahead. Eliminating the hidden adaptation current turns process noise into MA(1), so the true
model is *not* the one-step MSE optimum — a linear surrogate beats it at H=1. H>1 is what makes the loss
select the correct dynamics at all. (b) **No NaN padding.** Starts run from `max_length - 1`, so every
window is full and no mask is needed; leak-freeness comes from windows never crossing a *block*
boundary, not from a ceiling on `max_length`.

This is what is actually in `projects/synthetic_data_v2/evaluate/evaluate.py`:

```python
MAX_LENGTH = 2      # lags visible to the model
ROLLOUT_STEPS = 3   # H: steps rolled on the model's own predictions

def evaluate(model_fn, data, params):
    """data['x']: (n_samples, n_blocks, n_cells, block_len).
    Returns (preds, targets), both (n_samples, n_blocks, n_starts, ROLLOUT_STEPS, n_cells)."""
    def per_sample(sample, sample_params):
        return jax.vmap(partial(_rollout_block, model_fn, sample_params))(sample["x"])

    return jax.vmap(per_sample)(data, params)

def _rollout_block(model_fn, params, block):
    """Teacher-forced restarts within one block. block: (n_cells, block_len)."""
    n_cells, block_len = block.shape
    # start s needs history [s-MAX_LENGTH+1 .. s] and targets [s+1 .. s+ROLLOUT_STEPS]
    starts = jnp.arange(MAX_LENGTH - 1, block_len - ROLLOUT_STEPS)

    def rollout_from(s):
        window = jax.lax.dynamic_slice(
            block, (0, s - MAX_LENGTH + 1), (n_cells, MAX_LENGTH)
        )

        def step(w, _):
            pred = model_fn({"x": w}, params)          # (n_cells,)
            # slide the window: drop the oldest lag, append the model's OWN prediction
            return jnp.concatenate([w[:, 1:], pred[:, None]], axis=1), pred

        _, preds = jax.lax.scan(step, window, None, length=ROLLOUT_STEPS)
        targets = jax.lax.dynamic_slice(block, (0, s + 1), (n_cells, ROLLOUT_STEPS)).T
        return preds, targets  # both (ROLLOUT_STEPS, n_cells)

    return jax.vmap(rollout_from)(starts)
```

`evaluate` returns the **aligned targets**, so `loss_fn` does no indexing at all — the alignment is
written exactly once, here. It stays in `load_data.py` and is a pure comparison:

```python
def loss_fn(preds, targets):
    return jnp.mean((preds - targets) ** 2, axis=(1, 2, 3, 4))  # reduce all but the sample axis
```

## First step
Let's brainstorm by coming up with a reasonable synthetic dataset with a ground truth, autoregressive model. 

- [ ] Come up with a good candidate ground truth model that requires an autoregressive behaviour 
- [ ] Generate synthetic data
- [ ] Come up with appropriate seed models
- [ ] Jot down how we'll have to change the codebase to implement a model like this. 

## Next steps

We want to use this wrapper function $g$ whenever we evaluate the model. We evaluate the model when

- Calculating the loss
- Estimating the initial parameters
- Plotting imaging diagnostics

The question is, how do we incorporate g into these three processes? 

We can add an extra function under the project.  Current setup 

```markdown
projects/ 
	orientatin_tuning/ 
		data_loader/
			load_data.py
				load_data
				loss_fn
```

But we can change this to :

```markdown
projects/
	orientatin_tuning/
		data_loader/
			load_data.py
				load_data
				loss_fn        # STAYS HERE — evaluate returns aligned targets, so
				               # loss_fn does no indexing and moving it buys nothing
		evaluate/
			evaluate.py
				evaluate
```

`evaluate/evaluate.py` is **optional and per-project**. Its contract is
`evaluate(model_fn, data, params) -> (preds, targets)` — it returns the targets as well as the
predictions, so that the window/target alignment is written in exactly one place. Projects without one
fall back to `default_evaluate` in `edgar/scoring/scoring.py`, which preserves the old behaviour:

```python
# edgar/scoring/scoring.py — used when a project has no evaluate/evaluate.py
def default_evaluate(model_fn, data, params):
    """Non-autoregressive: vmap the model over samples, hand the data dict back as the target,
    so every existing project's loss_fn(model_output, data) is unchanged."""
    return jax.vmap(model_fn, in_axes=(0, 0))(data, params), data
```

`edgar/io/task_spec.py` loads it into a new `evaluate_fn` field (`None` if absent) and `run.py` passes
it to all three `score()` calls.

TODOS
- [x] Check if the model is evaluated at any other point
- [x] Make evaluate the single point of control through which any *model* is evaluated — the four sites
      that used to call `jax.vmap(model_fn, ...)` directly (`_optimize`, `_eval_loss`,
      `_eval_sample_losses`, `_eval_fingerprint`) now all go through `evaluate_fn`
- [ ] **The `parameter_estimator` was never wrapped.** Of the three call sites listed above, loss and
      plotting go through `evaluate`; `scoring._get_params` still calls the estimator on the raw
      `(n_blocks, n_cells, block_len)` sample. This is the gap behind the retrospective at the top of
      the doc — see `journal/2026-07-14.md`.

