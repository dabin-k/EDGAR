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

## How should we cross-validate?

We can cut up the data into N chunks. ~~Train and test sets should maintain their n_cells x n_times shape while having the other dataset masked to nan (~~ this is actually not a good idea. If we define the loss to be 0 when $x_t$ = nan, which we would, this would be equivalent to vmapping over the N chunks but with a downside that we waste time running the model on weird input data that contains a mixture of nan’s and floats).

By calculating n_times // N, we can establish a hardcoded MAX_LENGTH to ensure there is no leakage. 

In order to ensure that the LLM doesn’t cheat, we will evaluate the function under a wrapper function g. 
First pass. Writing this without vmap, you get :

```python
MAX_LENGTH_CEILING = X.shape[1] // n_chunks 

def g(X, f, beta):
	preds = []
	max_length = min(beta['max_length'], MAX_LENGTH_CEILING)
	for t in range(X.shape[1]):
		start_idx = max(0, t-max_length+1)
		end_idx = t+1		
		x = X[:, start_idx:end_idx]
		pred = f(x, beta)
		preds.append(pred)
	return preds # length T-1 
	
def loss_fn(preds, data):
	# Ignore t=0 because we don't predict the first 
	return jnp.mean((preds - data)**2, axis=-1)
	
def complexity_penalty(beta, alpha, gamma):
	return alpha * (len(beta) -1) + gamma * log(beta['max_length'])
```

I realised along the way that trying to parameterise the max_length was not very well thought through. The model changes discontinuous when the max_length changes. The options are either to 

- Do a discrete search  : for max_length in candidate_lengths: …
- If the evaluation at each t was somehow linear, we could have used a memory-decay parameter by using a differentiable weight according to lag: $w_k = exp(-k / \tau)$ where $\tau$ is learnt. But I think we can’t guarantee this - it’s possible that f simply takes the mean of all input data.
- There are some other alternatives, such as evaluating at all/sampled max_length and interpolating if there is a pattern.

I’m really not sure which method is best, so I’m going to fix MAX_LENGTH_CEILING = 2 and leave this for v2, and focus on the rest for now. 

Going back to vmapping, I think it’ll look something like : 

```python
max_length = X.shape[1] // n_chunks 

def make_windows(X, max_length):
	""" X : (n_cells, T)
	Returns : 
		windows: (T, n_cells, max_length)
	"""
	n_cells, T = X.shape
	X_pad = jnp.pad(X, ((0, 0), max_length -1, 0)), constant=jnp.nan)
	
	starts = jnp.arange(T -1) # nothing to predict on the final timestamp 

	def get_window(start):
		return jax.lax.dynamic_slice(X_pad, (0, start), (n_cells, max_length))
	
	return jax.vmap(get_window)(startS)

def g(X, f, beta):
	windows = make_windows(X, max_length)
	preds = jax.vmap(f, in_axes=(0, None))(windows, beta) # shape (T, n_cells)
	return preds.T
	
def loss_fn(preds, X):
	mask = ~jnp.isnan(X[:, 1:])
	
	clean_X = jnp.where(mas, X[:, 1:], 0.0)
	safe_model_output = jnp.where(mask, preds, 0.0) 
	
	total_error = jnp.sum((clean_X - safe_model_output)**2)
	valid_count = jnp.sum(mask)
	return total_error / valid_count
	
def complexity_penalty(beta, alpha, gamma):
	return alpha * (len(beta) -1)
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
				loss_fn (## move this down to evaluate in a separate feature)
		evaluate/
			evaluate.py
				evaluate
```

evaluate.py will contain the function evaluate : 

```python
# If model is not autoregressive
def evaluate_model(X, f, params): 
	return(f(X, params))
	
# If model is autoregressive -- basically the g function written above
def evaluate(X, f, params):
	preds = []
	max_length = min(params['max_length'], MAX_LENGTH_CEILING)
	for t in range(X.shape[1]):
		start_idx = max(0, t-max_length+1)
		end_idx = t+1		
		x = X[:, start_idx:end_idx]
		pred = f(x, params)
		preds.append(pred)
	return preds # length T-1 
```

TODOS 
- [ ] Check if the model is evaluated at any other point  
- [ ] Make evaluate the single point of control through which any model is evaluated 

