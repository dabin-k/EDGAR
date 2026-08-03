# Extending EDGAR's state-space DSL to accept an exogenous input signal

*A design plan. Written as the input-driven follow-on to `docs/state_space_dsl_leakage_fix.md` and companion to `docs/plans/multivariate_dsl.md`. Same audience: an engineer approaching EDGAR's scoring loop for the first time.*

---

## 1. Scope in one paragraph

EDGAR's state-space DSL (v1, September 2026) fixed temporal leakage by forcing each LLM-authored model to be a one-step causal update `model(state, y_prev, params) → (new_state, mean)` on a scalar observation. This plan extends that contract to accept a per-step **exogenous input vector** alongside the previous observation: `model(state, y_prev, u_prev, params) → (new_state, mean)` where `u_prev` is a JAX vector of shape `(d_u,)` giving the value of a known driving signal at time `t-1`. The framework does **not** pre-compute filtered / smoothed / delayed variants of `u`; the LLM is expected to *discover* the correct filter, delay, or coupling structure inside its state update. The change is designed to preserve every structural guarantee of v1 (leakage is still a scope error; the `s0_` convention still works; `WARMUP_STEPS` is still a Python constant), to require zero engine changes, and to be orthogonal to the pending multivariate-output extension. The testbed is a driven Van der Pol oscillator where `u` passes through a *hidden* first-order low-pass filter inside the data generator — an unambiguous "did the LLM figure out to filter u?" signal.

**Recommended approach up front:** feed `u_prev` as a second scanned input alongside `y_prev`, add `data["u"]` of shape `(n_samples, T, d_u)` to the load-data contract, extend the project-local `apply_model` to scan over `(y_traj[:-1], u_traj[:-1])`, and require the LLM to declare `d_u` implicitly through the size of a required `_u_shape_hint` in `DEFAULT_PARAMS` (see §3). **One-line rationale:** `jax.lax.scan` supports scanning over pytrees of arrays natively, so extending the scan carry is a project-local change that costs zero lines in `edgar/`.

## 2. The new contract

### 2.1 Signature

The LLM writes:

```python
def model(state, y_prev, u_prev, params):
    """
    state:   dict of arrays — the model's current belief.
             Values chosen by the LLM (scalars or arrays).
    y_prev:  scalar (v1 contract) — the previous observation y[t-1].
    u_prev:  jnp array of shape (d_u,) — the exogenous input at time t-1.
             d_u is fixed by the dataset. u_prev is causally legitimate:
             u is exogenous by assumption (see §3.2).
    params:  dict of learnable scalars/arrays including log_sigma_obs
             and any s0_* initial-state values.

    returns: (new_state, mean)
             new_state — same pytree structure as state
             mean      — scalar (v1 contract) — predicted mean of y[t]
    """
```

The framework then scans:

```python
def apply_model(model_fn, data, params):
    y = data["y"]                             # (n_samples, T)
    u = data["u"]                             # (n_samples, T, d_u)

    def per_sample(y_traj, u_traj, p):
        init_state, dyn_params = _split_params_s0(p)

        def scan_step(state, inputs):
            y_prev, u_prev = inputs           # y_prev: (), u_prev: (d_u,)
            new_state, mean = model_fn(state, y_prev, u_prev, dyn_params)
            return new_state, mean

        # lax.scan iterates over the leading axis of every leaf in `xs`.
        # Here xs = (y_traj[:-1], u_traj[:-1]) — each iteration yields
        # (scalar y_prev, vector u_prev) to scan_step.
        _, means = jax.lax.scan(scan_step, init_state, (y_traj[:-1], u_traj[:-1]))
        # ... rest identical to v1 ...
```

The key JAX fact this rests on is that `lax.scan(f, carry, xs)` accepts `xs` as any pytree whose leaves share a leading axis of the same length; at step `t` each leaf's `t`-th slice is passed to `f`. This is well-documented behaviour and used across the JAX ecosystem; no engine machinery is needed.

### 2.2 Vector `u` from day 1 (D2)

`u_prev` is a vector `(d_u,)`, not a scalar. This is baked in even for the recommended v1 testbed which will use `d_u = 1` or `d_u = 2` (see §7). Rationale for vector-from-the-start:

- **The alternative — scalar u in v1, vector u in v2 — would change the LLM contract twice**, forcing prompt rewrites and re-training the LLM on the new signature. That's the same objection the leakage-fix doc raised in §3.1 against layered prompt patches: an unstable contract is a contract violation waiting to happen.
- **Broadcasting handles the `d_u = 1` case cleanly**: at `d_u = 1`, `u_prev` has shape `(1,)`, and the LLM writes `u_prev[0]` or `jnp.sum(u_prev)` to get a scalar — both are one line. The prompt teaches the vector idiom from day one.
- **Distractor inputs need vector-u**: the recommended testbed variant (§7) includes a distractor input that the LLM must learn to ignore. That requires `d_u ≥ 2`. Committing to vector from day one lets us ship the distractor variant without a re-do.

At `d_u = 1`, `u_prev` is a length-1 array, not a Python scalar. The LLM must use `u_prev[0]` (JAX indexing) or vector arithmetic (`state["v"] + k * u_prev`) — never `float(u_prev)` (fails on traced arrays). The prompt teaches this pattern explicitly (§6).

### 2.3 A worked seed example — driven VdP with a low-pass filter

The scientific target: the data generator applies a hidden first-order low-pass to `u` before it drives velocity. The LLM sees raw `u_prev` and must discover both (a) the VdP dynamics and (b) the low-pass on `u` (whose time constant `tau` it has to learn).

Data-generating dynamics:
```
dx/dt = v
dv/dt = mu * (1 - x^2) * v - x + K * u_filt
tau * du_filt/dt = u - u_filt          # HIDDEN inside the data generator
```

A correct-in-principle LLM program:

```python
def model(state, y_prev, u_prev, params):
    x = state["x"]                                    # observable position
    v = state["v"]                                    # hidden velocity
    u_filt = state["u_filt"]                          # LLM-discovered latent filter

    mu, dt, K = params["mu"], params["dt"], params["K"]
    tau = params["tau"]
    k_gain = params["k_gain"]

    innovation = y_prev - x
    x_corr = x + k_gain * innovation

    # First-order low-pass on u — the "discovery":
    u_filt_new = u_filt + dt * (u_prev[0] - u_filt) / tau

    dv = mu * (1.0 - x_corr ** 2) * v - x_corr + K * u_filt_new
    v_new = v + dt * dv
    x_new = x_corr + dt * v_new

    new_state = {"x": x_new, "v": v_new, "u_filt": u_filt_new}
    mean = x_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "mu": 2.0, "dt": 0.05, "K": 1.0, "tau": 0.3, "k_gain": 0.3,
    "log_sigma_obs": -1.5,
    "s0_x": 2.0, "s0_v": 0.0, "s0_u_filt": 0.0,
}
```

Two things worth calling out:

- **The filter is a state variable, not a preprocessing step.** `u_filt` lives in the LLM's `state` dict; the LLM has to *choose* to introduce it and update it correctly. A seed that omits `u_filt` and uses raw `u_prev[0]` directly is a valid competitor — it will lose on discovery loss but not on scope.
- **`u_prev[0]` (not `u_prev`) enters the physics.** Even at `d_u = 1`, the vector idiom is enforced by the contract. The LLM extracts the scalar it wants and uses it. This matters for the multivariate composition (§10).

## 3. Design decisions and the alternatives we ruled out

### 3.1 Where the filter lives — LLM vs. framework (D1)

**The alternative**: precompute a menu of filtered versions of `u` (raw, exponential-smoothed with three timescales, one-step-delayed, two-step-delayed) and pass all of them into the model:

```python
def model(state, y_prev, u_features, params):
    # u_features["raw"], u_features["smooth_fast"], u_features["smooth_slow"], ...
```

**Why we rejected it**: the discovery target is precisely the filter structure. Handing the LLM a filter menu is handing it the answer — the LLM merely has to `argmin` over the menu items rather than discover the low-pass. This defeats the scientific purpose of the testbed and, more importantly, defeats the general principle that motivated the leakage-fix doc: **the state is the model's belief**. A pre-filtered u would be belief injected from outside, not derived by the model.

Additionally, the LLM already has a general-purpose mechanism for representing filters: it can put a state variable inside `state` and update it however it wants. The state-space DSL was designed exactly to support this (the leakage-fix doc §4.3's FHN Kalman-lite already carries a hidden `w` variable). We reuse this capability rather than adding parallel plumbing.

**What we did instead**: only raw `u_prev` at each step is passed. The LLM discovers filter, delay, or coupling structure by writing it into its state update. This is D1 in the requirements; it's a user call and it's the right call.

**Cost of this choice**: the LLM has more work to do, and early-generation programs will get the filter wrong. That's the *point* of an evolutionary discovery loop. Programs that get the filter structure closer to right beat programs that don't; evolution selects for the structure.

### 3.2 Leakage surface — is `u_prev` legitimate?

`u_prev = u[t-1]` is legitimate causal information *when u is exogenous*. That qualifier matters. Under the new contract two sub-cases exist:

- **u is truly exogenous** (planned experimental perturbation, known stimulus, engineered drive). Then u carries no information about `y[t]` beyond what causal filtering can extract, and `u_prev` in the model's scope reopens no leakage surface. The driven-VdP testbed sits in this category by construction.
- **u is a function of past y in the data-generating process** (e.g., an experimenter chose the next stimulus in response to the last observation, closed-loop). Then `u[t-1]` is a lag-1 function of `y[t-1], y[t-2], ...` — still causal, still safe.
- **u depends on `y[≥t]`** (retrospectively-annotated covariate, `u_prev = f(y[t])` post-hoc). Then u carries information about the future observation directly, and the "input" is actually a leaked target in disguise. This is the failure mode to worry about.

The framework cannot verify which case the user is in — u is a black box from the DSL's perspective. The **contract-level position** is: the user asserts (by putting `u` in `data["u"]`) that `u[t]` is a function only of `{y[<t], u[<t], time, exogenous state}`. This is the same assertion the current DSL makes about `y_prev` being a lag: nothing in the runtime verifies it, but the contract makes it the user's responsibility.

For the driven-VdP testbed, `u` is exogenous by construction (a chosen sinusoid / band-limited noise), so this is a plan-level concern and not a v1 blocker. The `leakage_check.py` invariant remains: perturbing `y[t:]` leaves `pred[<t]` bit-exact. We add the analogous invariant for `u` (§9).

### 3.3 Warmup — does `WARMUP_STEPS` need to grow?

Currently `WARMUP_STEPS = 100` in every state-space project's `load_data.py`. The purpose (per leakage-fix §5.3) is to give the initial-state prior time to be corrected by observations before the loss window opens.

`u_prev` does affect the state through the LLM's filter (e.g., `u_filt` in the driven-VdP example takes `~ tau / dt` steps to reach steady state after `s0_u_filt = 0`). Does warmup need to grow?

**Position**: keep `WARMUP_STEPS = 100` for v1. Rationale:

- The v1 warmup was tuned to ≈1 period of the FHN dynamics at `dt = 0.05`. For the driven-VdP testbed the fundamental period is comparable (mu = 2, period ≈ 7.6 time units ≈ 150 steps), so 100 steps still covers most of the transient.
- The LLM's filter timescale is *learnable* — if the LLM initializes `s0_u_filt = 0` and the true filter has `tau = 0.3` s (= 6 steps at dt = 0.05), the filter is at steady state within the first 10 steps. Even a slow filter with tau = 2 s converges within the 100-step warmup.
- The failure mode "warmup too short" surfaces as inflated NLL on the first 100 post-warmup steps, which is measurable. We can bump it later per project if we see it.

**Caveat**: this is an expectation, not a measurement. We haven't measured whether a hidden filter with `tau > 1` needs longer warmup than v1's 100 steps. If in practice the discovery signal is dominated by initial-transient error, revisit and consider making WARMUP_STEPS a per-project constant tuned to `max(v1_warmup, ~5 * expected_tau_max / dt)`.

### 3.4 Parameter estimator — should it see `data["u"]`?

The current `parameter_estimator(data)` sees `data["y"]` only, and returns per-sample initial parameter estimates.

**Position**: yes, the parameter estimator MUST also see `data["u"]`. Rationale:

- The parameter estimator's job is to give gradient descent a warm start. For a model with a `K` input-gain parameter and a `tau` filter time constant, a good warm start uses `u` — e.g., regress `y` on lagged smoothed `u` for a rough `K`; use `1 / omega_c` from the cross-spectrum of `y` vs. `u` for a rough `tau`. Without seeing `u`, the estimator can't produce these.
- The framework's `_get_params` in `edgar/scoring/scoring.py:141-149` already passes the full data dict to the parameter estimator: it iterates `param_est_fn({k: v[i] for k, v in data_np.items()} for i in range(n))`. So *no framework change is needed* — `data["u"]` is already in scope. This is only a prompt-level change (§6.3) and a convention.

**Cost**: prompts must instruct the LLM that `data["u"]` is available, with shape `(T, d_u)` per sample. If the LLM ignores it and returns a naive estimate, that's fine — it just gets a worse warm start.

### 3.5 Fingerprint — does `u_prev` change what fingerprints should look like?

Recall §5.7 of the leakage-fix doc and §4 of the multivariate plan: `apply_model` in fingerprint-only mode returns residuals `targets - means` of shape `(T-1,)` per sample; `island.py::_are_duplicates` (`edgar/evolution/island.py:361-368`) computes cosine similarity on the flattened residual vector.

Under the input-driven contract, the residual `y_true - mean` is still scalar-per-step. Its shape and dtype are unchanged. Cosine similarity still works. **No engine change; no fingerprint format change.**

But there is a subtle *discrimination* concern. In v1, two programs with different state structures produced visibly different means on the same trajectory because the driving signal (process noise) was different for each state trajectory. Under the input-driven contract, two programs with different filters on `u` produce visibly different means because the input `u` is deterministic and shared across programs — the filter difference shows up in the mean directly.

**Position**: this is a *feature*, not a bug. Two programs with different filters on `u` *should* be flagged as non-duplicates, and they will be. If anything, fingerprint discrimination should be *better* under the input-driven contract than under v1, because the input `u` gives a shared, well-controlled excitation that reveals filter differences that free-running dynamics would hide. No mitigation needed.

**Caveat (expectation, not measured)**: we haven't empirically checked that a driven system's residuals produce well-spread cosine-similarity distributions. If in practice we see all programs' residuals clustering (because a strong input drives all programs to predict roughly the same mean, dominated by the input rather than by their internal filter), consider dividing residuals by the residual std of the driven mean before flattening — analogous to the per-channel normalisation suggested in the multivariate plan §4.1.

### 3.6 GD tuning — do `gradient_clip_norm` / `learning_rate` / `max_iter` need retuning?

The v1 `vdp_relaxation` config (`projects/vdp_relaxation/config.yaml:23-29`) uses:
```
param_penalty_weight: 0.001
gradient_descent:
  max_iter: 500
  learning_rate: 0.005
  gradient_clip_norm: 5.0
```

Under the input-driven contract, the loss surface gains additional axes of sensitivity: the LLM has parameters that control the filter (`tau`), the input gain (`K`), and the coupling from `u_filt` into velocity. Each is a fresh gradient dimension.

**Position on each knob**:

- **`gradient_clip_norm`**: keep at 5.0 for v1. The leakage-fix doc §5.5 justified 5.0 as a standard safeguard for scans over long horizons; adding a few more parameters doesn't change the per-parameter gradient scale enough to warrant a retune yet.
- **`learning_rate`**: keep at 0.005 for v1. Rationale: with a well-scaled input (`u` normalised to unit std), the parameter gradients are on the same scale as the v1 dynamics parameters, so Adam's step size behaviour is unchanged. If we see slow convergence on `tau` or `K`, revisit.
- **`max_iter`**: keep at 500 for v1. The added parameters increase the ill-conditioning of the problem slightly, but 500 iterations should still be enough. If oracle-gap analysis shows we're leaving loss on the table, bump to 750 or 1000.

**Caveat**: all three of these are expectations from analogy with v1, not measurements. This is exactly the sort of thing a first smoke-run reveals; be prepared to retune on empirical evidence.

**One recommendation with more confidence**: **normalise `u` to unit std at data-load time.** Adam's per-parameter step size is scale-sensitive through the second-moment estimate; a `u` signal with std 10 will make the `K` gradient 10x larger than a `u` signal with std 1, requiring a smaller learning rate for stability. Normalising `u` at load time (`u = (u_raw - u_raw.mean()) / u_raw.std()`) removes this dependence and lets the same `learning_rate = 0.005` work across projects with different `u` scales. Bake this into `load_data`.

### 3.7 Engine changes — can we achieve zero?

The multivariate-output plan achieved zero engine changes. Can this one?

**Position: yes.** Every hinge point is in the project's `data_loader/load_data.py`:

- `apply_model` — extended to unpack `data["u"]` and scan over `(y_traj[:-1], u_traj[:-1])`. Project-local.
- `loss_fn` — unchanged (still consumes `(means, log_sigmas)` from the stacked output).
- `validate_step` — extended to feed a sample `u_prev` alongside a sample `y_prev`. Project-local.
- `load_data` — extended to return `data["u"]` alongside `data["y"]`. Project-local.
- Prompts — extended to describe the new signature. Project-local (`prompts.yaml`).

The engine sees `data` as an opaque dict passed through to the project's `apply_model_fn`. The `_get_params` path (`edgar/scoring/scoring.py:141-149`) iterates over `data_train.values()` to slice per-sample, which correctly handles `data["u"]` of shape `(n_samples, T, d_u)` — slicing axis-0 gives shape `(T, d_u)` per sample. `_worker` (line 280) passes `data` through unchanged. `ravel_pytree` handles arbitrary parameter shapes. Nothing in the engine cares about the DSL signature.

**Minimum engine surface if not zero**: none required. Confidence: high, based on the same file-by-file argument the multivariate plan §8.1 makes.

### 3.8 Validate_step signature

`validate_step` currently calls `model_fn(init_state_j, jnp.asarray(0.5), dyn_params_j)` (line 288 in `projects/fhn_excitable/data_loader/load_data.py`, line 253 in `projects/vdp_relaxation/data_loader/load_data.py`). It must now call:

```python
sample_y_prev = jnp.asarray(0.5)
sample_u_prev = jnp.zeros((d_u,), dtype=jnp.float32)
new_state, mean = model_fn(init_state_j, sample_y_prev, sample_u_prev, dyn_params_j)
```

Trivial change, but worth naming explicitly. Also add: check that `d_u` is available (either from a project-level constant or from `data["u"].shape[-1]` at load time — see §3.9).

### 3.9 How `d_u` gets known — no schema change

**The alternative**: add a `n_inputs: int` field to `ModelSchema` in `edgar/llm/response_schema.py`, forcing the LLM to declare `d_u` explicitly.

**Why we rejected it**: same argument as the multivariate plan §8.3 for `n_channels`. Adding a schema field is a five-file engine change (schema, translator prompts, `Program`, `_translate_one_model`, `_get_params`). Rejected on cost.

**What we do instead**: `d_u` is fixed by the project's `load_data` — it's whatever shape `data["u"]` has along its last axis. The project's `apply_model` and `validate_step` both know `d_u` at import time (via a module-level constant, symmetric to `WARMUP_STEPS`). The LLM does *not* have to declare `d_u`; the prompt tells the LLM what `d_u` is for this project ("d_u = 1" or "d_u = 2"), and the LLM writes code accordingly.

This mirrors exactly the way `WARMUP_STEPS` is handled — module-level Python int, baked into closures at import time, jit-safe. See leakage-fix doc §5.3.

### 3.10 ISOLATION invariant under the new contract

The v1 ISOLATION invariant is: perturbing `y[t:]` by ±100 leaves `mean[<t]` bit-exact. This passes because `y[≥t]` is not in scope inside `model_fn` at step `t` — see `projects/vdp_relaxation/leakage_check.py:67-87`.

Under the input-driven contract, the analogous statement for `u` is: **perturbing `u[t:]` by ±100 leaves `mean[<t]` bit-exact**. This should also pass, for the same structural reason: `u[t]` is not passed to `model_fn` at step `t`; only `u[t-1]` (via `u_traj[:-1]` in the scan) is. The scan reads `u_traj` positionally, one step at a time, and never has `u[≥t]` in scope during the computation of `mean[t]`.

**Position: yes, add this as a fifth structural invariant to `leakage_check.py`.** Trivial to test — same code path as the y-isolation check but perturb `data["u"][:, t:]` instead. This is important because it *proves* the scan's argument packing is correct; if someone accidentally writes `xs=(y_traj[:-1], u_traj)` (missing the `[:-1]` on u), the u-isolation check catches it immediately.

### 3.11 `_split_params_s0` — unchanged

The prefix-strip logic (`projects/vdp_relaxation/data_loader/load_data.py:191-200`) is dict-of-key-value and does not care about the model signature. Nothing to change.

## 4. Fingerprint / dedup implications

Recap of the fingerprint path (verified in `edgar/evolution/island.py:329-368`):

```python
y_i = p_i.eval_fingerprint.flatten()
y_j = p_j.eval_fingerprint.flatten()
# shape guard
if y_i.shape != y_j.shape: return False
cosine = np.dot(y_i, y_j) / (np.linalg.norm(y_i) * np.linalg.norm(y_j) + 1e-6)
return bool(cosine >= cosine_tol)
```

Under the input-driven contract, the fingerprint payload (residuals of shape `(n_samples, T-1)`) is unchanged. The residuals encode how each program's predictions differ from the truth on `X_eval`; if two programs have different filters on `u`, their residuals will differ on the driven segments of `u`.

**No engine change. No fingerprint format change.** The one *convention* change is that `X_eval` must include a `u` field alongside `y`; the framework already routes the full data dict into `apply_model_fn` (`edgar/scoring/scoring.py:277`), so this is a project-level load-data change.

**Fingerprint discrimination**: as noted in §3.5, having a shared exogenous input `u` is *good* for discrimination because it excites the same modes in all programs, revealing filter differences that pure autonomous dynamics would hide. This is an expectation, not a measurement — worth checking on the first driven-VdP run whether the observed pairwise cosine distribution is tighter or looser than on v1 VdP.

**One risk to name**: if `X_eval`'s `u` is short (T_eval = 200) and the LLM's filter has a slow time constant (tau > 50 steps), the filter never reaches steady state on `X_eval` and the residuals are dominated by the transient. This would make fingerprint discrimination poor. Mitigation: use `T_eval ≥ 5 * expected_tau_max / dt` (a testbed-specific number, ~100 for tau=0.5, dt=0.05). Bake into the project's `config.yaml`.

## 5. GD tuning implications

Repeated for emphasis (see §3.6):

- Normalise `u` at data-load time to unit std. This makes the `K` gradient dt-invariant across projects.
- Keep `gradient_clip_norm = 5.0`, `learning_rate = 0.005`, `max_iter = 500` for v1.
- `param_penalty_weight = 0.001` (unchanged from v1 vdp_relaxation; the extra `s0_u_filt` parameter is one scalar, not a whole channel).

If the LLM introduces a many-tap FIR filter as its state (a buffer of the last N `u_prev` values, so `state = {"u_buf": jnp.zeros(N)}` with an `s0_u_buf: [0.0] * N`), `n_params` inflates by N and `param_penalty_weight` may need to drop proportionally. But the LLM will typically choose the parsimony point itself under the current penalty; monitor and drop if the population converges on high-order FIRs that overfit.

**Expected, not measured.** First empirical run will tell us if these knobs are wrong.

## 6. Files to create / modify

*Engine*: zero changes. Confidence: high (§3.7).

*New project* (self-contained):

- `projects/vdp_driven/__init__.py`
- `projects/vdp_driven/config.yaml`
- `projects/vdp_driven/prompts.yaml`
- `projects/vdp_driven/data_loader/__init__.py`
- `projects/vdp_driven/data_loader/load_data.py`
- `projects/vdp_driven/seed_programs/__init__.py`
- `projects/vdp_driven/seed_programs/model{1..4}.py`
- `projects/vdp_driven/seed_programs/param_est{1..4}.py`
- `projects/vdp_driven/leakage_check.py`
- `projects/vdp_driven/scripts/vdp_driven_oracle_nll.py`
- `projects/vdp_driven/scripts/post_run_analysis.py`

Project name justification: `vdp_driven` — parallels `vdp_relaxation` (the undriven testbed we're extending), makes the "driven" nature clear from the directory name, and follows the existing project-naming convention (`fhn_excitable`, `oscillator_ss`).

*Prompts.yaml* — three sections updated relative to `vdp_relaxation/prompts.yaml`:

- `model.code_guidelines` — new signature, worked example with `u_prev`.
- `parameter_estimator.code_guidelines` — announce `data["u"]` availability.
- `jax_translator_model.code_guidelines` — preserve `u_prev` argument in translation.

Detailed prompt changes below.

### 6.1 `model.code_guidelines` — the key changes

The current univariate guideline (`projects/vdp_relaxation/prompts.yaml:43-102`) has this critical assertion:

> * Model signature must be `def model(state, y_prev, params):` — a SINGLE step of a state-space model. `state` is a dict of scalar floats (the current belief); `y_prev` is a scalar (the observation from the previous timestep); `params` is a dict of learnable scalars.

New:

> * Model signature must be `def model(state, y_prev, u_prev, params):` — a SINGLE step of a state-space model driven by an exogenous input. `state` is a dict of scalar floats (the current belief); `y_prev` is a scalar (the observation from the previous timestep); `u_prev` is a JAX array of shape `(d_u,)` — the exogenous input at time t-1; `params` is a dict of learnable scalars.
>
> * For this project, `d_u = 1`. `u_prev` is a length-1 vector: use `u_prev[0]` to get the scalar. Do NOT use `float(u_prev)` — this fails on traced arrays.
>
> * The exogenous input `u` drives the system dynamics but may enter through a filter, delay, or coupling that YOU MUST DISCOVER. Passing `u_prev[0]` directly into your velocity update is one option; low-passing it via a state variable is another; a delay buffer is a third. The correct structure is a scientific-discovery target.

Plus a worked example section using the driven-VdP template from §2.3.

### 6.2 Common pitfall section (new)

Adapted from the multivariate plan's §10.4 discussion:

> * COMMON PITFALL — vector-vs-scalar u_prev: `u_prev` is a length-`d_u` JAX array, not a Python scalar. Use `u_prev[i]` for integer `i` in `[0, d_u)` to access a channel, or vector arithmetic (`state["x"] + params["K"] * u_prev` if state["x"] is `(d_u,)`). Never `u_prev[t]` — `t` is not in scope inside a scan step.

### 6.3 `parameter_estimator.code_guidelines`

The current guideline (`projects/vdp_relaxation/prompts.yaml:104-134`) says `data` is one trajectory's training dict with `data["y"]`. New:

> * `data` is one trajectory's training dict with `data["y"]` (1-D array of length T, the observations) AND `data["u"]` (2-D array of shape (T, d_u), the exogenous input). Use `data["u"]` to warm-start input-related parameters — for example, cross-correlating `data["y"]` against smoothed `data["u"][:, 0]` gives a rough estimate of the input gain `K` and the filter delay.
>
> * Sensible defaults for input-driven keys:
>     - `K` (input gain): initial estimate from the ratio of `data["y"].std() / data["u"][:, 0].std()`.
>     - `tau` (filter time constant, if present): initial estimate of `0.3` seconds — gradient descent will refine.
>     - `s0_u_filt` (initial filtered-input value): `float(data["u"][0, 0])` — the first input value.

### 6.4 `jax_translator_model.code_guidelines`

The current guideline (`projects/vdp_relaxation/prompts.yaml:136-174`) says:

> * The model signature is `def model(state, y_prev, params)` and returns `(new_state, mean)` — a Python tuple whose second element is a jnp scalar. Preserve this signature and return-tuple structure exactly.

New:

> * The model signature is `def model(state, y_prev, u_prev, params)` and returns `(new_state, mean)` — a Python tuple whose second element is a jnp scalar. Preserve this signature and return-tuple structure exactly. `u_prev` is a JAX array of shape `(d_u,)`; treat it as an array, not a scalar. Never call `float(u_prev)` or `int(u_prev)` — use array indexing (`u_prev[0]`) or arithmetic.

## 7. Testbed rollout: driven VdP

### 7.1 Project name — `vdp_driven`

Justified in §6. Parallels the existing `vdp_relaxation`; the whole point is that the LLM should first solve VdP (as in `vdp_relaxation`) and then also discover the low-pass filter on `u`.

### 7.2 Data-generating dynamics — recap

```
dx/dt = v
dv/dt = mu * (1 - x^2) * v - x + K * u_filt
tau * du_filt/dt = u - u_filt          # HIDDEN
```

with mu = 2.0, K = 1.0, tau = 0.3 s (~6 steps at dt = 0.05, so a moderate filter — long enough that ignoring it hurts, short enough that the warmup covers it). Integrated with Euler at dt = 0.05, T = 2400 steps. Process noise `σ_p = 0.3` on `du` (matching `vdp_relaxation`). Observation noise `σ_o = 0.15` on `x`. `y = (x + obs_noise - y_shift) / y_scale`, per-trajectory normalisation identical to `vdp_relaxation/data_loader/load_data.py:70-76`.

### 7.3 Choice of u(t)

**Recommendation**: sum of three sinusoids at slow, medium, and fast frequencies, each with a small random phase per trajectory. Specifically:
```
u(t) = 0.5 * sin(2*pi*f1*t + phi1) + 0.5 * sin(2*pi*f2*t + phi2) + 0.3 * sin(2*pi*f3*t + phi3)
```
with f1 = 0.05 Hz, f2 = 0.3 Hz, f3 = 2.0 Hz. Then normalise `u` to unit std at data-load time.

Rationale:
- **The low-pass content is distinguishable in the loss.** The fast component (f3 = 2.0 Hz) is heavily attenuated by a tau = 0.3 s filter (cutoff frequency ~0.5 Hz); the slow components pass through nearly unchanged. A model that ignores the filter (passes raw `u_prev[0]` into velocity) will predict oscillations at f3 that the true system doesn't have — a clearly measurable NLL cost.
- **Band-limited noise** would work too but is harder to reproduce exactly across runs. Sinusoids-with-phase are deterministic modulo the phase, which makes debugging easier.
- **Step + ramp mixtures** are less good for this testbed because they excite the filter only at one moment (the step) rather than continuously. The sinusoids give a stationary input that lets us average over many filter periods for a robust signal.

### 7.4 `d_u` recommendation for v1

**Recommendation: `d_u = 2`, with the second channel a distractor.**

Data-generating dynamics extend to:
```
u(t) = [u_driving(t), u_distractor(t)]        # d_u = 2
u_driving:    sum of three sinusoids (see §7.3)
u_distractor: independent, similar spectrum
```

Only `u_driving` (u[:, 0]) enters the physics; `u_distractor` (u[:, 1]) is a red herring. The LLM must learn (a) that only the first channel drives `x`, and (b) that the driving channel passes through a low-pass.

Rationale:
- **Tests vector-u handling seriously.** With `d_u = 1`, the LLM can just write `u_prev[0]` mechanically. With `d_u = 2` and a distractor, the LLM has to *choose* which channel matters (or discover that both do, or neither does).
- **Matches the multivariate-composition story (§10).** Multi-input drives are more realistic and force the DSL to handle vector `u` end-to-end.
- **The distractor's contribution to the true `y` is exactly zero**, so a program that mistakenly uses `u_prev[1]` in its dynamics adds unpredictable variance to its mean and loses to a program that correctly ignores it. This is a clean discovery-vs-non-discovery signal.

Fallback: if v1 shows that distractor discrimination is *too* hard for the LLM (>50% of programs use both channels indiscriminately), fall back to `d_u = 1` for v1.5 and defer distractor variants.

### 7.5 Seed ladder (four programs)

Each seed occupies a rung on the discovery ladder from "no input awareness" to "correct filter":

**Seed 1: pure VdP, ignores `u_prev` entirely.**
```python
def model(state, y_prev, u_prev, params):
    # ignores u_prev; runs vdp_relaxation's seed 3
    x, v = state["x"], state["v"]
    mu, dt = params["mu"], params["dt"]
    k_gain = params["k_gain"]
    innovation = y_prev - x
    x_corr = x + k_gain * innovation
    dv = mu * (1.0 - x_corr**2) * v - x_corr
    v_new = v + dt * dv
    x_new = x_corr + dt * v_new
    return {"x": x_new, "v": v_new}, x_new
```

Baseline: shows the loss floor when u is ignored. Any program that beats this has extracted *something* from u.

**Seed 2: VdP with direct u-coupling, no filter.**
```python
# ... same as seed 1, except:
dv = mu * (1.0 - x_corr**2) * v - x_corr + params["K"] * u_prev[0]
```

Uses raw `u_prev[0]` as the drive. Ignores filter. Beats seed 1 on segments where the raw u happens to match the filtered u; loses on high-frequency segments.

**Seed 3: VdP with first-order low-pass on u.**
```python
# ... same as seed 1, except:
u_filt_new = state["u_filt"] + dt * (u_prev[0] - state["u_filt"]) / params["tau"]
dv = mu * (1.0 - x_corr**2) * v - x_corr + params["K"] * u_filt_new
# ... plus updated state dict with u_filt
```

The correct structural form. If `params["tau"]` is learned close to 0.3 s, this seed is near-oracle.

**Seed 4: VdP with cascaded two-pole filter on u.**
```python
# ... same as seed 3, plus a second cascaded LP:
u_filt1_new = state["u_filt1"] + dt * (u_prev[0] - state["u_filt1"]) / params["tau1"]
u_filt2_new = state["u_filt2"] + dt * (u_filt1_new - state["u_filt2"]) / params["tau2"]
dv = ... + params["K"] * u_filt2_new
```

Two-pole is over-parameterised relative to truth (which is one-pole). Under `param_penalty_weight = 0.001`, this seed pays for the extra state variable but has extra flexibility. Whether it beats seed 3 tells us how much slack the penalty leaves.

Together the four seeds span: ignores u (seed 1) < uses raw u (seed 2) < uses correctly-filtered u (seed 3) ≤ uses over-filtered u (seed 4). Evolution should target a program that matches or slightly beats seed 3.

### 7.6 Oracle NLL floor

Analogous to `vdp_relaxation/scripts/vdp_oracle_nll.py`. The oracle knows:
- True dynamics (mu, K, tau).
- True (x, v, u_filt) sequence per trajectory.
- The affine (y_shift, y_scale) that took raw x to normalised y.

Its one-step prediction is:
```
E[x_raw[t+1]] = x_raw[t] + dt * v_raw[t]
E[y[t+1]]     = (E[x_raw[t+1]] - y_shift) / y_scale
```

Identical to `vdp_relaxation`'s oracle. The oracle floor is essentially unchanged from `vdp_relaxation`'s (~-1.6 to -1.7 nat/bin depending on obs noise), because the *observation* is still `x` with additive noise, and the position update `dx/dt = v` has no process noise. **The u-filter discovery contest happens in the discovery *gap*, not in the oracle floor.**

Expected budget (this is an expectation from analogy with `vdp_relaxation`, not measured):
- Oracle: NLL ≈ -1.65
- Persistence: NLL ≈ -0.9
- Seed 1 (ignores u): NLL ≈ -1.2 (u contributes 0.3-0.5 nat to what a VdP-only model misses)
- Seed 2 (raw u): NLL ≈ -1.3 (better than seed 1 on slow segments; worse on fast segments)
- Seed 3 (correct 1-pole): NLL ≈ -1.55 (near oracle)
- Seed 4 (2-pole): NLL ≈ -1.55 (matches seed 3 modulo penalty)

Discovery budget: ~0.75 nat between persistence and oracle; ~0.35 nat between best undriven seed (seed 1) and oracle. **We haven't measured any of these numbers yet.** First smoke run will populate them; if the gap between seed 1 and seed 3 is < 0.1 nat, the low-pass filter isn't giving enough signal and we should either lengthen tau or increase K.

### 7.7 leakage_check.py invariants

Extend `vdp_relaxation/leakage_check.py` to five invariants:

1. **ISOLATION-y** (v1): perturbing `y[t:]` leaves every `pred[<t]` bit-exact. Passes because `y[≥t]` is not in scope.
2. **ISOLATION-u** (NEW): perturbing `u[t:]` leaves every `pred[<t]` bit-exact. Passes because `u[≥t]` is not in scope. This is the new invariant — it *proves* that `u_traj[:-1]` is scanned correctly (missing the `[:-1]` on u would fail this check immediately).
3. **SHAPE** (v1): `validate_step` passes for every seed with the new 4-arg signature.
4. **NLL sanity** (v1): seeds produce finite O(1) losses.
5. **Seed-vs-oracle gap** (v1, updated): the best seed's NLL is ≥ 0.03 nat above the oracle, confirming the DSL is not trivially solved.

Ordering matters: run ISOLATION-y and ISOLATION-u before touching scoring; they're structural and cheap. If they fail, the scan wiring is wrong and everything downstream is meaningless.

## 8. Composition with the multivariate-output plan

The multivariate-output plan (`docs/plans/multivariate_dsl.md`) extends the DSL to `y_prev: (d_y,)`, `mean: (d_y,)`. This plan extends the DSL to `u_prev: (d_u,)`. Both change `model`'s signature. Are they compatible?

**Yes, cleanly.** The combined contract is:
```python
def model(state, y_prev, u_prev, params) -> (new_state, mean)
    #             ^(d_y,)  ^(d_u,)                        ^(d_y,)
```

Concretely:
- `apply_model` scans `xs = (y_traj[:-1], u_traj[:-1])`, where `y_traj` has shape `(T, d_y)` and `u_traj` has shape `(T, d_u)`.
- At each scan step, `y_prev` has shape `(d_y,)` and `u_prev` has shape `(d_u,)`; `model_fn` returns `mean` of shape `(d_y,)`.
- The stacked output is `(T-1, d_y, 2)` as in the multivariate plan §3.1; `u` is not part of the output — it's an input.
- `loss_fn` is exactly the multivariate plan's version — unchanged, because it only reads model_output and `data["y"]`, not `data["u"]`.
- `validate_step` feeds `sample_y_prev` of shape `(d_y,)` and `sample_u_prev` of shape `(d_u,)`.

The two plans **compose without a fight**: the multivariate plan extends the *output* side of the DSL (and by extension the y-side of the input, since `y_prev` shape matches `mean` shape); this plan extends the *input* side (adding `u`). They touch orthogonal parts of `apply_model` and `validate_step`, and their prompt changes affect different lines of `model.code_guidelines`.

**Recommended shipping order**: (a) multivariate-output first, since it's the smaller signature change (add a dimension to an existing argument, rather than add a new argument); (b) input-driven second, layering on top. But nothing structural prevents either order — both PRs are self-contained project scaffolds with zero engine changes. **This plan does NOT propose merging the two PRs.** They should ship separately for the reasons in D4: easier to review, test, roll back, and gate independently.

## 9. Risks + mitigations

### 9.1 The LLM omits `u_prev` from its signature

The most likely LLM failure mode: the LLM writes `def model(state, y_prev, params):` (the old 3-arg signature) instead of the new 4-arg one. Under Python's arg-count checking, calling `model_fn(state, y_prev, u_prev, dyn_params)` on a 3-arg function raises `TypeError: model() takes 3 positional arguments but 4 were given`.

**Mitigation**: `validate_step` catches this eagerly with a clear error at load time (before scoring). The eager `model_fn(init_state_j, sample_y_prev, sample_u_prev, dyn_params_j)` call raises TypeError; the current try/except in `vdp_relaxation/data_loader/load_data.py:252-259` catches it and produces a legible AssertionError with the offending program's source. Prompts (§6.1) must be extremely explicit about the signature; include the 4-arg signature in the pattern block.

### 9.2 The LLM uses `float(u_prev)` and fails under trace

The second-most-likely LLM failure: the LLM writes `u_scalar = float(u_prev[0])` to convert the length-1 vector to a Python scalar. Under `jit`, this raises `ConcretizationTypeError`.

**Mitigation**: `jax_translator_model.code_guidelines` (§6.4) forbids `float(u_prev)`. The eager `validate_step` won't catch this (it runs outside jit), but the smoke run's first `_optimize` call will surface it as an inf-loss with a clear ConcretizationTypeError in the traceback. Evolution kills it; the program is dead-lettered.

### 9.3 The LLM confuses `u_prev` with `u[t]`

Analogous to the multivariate plan's §10.4: the LLM writes `u_prev[t]` thinking `t` is a time index. `t` is not in scope inside a scan step; this raises NameError. Evolution kills it.

**Mitigation**: prompt discipline (§6.2 "COMMON PITFALL" section). Include a worked example that uses `u_prev[0]` and `u_prev[1]` correctly.

### 9.4 Warmup is too short for the LLM's chosen filter

If the LLM picks `tau` much larger than the warmup permits (say, `s0_u_filt = 0` and `tau = 5.0`, so the filter takes ~500 steps to reach steady state), the first 500 post-warmup predictions are dominated by transient rather than steady-state error.

**Mitigation**: monitor per-seed loss curves — if seed 3 shows visible post-warmup transient, bump `WARMUP_STEPS` to 200. But most reasonable LLM-chosen taus are < 1.0 s, well within v1's warmup. **Not measured; expected to be a non-issue.**

### 9.5 `u` normalisation surprise

If we normalise `u` to unit std at data-load time but the user forgets to update the oracle NLL calculation accordingly (since the true `K` is now `K_raw * u_raw.std()`), the oracle NLL will be miscomputed.

**Mitigation**: oracle NLL script (`vdp_driven_oracle_nll.py`) reads `data["u"]` at the same normalisation as `apply_model` sees it. Since both go through `load_data`, they're consistent by construction. Add a docstring in `load_data` naming the normalisation contract.

### 9.6 Compile-cost blowup

Every new state shape → fresh JAX compile. The input-driven contract adds one new dimension (`u_prev: (d_u,)`) plus whatever state variables the LLM invents for its filter (`u_filt`, `u_buf`, etc). Compile-cache hits may drop.

**Expected impact**: modest. State shape is coarsely characterised (n_scalar_state_keys, has_u_filt); expect ~8-10 distinct trace signatures per generation vs. v1's ~4-6. Compile budget rises modestly. **Not measured.**

**Mitigation**: none for v1. If it becomes a bottleneck, revisit.

### 9.7 `X_eval` too short for slow filters

If `T_eval = 200` and the LLM's filter has `tau = 3.0 s = 60 steps`, the filter takes >5 tau (300 steps) to reach steady state — longer than `T_eval`. Fingerprint residuals are then dominated by transient, and dedup degrades.

**Mitigation**: bump `T_eval` to 400 or 500 for this project. Cheap; only affects fingerprint compute (which is a small fraction of the total). See §4.

## 10. Explicit deferred items

Not in v1:

- **Multivariate-output composition.** Ship separately per D4. The two plans compose cleanly (§8) but should be reviewed and tested independently.
- **Multi-cell homogeneous populations with per-cell inputs.** Appendix-A-style; different DSL contract, different testbed. Separate project.
- **Framework-provided filter menu.** Explicitly rejected (§3.1). The LLM discovers the filter.
- **Auto-scaling of `param_penalty_weight` with `d_u`.** Manual tuning per project; auto-scaling is out of scope.
- **Closed-loop u** (u driven by past y in the data generator). Contract-legal but not exercised by v1 testbed; deferred.
- **Non-Gaussian observation model.** Same limitation as v1 DSL; not touched by this plan.
- **Free-running rollout metric** (open-loop simulation with u as sole input). Would be a very strong test of filter discovery but requires additional engine plumbing; defer to v1.5.

## 11. Explicit checklist for the implementer

Before starting:
- [ ] Re-read `docs/state_space_dsl_leakage_fix.md` end to end, especially §4 (contract) and §5.3, 5.6, 5.7 (design decisions).
- [ ] Read `docs/plans/multivariate_dsl.md` end to end. The composition section (§8 of this doc) assumes the reader understands multivariate.
- [ ] Read `projects/vdp_relaxation/data_loader/load_data.py` — this is the template we're extending.
- [ ] Read `projects/vdp_relaxation/leakage_check.py` — the equivalent for input-driven is the acceptance test.

During implementation:
- [ ] `load_data` returns `data["u"]` of shape `(n_samples, T, d_u)` alongside `data["y"]` of shape `(n_samples, T)`.
- [ ] `u` is normalised to unit std at load time.
- [ ] `apply_model` scans over `xs = (y_traj[:-1], u_traj[:-1])`.
- [ ] `validate_step` feeds `sample_u_prev = jnp.zeros((d_u,))` alongside sample y_prev.
- [ ] `loss_fn` is unchanged from `vdp_relaxation`.
- [ ] `WARMUP_STEPS = 100` (same as v1).
- [ ] `d_u = 2` (one driving, one distractor) — or `d_u = 1` if we fall back per §7.4.
- [ ] Seed ladder includes seed 1 (ignores u), seed 2 (raw u), seed 3 (1-pole low-pass), seed 4 (2-pole low-pass).
- [ ] `leakage_check.py` has five invariants including ISOLATION-u (perturb `data["u"][:, t:]`).
- [ ] Oracle NLL script exists and matches `loss_fn`'s conventions; oracle sidecars include `_u_true`, `_u_filt_true`.
- [ ] Prompts explicitly state `d_u = X for this project` and give worked u_prev[0] / u_prev[1] indexing examples.
- [ ] `param_penalty_weight = 0.001` (v1 vdp_relaxation value; may need to drop for FIR-buffer states — monitor).

Acceptance:
- [ ] `leakage_check.py` passes on `vdp_driven`, all five invariants.
- [ ] A 1-generation smoke run produces at least one program with finite loss.
- [ ] A 4-generation full run produces at least one program that beats seed 1 by ≥ 0.1 nat (evidence that u is being used).
- [ ] Post-run leakage inspection confirms zero programs read `y[t]` or `u[t]` when predicting `y[t]`.

## 12. What could kill this plan

Two things, in order of concern:

1. **The LLM cannot reliably write vector-arithmetic on `u_prev`.** If, after prompt iteration, >50% of programs error out with shape mismatches or ConcretizationErrors, the discovery signal collapses. Mitigation: more prompt work, worked examples, explicit shape hints. If necessary, back off to `d_u = 1` for v1.
2. **The low-pass discovery signal is too weak to distinguish seed 2 from seed 3.** If the gap between "uses raw u" and "uses correctly-filtered u" is < 0.05 nat, evolution can't select for filter discovery and this whole testbed is uninformative. Mitigation: raise `K`, lengthen `tau`, increase the fast-component amplitude in `u`. All are project-level knobs.

Neither is catastrophic. Both are things we can iterate on in place.

---

*Plan ends. This is a specification for building; the empirical report should follow implementation in a separate document.*
