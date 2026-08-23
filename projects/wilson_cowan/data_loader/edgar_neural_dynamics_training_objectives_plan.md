# EDGAR Neural Dynamics Training-Objective Benchmark

## Goal

Test which training objective produces a neural dynamical model from which **EDGAR can most reliably recover a compact governing equation**.

Core experiment:

> same E/I population-rate data → same model architecture → different training objectives → EDGAR equation discovery → compare prediction, autonomous dynamics, and equation structure

The primary question is **not** “which loss gives the smallest forecasting error?” It is:

> Which training objective produces learned dynamics that are most useful for scientific equation discovery?

This plan is matched to the data layout described in `DATA.md`.

---

# 1. Actual neural dataset

Each session is stored as:

```text
results/population_rates_<animal_id>_s1.npz
```

Example:

```text
population_rates_M150605_ICTP1_s1.npz
```

Experiment types that may be present:

```text
single_E
single_I
paired_EE
paired_II
paired_EI
paired_IE
```

`flash` is not present in these files.

For each type:

```python
"{type}__responses"   # (n_conditions, n_folds, 2, n_bins)
"{type}__time_axis"   # (n_bins,)
"{type}__conditions"  # JSON list of dicts
```

Response axes are:

```text
axis 0 = stimulus condition
axis 1 = CV fold
axis 2 = population: 0 = E, 1 = I
axis 3 = time
```

The data are already:

- averaged over trials within each fold
- Hamming-smoothed with a 40 ms window
- baseline-normalised so pre-stimulus E and I rates are approximately 1
- normally sampled at 1 ms
- normally provided from -0.5 s to +1.5 s relative to first-pulse onset

The conditions list provides:

```text
pulse_type
ipi_ms
dur_ms
first_pop
second_pop
n_trials
n_trials_per_fold
```

Do not hard-code available pulse durations or IPIs. Read them from each session's condition metadata.

---

# 2. Scientific model

## State

Use an explicit latent dynamical state:

\[
z_t = [E_t, I_t, h_{1,t}, \ldots, h_{m,t}]
\]

where:

- `E`, `I` correspond directly to the measured population rates
- `h_1 ... h_m` are hidden dynamical variables
- hidden variables can capture slow inhibition, thalamic-like feedback, adaptation, or other missing mechanisms

The observation function is therefore deliberately simple:

\[
y_t = G(z_t) = [E_t, I_t]
\]

i.e. `G` selects the first two state coordinates.

This is preferable to allowing an arbitrary decoder if the goal is equation discovery, because EDGAR can then search for equations in coordinates where two dimensions already have clear biological meaning.

## Dynamics

Use a small continuous-time-inspired transition:

\[
\dot z = F_\theta(z,u)
\]

implemented initially with Euler integration:

\[
z_{t+1} = z_t + \Delta t\,F_\theta(z_t,u_t)
\]

with:

```python
dt = 0.001  # normally 1 ms; compute from time_axis rather than hard-coding
```

EDGAR's main target is `F_theta`.

This architecture should be kept identical across all four objectives.

---

# 3. Stimulus input u(t)

Construct stimulus channels directly from the condition metadata.

Use two exogenous channels:

```text
u_E(t)
u_I(t)
```

A pulse driving E sets `u_E = 1` for the pulse duration.
A pulse driving I sets `u_I = 1` for the pulse duration.

For all conditions:

```text
pulse 1 onset = 0 s
```

For paired conditions:

```text
pulse 2 onset = ipi_ms / 1000
```

Use `dur_ms` to determine the duration. If `dur_ms` is a two-element list, use the corresponding duration for each pulse.

Do not use `pulse_type` as the primary model input; use the physically interpretable E/I pulse channels.

Example helper:

```python
def make_stimulus(time_axis, condition):
    u = np.zeros((len(time_axis), 2), dtype=np.float32)

    durations = condition["dur_ms"]
    if np.isscalar(durations):
        durations = [durations, durations]

    pulses = [
        (0.0, condition.get("first_pop"), durations[0]),
    ]

    if condition["ipi_ms"] > 0:
        pulses.append(
            (
                condition["ipi_ms"] / 1000.0,
                condition.get("second_pop"),
                durations[1],
            )
        )

    for onset, pop, dur_ms in pulses:
        if pop not in ("E", "I"):
            continue
        channel = 0 if pop == "E" else 1
        offset = onset + dur_ms / 1000.0
        active = (time_axis >= onset) & (time_axis < offset)
        u[active, channel] = 1.0

    return u
```

---

# 4. Convert the NPZ into trajectory samples

Treat each:

```text
animal × experiment type × condition × fold
```

as one noisy replicate of the same population-level response.

A single training sample should look like:

```python
sample = {
    "target_y": ...,       # [T, 2]
    "u": ...,              # [T, 2]
    "time": ...,           # [T]
    "dt": ...,             # scalar
    "sample_weight": ...,  # scalar, normally n_trials in this fold

    # metadata, useful for grouping/evaluation
    "animal_id": ...,
    "experiment_type": ...,
    "condition_index": ...,
    "fold": ...,
    "ipi_ms": ...,
    "dur_ms": ...,
    "first_pop": ...,
    "second_pop": ...,
}
```

Use:

```python
sample_weight = condition["n_trials_per_fold"][fold]
```

so folds containing more trials can contribute proportionally if desired.

Because the folds are almost evenly split, equal fold weighting is also acceptable. Pick one convention and use it for every objective.

---

# 5. Cross-validation and held-out conditions

The saved files contain `n_folds = 3` and a deterministic trial-to-fold split.

## Primary cross-validation

Use three-fold CV:

```text
train on 2 folds
test on 1 fold
rotate test fold
```

Within each CV run, the model sees the same stimulus conditions in train and test, but the target PSTHs come from disjoint sets of trials.

This evaluates robustness to trial noise.

## Stronger secondary test: held-out perturbation conditions

For paired-pulse experiments, also evaluate generalization to unseen IPIs.

For example:

```text
train on a subset of paired-condition IPIs
test on held-out IPIs
```

Do this identically for all training objectives.

This is scientifically more informative than another random time-point split because the intended object is a dynamical law that should generalize across perturbations.

Do not assume every animal has the same set of IPIs.

---

# 6. Initial state / burn-in

The pre-stimulus interval is useful for synchronizing hidden state.

Default strategy:

```text
burn-in window: -0.5 s to 0 s
scored response: 0 s onward
```

During burn-in:

- clamp observed coordinates `E, I` to the measured data
- allow hidden coordinates to evolve
- do not include burn-in error in the headline loss unless needed

At `t = 0`, the model has a hidden state inferred from the pre-stimulus baseline.

This avoids learning a separate free initial latent vector for every trajectory.

Because the rates are baseline-normalised, a simple alternative for the very first implementation is:

```python
z_init[:2] = target_y[0]
z_init[2:] = 0.0
```

followed by the pre-stimulus burn-in.

---

# 7. What “teacher forcing” means in this model

The state contains observed and hidden coordinates:

\[
z_t = [E_t,I_t,h_t].
\]

After predicting the next state:

\[
\tilde z_{t+1}
=
z_t + \Delta t F_\theta(z_t,u_t),
\]

teacher forcing means replacing the predicted observed coordinates with the actual data before taking the next transition:

\[
z_{t+1}^{TF}
=
[y_{t+1}^{data},\ \tilde h_{t+1}].
\]

The hidden coordinates are **not** replaced.

Free-running mode instead uses:

\[
z_{t+1}^{free} = \tilde z_{t+1}.
\]

This gives a clean comparison while keeping exactly the same `F_theta`.

---

# 8. Model-output interface

Recommended common output:

```python
model_output = {
    # one-step prediction from the teacher-forced state
    "pred_y_1step": ...,       # [B, T, 2]

    # autonomous predictions starting from selected anchors
    "pred_y_rollout": ...,     # [B, A, K, 2]

    # latent state inferred using the observed trajectory / clamping
    "z_inferred": ...,         # [B, T, z_dim]

    # autonomous latent rollout from the same anchors
    "z_rollout": ...,          # [B, A, K, z_dim]

    # corresponding inferred future states
    "z_target_future": ...,    # [B, A, K, z_dim]
}
```

where:

```text
B = batch size
T = full trajectory length
A = number of rollout anchor times
K = rollout horizon in bins
```

The corresponding loss input can be:

```python
data = {
    "target_y": ...,             # [B, T, 2]
    "target_y_future": ...,      # [B, A, K, 2]
    "u": ...,                    # [B, T, 2]
    "sample_weight": ...,        # [B]
}
```

The training code, not the NPZ loader, can construct `target_y_future` windows from `target_y`.

---

# 9. Core MSE helper

The saved population-rate trajectories are complete, so the primary losses can use ordinary MSE.

```python
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)
```

If fold trial counts are used as sample weights, apply them at the batch level in the training loop rather than complicating each loss function.


# 10. Four training objectives

## Objective A — teacher-forced one-step MSE

### Scientific question

Is accurate local prediction sufficient for EDGAR to recover the correct dynamical equation?

### Training

At every step:

1. predict `z_{t+1}`
2. score predicted E/I against the real E/I
3. replace predicted E/I with the real E/I before the next transition

Loss:

\[
L_A
=
\sum_t
\|\hat y_{t+1}-y_{t+1}\|^2.
\]

### Loss

```python
def loss_A_one_step_tf(model_output, data):
    pred = model_output["pred_y_1step"]
    target = data["target_y"]
    return jnp.mean((pred - target) ** 2)
```

Align `pred_y_1step[:, t]` and `target_y[:, t]` consistently in the model code.

### Prior

Expected:

- easiest optimization
- best one-step prediction
- potentially poor autonomous dynamics
- potentially poor or unnecessarily complicated EDGAR equations

This is the baseline.

---

## Objective B — autonomous multi-step rollout MSE

### Scientific question

Does explicitly training autonomous trajectories improve recovery of the governing equation?

### Training

Use observed data only to synchronize the latent state at the anchor.

Then roll the model forward without clamping E/I.

\[
L_B
=
\sum_{a,k}
\|\hat y^{free}_{a+k}-y_{a+k}\|^2.
\]

### Loss

```python
def loss_B_rollout(model_output, data):
    pred = model_output["pred_y_rollout"]
    target = data["target_y_future"]
    return jnp.mean((pred - target) ** 2)
```

### Rollout horizons

Because the data are sampled at 1 ms and smoothed over 40 ms, a 5–10 step rollout is too short to be scientifically meaningful.

Recommended curriculum:

```text
early:   25–50 ms
middle:  100–150 ms
late:    300–500 ms
```

Example for 1 ms bins:

```python
K_schedule = [50, 150, 300]
```

Evaluation should include the full post-stimulus response out to 1.5 s even if the training horizon is shorter.

### Prior

Expected:

- harder optimization
- possibly worse one-step MSE
- better autonomous response dynamics
- stronger candidate for useful EDGAR recovery

---

## Objective C — rollout + latent-dynamics consistency

### Scientific question

Does explicitly making the hidden state obey a consistent autonomous flow improve symbolic discoverability?

### Training

From an inferred/clamped state `z_t`, autonomously predict:

\[
\hat z_{t+k}=F_\theta^k(z_t).
\]

Separately obtain the state reached by running the model along the observed trajectory:

\[
z^{inf}_{t+k}.
\]

Penalize disagreement:

\[
L_C
=
L_{rollout}
+
\lambda_z
\sum_{t,k}
\|\hat z_{t+k}-z^{inf}_{t+k}\|^2.
\]

### Loss

```python
def loss_C_latent_consistency(
    model_output,
    data,
    lambda_z=1.0,
):
    pred_y = model_output["pred_y_rollout"]
    target_y = data["target_y_future"]

    z_rollout = model_output["z_rollout"]
    z_target = model_output["z_target_future"]

    obs_loss = jnp.mean((pred_y - target_y) ** 2)
    latent_loss = jnp.mean((z_rollout - z_target) ** 2)

    return obs_loss + lambda_z * latent_loss
```

### Important caveat

Only E and I coordinates are directly identifiable.

Hidden coordinates may be transformed without changing the observations, so do not claim that a hidden coordinate literally represents a particular biological mechanism unless that interpretation is separately constrained.

The useful question is whether EDGAR finds a simpler/more stable dynamical law from this representation.

---

## Objective D — rollout + perturbation-response signature loss

### Scientific question

Does training directly on the macroscopic response features used to distinguish Wilson–Cowan-like mechanisms improve equation discovery?

The data structure is especially suitable for this because it contains:

- single E pulses
- single I pulses
- all four paired-pulse orders
- multiple pulse durations and IPIs

### Base loss

Always retain autonomous rollout MSE:

\[
L_D
=
L_{rollout}
+
\lambda_{dyn}L_{signature}.
\]

Do not use a signature-only objective.

### Recommended differentiable response signatures

Start with features that are easy to compute from baseline-normalised E/I traces:

1. **suppression area**
2. **rebound amplitude**
3. **late baseline offset**

For population trace `r(t)` with baseline 1:

\[
A_{supp} = \int \max(1-r(t),0)\,dt
\]

A differentiable rebound approximation can use a soft maximum.

Example:

```python
def soft_max(x, temperature=20.0, axis=-1):
    w = jax.nn.softmax(temperature * x, axis=axis)
    return jnp.sum(w * x, axis=axis)
```

Example feature extractor:

```python
def response_features(y, time_axis):
    # y: [..., T, 2]
    e = y[..., 0]

    post = time_axis >= 0.0
    late = time_axis >= 0.8

    e_post = e[..., post]
    e_late = e[..., late]

    suppression_area = jnp.mean(
        jax.nn.relu(1.0 - e_post),
        axis=-1,
    )

    rebound = soft_max(e_post, axis=-1)

    late_offset = jnp.mean(e_late - 1.0, axis=-1)

    return jnp.stack(
        [suppression_area, rebound, late_offset],
        axis=-1,
    )
```

### Loss

```python
def loss_D_dynamics_aware(
    model_output,
    data,
    lambda_dyn=1.0,
):
    pred = model_output["pred_y_rollout"]
    target = data["target_y_future"]

    rollout_loss = jnp.mean((pred - target) ** 2)

    # For this objective it is simplest to include one full
    # post-stimulus rollout per sample in model_output.
    pred_full = model_output["pred_y_full_rollout"]
    target_full = data["target_y"]
    time_axis = data["time"]

    pred_feat = response_features(pred_full, time_axis)
    target_feat = response_features(target_full, time_axis)

    feature_loss = jnp.mean((pred_feat - target_feat) ** 2)

    return rollout_loss + lambda_dyn * feature_loss
```

### Paired-pulse feature for evaluation

For `paired_IE` and `paired_EI`, compute recovery time versus `ipi_ms`.

The Lin-style mechanistic question is whether recovery is locked to a particular pulse rather than simply following the latest perturbation.

Use the slope of:

```text
recovery_time ~ ipi_ms
```

as an **evaluation metric first**, rather than putting the hard threshold-crossing operation into the training loss.

If Objective D is promising, a differentiable paired-pulse timing loss can be added later.

---

# 11. Do not use Poisson NLL for these saved rates

The stored observations are **not raw spike counts**.

They are:

- trial-averaged
- Hamming-smoothed
- baseline-normalised continuous rates

Therefore Poisson NLL is not the appropriate default likelihood.

Use MSE for the primary benchmark.

A Gaussian likelihood with fixed variance is effectively equivalent to scaled MSE.

## Optional noise-aware extension

Because the dataset stores three independent fold means, it is possible to estimate condition/time-dependent empirical variability across folds and use an uncertainty-weighted Gaussian loss.

This is interesting but should **not** be one of the first four objectives because only three folds provide a noisy variance estimate.

Treat it as a robustness analysis if needed.

---

# 12. Synthetic benchmark matched to the real data

Generate synthetic data using the same observable structure as the real NPZs:

```text
observed channels = E, I
stimuli = single/paired E/I pulses
sample interval = 1 ms
window = -0.5 to +1.5 s
baseline ≈ 1
```

Optionally apply 40 ms smoothing to synthetic observations before training so the model sees data with similar temporal resolution.

## Synthetic system 1 — standard Wilson–Cowan

Ground truth:

\[
\dot E = F_E(E,I,u)
\]

\[
\dot I = F_I(E,I,u)
\]

Observe both E and I.

Purpose:

- pipeline sanity check
- verify EDGAR can recover a known 2D system

## Synthetic system 2 — Wilson–Cowan + one hidden slow variable

Ground truth:

\[
\dot E = F_E(E,I,S,u)
\]

\[
\dot I = F_I(E,I,S,u)
\]

\[
\tau_S\dot S=-S+I
\]

Observe:

```text
E, I
```

Hide:

```text
S
```

This should be the main benchmark because it mirrors the scientific problem:

> measured E/I dynamics appear non-Markovian because an unobserved slower process influences them.

Generate the same `single_*` and `paired_*` stimulus conditions as used in the real-data experiment when practical.

---

# 13. EDGAR extraction protocol

For every trained model, extract state/derivative pairs:

```python
z
u
dz_dt = F_theta(z, u)
```

using states visited during:

1. teacher-forced / inferred trajectories
2. autonomous rollouts

Keep these two extraction sets distinguishable.

Primary EDGAR input should use the autonomous dynamical field `F_theta`, not an observation-correction rule.

For each training objective:

```text
trained neural dynamics
    ↓
sample (z, u, dz/dt)
    ↓
EDGAR
    ↓
symbolic candidate equations
```

Keep EDGAR:

- search operators
- population size
- evolutionary budget
- LLM calls
- scoring function
- random seeds / number of repeats

fixed across training objectives.

---

# 14. Evaluation

Evaluate at three levels.

## A. Prediction

Per held-out fold:

```text
1-step MSE
50 ms autonomous MSE
150 ms autonomous MSE
300 ms autonomous MSE
full post-stimulus rollout MSE
```

Report separately for:

```text
single_E
single_I
paired_EE
paired_II
paired_EI
paired_IE
```

and then aggregate.

---

## B. Dynamical-response behavior

For both neural-model rollouts and EDGAR equations report:

```text
suppression area
rebound amplitude
late baseline offset
recovery time
```

For paired conditions additionally report:

```text
recovery time vs IPI slope
```

for each pulse ordering.

Also test:

```text
stability of autonomous rollout
return to baseline / fixed point
```

---

## C. Equation recovery

### Synthetic data

Primary metrics:

```text
ground-truth term recovery
structural similarity
coefficient error where identifiable
equation complexity
free-running trajectory error of discovered equation
fixed-point / local stability agreement
```

Do not require literal algebraic identity if two equations are dynamically equivalent.

### Real data

There is no ground-truth equation.

Compare:

```text
held-out fold prediction
held-out IPI prediction
autonomous response dynamics
equation complexity
equation stability
consistency across training seeds
consistency across animals
```

The most interesting real-data result is not necessarily the lowest MSE. It is a simple EDGAR equation that generalizes across perturbations and reproduces the characteristic response dynamics.

---

# 15. Main hypotheses

## H1 — local predictive fit and equation quality will diverge

Teacher-forced one-step training will probably achieve the best local MSE but not necessarily the best EDGAR recovery.

## H2 — autonomous rollout training will improve dynamical fidelity

Objective B should better reproduce long-timescale suppression/rebound behavior and should give EDGAR a more faithful dynamical field.

## H3 — latent consistency may improve symbolic simplicity

Objective C may produce hidden-state dynamics that are easier for EDGAR to approximate compactly.

## H4 — perturbation-aware training may recover the most scientifically useful equations

Objective D may sacrifice some pointwise MSE while improving response features that distinguish different mechanisms.

---

# 16. Minimal experiment matrix

Run exactly these four first:

```text
A. teacher-forced one-step MSE
B. autonomous multi-step rollout MSE
C. autonomous rollout + latent consistency
D. autonomous rollout + perturbation-response signature loss
```

This progression is more informative than a simple teacher-forcing × horizon factorial:

```text
local transition accuracy
        ↓
autonomous trajectory accuracy
        ↓
latent dynamical consistency
        ↓
scientifically relevant perturbation dynamics
```

---

# 17. Implementation order

## Phase 0 — data loader

1. Load every available `population_rates_*_s1.npz`.
2. Enumerate available experiment types rather than assuming they exist.
3. Parse condition JSON.
4. Convert each condition/fold into a trajectory sample.
5. Construct `u_E(t), u_I(t)` from condition metadata.
6. Verify:
   - `target_y.shape == (T, 2)`
   - baseline ≈ 1
   - pulse timing matches `ipi_ms`
   - `dt` matches `time_axis`
7. Plot a few trajectories before training.

## Phase 1 — model

1. Implement `z = [E, I, hidden...]`.
2. Implement `F_theta(z, u)`.
3. Implement Euler step.
4. Implement clamped teacher-forced step.
5. Implement autonomous step.
6. Implement pre-stimulus burn-in.

## Phase 2 — objectives

Implement in this order:

```text
A → B → C → D
```

Do not implement all four before verifying that A and B learn sensible trajectories.

## Phase 3 — synthetic benchmark

1. WC
2. WC + hidden slow variable
3. Train A–D with identical model capacity
4. Run EDGAR
5. measure ground-truth equation recovery

## Phase 4 — real neural data

0. Chop timeseries data to only -50ms to 400ms around the first stimulus (at t=0ms)
1. 3-fold trial CV
2. train A–D
3. run EDGAR
4. evaluate held-out folds
5. evaluate held-out IPIs where feasible
6. compare equation simplicity and dynamical fidelity

---

# 18. First-pass hyperparameters

The original generic horizon of `K=10` is too short for these data.

Start with:

```python
hidden_dim = 1  # then test 2–3 if necessary
lambda_z = 1.0
lambda_dyn = 1.0
```

Rollout curriculum at 1 ms sampling:

```text
50 ms → 150 ms → 300 ms
```

Evaluate out to:

```text
1.5 s post-stimulus
```

Keep the first pass small. Do not perform large hyperparameter sweeps before confirming that the training objectives produce meaningfully different EDGAR equations.

---

# 19. Key figures

## Figure 1 — experiment

```text
E/I pulse-response dataset
        ↓
same latent RNN
        ↓
A / B / C / D objectives
        ↓
EDGAR
        ↓
candidate dynamical equations
```

## Figure 2 — prediction versus equation recovery

Synthetic benchmark:

```text
x-axis = neural-model prediction error
y-axis = EDGAR ground-truth equation recovery
```

The most interesting outcome is imperfect alignment.

## Figure 3 — real perturbation responses

For representative:

```text
single_E
single_I
paired_EI
paired_IE
```

show:

```text
held-out neural data
neural-model rollout
EDGAR-equation rollout
```

## Figure 4 — discovered equations

Show the best EDGAR equation from each objective, with:

```text
complexity
held-out error
dynamical-response metrics
```

For synthetic data also show the known ground-truth equation.

---

# 20. Main workshop claim to test

Avoid making the claim:

> Loss X gives the lowest MSE.

The intended claim is:

> **Training objectives impose different inductive biases on the dynamical representation learned from the same neural perturbation data, and those biases materially change which governing equations EDGAR can recover. Predictive accuracy alone is not sufficient to select a scientifically useful equation.**

That is the result the experiment should be designed to test.
