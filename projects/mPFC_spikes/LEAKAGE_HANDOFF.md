# Handoff: mPFC point-process causal leakage in EDGAR

Briefing for agents brainstorming a better solution. Written from the `windowed-scoring` worktree, Jul 2026.

---

## Context

| Item | Value |
|------|-------|
| **Repo / worktree** | `/groups/ahrens/home/ruttenv/python_packages/EDGAR_windowed_scoring` |
| **Branch** | `windowed-scoring` (commit `6557181` + uncommitted WIP) |
| **Engine** | [EDGAR](https://github.com/reillytilbury/EDGAR) — LLM-driven evolution of JAX models, scored in subprocesses with gradient descent |

**Related worktrees (awareness only):**

- `EDGAR` → `vmsr_gamma`: main integration branch; old full-series mPFC runs live in `cluster_logs/` only (project never committed there)
- `EDGAR_review_fixes` → `review_fixes`: upstream PR fixes (RNG resume, importlib loading)
- `EDGAR_mdgar_gamma` → `mdgar_gamma`: unrelated pipeline/dashboard work

---

## Scientific task

**Project:** `projects/mPFC_spikes/`

Mouse mPFC single-unit spike trains during slow-wave sleep (Buzsáki-lab `.mat` files). No stimulus — each neuron's spikes should be predicted from its **own past** via a conditional intensity λ(t | history).

**Intended loss:** discrete-time point-process NLL (Poisson), per neuron:

```
NLL = Σ_t ( μ[t] - y[t]·log(μ[t]) ) / Σ_t y[t]   # per-spike normalized, lower is better
```

where `y[t]` is the binned spike count in bin `t` and `μ[t]` is the model's predicted intensity for that bin.

**Constant-rate baseline:** ~6.8 NLL/spike (from loader printout in `config.yaml` comments).

**Data scale:** ~49–55 neurons kept after filtering; SWS segment binned at 5 ms → **T ≈ 50,000 bins** (~4 min), or 20 ms → **T ≈ 35,000 bins** (~12 min).

See also `EDGAR_submission.md` for the original scientific spec (full-series NLL, Hawkes-style seeds).

---

## The leakage problem (original "full-series" formulation)

### Contract (v1 — never committed; evidence in `EDGAR/projects/mPFC_spikes/cluster_logs/`)

```python
def model(data, params):
    x = data["x"]          # shape (T,) — full binned spike count series
    ...
    return mu              # shape (T,) — predicted intensity at each bin
```

- Standard EDGAR scoring: `jax.vmap(model)` over neurons; each program sees the **entire** spike series.
- Prompts repeatedly warned: `μ[t]` must depend only on `x[0..t-1]`, use 1-bin shifts, causal convolutions, etc.
- **Nothing in the engine enforced this.** The model had access to `x[t]` at the same index where loss is evaluated.

### How leakage happens (LLM-generated code)

Common failure modes seen in cluster logs:

1. **Missing 1-bin shift** — `jnp.convolve(x, kernel)` without shifting → `μ[t]` uses `x[t]`.
2. **Explicit current-bin features** — "Lag-1" terms, cumulative means, or box filters over `x[0:t]` that include `x[t]`.
3. **Off-by-one in "causal" helpers** — comments say causal but implementation uses `x` directly at `t`.
4. **Subtle bugs even in "causal" code** — e.g. `conv_full[t]` includes `x[t]` at lag 0 before slicing.

Even models that *look* causal can cheat because the full series is in scope and JAX makes it easy to accidentally index the wrong lag.

### Evidence leakage was real and mattered

| Run | Formulation | Best validate NLL | Notes |
|-----|-------------|-------------------|-------|
| Full-series (Jul 8, 5 ms, T=50k) | `model(data)` → `μ(T,)` | **2.14** (Program #374, "Lag-1 … UP-State Interaction") | Massive outlier; next-best ~5.34; most programs ~5.3–6.7 |
| Windowed (Jul 9, 5 ms, W=300, A=4000) | `model(window)` → scalar | **6.49** | All top programs ~6.49–6.55, essentially at baseline |
| Windowed (Jul 10, 20 ms, W=100, A=15000) | same | run crashed ~gen 11 | Losses still ~6.5; no real evolution headroom |

The **2.14 vs ~6.5** gap is the smoking gun: the full-series winner is far below the constant-rate baseline (~6.8), which is only plausible with lookahead (effectively predicting `y[t]` from `x[t]` or near-perfect knowledge of the target bin).

Log paths:

- Full-series: `../EDGAR/projects/mPFC_spikes/cluster_logs/full_152023729.log`
- Windowed 5 ms: `cluster_logs/full_0709_172550.log`
- Windowed 20 ms: `cluster_logs/full_20ms_0710_070243.log`

### Prior mitigation (soft — didn't work)

1. **Prompt engineering** — extensive causality instructions in `prompts.yaml`; LLMs still produced leaky or buggy code.
2. **"Perturb-the-timeseries" causality gate** (removed/replaced) — perturb future bins in `x` and check whether `μ` changes; mentioned in `leakage_check.py` as the *old* approach. Hard to make reliable as a filter (false positives/negatives, doesn't catch all leak paths, doesn't fix the scoring contract).
3. **Manual inspection of evolved programs** — confirmed many "causal" implementations were wrong or suspicious; evolution **rewarded** leaky models because they got artificially low NLL.

---

## The fix we tried: windowed next-bin scoring (structural anti-leakage)

**Commit:** `6557181` on `windowed-scoring`

### Idea

Make leakage **impossible by construction**: never give the model the target bin or a time index into the full series.

**New data contract:**

```python
data["history"][n, a] = x[n, t_a-W : t_a]   # (W,) strictly past
data["target_y"][n, a] = x[n, t_a]          # held out — loss only
```

**New model contract:**

```python
def model(window, params) -> scalar:   # predict NEXT bin only
    ...
```

**Engine changes (backward-compatible):**

- Pluggable `apply_model_fn` in `edgar/scoring/scoring.py` (default unchanged)
- Project provides `apply_model()` in `load_data.py` — nested vmap: outer over neurons, inner over anchor windows
- `leakage_check.py` — structural invariants:
  1. **ISOLATION** — perturb `target_y` → output unchanged; perturb `history` → output changes
  2. **SHAPE** — model only ever receives 1-D `(W,)` window
  3. **GUARD BAND** — train/test time split with W-bin gap; no overlap
  4. **SCORING** — seeds load → optimize → finite O(1) losses on real data

**Train/test split:** contiguous time split with **W-bin guard band** so no train window overlaps any test target.

Run the gate locally:

```bash
python projects/mPFC_spikes/leakage_check.py
```

### Uncommitted WIP (still on disk)

- **`edgar/scoring/scoring.py`:** spill `(train, test)` arrays >32 MB to temp `.npz` so subprocess spawn doesn't re-pickle huge tensors
- **`projects/dynamical_1d/`:** synthetic 1D toy (van der Pol) using the same windowed machinery — simpler testbed
- **`tests/scoring/test_mpfc_windowed.py`**, **`scripts/score_seeds.py`**

---

## Why the windowed approach is problematic / doesn't scale

### 1. Evolution signal collapsed

Windowed runs produce losses **tight around the constant-rate baseline (~6.5–6.8)**. After 12 generations, best validate ≈ 6.49 vs seeds ≈ 6.52 — essentially **no headroom**. Evolution cannot distinguish good dynamical models from trivial predictors.

Full-series (leaky) runs showed apparent headroom (2.14 best) — but that headroom was **fake**.

### 2. Wrong scientific formulation

Real point-process / Hawkes models are **sequential**: state updates via scan, IIR filters, or recurrence over T. The natural EDGAR contract is `model(data) → μ(T,)`.

Windowed next-bin turns this into **independent supervised samples** `(window → y)`. That:

- Breaks temporal coupling between consecutive bins in the loss
- Pushes the LLM toward static GLMs (`np.dot(window, kernel)`) instead of dynamical systems
- Does not match `EDGAR_submission.md`, which describes continuous-time / full-series NLL

### 3. Memory / compute blow-up

To get enough spike events in the loss for sparse neurons, you need **many anchors per neuron**:

```
memory ≈ N_neurons × A_anchors × W × 4 bytes  (per split)
```

Example config (`config.yaml`): 49 neurons × 15,000 anchors × 100 window ≈ **300 MB per split**, ~600 MB discover — and that still required coarser 20 ms bins. At 5 ms with W=300, A=4000: similar order of magnitude.

Each GD step runs the model on **all** `(neuron, anchor)` pairs via nested `vmap` → expensive compile + memory pressure. Had to add `.npz` spill for subprocess pickling.

### 4. Data efficiency vs independence tradeoff

`min_stride=W` gives non-overlapping windows (independent samples) but **throws away ~W−1 of every W bins**. Overlapping windows recover data but complicate train/test leakage and inflate `(N, A, W)` further.

### 5. LLM / JAX friction persists

Windowed contract requires **fixed static shapes** (`W = window.shape[0]`). LLMs still emit `np.arange(5*tau)`, dynamic slicing under `vmap`, etc. — failures in logs:

- `ConcretizationTypeError` (dynamic `jnp.arange`)
- `Slice entries must be static integers` (dynamic slice inside nested vmap)

Different failure mode, same root cause: LLM code + JAX tracing under batching.

### 6. Scoring still slow at scale

- Full-series: ~70–130 s/program (O(T) convolutions, T=50k)
- Windowed: ~15–20 s/program — better, but 48 programs × 8 islands × 12 gens × 2 splits is still hours, and **you're mostly measuring noise around baseline**

### 7. Doesn't solve the real goal

We want: **evolve legitimate causal point-process models** that fit ISI structure, bursting, long-range dependence — scored by proper NLL over time — **without** the model ever seeing `x[t]` when predicting `μ[t]`.

Windowed scoring **guarantees no peeking** but **sacrifices the formulation** that makes Hawkes/renewal models natural and evolvable.

---

## What a good solution might need to satisfy

**Hard requirements:**

1. **Causal by construction** — model cannot access `x[t]` (or spike times ≥ t) when producing `μ[t]`
2. **Proper point-process (or equivalent) likelihood** over time — not independent next-bin samples unless there's a clear equivalence argument
3. **Scales** to T ~ 50k, N ~ 50 neurons, EDGAR's subprocess scoring + vmap + GD loop
4. **Evolution headroom** — loss should span a range where better dynamical models beat baseline and seeds (cf. `mdgar_gamma` mdgar_demo: hand-crafted ceiling ~0.83 k-NN vs naive seeds ~0.62–0.64)
5. **LLM-friendly** — minimize foot-guns (causal shift, static shapes, scipy in traced code)

**Existing assets to build on:**

- `apply_model_fn` hook (general, upstream-worthy)
- `leakage_check.py` pattern (structural invariants + local no-LLM gate)
- `projects/dynamical_1d/` — simpler sandbox for testing contracts
- Full-series cluster logs — corpus of what LLMs actually generate

**Constraints of EDGAR architecture:**

- Scoring is **subprocess-isolated** with cloudpickle; large data must be passed efficiently
- Models are **JAX**; must `vmap` over neurons; GD on shared per-neuron params
- LLM generates Python source → translated to JAX; failures are common

---

## Brainstorm prompts

1. **Engine-enforced causal API** — e.g. model receives only `x[:t]` or a scan carry, never full `x` with index `t`; or `lax.scan` wrapper that hides current bin. Can EDGAR provide a causal primitive instead of raw arrays?

2. **Sequential scoring path** — first-class support for `model(carry, x_t) → (carry', μ_t)` scanned over T, with the engine guaranteeing `x_t` is past-only. Keeps full NLL, removes index arithmetic from LLM.

3. **Automated leakage detection in scoring** — runtime check: perturb `x[t+1:]` and assert `μ[:t+1]` unchanged; reject or penalize programs that fail. Complements prompts; doesn't require windowed reformulation.

4. **Loss / data reformulation** — predict spikes at **event times** (continuous-time NLL) instead of dense bins? Or coarse bins with scan-based state so windows aren't materialized?

5. **Calibrated difficulty** — like mdgar_demo: tune bin width / normalization so seeds sit ~X and hand-crafted ceiling ~Y with real headroom, *without* allowing leakage.

6. **Hybrid** — full-series scan internally, but LLM only writes per-step update function (smaller API surface).

---

## Key file paths

| Path | Role |
|------|------|
| `projects/mPFC_spikes/data_loader/load_data.py` | Windowed loader + `apply_model` + `loss_fn` |
| `projects/mPFC_spikes/leakage_check.py` | Structural anti-leakage gate |
| `projects/mPFC_spikes/config.yaml` | 20 ms, W=100, A=15000 |
| `projects/mPFC_spikes/EDGAR_submission.md` | Original scientific spec (full-series NLL) |
| `projects/mPFC_spikes/prompts.yaml` | Windowed model/param_est prompts |
| `edgar/scoring/scoring.py` | `apply_model_fn`, GD, subprocess scoring, npz spill (WIP) |
| `edgar/io/task_spec.py` | Loads optional `apply_model()` from project |
| `edgar/run.py` | Threads `spec.apply_model_fn` through score sites |
| `projects/dynamical_1d/` | Synthetic windowed toy |
| `../EDGAR/projects/mPFC_spikes/cluster_logs/full_152023729.log` | Full-series 12-gen run (validate best **2.14**) |
| `projects/mPFC_spikes/cluster_logs/full_0709_172550.log` | Windowed 5 ms run (validate best **6.49**) |
| `../EDGAR/meeting_report.md` | Workstream 3 summary (pluggable `apply_model_fn`) |

---

## Bottom line

**Problem:** Full-series point-process scoring let LLM models peek at the current bin; evolution rewarded leaky code (validate NLL ~2 vs baseline ~6.8).

**Fix tried:** Windowed next-bin prediction with held-out targets — leakage structurally impossible.

**Why it failed to scale:** Wrong scientific object, no evolution headroom (all losses ≈ baseline), memory/compute cost of materializing `(N, A, W)`, and LLM/JAX fragility under nested vmap — without recovering the sequential point-process formulation that makes this task interesting.

**Open problem:** causal guarantees + full-time NLL + evolvable signal + EDGAR's LLM/JAX/subprocess constraints.
