# What I need from you to model your spiking data with EDGAR

EDGAR takes a dataset plus a couple of candidate ("seed") models, then uses an
LLM-driven evolutionary loop to invent better models that fit your data with
lower loss. To set that up I need 5 things from you. **Don't reformat or clean
anything — send it in whatever format you already have and I'll do the
conversion.** Just tell me clearly what's in it.

---

## 1. The data

- Send the file(s) **as-is** — `.mat`, HDF5, `.npz`, pandas, `.pkl`, whatever.
  No need to convert or reshape anything.
- Just tell me **what's inside**, roughly:
  - The **spikes**: spike times per neuron, or an already-binned spike-count /
    firing-rate matrix. Whatever you have.
  - The **covariates/inputs** the model should predict spikes *from* — e.g.
    stimulus value, position, running speed, time, another population's
    activity. List each one.
  - A **time vector / sampling rate** so spikes and covariates can be aligned.
- A one-liner on scale: number of neurons, number of trials/timepoints,
  sampling rate, session length.

> **The one thing I really need conceptually:** what is the **response** the
> model should predict (usually firing rate), and what are the **inputs** it
> depends on? e.g. *"predict each neuron's firing rate from stimulus
> orientation and contrast."*

## 2. A few sentences describing the data

Plain text — this becomes the LLM's task prompt, so domain context directly
improves the models it invents:

- Brain region / cell type / recording method.
- What the subject was doing (stimulus or behavioral paradigm).
- What each covariate means and its units/range.
- What **"a good fit"** means scientifically — what should the model capture.

## 3. How to read the data (a snippet is enough)

Even 3–5 lines showing how you normally open the file and pull out the arrays,
e.g.:

```python
import scipy.io as sio
d = sio.loadmat("session.mat")
spikes = d["spike_times"]   # neuron -> spike times (s)
stim   = d["orientation"]   # stimulus per timepoint
t      = d["t"]             # time (s)
```

Plus one line on any preprocessing you'd normally apply (bin size, z-scoring,
speed threshold, trial exclusion). I'll bake that into the loader — you don't
have to apply it yourself.

## 4. Your two model ideas (the seed models)

You already have candidate models — great. **You don't need to write any Python.**
Give me **two** models (they should differ, e.g. a simple one and a richer one),
each as:

- The **equation** — LaTeX, or just a clear photo/scan of the math:
  `predicted_firing_rate = f(inputs, parameters)`.
- The **free parameters** and their **plausible ranges** (so I can clip them,
  e.g. `sigma ∈ [0.05, 1.0]`).
- Sensible **default values** for each parameter.
- One sentence of intuition for each.

That's all — I'll translate the equations into the code EDGAR needs.

## 5. The loss function (how fit quality is scored)

One line telling me how a prediction should be compared to the real data. Pick one:

- **Mean squared error** on firing rate (good default, esp. for z-scored rates),
- **Poisson negative log-likelihood** (natural for raw spike counts),
- **Correlation** between predicted and actual rate, or
- something custom (describe it).

---

## Optional / nice-to-have
- **Parameter-guessing tricks:** if you know a cheap way to estimate a model's
  parameters from data (e.g. "center = spike-weighted mean of the covariate"),
  tell me — it speeds up fitting. Otherwise I'll write a generic one.
- **A go-to plot** you'd use to eyeball a fit — EDGAR can render figures and
  feed them back to the LLM as visual feedback.

---

## TL;DR — minimum to get started
1. The data file(s), in any format, + a note on what's inside and its size.
2. 3–4 sentences on the data and what a good fit means.
3. A snippet showing how you read it.
4. **Two model equations** (LaTeX/photo) with parameter ranges + defaults.
5. One line on the loss (MSE / Poisson / correlation / custom).

Send those and I'll stand up the project. All the plumbing (data conversion,
train/validate/eval splits, JAX, the evolutionary config) is on my side.
