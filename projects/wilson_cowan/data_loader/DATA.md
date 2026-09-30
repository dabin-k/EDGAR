# Structure of population_rates npz files

Saved E/I mean population firing rates , grouped by experiment type and stimulus condition.
Produced by `regenerate_population_rates.py` from `data_loader.get_population_responses`.
Live in `/home/dabin/code/ichun_opto/results/`.

## Naming convention
- `population_rates_<animal_id>_s<session>.npz`
- `<animal_id>` = mouse + protocol, e.g. `M150605_ICTP1`, `M150609_ICTP2`, `M151020_ICTP1`.
- Session is always `s1` so far.

## Available fields
Per experiment type present in the session, three keys:
- `{type}__responses` — float64 array, the mean population firing rates (see shapes below).
- `{type}__time_axis` — float64 (n_bins,), bin centres.
- `{type}__conditions` — 0-d string array holding `json.dumps(list-of-dicts)`; `json.loads` it.

Types: `single_E`, `single_I`, `paired_EE`, `paired_II`, `paired_EI`, `paired_IE`.
- `single_*` = single pulse (ipi = 0); `paired_XY` = two pulses driving pop X then pop Y.
- E = excitatory (wide-spike / pyramidal), I = inhibitory (narrow-spike / PV).
- Types with no trials in the session are omitted. `flash` (visual) is never saved.

Plus provenance scalars (0-d arrays):
- `ei_cache_key` — str, the classified-cache key (session + boxes + classification version) that produced the E/I labels.
- `n_folds` — int, number of CV folds (3).
- `fold_seed` — int, RNG seed fixing the trial→fold split (0).

## Shape of responses
`{type}__responses` shape `(n_conditions, n_folds, 2, n_bins)`:
- axis 0: condition — aligned to the `{type}__conditions` list.
- axis 1: fold — one mean population firing rate per CV fold (not a single all-trials mean). But each fold is itself a mean of a subset of trials to reduce noise.
- axis 2: population — `0 = E` (wide), `1 = I` (narrow).
- axis 3: time bin — aligned to `time_axis`.

Values are per-fold trial-averaged firing rates, Hamming-smoothed (40 ms) and
baseline-normalised (divided by mean rate in −0.5 to −0.1 s, so baseline ≈ 1.0).
A fold with no trials (condition with fewer trials than `n_folds`) is all-NaN.

`time_axis`: seconds relative to pulse onset. Default window −0.5 to +1.5 s,
1 ms bins → 2000 bins, centres −0.4995 … +1.4995.

## k-fold splitting logic
- Per condition, trials are permuted with `np.random.default_rng(fold_seed)` then split by `np.array_split(perm, n_folds)`.
- As-even-as-possible; earlier folds absorb the remainder. Empty fold → NaN population firing rate, count 0.
- Split is deterministic from `fold_seed` alone (one RNG stream consumed in sorted-condition order).
- Recover the all-trials mean = average of fold mean population firing rates weighted by `n_trials_per_fold`.

## Shape of conditions
`json.loads(str(d['{type}__conditions']))` → list of dicts, length `n_conditions`,
aligned to axis 0 of `responses`. Each dict:
- `pulse_type` — int, protocol pulse-type code.
- `ipi_ms` — int, inter-pulse interval (0 for `single_*`).
- `dur_ms` — pulse duration; scalar, or `[d1, d2]` when the two pulses differ.
- `first_pop`, `second_pop` — "E"/"I", populations driven by pulse 1 and pulse 2.
- `n_trials` — int, total trials across folds.
- `n_trials_per_fold` — list of ints (length `n_folds`), aligned to axis 1.

Conditions are sorted by `(ipi_ms, dur_ms)`.

## Available stimulus conditions
- `single_E` / `single_I`: differ by `dur_ms` (e.g. 1 ms, 2 ms).
- `paired_*`: differ by `ipi_ms` (e.g. 5, 8, … ms) and `dur_ms`.
- Exact set is per-mouse — read them off the `conditions` list rather than assuming.

## How to pull in data
```python
import json, numpy as np

d = np.load("results/population_rates_M150605_ICTP1_s1.npz", allow_pickle=True)

resp = d["paired_EE__responses"]        # (n_cond, n_folds, 2, n_bins)
t    = d["paired_EE__time_axis"]         # (n_bins,)
cond = json.loads(str(d["paired_EE__conditions"]))   # list of dicts

# All-trials mean E population firing rates for condition 0 (weight folds by trial count):
w = np.array(cond[0]["n_trials_per_fold"], float)
e_r = np.nansum(resp[0, :, 0, :] * w[:, None], axis=0) / w.sum()
```
