# mPFC spiking data + two seed models for EDGAR

Spontaneous activity, no stimulus, so the model predicts each neuron's spiking
from its **own past spikes**. Response = spike train; the models below specify
the conditional intensity lambda(t | history). Fit one neuron at a time.

## 1. The data

Buzsaki-lab `.mat` files (sending as-is):

- `..._SAll.mat` - `S_CellFormat`: spike times (s), one entry per neuron; `shank`.
- `..._CellIDs.mat` - `CellIDs.EAll` / `.IAll`: exc / inh unit indices (1-based).
- `..._WSRestrictedIntervals.mat` - Wake/SWS/REM `[start, stop]` second pairs.
- `..._Spindles.mat` - `SpindleData.normspindles` = `[start, peak, stop, freq]`.

Scale: 114 units (104 exc, 9 inh), ~327 min, spike times in seconds; spikes per
neuron median ~17k.

## 2. A few sentences describing the data

Mouse mPFC single units across natural sleep-wake, spike-sorted, labelled
exc/inh. No stimulus - the structure is temporal: heavy-tailed ISIs, non-renewal
spike trains (consecutive ISIs correlated, long-range dependence), and relaxation
back toward the mean after a short/long interval. A good fit reproduces the ISI
survival function (with its tail) and the regression-to-the-mean, not just the
mean rate. Activity is non-stationary across states (can restrict to SWS).

## 3. How to read the data

```python
import scipy.io as sio, numpy as np
base = "Dino_061814_mPFC"
spikes = [np.asarray(c).ravel().astype(float)          # spike times (s) per neuron
          for c in sio.loadmat(base + "_SAll.mat")["S_CellFormat"][0]]
ids = sio.loadmat(base + "_CellIDs.mat", struct_as_record=False,
                  squeeze_me=True)["CellIDs"]
exc = np.atleast_1d(ids.EAll).astype(int) - 1          # 1-based -> 0-based
```

## 4. The two seed models

Both give the intensity `lam(t) > 0` of a self-exciting point process; they
differ in how memory is stored.

### Model A - nonlinear reset process

Between spikes: `dlam/dt = -a * lam**p + drive`.
At each spike: `lam -> c0 - c1/(c2 + lam)` (reset from the pre-spike value).

| param | range | default |
|-------|-------|---------|
| `a`     | `[0.01, 100]` | `1.0` |
| `p`     | `[0.5, 3.0]`  | `1.5` |
| `drive` | `[0.01, 100]` | `1.0` |
| `c0`    | `[0.1, 500]`  | `10*rate` |
| `c1`    | `[0, 1000]`   | `5.0` |
| `c2`    | `[0.01, 100]` | `rate` |

Intuition: each spike remembers the intensity it fired at (the reset), giving
correlated ISIs and mean-reversion; `p` sets how fast the hazard decays between
spikes, hence the ISI tail (`p=1` exponential, `p>1` power-law).

### Model B - power-law Hawkes with drive

`lam(t) = mu + sum_over_past_spikes K / (t - t_i + u0)**(1 + gamma)`

| param | range | default |
|-------|-------|---------|
| `mu`    | `[0.001, 50]` | `rate` |
| `K`     | `[0, 1000]`   | `0.5` |
| `gamma` | `[0.05, 2.0]` | `0.5` |
| `u0`    | `[0.001, 1.0]`| `0.01` |

Intuition: every past spike lifts future intensity by a power-law-decaying
amount, so memory spans many timescales - the natural source of long-range
dependence and heavy ISI tails.

## 5. Loss

Point-process negative log-likelihood: `NLL = integral(lam dt) - sum(log lam(t_i))`,
per neuron. (Not MSE - it wouldn't distinguish the two models.)

## Nice-to-have

- Cheap init: `rate = n_spikes / duration`; `gamma ~ alpha - 1` where `alpha` is
  the empirical ISI survival-tail slope.
- Go-to plot: ISI survival (log-log) and the regression-to-the-mean curve,
  simulated vs real - both already implemented in `spike_analysis.py`.
