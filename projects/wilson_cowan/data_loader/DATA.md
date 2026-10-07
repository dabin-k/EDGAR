# Real opto data as EDGAR sees it

Source files: `trial_counts_b30_<animal_id>_s1.npz`, one per mouse. They are produced in
`/home/dabin/code/ichun_opto` (`regenerate_population_rates.py`), which also documents their
format in **`ichun_opto/DATA.md`**. In short, each file is a flat trial table: raw E/I population
spike counts `counts (n_trials, 2, n_bins)` (1 ms bins, −0.5 to +1.5 s around the first pulse),
with each trial linked to a condition table (`cond_*` columns) via `cond_idx`. Nothing in the
file is averaged, smoothed or normalised.

`config.yaml` `io.data_path` points at `ichun_opto/edgar_b30_s1/`, which holds copies of only the
five session files. That keeps uploads (GCP) small.

## Transforms (`neural_data.build_cv_samples`, per mouse and CV unit)
1. **Filter**: keep conditions with `cond_contrast == contrast` (default 0, the paper's blank
   screen; `c > 0` trials had a screen stimulus).
2. **Bin**: `bin_ms` (default 1 = the file's native bins). Coarser bins are exact count sums, and
   t = 0 must stay a bin edge.
3. **Split trials** (no averaging; every row is one trial):
   - `k_fold`: trials are grouped into `n_folds` folds, stratified by condition and seeded by
     `fold_seed`. Test = the trials of `held_out_fold`; train = the trials of the other folds.
     Folds are not rotated: rotating would give one parameter set per fold, and those can't be merged.
   - `exp_cond`: train = trials of `train_types`, test = trials of `test_types`.
4. **Normalise**: rate = counts / bin width, divided per population (E, I) by the mean rate over
   −0.5 to −0.1 s across the **train** trials only. Test trials use the same divisor, so the
   normaliser never sees test data. Train baseline = 1 by construction.
5. **Chop** to `[chop_pre_ms, chop_post_ms)`. Build each trial's stimulus `(u_E, u_I)` from its
   condition: pulse 1 drives `first_pop` from 0 for `dur1_ms`; pulse 2 drives `second_pop` from
   **`onset_ipi_ms`** (= intT + dur1, the true onset-to-onset interval) for `dur2_ms`. A bin is on
   if a pulse overlaps any part of it.

## EDGAR dicts (`load_data._load_real`)
- One sample = one mouse, for both `k_fold` and `exp_cond`, so one parameter set per mouse. Mice are
  split 50/50 into discover / validate.
- The `n_stim` axis holds that sample's **individual trials**. Trial counts differ across samples,
  so each split is padded to its largest sample:
  - padding rows are cyclic copies of the sample's own trials, so parameter estimators see only
    real data;
  - `mask [n, C]` = 1 for real trials and 0 for padding;
  - `cond_id [n, C]` = the trial's condition row (−1 for padding).
- The losses (`losses/loss_common.per_sample_loss`) average over real rows only, so padding
  never changes a loss value.
- **Per-mouse loss scale.** `loss_scale [n]` is the mouse's mean per-trial variance over the scored
  bins (t ≥ 1 + warmup), averaged over E and I and computed on **train** trials only. Both the train
  and test dicts carry the same value. Each objective's loss is mean(per-trial MSE) / `loss_scale`:
  the fraction of variance unexplained. Mice with different firing rates then count equally in the
  score, and trials count equally within a mouse. It is one constant per mouse, with nothing
  condition- or trial-specific. `load_data._loss_scale` raises at load time if it isn't finite and
  positive; the losses require it (synthetic data gets the same scale from its train curves).
- Other keys are unchanged: `target_y [n, C, T, 2]`, `stim_E`/`stim_I [n, C, T]`,
  `target_y_future [n, C, A, K, 2]`, `time [n, T]`.

Laser-only trial counts per mouse (before the split): M150605_ICTP1 1223, M150609_ICTP1 1627,
M150609_ICTP2 1533, M150823_ICTP2 2593, M151020_ICTP1 2370.
