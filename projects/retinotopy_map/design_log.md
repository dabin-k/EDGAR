# Loader design log — <project name>

Working copy of the data-loader-helper design log, living at `projects/<name>/design_log.md`.
Seeded from this template by `setup_design_log.sh` once the project name is decided; filled in
as decisions lock. See the skill's `SKILL.md` ("Keep a running design log") for the lifecycle.
This is the anti-drift anchor — re-read it before proposing the (sample, trial) mapping, and
verify Section 2 against the loader you actually wrote before claiming done.

## Section 1 — Decisions table

Fill `Decision` + `Rationale` as each is settled. `Status` is one of:
`proposed` (a candidate, not yet agreed) → `confirmed` (agreed with the user) ·
`broken` (an *existing* loader makes this choice but it violates a contract / doesn't
satisfy the conditions — record what's wrong in `Rationale`, then resolve to `confirmed`).
One or two lines per row — a ledger, not prose.

| Item | Decision | Rationale | Status |
|---|---|---|---|
| Raw data axes + sizes | per recording: dense pixel maps azimuth/elevation each `(H≈1500, W≈2100, 1)` deg; ~60–86% NaN outside responsive region. Data dir = one folder per animal (CR031, SP068 so far → 2 recordings; more later). | real, preprocessed (no cells); valid pixels are the data points | confirmed |
| Target equation / hypothesis | retinotopic map cortex→visual field: input = cortical pixel pos (col,row) px, target = (azimuth, elevation) deg; matches existing dipole fit `f(x,y)→vf` in data/code/fit_model_isolines.py | one shared map per recording | confirmed |
| **Sample** = | one recording | map params shared across all pixels in a recording, differ across recordings/animals | confirmed |
| **Trial** = | one valid cortical pixel (subsampled to fixed N per recording) | the axis train/test proves generalisation over (held-out cortical locations) | confirmed |
| Trailing-axis layout | `cortical_pos (n_samples, N, 2)`, `visual_field (n_samples, N, 2)` | 2D in / 2D out; subsample → rectangular, no NaN-pad | confirmed |
| Pointwise vs integrative | pointwise: each pixel's (az,el) from its own (x,y) alone | no neighbour dependence; confirmed by existing per-pixel least_squares dipole fit | confirmed |
| Train/test split granularity | checkerboard spatial blocks: tile cortical area into grid, alternate whole blocks to train/test; subsample N_train / N_test valid pixels from each | dense smooth grid → per-pixel random leaks (neighbours near-duplicate); blocks cover full field, limit boundary leakage | confirmed |
| Discover/validate sample counts | hold out 1 whole animal = validate; rest = discover (currently 1 animal each → 1 discover / 1 validate; scales as animals added) | tests map FORM transfers to unseen animal; no animal-specific leakage | confirmed |
| X_eval subset size | min(3, n_discover) samples from X_disc_train | small fingerprint subset | confirmed |

## Section 2 — Loader invariants checklist

Verify against the *written* loader before claiming done (a couple can be checked mid-way).
Any unticked box is a blocker, not a footnote.

- [x] All four split dicts share the same keys (`cortical_pos`, `visual_field`); JAX arrays, leading axis = samples (shapes `(1, 3000, 2)`)
- [x] `X_disc_*` (CR031) and `X_val_*` (SP068) hold **disjoint** sample sets — animal-level holdout
- [x] train/test hold the **same** recording(s), split along the pixel axis via checkerboard blocks
- [x] `X_eval` carries `_sample_indices` indexing into `X_disc_train`'s sample axis
- [x] `loss_fn` returns shape `(n_samples,)`, reducing pixel + coord axes (perfect→0, zeros→260)
- [x] every `project_params` knob is a named `load_data` kwarg with a default
- [x] split figure rendered (`projects/retinotopy_map/split.png`) — checkerboard + animal-disjoint

Validated: `uv run edgar validate retinotopy_map` ✓ ; smoke run `uv run edgar test
projects/retinotopy_map/config.yaml` ✓ (4/4 programs scored, discovery improved validate
loss 23.99/13.16 seeds → 7.18). Known gap: scaffold `image_feedback/plot.py` is generic and
errors (non-fatal); needs a project-specific `plot_model_fits` before image feedback works.
