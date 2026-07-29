"""Generate the shared unforced-Burgers benchmark datasets (EDGAR / SINDy / STENCIL-NET).

Writes one npz per (initial condition, noise level) to `--out-dir`:

    ic_seed_{ic}_nl_{nl}.npz

Each file holds `len(D_VALUES)` samples that share one initial condition and differ
only in the viscosity `D` — the per-sample parameter each method has to fit. The same
`D_VALUES` and the same clean fields are reused across every noise level, and the noise
seed does not depend on the noise level, so the files for one `ic_seed` differ *only* in
noise amplitude. That is what makes "all three methods saw the same data, including the
same noise" true by construction rather than by convention.

Unforced throughout (`f = 0`); `A/w/phi/l` are stored as zeros so a consumer that still
reconstructs the forcing from the bundle gets `f = 0` instead of the wrong forcing.

Run:  uv run python projects/burgers/data_loader/generate_datasets.py
      uv run python projects/burgers/data_loader/generate_datasets.py --check
Wall time ~1.5 min on CPU (8 simulations; the 3 noise levels reuse them).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import burgers_sim as bs  # same directory


D_VALUES = (0.005, 0.01, 0.02, 0.04)
# Seed 1 is skipped deliberately: draw_ic gives it amp=1.132 vs seed 0's amp=1.129, so it
# is the same bump translated in x — on a periodic domain that is a symmetry, not a second
# experiment. Seed 2 draws amp=0.716, a genuinely different initial amplitude.
IC_SEEDS = (0, 2)
NOISE_LEVELS = (0.0, 0.01, 0.1)
OUT_DIR = "/home/dabin/data/burgers_simulated"

LX = 256
L = 2.0 * np.pi
DT = 0.002
TSIM = 20001  # t <= 40; the unforced field is nearly flat well before the old t <= 160
S_FACTOR = 4
T_FACTOR = 20


def dataset_path(out_dir: str, ic_seed: int, noise_level: float) -> str:
    return os.path.join(out_dir, f"ic_seed_{ic_seed}_nl_{noise_level}.npz")


def noise_seed(ic_seed: int, sample_idx: int) -> int:
    """Noise seed for one sample. Deliberately independent of the noise level."""
    return 100 * ic_seed + sample_idx


def simulate_samples(ic_seed, D_values=D_VALUES, Lx=LX, L=L, dt=DT, Tsim=TSIM,
                     s_factor=S_FACTOR, t_factor=T_FACTOR):
    """Simulate one initial condition at each viscosity in `D_values`.

    Args:
        ic_seed: Seed for `burgers_sim.draw_ic`; the same initial condition is used
            for every viscosity, so the samples differ only in `D`.
        D_values: Per-sample viscosities.
        Lx: Fine-grid spatial resolution.
        L: Domain length.
        dt: Fine-grid time step.
        Tsim: Number of fine-grid time steps.
        s_factor: Spatial coarsening factor.
        t_factor: Temporal coarsening factor.

    Returns:
        Tuple of `(u_clean, meta)`, where `u_clean` has shape
        `(len(D_values), Lx // s_factor, Tsim // t_factor + 1)` and `meta` carries the
        coarse grid and the reproducibility metadata for the npz.
    """
    x = np.linspace(0, L, Lx)
    u0 = bs.draw_ic(x, L, ic_seed)

    fields, coarse = [], None
    for D in D_values:
        sim = bs.simulate(Lx=Lx, L=L, D=D, dt=dt, Tsim=Tsim, forcing_seed=None, ic_seed=ic_seed)
        coarse = bs.coarsen(sim, s_factor=s_factor, t_factor=t_factor)
        fields.append(coarse["u_coarse"])
        print(f"  ic_seed={ic_seed} D={D:<7g} field {coarse['u_coarse'].shape} "
              f"std(t=0)={np.std(coarse['u_coarse'][:, 0]):.4f} "
              f"std(t=end)={np.std(coarse['u_coarse'][:, -1]):.4f}")

    meta = {
        "D": np.asarray(D_values, float),
        "u0_coarse": u0[np.arange(0, Lx, s_factor)],
        "x_coarse": coarse["x_coarse"],
        "dxc": coarse["dxc"], "dtc": coarse["dtc"],
        "ic_seed": ic_seed,
        "L": L, "Lx": Lx, "dt": dt, "Tsim": Tsim,
        "s_factor": s_factor, "t_factor": t_factor,
        "forced": False,
        "A": np.zeros(20), "w": np.zeros(20), "phi": np.zeros(20), "l": np.zeros(20), "N": 20,
    }
    return np.stack(fields), meta


def write_dataset(out_dir, ic_seed, noise_level, u_clean, meta):
    """Add noise to the clean samples and write one npz."""
    seeds = [noise_seed(ic_seed, j) for j in range(u_clean.shape[0])]
    if noise_level in (0, 0.0):
        u_noisy = u_clean.copy()
    else:
        u_noisy = np.stack([bs.add_noise(u_clean[j], noise_level, seed=s)
                            for j, s in enumerate(seeds)])
    path = dataset_path(out_dir, ic_seed, noise_level)
    np.savez_compressed(
        path,
        u_clean=u_clean, u_noisy=u_noisy,
        noise_level=noise_level, noise_seeds=np.asarray(seeds, int),
        **meta,
    )
    print(f"wrote {path}  {u_clean.shape}  nl={noise_level}")
    return path


def verify(out_dir=OUT_DIR, ic_seeds=IC_SEEDS, noise_levels=NOISE_LEVELS,
           D_values=D_VALUES, block_len=200):
    """Check the written datasets satisfy every invariant the benchmark relies on."""
    import load_data as ld  # same directory; only used for block_split

    bundles = {}
    for ic in ic_seeds:
        for nl in noise_levels:
            path = dataset_path(out_dir, ic, nl)
            assert os.path.exists(path), f"missing {path}"
            with np.load(path) as z:
                bundles[(ic, nl)] = {k: z[k] for k in z.files}

    n_s, n_x, n_t = len(D_values), None, None
    for (ic, nl), b in bundles.items():
        assert b["u_clean"].shape == b["u_noisy"].shape
        assert b["u_clean"].shape[0] == n_s, (ic, nl, b["u_clean"].shape)
        n_x, n_t = b["u_clean"].shape[1:]
        assert not bool(b["forced"]) and not b["A"].any()

    # (2) the 3 noise levels of one ic_seed share bitwise-identical clean data
    for ic in ic_seeds:
        ref = bundles[(ic, noise_levels[0])]
        for nl in noise_levels[1:]:
            b = bundles[(ic, nl)]
            for k in ("u_clean", "D", "u0_coarse", "x_coarse"):
                assert np.array_equal(ref[k], b[k]), f"{k} differs between nl={noise_levels[0]} and nl={nl} at ic_seed={ic}"

    # (3) D identical across all files; (4) ICs genuinely differ
    for b in bundles.values():
        assert np.array_equal(b["D"], np.asarray(D_values, float))
    ics = [bundles[(ic, noise_levels[0])]["u0_coarse"] for ic in ic_seeds]
    for a, c in zip(ics, ics[1:]):
        assert not np.allclose(a, c), "initial conditions are not distinct"

    # (5) noise amplitude and shared realisation
    print("\nnoise check (std(resid) / [nl * std(u_clean)], want ~1.0):")
    for ic in ic_seeds:
        for nl in noise_levels:
            b = bundles[(ic, nl)]
            resid = b["u_noisy"] - b["u_clean"]
            if nl in (0, 0.0):
                assert np.array_equal(b["u_noisy"], b["u_clean"])
                print(f"  ic={ic} nl={nl}: exact (clean)")
                continue
            ratios = [np.std(resid[j]) / (nl * np.std(b["u_clean"][j])) for j in range(n_s)]
            assert all(abs(r - 1.0) < 0.02 for r in ratios), ratios
            print(f"  ic={ic} nl={nl}: " + " ".join(f"{r:.3f}" for r in ratios))
    nz = [nl for nl in noise_levels if nl not in (0, 0.0)]
    for ic in ic_seeds:
        for a, c in zip(nz, nz[1:]):
            ra = bundles[(ic, a)]["u_noisy"] - bundles[(ic, a)]["u_clean"]
            rc = bundles[(ic, c)]["u_noisy"] - bundles[(ic, c)]["u_clean"]
            assert np.allclose(rc, ra * (c / a)), f"noise realisation differs between nl={a} and nl={c}"
    print(f"  noise realisation shared across {nz} (amplitude-scaled only)")

    # (6) physics: decay is monotone and faster at larger D
    dtc = float(bundles[(ic_seeds[0], noise_levels[0])]["dtc"])
    probes = [t for t in (0, 10, 20, 40) if int(round(t / dtc)) < n_t]
    print(f"\nfield std over time (rows = D, cols = t):\n  {'D':>7} " + " ".join(f"t={t:<7g}" for t in probes))
    for ic in ic_seeds:
        b = bundles[(ic, noise_levels[0])]
        curves = []
        for j, D in enumerate(D_values):
            stds = [float(np.std(b["u_clean"][j][:, int(round(t / dtc))])) for t in probes]
            curves.append(stds)
            print(f"  ic={ic} {D:>7g} " + " ".join(f"{s:<9.4f}" for s in stds))
            assert all(x >= y for x, y in zip(stds, stds[1:])), f"non-monotone decay at ic={ic} D={D}"
        assert all(a[-1] > c[-1] for a, c in zip(curves, curves[1:])), \
            f"decay is not monotone in D at ic_seed={ic}"

    # (7) block split on the new time axis
    tr, te = ld.block_split(n_t, block_len=block_len)
    print(f"\nblock_split({n_t}, {block_len}): {tr.size} train cols, {te.size} test cols, "
          f"{n_t - tr.size - te.size} orphaned")

    print(f"\nOK — {len(bundles)} files, {n_s} samples each, field ({n_x}, {n_t})")


def main(out_dir=OUT_DIR, ic_seeds=IC_SEEDS, D_values=D_VALUES, noise_levels=NOISE_LEVELS,
         Tsim=TSIM, s_factor=S_FACTOR, t_factor=T_FACTOR, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    for ic_seed in ic_seeds:
        paths = [dataset_path(out_dir, ic_seed, nl) for nl in noise_levels]
        if not overwrite and all(os.path.exists(p) for p in paths):
            print(f"ic_seed={ic_seed}: all noise levels present, skipping (use --overwrite)")
            continue
        u_clean, meta = simulate_samples(ic_seed, D_values=D_values, Tsim=Tsim,
                                         s_factor=s_factor, t_factor=t_factor)
        for nl in noise_levels:
            write_dataset(out_dir, ic_seed, nl, u_clean, meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--ic-seeds", type=int, nargs="+", default=list(IC_SEEDS))
    p.add_argument("--noise-levels", type=float, nargs="+", default=list(NOISE_LEVELS))
    p.add_argument("--D", type=float, nargs="+", default=list(D_VALUES))
    p.add_argument("--Tsim", type=int, default=TSIM)
    p.add_argument("--s-factor", type=int, default=S_FACTOR)
    p.add_argument("--t-factor", type=int, default=T_FACTOR)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check", action="store_true", help="only verify existing files")
    a = p.parse_args()

    if not a.check:
        main(out_dir=a.out_dir, ic_seeds=tuple(a.ic_seeds), D_values=tuple(a.D),
             noise_levels=tuple(a.noise_levels), Tsim=a.Tsim,
             s_factor=a.s_factor, t_factor=a.t_factor, overwrite=a.overwrite)
    verify(out_dir=a.out_dir, ic_seeds=tuple(a.ic_seeds),
           noise_levels=tuple(a.noise_levels), D_values=tuple(a.D))
