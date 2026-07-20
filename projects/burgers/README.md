# Burgers benchmark (STENCIL-NET vs SINDy vs EDGAR)

Benchmark for quantifying EDGAR on near-equilibrium, low-but-nonzero-noise systems.
Full plan: `journal/burgers_benchmark_plan.md`.

Layout:
- `data_loader/` — forced-Burgers generator (physics vendored verbatim from
  mosaic-group/STENCIL-NET), shared data-access layer, and the shared forecast-MSE
  metric. `burgers_clean.npz` is **git-ignored**; regenerate it per machine.
- `bench/stencilnet/` — Step 2. STENCIL-NET in isolation. `stencilnet_src/` holds
  their `MLPConv` + RK3 time-stepping copied unmodified (see its PROVENANCE.md);
  `runner.py`/`sweep.py` are the only original code and just wire their model to our
  data. Device auto-detects (CUDA > MPS > CPU).

## Setup on a new machine

```bash
# 1. Python deps (GPU box: install the CUDA build of torch for your CUDA version)
pip install numpy scipy matplotlib pysindy
pip install torch            # or the cu121/cu118 wheel from pytorch.org for your GPU

# 2. Regenerate the data bundle (~40 s; writes data_loader/burgers_clean.npz)
python projects/burgers/data_loader/regenerate.py
```

The STENCIL-NET model/physics are **vendored** into this repo, so you do NOT need to
clone github.com/mosaic-group/STENCIL-NET to run the benchmark. Clone it only if you
want their original notebooks / pretrained `.pth` models for reference:

```bash
git clone https://github.com/mosaic-group/STENCIL-NET.git
```

## Run Step 2 (STENCIL-NET)

```bash
cd projects/burgers/bench/stencilnet
# single noise level
PYTHONPATH=. python runner.py --noise 0.1 --epochs 30000
# full noise sweep (writes results/stencilnet_sweep.json)
PYTHONPATH=. python sweep.py --epochs 30000 --levels 0.0 0.01 0.05 0.1 0.3
```

Reference recipe (ForcedBurgers* notebooks): Adam lr=1e-3, 30k epochs, m=4 rollout,
decay 0.9, l_wd=1e-7, l_n=1e-5 (noisy only), train on first 1001 coarse time steps.
Clean levels use forcing-aware RK3 loss; noisy levels use TVD-RK3 + noise-as-latent.
