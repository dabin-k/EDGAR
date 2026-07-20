# Vendored STENCIL-NET source (verbatim)

`network.py`, `timestepping.py`, `__init__.py` are copied **unmodified** from
github.com/mosaic-group/STENCIL-NET (Maddu et al., Sci. Rep. 13:12787, 2023).

- `network.py::MLPConv` — the stencil MLP `N_θ`. `_preprocess` builds a `fs`-point
  stencil per grid location via `torch.roll` (periodic), then a plain MLP maps
  `fs → 64 → 64 → 64 → 1`. Optional `noise` is the latent noise field (a learnable
  `nn.Parameter`) used for the noisy case.
- `timestepping.py` — the RK3 propagators. `forward_rk3_error` / `backward_rk3_error`
  roll the net forward/backward `m` steps with decaying weights `wd`, supporting known
  forcing terms (`fc*`) and latent noise. The `*_tvd_error` variants are the TVD-RK3
  used for the noisy (denoising) runs.

Only `runner.py` (one directory up) is ours — it wires their code to our shared
Burgers data and metric. Do not edit the files in this directory; re-copy from the
reference if they need updating.
