"""Regenerate burgers_clean.npz from scratch (Step 1 reproducibility).

Run:  python projects/burgers/data_loader/regenerate.py
Wall time ~40 s on CPU. Produces the same bundle load_data.py consumes.
"""

from __future__ import annotations

import os
import numpy as np

import burgers_sim as bs  # same directory


def main(out_path: str | None = None,
         s_factor: int = 4, t_factor: int = 20,
         noise_levels=(0.0, 0.01, 0.05, 0.1, 0.3),
         noise_seed: int = 0):
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "burgers_clean.npz")
    sim = bs.simulate()                                  # reference forced-Burgers fine field
    co = bs.coarsen(sim, s_factor=s_factor, t_factor=t_factor)
    uc = co["u_coarse"]
    noisy = {f"u_noisy_{nl}": bs.add_noise(uc, nl, seed=noise_seed)
             for nl in noise_levels if nl not in (0, 0.0)}
    np.savez_compressed(
        out_path,
        u_coarse=uc, x_coarse=co["x_coarse"],
        dxc=co["dxc"], dtc=co["dtc"], s_factor=co["s_factor"], t_factor=co["t_factor"],
        D=sim["D"], L=sim["L"], Lx=sim["Lx"], dt=sim["dt"], Tsim=sim["Tsim"],
        A=sim["A"], w=sim["w"], phi=sim["phi"], l=sim["l"], N=sim["N"], forcing_seed=0,
        noise_levels=np.array(list(noise_levels)),
        **noisy,
    )
    print(f"wrote {out_path}  field {uc.shape}  levels {list(noise_levels)}")


if __name__ == "__main__":
    main()
