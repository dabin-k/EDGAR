#!/usr/bin/env python
"""Generate diagnostic figures for oscillator_ss without running full pipeline.

Produces three sets of figures under
`projects/oscillator_ss/figures/`:

    raw_data.png     : the noisy drifting-freq oscillator trajectories
    P<i>_init.png    : one-step predictions using param_estimator init params
    P<i>_final.png   : one-step predictions after 500 GD iterations

Use this when you want to inspect fits without waiting for a real `edgar run`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from projects.oscillator_ss.data_loader.load_data import (
    load_data, apply_model, loss_fn, WARMUP_STEPS,
)
from edgar.scoring.scoring import _optimize


OUTPUT_DIR = REPO / "projects" / "oscillator_ss" / "figures"


def _numpy_to_jax_source(src: str) -> str:
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def _load_seed(i: int):
    src = (REPO / "projects" / "oscillator_ss" / "seed_programs" / f"model{i}.py").read_text()
    pe_src = (REPO / "projects" / "oscillator_ss" / "seed_programs" / f"param_est{i}.py").read_text()
    ns_m = {}
    exec(_numpy_to_jax_source(src), ns_m)
    ns_p = {}
    exec(pe_src, ns_p)
    return ns_m["model"], ns_p["parameter_estimator"]


def plot_raw_data(y_disc: np.ndarray, save_path: Path, max_show: int = 4, window: int = 400):
    n_samples, T = y_disc.shape
    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)
    T_show = int(min(window, T))
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.4 * n_show), squeeze=False)
    for row, s in enumerate(show_idx):
        ax = axes[row, 0]
        ax.plot(np.arange(T_show), y_disc[s, :T_show], "k-", lw=0.7)
        ax.set_ylabel(f"traj {s}")
        if row == 0:
            ax.set_title(f"Raw noisy drifting-frequency oscillator (first {T_show} of T={T} bins, "
                         f"{n_show} of {n_samples} discover trajectories)")
        if row == n_show - 1:
            ax.set_xlabel("time bin")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_one_step_fit(y: np.ndarray, model_fn, params, title: str, save_path: Path,
                      max_show: int = 4, window: int = 400):
    n_samples, T = y.shape
    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)
    T_show = int(min(window, T - 1))
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.6 * n_show), squeeze=False)
    for row, s in enumerate(show_idx):
        ax = axes[row, 0]
        sample_data = {"y": jnp.asarray(y[s:s + 1])}
        sample_params = {k: jnp.asarray(np.asarray(v)[s:s + 1]) for k, v in params.items()}
        out = np.asarray(apply_model(model_fn, sample_data, sample_params))
        means = out[0, :T_show, 0]
        ax.plot(np.arange(T_show + 1), y[s, :T_show + 1], "k-", lw=0.6, alpha=0.5, label="true")
        ax.plot(np.arange(1, T_show + 1), means, color="tab:orange", lw=0.9, label="mean")
        ax.axvspan(0, WARMUP_STEPS, color="0.9", alpha=0.6, label=f"warmup (skipped in loss)")
        ax.set_ylabel(f"traj {s}")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(title)
        if row == n_show - 1:
            ax.set_xlabel("time bin")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (disc_tr, disc_te), _, _ = load_data()
    y_disc = np.asarray(disc_tr["y"])
    n_disc = y_disc.shape[0]

    raw_path = OUTPUT_DIR / "raw_data.png"
    plot_raw_data(y_disc, raw_path)
    print(f"  wrote {raw_path.relative_to(REPO)}")

    for i in [1, 2, 3, 4]:
        print(f"[seed {i}]")
        model_fn, pe_fn = _load_seed(i)
        per_sample = [pe_fn({"y": y_disc[j]}) for j in range(n_disc)]
        params_init = {
            k: jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]
        }
        init_loss = float(jnp.mean(loss_fn(apply_model(model_fn, disc_te, params_init), disc_te)))

        params_final = _optimize(
            model_fn, loss_fn, params_init, disc_tr,
            gd_config={"max_iter": 500, "learning_rate": 0.005, "gradient_clip_norm": 5.0},
            apply_model_fn=apply_model,
        )
        final_loss = float(jnp.mean(loss_fn(apply_model(model_fn, disc_te, params_final), disc_te)))
        print(f"  init loss = {init_loss:+.4f}   final loss = {final_loss:+.4f}   delta = {init_loss - final_loss:+.4f}")

        init_path = OUTPUT_DIR / f"P{i}_init.png"
        plot_one_step_fit(y_disc, model_fn, params_init,
                          f"seed{i} initial params — NLL={init_loss:+.4f}", init_path)
        print(f"  wrote {init_path.relative_to(REPO)}")

        final_path = OUTPUT_DIR / f"P{i}_final.png"
        plot_one_step_fit(y_disc, model_fn, params_final,
                          f"seed{i} after 500 GD steps — NLL={final_loss:+.4f}", final_path)
        print(f"  wrote {final_path.relative_to(REPO)}")

    print(f"\nAll figures under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
