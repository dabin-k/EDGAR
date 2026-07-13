"""Verify projects/synthetic_data_v2 without needing the edgar-side `evaluate` change.

Exercises the project's own contract: loader shapes/splits, the autoregressive
wrapper, the loss, both seed programs, and the image feedback plot.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJ = Path("projects/synthetic_data_v2")
sys.path.insert(0, str(PROJ / "evaluate"))
sys.path.insert(0, str(PROJ / "data_loader"))
sys.path.insert(0, str(PROJ / "image_feedback"))

from evaluate import MAX_LENGTH, ROLLOUT_STEPS, evaluate  # noqa: E402
from load_data import load_data, loss_fn  # noqa: E402
from plot import plot_model_fits  # noqa: E402


def load_seed(n):
    ns = {}
    exec((PROJ / "seed_programs" / f"model{n}.py").read_text(), ns)
    exec((PROJ / "seed_programs" / f"param_est{n}.py").read_text(), ns)
    return ns["model"], ns["parameter_estimator"]


def batch_params(param_est, data):
    n = next(iter(data.values())).shape[0]
    per = [param_est({k: np.asarray(v)[i] for k, v in data.items()}) for i in range(n)]
    return (
        {k: jnp.stack([jnp.asarray(p[k]) for p in per]) for k in per[0]}
        if per[0]
        else {}
    )


(disc_train, disc_test), (val_train, val_test), X_eval = load_data("unused")

print("── loader ──")
for name, d in [
    ("disc_train", disc_train),
    ("disc_test", disc_test),
    ("val_train", val_train),
    ("val_test", val_test),
]:
    print(f"  {name:<11} x {tuple(d['x'].shape)}  (samples, blocks, cells, time)")
print(f"  X_eval      x {tuple(X_eval['x'].shape)}  idx {X_eval['_sample_indices']}")

print("\n── seeds ──")
for n in (1, 2):
    model, param_est = load_seed(n)
    params = batch_params(param_est, disc_train)
    preds, targets = evaluate(model, disc_train, params)
    train_loss = loss_fn(preds, targets)

    preds_te, targets_te = evaluate(model, disc_test, params)
    test_loss = loss_fn(preds_te, targets_te)

    fitted = {k: float(np.mean(v)) for k, v in params.items()}
    print(
        f"  model{n}: preds {tuple(preds.shape)}  "
        f"train {float(train_loss.mean()):.5f}  test {float(test_loss.mean()):.5f}  {fitted}"
    )

print("\n── the wrapper cannot see the future ──")
# A model that returns its most recent input must reproduce persistence exactly; a
# model that could see x(t+1) would score ~0. Check the window really ends at t.
model1, _ = load_seed(1)
preds, targets = evaluate(model1, disc_test, {})
step1 = float(jnp.mean((preds[:, :, :, 0] - targets[:, :, :, 0]) ** 2))
step3 = float(jnp.mean((preds[:, :, :, 2] - targets[:, :, :, 2]) ** 2))
print(f"  persistence error at h=1: {step1:.5f}   at h=3: {step3:.5f}   (must grow)")
assert step3 > step1 > 1e-4, "window alignment is wrong — the model can see its target"

n_starts = preds.shape[2]
expected = disc_test["x"].shape[-1] - ROLLOUT_STEPS - (MAX_LENGTH - 1)
print(f"  starts per block: {n_starts} (expected {expected})")
assert n_starts == expected

print("\n── gradient flows through the rollout ──")
model2, param_est2 = load_seed(2)
params2 = batch_params(param_est2, disc_train)


def total(p):
    preds, targets = evaluate(model2, disc_train, p)
    return jnp.mean(loss_fn(preds, targets))


g = jax.grad(total)(params2)
print(f"  d(loss)/d(decay) = {np.asarray(g['decay'])}")
assert np.all(np.isfinite(g["decay"])) and np.any(g["decay"] != 0)

print("\n── image feedback ──")


class FakeProgram:
    def __init__(self, n, name):
        self.n, self.name = n, name

    def compile_model(self):
        return load_seed(self.n)[0]


progs = [FakeProgram(1, "Persistence"), FakeProgram(2, "Leaky decay")]
prog_params = [{}, batch_params(load_seed(2)[1], disc_train)]
out = "journal/2026-07-13_synthetic_data_v2_feedback.png"
plot_model_fits(
    disc_train,
    progs,
    save_path=out,
    losses=[0.047, 0.047],
    params=prog_params,
)
print(f"  wrote {out}")
print("\nall checks passed")
