#!/bin/bash
# Run the mPFC_spikes EDGAR project on a Janelia GPU node.
#
# Usage (submit from a login node):
#   bsub -n 4 -gpu "num=1:mode=shared" -q gpu_l4 -W 60  -J mpfc_smoke -o <log> run_cluster.sh test
#   bsub -n 8 -gpu "num=1:mode=shared" -q gpu_l4 -W 720 -J mpfc_full  -o <log> run_cluster.sh run
#
# GPU MUST be mode=shared: scoring runs each program in a GPU subprocess while the
# engine also holds a CUDA context; exclusive_process would starve the children.
set -uo pipefail

MODE="${1:-run}"; shift || true

REPO=/groups/ahrens/home/ruttenv/python_packages/EDGAR_windowed_scoring
source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh
conda activate edgar_vmsr          # edgar_vmsr is editable-installed FROM this worktree
cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XDG_RUNTIME_DIR

echo "=== node: $(hostname)  $(date) ==="
echo "=== which python: $(which python) ==="
echo "=== which edgar:  $(which edgar) ==="

# Guard: fail loudly unless `edgar` resolves to THIS worktree (the framework
# apply_model_fn change + windowed scoring live here; the wrong checkout silently
# no-ops it).
python - <<'PY'
import edgar
print("edgar.__file__ =", edgar.__file__)
assert edgar.__file__.startswith(
    "/groups/ahrens/home/ruttenv/python_packages/EDGAR_windowed_scoring/"
), f"WRONG edgar: {edgar.__file__} (expected the windowed-scoring worktree; run pip install -e . from it)"
PY
[ $? -ne 0 ] && { echo "ABORT: edgar import guard failed"; exit 3; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "no nvidia-smi"
python -c "import jax; print('JAX_DEVICES', jax.devices())"

CONFIG=projects/mPFC_spikes/config.yaml

if [ "$MODE" = "test" ]; then
  # `edgar test` forces Gemini via TEST_OVERRIDES; append cheap-Claude overrides
  # (last-wins) so the smoke exercises the real Anthropic path, and shrink the
  # window so it finishes in minutes.
  echo "=== SMOKE TEST start $(date +%H:%M:%S) ==="
  python -m edgar.cli test "$CONFIG" \
    --project_params.anchors_per_neuron=1000 \
    --llms.model_llm=claude-haiku-4-5 \
    --llms.param_est_llm=claude-haiku-4-5 \
    --llms.jax_model_translator_llm=claude-haiku-4-5 \
    "$@"
  code=$?
  echo "=== SMOKE TEST exit=$code end $(date +%H:%M:%S) ==="
else
  echo "=== FULL RUN start $(date +%H:%M:%S) ==="
  python -m edgar.cli run "$CONFIG" "$@"
  code=$?
  echo "=== FULL RUN exit=$code end $(date +%H:%M:%S) ==="
fi
exit $code
