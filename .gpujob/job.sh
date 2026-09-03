#!/usr/bin/env bash
# Runs inside a Slurm allocation on one H200. SLURM_SUBMIT_DIR, not $0 --
# Slurm copies this into its own spool dir, so anything shipped alongside it
# (repo checkout, sweep.py) is invisible from $(dirname $0).
set -uo pipefail
HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }

PY=$HOME/arxiv-jax/venv/bin/python
export PYTHONPATH="$HERE/repo:$HERE/repo/src:${PYTHONPATH:-}"

echo "== nvidia-smi =="
nvidia-smi -L
echo "== python =="
$PY -c "import jax; print('jax', jax.__version__); print(jax.devices())"

MODE=${MODE:-sweep}

if [ "$MODE" = "sweep" ]; then
  N_SEEDS=${N_SEEDS:-20}
  OFFSET=${OFFSET:-0}
  $PY "$HERE/sweep.py" "$N_SEEDS" "$HERE/results/gpu_sweep.json" "$OFFSET"
elif [ "$MODE" = "pytest" ]; then
  cd "$HERE/repo"
  OUT_NAME=${OUT_NAME:-pytest_out.txt}
  $PY -m pytest tests/ -q > "$HERE/results/$OUT_NAME" 2>&1
  echo "pytest exit $?"
  tail -50 "$HERE/results/$OUT_NAME"
elif [ "$MODE" = "identical_input" ]; then
  OUT_NAME=${OUT_NAME:-gpu_identical.json}
  $PY "$HERE/identical_input.py" "$HERE/upstream.npz" "$HERE/results/$OUT_NAME"
fi
