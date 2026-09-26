#!/usr/bin/env bash
# One-time setup on Ubuntu/Debian (or any Linux with Python >= 3.10):
#   bash setup_ubuntu.sh
# Then run the pipeline (inside tmux/screen so it survives an SSH disconnect):
#   .venv/bin/python code/business_entity_resolution/run_pipeline.py
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found: sudo apt install python3 python3-venv python3-pip"; exit 1
fi
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python >= 3.10 required (found {sys.version.split()[0]}); install python3.10+ and rerun "
             f"with PYTHON=python3.x bash setup_ubuntu.sh")
EOF
if ! "$PY" -m venv --help >/dev/null 2>&1 || ! "$PY" -c "import ensurepip" >/dev/null 2>&1; then
  echo "The venv module is missing: sudo apt install python3-venv   (or python3.X-venv for your version)"; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt

if ! .venv/bin/python -c "import lightgbm" >/dev/null 2>&1; then
  echo "lightgbm failed to import -- it needs the OpenMP runtime: sudo apt install libgomp1"; exit 1
fi

if [ ! -f dataset/train/train_source1.tsv ]; then
  echo "WARNING: dataset/ not found. Copy the challenge dataset folder (train/ + test/ TSVs) to $(pwd)/dataset/"
fi

.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "code/business_entity_resolution/src")
import resources
print(resources.describe())
EOF

cat <<'EOF'

Setup done. Next:
  tmux new -s ber                      # optional, keeps the run alive if SSH drops
  .venv/bin/python code/business_entity_resolution/run_pipeline.py
Progress: output/logs/pipeline_status.json and output/logs/<step>.log
Scratch/spill defaults to data/scratch; point it at a big disk with
  export BER_SCRATCH_DIR=/path/with/150GB/free
EOF
