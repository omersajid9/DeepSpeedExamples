#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
set -u
cd /workspace/DeepSpeedExamples/training/opsd
source /workspace/opsd_env.sh 2>/dev/null
export HF_HOME=/dev/shm/hf_cache
export PYTHONDONTWRITEBYTECODE=1
for round in 1 2 3; do
    echo "[watchdog] round $round"
    python scripts/run_rollout_matrix.py --resume --tasks retool
    python - <<'PY'
import json
import sys
from pathlib import Path
raw = Path("runs/rollout_matrix_2026-11-10/raw")
valid = []
for path in raw.glob("*.json"):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    if data.get("iterations"):
        valid.append(path.name)
print(f"[watchdog] valid cells {len(valid)}/23")
sys.exit(0 if len(valid) >= 23 else 1)
PY
    if [ $? -eq 0 ]; then
        break
    fi
    sleep 30
done
