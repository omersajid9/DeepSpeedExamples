#!/usr/bin/env bash
# B300 FULL MATRIX: all 60 cells
#   models (qwen35_4b, qwen35_9b) x batches (2,4,8,16,32) x levels
#   (A,B,C,D,E + vllm reference), 3072-token context -> 1024-token decode.
#
# Resumable: cells already present in $OUT/raw are skipped, so re-running
# after an interruption (or a crash) continues where it stopped. The 6
# smoke cells from run_smoke.sh count toward the 60.
#
# Run it under nohup / tmux / screen if your session might drop:
#   nohup bash b300/run_matrix.sh > b300/matrix.out 2>&1 &
# Estimated wall time on a B300: ~2-4 h (dominated by per-cell model loads).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source b300/b300.env
OUT="${PFUSE_B300_OUT:-runs/paged_fused_ablation_2026-11-15_b300}"
mkdir -p "$OUT"
{
  echo "=== B300 matrix: $(hostname) $(date -Is) ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$OUT/start.log"
python3 scripts/run_paged_fused_ablation.py \
  --out "$OUT" --cuda-arch auto \
  2>&1 | tee "$OUT/matrix.log"
echo
echo "matrix finished — results in $OUT (raw/ + summary md/csv), logs in $OUT/logs/"
echo "fetch back to Delta: see b300/README_B300.md"
