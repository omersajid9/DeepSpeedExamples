#!/usr/bin/env bash
# B300 SMOKE TEST: 6 cells — qwen35_4b b02, levels A,B,C,D,E + vllm reference.
#
# Purpose: force every JIT build for sm_103 (fused_glu / segki kernels,
# flashinfer decode, dskernels, torch inductor) and exercise every engine
# path before committing to the full matrix. Expect ~30-60 min, most of it
# JIT cold start + 4B model download on first run.
#
# If all 6 cells pass (check $OUT/raw/*.json), run b300/run_matrix.sh.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source b300/b300.env
OUT="${PFUSE_B300_OUT:-runs/paged_fused_ablation_2026-11-15_b300}"
mkdir -p "$OUT"
{
  echo "=== B300 smoke: $(hostname) $(date -Is) ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$OUT/start.log"
python3 scripts/run_paged_fused_ablation.py \
  --out "$OUT" --cuda-arch auto --only "qwen35_4b,2" \
  2>&1 | tee "$OUT/smoke.log"
echo
echo "smoke finished — results in $OUT (raw/ + summary), logs in $OUT/logs/"
