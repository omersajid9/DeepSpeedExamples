#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Steady-state continuous-batch + graph rollout benchmark on LongBench multifieldqa_en.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPSD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /workspace/opsd_env.sh
cd "$OPSD_DIR"
mkdir -p runs

python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B \
  --data data/longbench_mfqa.jsonl \
  --batch-size 2 \
  --continuous-batch-size 1 \
  --max-prompt-length 8192 \
  --max-new-tokens 256 \
  --warmup 1 \
  --iterations 3 \
  --output runs/qwen3_4b_longbench_continuous_graph.json
