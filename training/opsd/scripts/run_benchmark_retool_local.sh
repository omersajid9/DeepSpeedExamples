#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Steady-state continuous-batch + graph rollout benchmark on ReTool-SFT prompts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPSD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /workspace/opsd_env.sh
cd "$OPSD_DIR"
mkdir -p runs

python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B \
  --data data/retool_prompts.jsonl \
  --batch-size 4 \
  --continuous-batch-size 2 \
  --max-prompt-length 2048 \
  --max-new-tokens 512 \
  --warmup 1 \
  --iterations 3 \
  --output runs/qwen3_4b_retool_continuous_graph.json
