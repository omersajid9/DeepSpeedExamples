#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# One-step Qwen3-4B LongBench continuous-batch + graph-capture run on one local GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPSD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-configs/baseline_qwen3_4b_longbench_continuous_graph.json}"

source /workspace/opsd_env.sh
cd "$OPSD_DIR"
mkdir -p runs

deepspeed --num_gpus 1 --master_port 29500 main.py \
  --config "$CONFIG" \
  2>&1 | tee "runs/$(basename "$CONFIG" .json).stdout.log"
