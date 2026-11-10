#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# One-step Qwen3-4B continuous-batch + graph-capture run on Delta.
# Usage (from a login node):
#   bash scripts/srun_qwen3_4b_continuous_graph.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPSD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPSD_ENV="${OPSD_ENV:-$HOME/opsd_env.sh}"
export OPSD_DIR OPSD_ENV

# One task only: `deepspeed --num_gpus 1` spawns the single rank.
# Do NOT use --ntasks-per-node=16 here: that runs 16 copies of DeepSpeed
# on the same node, and they all race for --master_port 29500 (EADDRINUSE).
srun -A bcjw-delta-gpu --time=04:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=128g \
  bash -lc 'set -euo pipefail
source "$OPSD_ENV"
module load anaconda3_gpu 2>/dev/null || true
conda activate "$WORK/conda/opsd"
cd "$OPSD_DIR"
mkdir -p runs
deepspeed --num_gpus 1 --master_port 29500 main.py \
  --config configs/baseline_qwen3_4b_continuous_graph.json \
  2>&1 | tee runs/qwen3_4b_continuous_graph.stdout.log'
