#!/bin/bash
# vLLM oracles for the long-generation CB study (runs/3k3k_2026-11-12).
# 3072 prompt -> 3072 new tokens, batch 4 (C=4), Gumbel T=0.7/PP=0.9, both models.
# Same env/flags convention as r_vllm_cb_oracles.sh.
set -uo pipefail
cd /u/msajid/projects/DeepSpeedExamples/training/opsd

VLLM_PY=/u/msajid/vllm_env/bin/python
OUT=runs/3k3k_2026-11-12/vllm_oracles
LOGS=$OUT/logs
mkdir -p "$LOGS"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/u/msajid/cuda-12.9-shim
export CPATH=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages/nvidia/cu13/include:/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages/nvidia/cu13/include/cccl
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH
export HF_HOME=/work/nvme/bcjw/msajid/hf_cache
export VLLM_CACHE_ROOT=/u/msajid/vllm_cache
export TMPDIR=/u/msajid/tmp
export PYTHONUNBUFFERED=1
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

FAILED=()
run() {  # run NAME COMMAND...
  local name="$1"; shift
  echo "=== [run] $name  $(date '+%H:%M:%S')"
  local t0=$(date +%s)
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "=== [ok]   $name  wall=$((t1 - t0))s  $(date '+%H:%M:%S')"
  else
    echo "=== [FAIL] $name  rc=$rc  $(date '+%H:%M:%S')  see $LOGS/$name.log"
    FAILED+=("$name")
  fi
}

run vllm_4b_3k_c4 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 4 \
  --max-prompt-length 3072 --max-new-tokens 3072 --prefill-chunk-size 4096 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_4b_3k_c4.json"

run vllm_8b_3k_c4 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-8B --data data/longbench_20k.jsonl --batch-size 4 \
  --max-prompt-length 3072 --max-new-tokens 3072 --prefill-chunk-size 4096 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_8b_3k_c4.json"

echo "=== vllm 3k oracles done: ${#FAILED[@]} failures (${FAILED[*]:-none})"
