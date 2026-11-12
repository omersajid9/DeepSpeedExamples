#!/bin/bash
# vLLM oracles at continuous-batching width for the CB ablation (2026-11-12).
#
# The CB study (runs/cb_ablation_2026-11-12) has no vLLM cell; these run the
# vLLM offline engine at the same concurrency as the CB levels (C=4 for 6k,
# C=2 for 20k) so the "vLLM = 1.0" figure set can be built on the same
# 6-step progression (A_eager ... F_flashinfer_sampler) instead of the C=1
# A100 ladder.
#
# Same env/flags convention as r_a100_postopt.sh vLLM section:
#   6k: P=6144 N=16 T=0.7 PP=0.9, prefill chunk 8192
#   20k: P=20480 N=8 T=0.0 PP=1.0, prefill chunk 20488
# kv-cache left to vLLM auto-sizing (0), gpu-memory-utilization 0.85.
set -uo pipefail
cd /u/msajid/projects/DeepSpeedExamples/training/opsd

VLLM_PY=/u/msajid/vllm_env/bin/python
OUT=runs/cb_ablation_2026-11-12/vllm_oracles
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

run vllm_4b_6k_c4 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 4 \
  --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_4b_6k_c4.json"

run vllm_4b_20k_c2 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 2 \
  --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_4b_20k_c2.json"

run vllm_8b_6k_c4 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-8B --data data/longbench_20k.jsonl --batch-size 4 \
  --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_8b_6k_c4.json"

run vllm_8b_20k_c2 "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-8B --data data/longbench_20k.jsonl --batch-size 2 \
  --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-gib 0 \
  --output "$OUT/vllm_8b_20k_c2.json"

echo "=== vllm cb-width oracles done: ${#FAILED[@]} failures (${FAILED[*]:-none})"
