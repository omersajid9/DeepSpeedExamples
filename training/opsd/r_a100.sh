#!/bin/bash
# A100 re-run of the clean rollout optimization ablation.
# Reference A40 results: runs/optimization_ablation.md (2026-11-10, A40 48 GB).
# Outputs: runs/a100_2026-11-11/<cell>.json  (same cell names as the A40 run).
set -uo pipefail
cd "$(dirname "$0")"

PY=/work/nvme/bcjw/msajid/conda/opsd/bin/python
OUT=runs/a100_2026-11-11
LOGS=$OUT/logs
mkdir -p "$LOGS"

export CUDA_VISIBLE_DEVICES=0
SP=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages
export CUDA_HOME=/u/msajid/cuda-12.9-shim # shim: node's 12.9 nvcc + pip cudart lib (partial node toolkit)
export CPATH=$SP/nvidia/cu13/include:$SP/nvidia/cu13/include/cccl  # full CUDA headers from pip
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc    # bypass Cray cc wrapper (injects 13.2 -I flags)
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH
export HF_HOME=/u/msajid/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache
export TMPDIR=/u/msajid/tmp_bench
export PYTHONUNBUFFERED=1
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

# bench NAME P N T PP [DATA] [--flag ...]
# Fixed: batch 1, continuous 1, warmup 1, 1 measured iteration, seed 0
# (matches every A40 clean-ablation cell; prefix cells override batch/C).
bench() {
  local name="$1" P="$2" N="$3" T="$4" PP="$5"; shift 5
  local data="data/longbench_20k.jsonl"
  case "$name" in
    prefix_*) data="data/prefix_reuse_6k.jsonl";;
  esac
  echo "=== [bench] $name  P=$P N=$N T=$T PP=$PP data=$data $*  $(date '+%H:%M:%S')"
  local t0=$(date +%s)
  "$PY" benchmarks/bench_continuous_graph.py \
    --data "$data" \
    --batch-size 1 --continuous-batch-size 1 --warmup 1 --iterations 1 \
    --max-prompt-length "$P" --max-new-tokens "$N" \
    --temperature "$T" --top-p "$PP" \
    --output "$OUT/$name.json" "$@" \
    > "$LOGS/$name.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  if [ $rc -eq 0 ] && [ -f "$OUT/$name.json" ]; then
    echo "=== [ok]   $name  wall=$((t1 - t0))s  $(date '+%H:%M:%S')"
  else
    echo "=== [FAIL] $name  rc=$rc  $(date '+%H:%M:%S')  see $LOGS/$name.log"
    FAILED+=("$name")
  fi
}

FAILED=()

have_flashinfer=0
if "$PY" -c "import flashinfer" 2>/dev/null; then have_flashinfer=1; fi
echo "flashinfer available: $have_flashinfer"

# ---------------- 4B 6k clean ablation (LongBench truncated to 6144) ----------
P=6144; N=16; T=0.7; PP=0.9
bench ablation_6k_dense_eager_nochunk         $P $N $T $PP --no-graph-capture
bench ablation_6k_dense_graph_nochunk         $P $N $T $PP
bench ablation_6k_paged_decode_graph_nochunk  $P $N $T $PP --use-paged-cache
bench ablation_6k_paged_prefill_graph_nochunk $P $N $T $PP --use-paged-cache --enable-paged-prefill
bench compile_6k_graph                        $P $N $T $PP --use-paged-cache --enable-paged-prefill --torch-compile
bench compiled_sampler_6k                     $P $N $T $PP --use-paged-cache --enable-paged-prefill --compiled-sampler
if [ $have_flashinfer -eq 1 ]; then
  bench flashinfer_6k_baseline                $P $N $T $PP --use-paged-cache --enable-paged-prefill
  bench flashinfer_6k                         $P $N $T $PP --use-paged-cache --enable-paged-prefill --flashinfer-prefill
else
  echo "=== [skip] flashinfer_6k* (flashinfer not installed)"
fi
# 6k chunk-penalty rows
bench chunk_paged_prefill_6k                  $P $N $T $PP --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill
bench chunk_dense_staging_6k                  $P $N $T $PP --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache

# ---------------- 4B 20k clean ablation (LongBench 20480, greedy) -------------
P=20480; N=8; T=0.0; PP=1.0
bench ablation_20k_dense_eager_nochunk        $P $N $T $PP --no-graph-capture
bench ablation_20k_dense_graph_nochunk        $P $N $T $PP
bench ablation_20k_paged_decode_eager_nochunk $P $N $T $PP --use-paged-cache --no-graph-capture
bench ablation_20k_paged_decode_graph_nochunk $P $N $T $PP --use-paged-cache
bench ablation_20k_paged_prefill_graph_nochunk $P $N $T $PP --use-paged-cache --enable-paged-prefill
bench compile_20k_graph                       $P $N $T $PP --use-paged-cache --enable-paged-prefill --torch-compile
if [ $have_flashinfer -eq 1 ]; then
  bench flashinfer_20k                        $P $N $T $PP --use-paged-cache --enable-paged-prefill --flashinfer-prefill
else
  echo "=== [skip] flashinfer_20k (flashinfer not installed)"
fi
bench chunk_paged_prefill_20k_greedy          $P $N $T $PP --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill

# ---------------- prefix reuse (6k, two identical prompts, batch 2) ----------
# --enable-paged-prefill matches the A40 effective config: on A40 these cells ran
# before paged prefill became opt-in, when --use-paged-cache alone took the paged
# prefill path. Without the flag the A100 cells fell back to dense-staging chunks.
bench prefix_no_reuse 6144 8 0.0 0.9 \
  --batch-size 2 --continuous-batch-size 2 --enable-chunked-prefill \
  --prefill-chunk-size 512 --no-graph-capture --use-paged-cache --enable-paged-prefill
bench prefix_reuse 6144 8 0.0 0.9 \
  --batch-size 2 --continuous-batch-size 2 --enable-chunked-prefill \
  --prefill-chunk-size 512 --no-graph-capture --use-paged-cache --enable-paged-prefill \
  --enable-prefix-reuse

# ---------------- vLLM reference (separate env, optional) --------------------
# VLLM_PY must point at a venv with vllm 0.33.0 (like the A40 /workspace/vllm_env).
if [ -n "${VLLM_PY:-}" ] && [ -x "$VLLM_PY" ]; then
  export VLLM_CACHE_ROOT=/u/msajid/vllm_cache
  mkdir -p "$VLLM_CACHE_ROOT"
  echo "=== [vllm] $VLLM_PY"
  $VLLM_PY benchmarks/bench_vllm_offline.py \
    --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
    --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
    --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
    --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 \
    --output "$OUT/vllm_6k_nochunk.json" > "$LOGS/vllm_6k_nochunk.log" 2>&1 \
    && echo "=== [ok] vllm_6k_nochunk" || { echo "=== [FAIL] vllm_6k_nochunk"; FAILED+=("vllm_6k_nochunk"); }
  $VLLM_PY benchmarks/bench_vllm_offline.py \
    --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
    --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
    --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
    --gpu-memory-utilization 0.75 --kv-cache-memory-gib 3.3 \
    --output "$OUT/vllm_20k_nochunk_greedy.json" > "$LOGS/vllm_20k_nochunk_greedy.log" 2>&1 \
    && echo "=== [ok] vllm_20k_nochunk_greedy" || { echo "=== [FAIL] vllm_20k_nochunk_greedy"; FAILED+=("vllm_20k_nochunk_greedy"); }
else
  echo "=== [skip] vllm reference (VLLM_PY not set)"
fi

echo
echo "================ summary ================"
echo "cells: $(ls "$OUT"/*.json 2>/dev/null | wc -l) wrote, ${#FAILED[@]} failed: ${FAILED[*]:-none}"
exit ${#FAILED[@]}
