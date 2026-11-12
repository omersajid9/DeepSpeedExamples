#!/bin/bash
# A100 post-optimization remeasure + correctness + prefix re-run + fresh vLLM oracles.
set -uo pipefail
cd /u/msajid/projects/DeepSpeedExamples/training/opsd

PY=/work/nvme/bcjw/msajid/conda/opsd/bin/python
VLLM_PY=/u/msajid/vllm_env/bin/python
OUT=runs/a100_2026-11-11
LOGS=$OUT/logs
mkdir -p "$LOGS"

export CUDA_VISIBLE_DEVICES=0
SP=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages
export CUDA_HOME=/u/msajid/cuda-12.9-shim
export CPATH=$SP/nvidia/cu13/include:$SP/nvidia/cu13/include/cccl
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH
export HF_HOME=/u/msajid/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache
export TMPDIR=/u/msajid/tmp_bench
export PYTHONUNBUFFERED=1
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export VLLM_CACHE_ROOT=/u/msajid/vllm_cache
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

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

DS_COMMON=(--batch-size 1 --continuous-batch-size 1 --warmup 1 --iterations 3 --seed 0)

# ---- 1. official remeasure: old best vs new best, 3 iterations each ----
run remeasure_oldbest_6k  "$PY" benchmarks/bench_continuous_graph.py \
  --data data/longbench_20k.jsonl "${DS_COMMON[@]}" \
  --max-prompt-length 6144 --max-new-tokens 16 --temperature 0.7 --top-p 0.9 \
  --use-paged-cache --enable-paged-prefill --torch-compile \
  --output "$OUT/remeasure_oldbest_6k.json"
run remeasure_oldbest_20k "$PY" benchmarks/bench_continuous_graph.py \
  --data data/longbench_20k.jsonl "${DS_COMMON[@]}" \
  --max-prompt-length 20480 --max-new-tokens 8 --temperature 0.0 --top-p 1.0 \
  --use-paged-cache --enable-paged-prefill --torch-compile \
  --output "$OUT/remeasure_oldbest_20k.json"
run remeasure_newbest_6k  "$PY" benchmarks/bench_continuous_graph.py \
  --data data/longbench_20k.jsonl "${DS_COMMON[@]}" \
  --max-prompt-length 6144 --max-new-tokens 16 --temperature 0.7 --top-p 0.9 \
  --use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode \
  --output "$OUT/remeasure_newbest_6k.json"
run remeasure_newbest_20k "$PY" benchmarks/bench_continuous_graph.py \
  --data data/longbench_20k.jsonl "${DS_COMMON[@]}" \
  --max-prompt-length 20480 --max-new-tokens 8 --temperature 0.0 --top-p 1.0 \
  --use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode \
  --output "$OUT/remeasure_newbest_20k.json"

# ---- 2. prefix re-run with --enable-paged-prefill (A40 effective config) ----
run prefix_no_reuse "$PY" benchmarks/bench_continuous_graph.py \
  --data data/prefix_reuse_6k.jsonl \
  --batch-size 2 --continuous-batch-size 2 --warmup 1 --iterations 1 --seed 0 \
  --max-prompt-length 6144 --max-new-tokens 8 --temperature 0.0 --top-p 0.9 \
  --enable-chunked-prefill --prefill-chunk-size 512 --no-graph-capture \
  --use-paged-cache --enable-paged-prefill \
  --output "$OUT/prefix_no_reuse.json"
run prefix_reuse "$PY" benchmarks/bench_continuous_graph.py \
  --data data/prefix_reuse_6k.jsonl \
  --batch-size 2 --continuous-batch-size 2 --warmup 1 --iterations 1 --seed 0 \
  --max-prompt-length 6144 --max-new-tokens 8 --temperature 0.0 --top-p 0.9 \
  --enable-chunked-prefill --prefill-chunk-size 512 --no-graph-capture \
  --use-paged-cache --enable-paged-prefill --enable-prefix-reuse \
  --output "$OUT/prefix_reuse.json"

# ---- 3. fresh vLLM oracles (back up the 11-11 ones first) ----
cp -n "$OUT/vllm_6k_nochunk.json" "$LOGS/vllm_6k_nochunk.2026-11-11.json" 2>/dev/null || true
cp -n "$OUT/vllm_20k_nochunk_greedy.json" "$LOGS/vllm_20k_nochunk_greedy.2026-11-11.json" 2>/dev/null || true
run vllm_6k_nochunk "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 \
  --output "$OUT/vllm_6k_nochunk.json"
run vllm_20k_nochunk_greedy "$VLLM_PY" benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 3.3 \
  --output "$OUT/vllm_20k_nochunk_greedy.json"

# ---- 4. 6k torch profile of the new best (post-optimization) ----
run profile_6k_postopt "$PY" benchmarks/profile_torch_rollout.py \
  --data data/longbench_20k.jsonl --workload 6k \
  --use-flashinfer-decode --compiled-sampler \
  --out "$OUT/profile_raw/ds_6k_postopt"

echo
echo "================ summary ================"
echo "failed: ${FAILED[*]:-none}"
exit ${#FAILED[@]}
