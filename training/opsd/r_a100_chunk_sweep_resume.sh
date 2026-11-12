#!/bin/bash
# Resume the chunk-size sweep after the server restart killed it mid-run.
# Only the three missing cells: 20k c4096, 20k c8192, and the 6k greedy isolation cell.
set -uo pipefail
cd "$(dirname "$0")"

PY=/work/nvme/bcjw/msajid/conda/opsd/bin/python
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
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

bench() {  # bench NAME P N T PP [--flag ...]
  local name="$1" P="$2" N="$3" T="$4" PP="$5"; shift 5
  echo "=== [bench] $name  P=$P N=$N T=$T PP=$PP data=data/longbench_20k.jsonl $*  $(date '+%H:%M:%S')"
  local t0=$(date +%s)
  "$PY" benchmarks/bench_continuous_graph.py \
    --data data/longbench_20k.jsonl \
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
    exit 1
  fi
}

NB="--use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode"
bench chunk_sweep_20k_c4096 20480 8 0.0 1.0 --enable-chunked-prefill --prefill-chunk-size 4096 $NB
bench chunk_sweep_20k_c8192 20480 8 0.0 1.0 --enable-chunked-prefill --prefill-chunk-size 8192 $NB
bench newbest_6k_greedy 6144 16 0.0 1.0 $NB

echo "=== resume complete: 3/3"
