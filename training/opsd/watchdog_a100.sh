#!/bin/bash
# Watchdog for the A100 ablation re-run:
#   1. waits for the r_a100.sh queue (PID $1) to finish
#   2. retries any [FAIL] cells once, using cells_a100.tsv
#   3. runs the two vLLM no-chunk reference cells (separate venv, GPU-free)
#   4. regenerates runs/a100_2026-11-11/optimization_ablation_a100.md
set -uo pipefail
cd "$(dirname "$0")"

PY=/work/nvme/bcjw/msajid/conda/opsd/bin/python
VLLM_PY=/u/msajid/vllm_env/bin/python
OUT=runs/a100_2026-11-11
LOGS=$OUT/logs
WLOG=$OUT/watchdog.log
QUEUE_PID="$1"
shift

export CUDA_VISIBLE_DEVICES=0
SP=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages
export CUDA_HOME=/u/msajid/cuda-12.9-shim
export CPATH=$SP/nvidia/cu13/include:$SP/nvidia/cu13/include/cccl:$SP/nvidia/curand/include
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH
export HF_HOME=/u/msajid/hf_cache
export VLLM_CACHE_ROOT=/u/msajid/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache
export TMPDIR=/u/msajid/tmp_bench
export PYTHONUNBUFFERED=1
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

log() { echo "[watchdog $(date '+%H:%M:%S')] $*" | tee -a "$WLOG"; }

run_cell() { # name P N T PP data flags...
  local name="$1" P="$2" N="$3" T="$4" PP="$5" data="$6"; shift 6
  log "retry $name"
  "$PY" benchmarks/bench_continuous_graph.py \
    --data "$data" \
    --max-prompt-length "$P" --max-new-tokens "$N" \
    --temperature "$T" --top-p "$PP" \
    --warmup 1 --iterations 1 \
    --output "$OUT/$name.json" "$@" \
    >> "$LOGS/$name.log" 2>&1
  if [ $? -eq 0 ] && [ -f "$OUT/$name.json" ]; then
    log "retry ok: $name"
  else
    log "retry FAILED: $name (see $LOGS/$name.log)"
  fi
}

# ---- 1. wait for the queue -------------------------------------------------
log "watchdog started; waiting for queue pid $QUEUE_PID"
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 20; done
log "queue finished"

# ---- 2. retry failed cells -------------------------------------------------
FAILED=$(grep -o '=== \[FAIL\] *[a-z0-9_]*' "$OUT/queue.log" 2>/dev/null | awk '{print $3}' | sort -u)
if [ -n "$FAILED" ]; then
  for name in $FAILED; do
    row=$(awk -F'\t' -v n="$name" '$1==n {print; exit}' cells_a100.tsv)
    if [ -z "$row" ]; then log "no tsv row for $name, skipping"; continue; fi
    # shellcheck disable=SC2086
    run_cell $row
  done
else
  log "no failed cells to retry"
fi

# ---- 3. vLLM reference cells (only after DeepSpeed cells finished) ---------
run_vllm() { # name args...
  local name="$1"; shift
  log "vllm $name"
  "$VLLM_PY" benchmarks/bench_vllm_offline.py \
    --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
    --warmup 1 --iterations 1 --seed 0 --gpu-memory-utilization 0.75 \
    --output "$OUT/$name.json" "$@" > "$LOGS/$name.log" 2>&1
  if [ $? -eq 0 ] && [ -f "$OUT/$name.json" ]; then
    log "vllm ok: $name"
  else
    log "vllm FAILED: $name (see $LOGS/$name.log)"
  fi
}
if [ -x "$VLLM_PY" ]; then
  run_vllm vllm_6k_nochunk \
    --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
    --temperature 0.7 --top-p 0.9 --kv-cache-memory-gib 1.1
  run_vllm vllm_20k_nochunk_greedy \
    --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
    --temperature 0.0 --top-p 1.0 --kv-cache-memory-gib 3.3
else
  log "vllm env missing, skipping reference cells"
fi

# ---- 4. regenerate the report ----------------------------------------------
"$PY" scripts/aggregate_a100_ablation.py >> "$WLOG" 2>&1
log "watchdog done; report at $OUT/optimization_ablation_a100.md"
