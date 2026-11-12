#!/bin/bash
# Fresh re-run of the A100 ablation on the POST-optimization code (2026-11-12).
# The 11-11 cells were measured before the decode elementwise cleanup landed, so the
# study was internally inconsistent; this reruns every DS cell on one code base and
# adds the new-best cells (FlashInfer decode + compiled sampler). vLLM cells are
# already fresh from today's run and are skipped.
set -uo pipefail
cd "$(dirname "$0")"

PY=/work/nvme/bcjw/msajid/conda/opsd/bin/python
OUT=runs/a100_2026-11-11
LOGS=$OUT/logs
BAK=$LOGS/ablation_11-11_original
mkdir -p "$LOGS" "$BAK"

# Preserve the original 11-11 ablation JSONs (pre-optimization code).
for f in ablation_6k_dense_eager_nochunk ablation_6k_dense_graph_nochunk \
         ablation_6k_paged_decode_graph_nochunk ablation_6k_paged_prefill_graph_nochunk \
         compile_6k_graph compiled_sampler_6k flashinfer_6k_baseline flashinfer_6k \
         chunk_paged_prefill_6k chunk_dense_staging_6k \
         ablation_20k_dense_eager_nochunk ablation_20k_dense_graph_nochunk \
         ablation_20k_paged_decode_eager_nochunk ablation_20k_paged_decode_graph_nochunk \
         ablation_20k_paged_prefill_graph_nochunk compile_20k_graph flashinfer_20k \
         chunk_paged_prefill_20k_greedy prefix_no_reuse prefix_reuse; do
  [ -f "$OUT/$f.json" ] && cp -n "$OUT/$f.json" "$BAK/$f.json"
done

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

bench() {  # bench NAME P N T PP [DATA] [--flag ...]
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
  # NEW BEST: old best + FlashInfer decode + compiled sampler
  bench newbest_6k                            $P $N $T $PP --use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode
else
  echo "=== [skip] flashinfer_6k* + newbest_6k (flashinfer not installed)"
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
  # NEW BEST: old best + FlashInfer decode + compiled sampler
  bench newbest_20k                           $P $N $T $PP --use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode
else
  echo "=== [skip] flashinfer_20k + newbest_20k (flashinfer not installed)"
fi
bench chunk_paged_prefill_20k_greedy          $P $N $T $PP --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill

# ---------------- prefix reuse (6k, two identical prompts, batch 2) ----------
# --enable-paged-prefill matches the A40 effective config (see r_a100.sh).
bench prefix_no_reuse 6144 8 0.0 0.9 \
  --batch-size 2 --continuous-batch-size 2 --enable-chunked-prefill \
  --prefill-chunk-size 512 --no-graph-capture --use-paged-cache --enable-paged-prefill
bench prefix_reuse 6144 8 0.0 0.9 \
  --batch-size 2 --continuous-batch-size 2 --enable-chunked-prefill \
  --prefill-chunk-size 512 --no-graph-capture --use-paged-cache --enable-paged-prefill \
  --enable-prefix-reuse

echo
echo "================ summary ================"
echo "cells: $(ls "$OUT"/*.json 2>/dev/null | wc -l) present, ${#FAILED[@]} failed: ${FAILED[*]:-none}"
exit ${#FAILED[@]}
