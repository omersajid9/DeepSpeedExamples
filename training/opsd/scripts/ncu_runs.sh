#!/bin/bash
# ncu (Nsight Compute) roofline capture of the decode-attention kernel, DS vs vLLM.
# GPU must be free. ncu replays kernels, so use launch-skip/count to grab a few
# decode-step instances (after the prefill instances have passed).
set -uo pipefail
cd "$(dirname "$0")/.."
SP=/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages
RAW=runs/a100_2026-11-11/profile_raw
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
COMMON="--set full --csv --page details"
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/u/msajid/cuda-12.9-shim
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export HF_HOME=/u/msajid/hf_cache PYTHONUNBUFFERED=1 TMPDIR=/u/msajid/tmp_bench
export TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache VLLM_CACHE_ROOT=/u/msajid/vllm_cache
export CPATH=$SP/nvidia/cu13/include:$SP/nvidia/cu13/include/cccl:$SP/nvidia/curand/include
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:/u/msajid/cuda-12.9-shim/bin:$PATH

# --- DeepSpeed decode flash attention (no-graph, direct launches) ------------
# 6k prefill = 36 flash instances (one per layer), then 16 decode steps x 36.
# Skip 36 (prefill) + grab 4 decode instances.
echo "=== ncu ds_6k flash-decode start $(date +%T)"
"$NCU" $COMMON -k "regex:flash_fwd_kernel" --launch-skip 36 --launch-count 4 \
  -o "$RAW/ncu_ds_flash" --force-overwrite true \
  /work/nvme/bcjw/msajid/conda/opsd/bin/python benchmarks/profile_torch_rollout.py \
    --data data/longbench_20k.jsonl --workload 6k --no-graph-capture \
    --no-torch-profile --out "$RAW/ncu_ds_flash_run" > "$RAW/ncu_ds_flash.log" 2>&1
echo "=== ncu ds_6k flash-decode done rc=$? $(date +%T)"

# --- vLLM decode flash attention ---------------------------------------------
# vLLM prefill = flash_fwd_splitkv (chunked), decode also splitkv. Grab 4 decode.
echo "=== ncu vllm_6k flash-decode start $(date +%T)"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
"$NCU" $COMMON -k "regex:flash_fwd" --launch-skip 36 --launch-count 4 \
  -o "$RAW/ncu_vllm_flash" --force-overwrite true \
  /u/msajid/vllm_env/bin/python benchmarks/profile_torch_vllm.py \
    --data data/longbench_20k.jsonl --workload 6k --no-torch-profile --enforce-eager \
    --out "$RAW/ncu_vllm_flash_run" > "$RAW/ncu_vllm_flash.log" 2>&1
echo "=== ncu vllm_6k flash-decode done rc=$? $(date +%T)"

echo ALL_NCU_DONE
