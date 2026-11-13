#!/bin/bash
# DS F-level 4B profiling runs (nsys + torch profiler), 3072 prompt -> 1024 new, C=4.
#
# Build env: the login modules set CPATH=<HPC-SDK>/cuda/13.2/include, and every
# compiler (gcc and cicc) searches CPATH BEFORE -I/-isystem, so cooperative_groups.h
# ALWAYS resolves to the 13.2 tree regardless of CUDA_HOME. The 13.2 tree lacks the
# cuda/std (CCCL) headers its CG header includes; /usr/local/cuda-12.9 and the
# cuda-12.9-shim are both partial. The only consistent combo (and exactly how the
# ~/.cache/flashinfer/0.7.4 cache was originally built Nov 11 under lmod 13.2
# defaults) is:
#   13.2 nvcc  +  CPATH-13.2 headers  +  cuda/std from flashinfer's bundled CCCL
# Keep CUDA_HOME + CPATH at these values for all future DS runs so the JIT cache
# stays warm (ninja no-op).
cd /u/msajid/projects/DeepSpeedExamples/training/opsd || exit 1
P=/work/nvme/bcjw/msajid/conda/opsd/bin/python
OUT=runs/profiling_2026-11-12
export HF_HOME=/work/nvme/bcjw/msajid/hf_cache
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2
export CPATH="$CUDA_HOME/include:/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON4B="--model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 4 --max-prompt-length 3072 --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 --top-k -1 --seed 0 --warmup 1 --iterations 1"

echo "=== [0/3] plain run (warms JIT cache): DS F 4B $(date +%H:%M:%S)"
$P benchmarks/bench_continuous_graph.py $COMMON4B \
  --continuous-batch-size 4 --output $OUT/ds_f4b/result_plain.json --ignore-eos \
  --use-paged-cache --enable-paged-prefill --flashinfer-prefill --use-flashinfer-decode \
  --compiled-sampler
rc=$?
[ $rc -ne 0 ] && { echo "PLAIN RUN FAILED rc=$rc"; exit $rc; }

echo "=== [1/3] nsys: DS F 4B $(date +%H:%M:%S)"
nsys profile --force-overwrite=true -t cuda \
  -o $OUT/ds_f4b/nsys_ds_f4b $P benchmarks/bench_continuous_graph.py $COMMON4B \
  --continuous-batch-size 4 --output $OUT/ds_f4b/result.json --save-response-ids --ignore-eos \
  --use-paged-cache --enable-paged-prefill --flashinfer-prefill --use-flashinfer-decode \
  --compiled-sampler
rc=$?
[ $rc -ne 0 ] && { echo "NSYS RUN FAILED rc=$rc"; exit $rc; }

echo "=== [2/3] torch profiler: DS F 4B $(date +%H:%M:%S)"
$P benchmarks/bench_continuous_graph.py $COMMON4B \
  --continuous-batch-size 4 --output $OUT/ds_f4b/result_torch.json --save-response-ids --ignore-eos \
  --use-paged-cache --enable-paged-prefill --flashinfer-prefill --use-flashinfer-decode \
  --compiled-sampler --torch-profile $OUT/ds_f4b/torch_ds_f4b.json
rc=$?
[ $rc -ne 0 ] && { echo "TORCH RUN FAILED rc=$rc"; exit $rc; }

echo "=== [3/3] nsys stats $(date +%H:%M:%S)"
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
  -o $OUT/ds_f4b/nsys_ds_f4b $OUT/ds_f4b/nsys_ds_f4b.nsys-rep >/dev/null 2>&1
echo "=== DS PROFILING DONE $(date +%H:%M:%S)"
