# A100 reproduction runbook — vLLM comparison + rollout optimization ablation

Reproduces the measured tables in `runs/vllm_comparison.md` and
`runs/optimization_ablation.md` on a single **A100-40GB** interactive node.

The original measurements were taken on an **A40-48GB** with cluster-local paths
(`/workspace/DeepSpeed`, `/workspace/DeepSpeedExamples`, `/workspace/opsd_env.sh`,
`/workspace/vllm_env`, `/dev/shm/hf_cache`). This runbook maps those onto a home-directory
layout and keeps every flag identical to the source artifacts (each `runs/*.json` embeds
the CLI it was produced with).

Expected differences on A100: absolute milliseconds (A100 ≈ 2.2x HBM bandwidth of A40, more
bf16 throughput) — **not** the response tokens. Token identity is the correctness gate:

- all 6k rows → `d4a324e903ed2fe8e2d5b676b43ec94a22efbbdd5663c30a310d93d0144cf02c`
- all 20k rows → `c4291b8013dd0839db4fb6ac81bce4258b9a032ff8014cb72e8c9068cac2b258`

`--flashinfer-prefill` and `--torch-compile` may flip a last-ULP token on a different
architecture; the other rows must stay exact.

## 0. Path map

| original (A40 box) | here |
|---|---|
| `/workspace/DeepSpeed` | `$HOME/projects/DeepSpeed` |
| `/workspace/DeepSpeedExamples/training/opsd` | `$HOME/projects/DeepSpeedExamples/training/opsd` |
| `/workspace/opsd_env.sh` | `$HOME/opsd_env.sh` |
| `/workspace/vllm_env` | `$HOME/vllm_env` (must be created) |
| `/dev/shm/hf_cache` | `$HOME/hf_cache` (`/dev/shm` also works, faster, per-boot) |

Preconditions: the DeepSpeed checkout must contain `deepspeed/runtime/rollout/paged_cache.py`
with the paged/prefix/torch-compile/FlashInfer options (this repo's tip: `6750ed298
rollout: optional FlashInfer prefill backend`), and the examples checkout must contain
`benchmarks/bench_continuous_graph.py` with `--flashinfer-prefill` / `--compiled-sampler`
(this repo's tip: `2b0c09f bench: add FlashInfer prefill backend profile`).

## 1. Environment

```bash
export PROJECTS=$HOME/projects
export DS=$PROJECTS/DeepSpeed
export OPSD=$PROJECTS/DeepSpeedExamples/training/opsd
source "$HOME/opsd_env.sh"
export PYTHONPATH="$DS:$OPSD:${PYTHONPATH:-}"          # checkout wins over any pip deepspeed
export HF_HOME=$HOME/hf_cache
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor
export TORCH_EXTENSIONS_DIR=$HOME/torch_extensions    # JIT cache for the blocked-flash op
export TMPDIR=$HOME/tmp
export TORCH_CUDA_ARCH_LIST=8.0                       # A100 = sm80
export TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0   # gcc-toolset breaks the PCH step; see 1.4
cd "$OPSD"
```

Verify before running anything:

```bash
git -C "$DS" log --oneline -1
git -C "$PROJECTS/DeepSpeedExamples" log --oneline -1
python - <<'PY'
import importlib
for m in ("torch", "transformers", "deepspeed"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), getattr(mod, "__file__", "?"))
    except Exception as exc:
        print(m, "MISSING:", exc)
try:
    import dskernels
    print("dskernels", dskernels.__file__, dskernels.library_path())
except Exception as exc:
    print("dskernels MISSING:", exc)
PY
which ninja nvcc
```

`deepspeed.__file__` must resolve inside `$DS`. `dskernels` (pip: `deepspeed-kernels`) is a
hard requirement of the paged path — `RaggedOpsBuilder` links `-lblockedflash` against it;
without it every `--use-paged-cache` run fails at kernel load.

### 1.1 Build the environment from scratch (only if the check fails)

```bash
conda create -y -n opsd python=3.12 && conda activate opsd
pip install -e "$DS"                    # DS runtime requirements (incl. ninja)
pip install deepspeed-kernels           # provides the `dskernels` import
pip install -r "$OPSD/requirements.txt" # datasets, numpy, transformers>=5
# nvcc: module load cuda/13.x (or a conda cuda-toolkit matching torch's CUDA)
```

### 1.2 Paged-path prerequisite: `dskernels` (blocked-flash kernels)

The pip name and the import name differ — this is the first failure every paged run hits on a
fresh env:

- `import dskernels` → `ModuleNotFoundError: No module named 'dskernels'`
- `pip install dskernels` → `No matching distribution found for dskernels` (that name is not on
  PyPI)
- correct install: **`pip install deepspeed-kernels`** (declared unpinned in
  `$DS/requirements/requirements-dev.txt`; ships prebuilt `libblockedflash.so` / `libdeepspeedft.so`)

`RaggedOpsBuilder.extra_ldflags()` links `-lblockedflash` at
`<repo_root>/deepspeed/<dskernels.library_path()>` (or the absolute path, if
`library_path()` returns one). Verify before running anything:

```bash
pip install deepspeed-kernels
python - <<'PY'
import os, dskernels
from pathlib import Path
ab_root = Path(os.environ["DS"])                       # op_builder's parent.parent == repo root
lp = dskernels.library_path()
resolved = Path(lp) if os.path.isabs(lp) else ab_root / ("deepspeed" if (ab_root / "deepspeed").is_dir() else "..") / lp
print("dskernels:", dskernels.__file__)
print("library_path():", lp, "->", resolved, resolved.is_dir())
print(sorted(os.listdir(resolved)) if resolved.is_dir() else "MISSING libblockedflash.so")
PY
python -c "import torch.utils.cpp_extension as e; print('CUDA_HOME', e.CUDA_HOME)"
which ninja nvcc
```

`CUDA_HOME` must be non-`None` and `nvcc` on `PATH` (the ragged ops JIT-compile for sm80 on
first paged use). `cannot find -lblockedflash` at link time means the resolved directory above
is wrong — reinstall the wheel (`pip install --force-reinstall --no-deps deepspeed-kernels`),
do not change bench flags.

### 1.3 `$HOME/opsd_env.sh` reference content

Diff against the file already on the node; keep whatever conda activation it has.

```bash
#!/usr/bin/env bash
export PROJECTS=$HOME/projects
export DS=$PROJECTS/DeepSpeed
export OPSD=$PROJECTS/DeepSpeedExamples/training/opsd
export PYTHONPATH=$DS:$OPSD:${PYTHONPATH:-}
export HF_HOME=$HOME/hf_cache
export TORCHINDUCTOR_CACHE_DIR=$HOME/torchinductor
export TORCH_EXTENSIONS_DIR=$HOME/torch_extensions
export TMPDIR=$HOME/tmp
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0   # see 1.4
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
# conda activate opsd        # env holding torch/transformers/deepspeed
# module load cuda/13.x      # nvcc for the ragged-ops JIT build
```

### 1.4 `--torch-compile` prerequisite: inductor's C++ precompile-headers step

`--torch-compile` (`torch.compile(mode="max-autotune-no-cudagraphs")`) fails on toolchains whose
gcc driver mishandles `-x c++-header`. Symptom, after triton autotune succeeds:

```
torch._inductor.exc.InductorError: CppCompileError: C++ compile error
Command:
<cc> -x c++-header <TORCHINDUCTOR_CACHE_DIR>/precompiled_headers/<hash>.h -D TORCH_INDUCTOR_CPP_WRAPPER ...
Output:
ld: /lib/../lib64/crt1.o: in function `_start': undefined reference to `main'
```

The PCH build is a compile-time-only optimization (it speeds up compiling the generated C++
helper/`cpp_wrapper` translation units); disabling it does not change graph semantics or the
measured iteration. After passing autotune, the run then compiles the C++ pybinding directly:

```bash
export TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0
python -c "from torch._inductor import config; print(config.cpp_cache_precompile_headers)"  # False
echo "CXX=${CXX:-<unset, torch uses g++>}"     # torch uses CXX verbatim: config.cpp.cxx
command -v g++ && g++ --version | head -1      # must be a real C++ driver with cc1plus
```

If the pybinding compile then fails with `'omp.h' file not found`, the `-fopenmp` flag needs the
toolset's headers — fix the toolchain/`CXX`, do not add compiler flags to the bench.

## 2. Data

The LongBench / ReTool / prefix-reuse prompts are **git-tracked** in the examples repo —
nothing to download. Only the model comes from the Hub:

```bash
source "$HOME/opsd_env.sh"
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-4B", max_workers=8))
PY
wc -l "$OPSD"/data/*.jsonl     # 64 150 2 238 2000  (longbench_20k, longbench_mfqa,
                               #                      prefix_reuse_6k, prompts, retool)
du -sh "$HF_HOME"              # ~8 GB after prefetch
```

No `HF_TOKEN` needed. Runtime reads the local `data/*.jsonl` only.

## 3. Smoke test (forces the ragged-ops JIT build)

Run this before the full matrix — it is where a missing `dskernels`/`nvcc` shows up.

```bash
cd "$OPSD"
COMMON="--model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 --save-response-ids"

python benchmarks/bench_continuous_graph.py $COMMON --no-graph-capture --output runs/_smoke_6k.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --output runs/_smoke_6k_paged.json

jq -r '[(.iterations[0].wall_ms|tostring), .response_sha256[0:8]]|@tsv' \
  runs/_smoke_6k.json runs/_smoke_6k_paged.json     # expect d4a324e9 on both
rm runs/_smoke_6k.json runs/_smoke_6k_paged.json
```

## 4. DeepSpeed runs — both docs

Single process, one GPU (`bench_continuous_graph.py` is not launched through `deepspeed`).
**Run strictly sequentially**; nothing else on the GPU. Every row below writes the artifact
name the docs' source map expects.

### 4.1 Clean 4B 6k — `optimization_ablation.md` §Clean 4B 6k, `vllm_comparison.md` Results (DeepSpeed rows)

Settings: batch 1, C=1, prompt 6144, 16 new, Gumbel `temperature=0.7 top_p=0.9`, seed 0,
warmup 1, one measured iteration, chunked prefill off.

```bash
COMMON="--model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 --save-response-ids"

# dense eager (no graph)
python benchmarks/bench_continuous_graph.py $COMMON --no-graph-capture \
  --output runs/ablation_6k_dense_eager_nochunk.json
# + CUDA graph capture
python benchmarks/bench_continuous_graph.py $COMMON \
  --output runs/ablation_6k_dense_graph_nochunk.json
# + paged decode
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache \
  --output runs/ablation_6k_paged_decode_graph_nochunk.json
# + paged prefill
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --output runs/ablation_6k_paged_prefill_graph_nochunk.json
# + torch.compile
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --torch-compile --output runs/compile_6k_graph.json
# FlashInfer prefill (baseline = paged prefill re-run under the same build)
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --output runs/flashinfer_6k_baseline.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --flashinfer-prefill --output runs/flashinfer_6k.json
# compiled sampler
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --compiled-sampler --output runs/compiled_sampler_6k.json
```

### 4.2 Clean 4B 20k — `optimization_ablation.md` §Clean 4B 20k

Settings: batch 1, C=1, prompt 20480, 8 new, greedy (`temperature=0.0 top_p=1.0`), seed 0,
warmup 1, one measured iteration, chunked prefill off.

```bash
COMMON="--model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 20480 --max-new-tokens 8 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 --save-response-ids"

python benchmarks/bench_continuous_graph.py $COMMON --no-graph-capture \
  --output runs/ablation_20k_dense_eager_nochunk.json
python benchmarks/bench_continuous_graph.py $COMMON \
  --output runs/ablation_20k_dense_graph_nochunk.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --no-graph-capture \
  --output runs/ablation_20k_paged_decode_eager_nochunk.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache \
  --output runs/ablation_20k_paged_decode_graph_nochunk.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --output runs/ablation_20k_paged_prefill_graph_nochunk.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --torch-compile --output runs/compile_20k_graph.json
python benchmarks/bench_continuous_graph.py $COMMON --use-paged-cache --enable-paged-prefill \
  --flashinfer-prefill --output runs/flashinfer_20k.json
```

### 4.3 Chunked 512 rows — the "Chunk penalty" table

Chunked prefill on, `prefill_chunk_size=512`, graph capture on, batch 1.

```bash
# 6k, dense staging + paged decode  (the uniquely bad path, +36.5%)
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 --save-response-ids \
  --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache \
  --output runs/fourway_current_best_6k.json

# 6k, paged prefill + paged decode
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 --save-response-ids \
  --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill \
  --output runs/fourway_paged_prefill_6k.json

# 20k greedy, paged prefill + paged decode (3871.2 ms on A40)
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --continuous-batch-size 1 --max-prompt-length 20480 --max-new-tokens 8 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 --save-response-ids \
  --enable-chunked-prefill --prefill-chunk-size 512 --use-paged-cache --enable-paged-prefill \
  --output runs/fourway_paged_prefill_20k_greedy.json
```

### 4.4 Prefix reuse — `optimization_ablation.md` §Prefix reuse

Two identical prompts (`data/prefix_reuse_6k.jsonl` has 2 lines), chunked, no graph.

```bash
COMMON="--model Qwen/Qwen3-4B --data data/prefix_reuse_6k.jsonl --batch-size 2 \
  --continuous-batch-size 2 --max-prompt-length 6144 --max-new-tokens 8 \
  --temperature 0.0 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 --save-response-ids \
  --enable-chunked-prefill --prefill-chunk-size 512 --no-graph-capture --use-paged-cache"

python benchmarks/bench_continuous_graph.py $COMMON --output runs/prefix_no_reuse.json
python benchmarks/bench_continuous_graph.py $COMMON --enable-prefix-reuse \
  --output runs/prefix_reuse.json
```

## 5. vLLM runs — `vllm_comparison.md`

Separate environment, isolated from DeepSpeed. Create it once:

```bash
python -m venv "$HOME/vllm_env"
"$HOME/vllm_env/bin/pip" install -U pip
"$HOME/vllm_env/bin/pip" install "vllm==0.33.0"
"$HOME/vllm_env/bin/pip" install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
"$HOME/vllm_env/bin/python" -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Match the doc's versions (`vLLM 0.33.0`, `torch 2.14.0+cu130`); a different resolver result is
not comparable to the table.

```bash
cd "$OPSD"
export HF_HOME=$HOME/hf_cache VLLM_CACHE_ROOT=$HOME/vllm_cache TMPDIR=$HOME/tmp
V=$HOME/vllm_env/bin/python

# 6k, 16 new (Gumbel): no-chunk (8192 = one prefill forward) and chunk 512
$V benchmarks/bench_vllm_offline.py --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl \
  --batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 --output runs/vllm_6k_nochunk.json

$V benchmarks/bench_vllm_offline.py --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl \
  --batch-size 1 --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 512 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 --output runs/vllm_6k_kv1g.json

# 20k, 8 new greedy: no-chunk (20488) and chunk 512
$V benchmarks/bench_vllm_offline.py --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl \
  --batch-size 1 --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 3.3 \
  --output runs/vllm_20k_nochunk_greedy.json

$V benchmarks/bench_vllm_offline.py --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl \
  --batch-size 1 --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 512 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 3.3 --output runs/vllm_20k_greedy.json
```

`--kv-cache-memory-gib` / `--gpu-memory-utilization` approximate the DeepSpeed cache budget
(1.1 GiB at 6k, 3.3 GiB at 20k); they are engine-independent and must not be retuned for A100.

## 6. Validation

```bash
cd "$OPSD"
for f in runs/ablation_*_nochunk.json runs/compile_*_graph.json runs/flashinfer_*.json \
         runs/fourway_*.json runs/prefix_*.json runs/compiled_sampler_6k.json; do
  jq -r --arg f "$(basename "$f")" \
    '[$f, (.max_prompt_length|tostring), (.iterations[0].wall_ms|tostring),
      (.peak_memory_mb|tostring), .response_sha256[0:8]] | @tsv' "$f"
done | column -t
```

Columns: artifact, prompt length, wall ms, peak MiB, response digest. Compare wall ms against
the A40 tables only as ratios (A100 is faster); the 8-hex digests must match per workload.

Artifact → table map:

| artifact(s) | documented in |
|---|---|
| `ablation_6k_*_nochunk.json`, `compile_6k_graph.json` | `optimization_ablation.md` §Clean 4B 6k; `vllm_comparison.md` Results / Distance to vLLM |
| `ablation_20k_*_nochunk.json`, `compile_20k_graph.json` | `optimization_ablation.md` §Clean 4B 20k; `vllm_comparison.md` Results / Distance to vLLM |
| `flashinfer_6k.json`, `flashinfer_6k_baseline.json`, `flashinfer_20k.json` | `optimization_ablation.md` §FlashInfer prefill backend |
| `compiled_sampler_6k.json` | `optimization_ablation.md` §Compiled sampler |
| `fourway_paged_prefill_6k.json`, `fourway_current_best_6k.json`, `fourway_paged_prefill_20k_greedy.json` | §Chunk penalty: ours vs vLLM; §6k/20k four-way comparison |
| `prefix_no_reuse.json`, `prefix_reuse.json` | §Prefix reuse |
| `vllm_6k_nochunk.json`, `vllm_20k_nochunk_greedy.json` | `vllm_comparison.md` Results (vLLM no-chunk rows) |
| `vllm_6k_kv1g.json`, `vllm_20k_greedy.json` | chunk-512 rows of the chunk-penalty table |

## 7. A100 notes and caveats

- **Memory:** 40 GB is sufficient for every row here. The largest documented DeepSpeed peak is
  16.2 GB (`compile_20k_graph`); the only OOM in the docs is the 20k chunked bucket-graph
  configuration at 8 active slots, which already OOMs at 48 GB.
- **Architecture:** A100 is sm80, the docs' A40 is sm86. Absolute ms will not match (A100 has
  roughly 2.2x the HBM bandwidth and more bf16 tensor-core throughput); the response digests
  are the reproducibility contract. `--flashinfer-prefill` / `--torch-compile` may legitimately
  pick different kernels and flip a last-ULP token — inspect before treating as a regression.
- **First paged run builds the ragged ops JIT** (blocked-flash kernels, `dskernels` +
  `-lblockedflash`) for sm80; expect a few minutes of `nvcc` before the first print. If it dies
  with `ModuleNotFoundError: No module named 'dskernels'`, install `deepspeed-kernels` (see
  §1.2 — `pip install dskernels` does not exist). `TORCH_CUDA_ARCH_LIST=8.0` keeps that build
  single-arch.
- **`--flashinfer-prefill`** requires `flashinfer` importable in the DeepSpeed env or the run
  raises at the first prefill.
- **`--torch-compile`** autotunes inside the excluded warmup (170 s at 6k / 276 s at 20k on the
  A40); peak memory rises materially (9.1 → 10.4 GB at 6k, 12.0 → 16.2 GB at 20k). Flag it as a
  latency, not memory, option. If it dies inside inductor's C++ precompile-headers step
  (`undefined reference to 'main'`), see §1.4 — `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0`.
- **Sequential execution only.** Every row is a single-GPU timing run; concurrency invalidates
  the numbers. Keep the GPU otherwise idle and reuse the same shell environment for all rows.
- **Flag defaults that affect comparability:** `--graph-capture` is on by default (use
  `--no-graph-capture` for the eager rows); `--paged-block-size 0` (default) selects the
  auto block size (64) the artifacts record; `--top-k -1` and `--seed 0` match the artifacts.
- **Caches:** `$HOME/hf_cache` (model, ~8 GB), `$HOME/torchinductor` (compile artifacts),
  `$HOME/torch_extensions` (JIT kernels). All are safe to delete; they repopulate. `/dev/shm`
  variants are faster but per-boot — then set `HF_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
  `TORCH_EXTENSIONS_DIR`, `TMPDIR` accordingly for **all** rows, and keep one choice across
  DeepSpeed and vLLM runs.

## 8. Rough wall time (A100-40GB, interactive)

| section | rows | time |
|---|---:|---:|
| §3 smoke (incl. first JIT build) | 2 | 5–10 min |
| §4.1 6k clean set | 8 | ~10 min |
| §4.2 20k clean set | 7 | ~25 min (`compile_20k_graph` dominates) |
| §4.3 chunked 512 | 3 | ~10 min |
| §4.4 prefix reuse | 2 | ~5 min |
| §5 vLLM set | 4 | ~10 min incl. engine warmup |
