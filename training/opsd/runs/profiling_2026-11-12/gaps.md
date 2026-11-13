# Where the DeepSpeed-vs-vLLM decode gap lives (4B profiled study)

Profiled on a single A100-80GB, Nov 12 2026. Task: 4 prompts × 3072 tokens
(`data/longbench_20k.jsonl`), 1024 new tokens each, continuous batch C=4,
Gumbel sampler T=0.7 / PP=0.9, seed 0, `--ignore-eos`, bf16 — the F-level
rollout config vs vLLM 0.33.0 (V1 engine, FLASH_ATTN v2, chunked prefill
`max_num_batched_tokens=512`, FULL+PIECEWISE CUDA graphs, FlashInfer
top-p/top-k + Gumbel sampler).

## 1. Wall times and the reproduced production gap

| Engine | plain | under nsys | under torch profiler |
|---|---|---|---|
| DS F-level 4B | **17,026.7 ms** | 18,086.4 ms (+6.2 %) | 20,880.8 ms (+22.6 %) |
| vLLM 4B | **10,673.0 ms** | 10,704.0 ms (+0.3 %) | 11,291.1 ms (+5.8 %) |

**Plain-equivalent gap: 17,027 / 10,673 = 1.595×** — reproduces the
production 3k/3k F-vs-vLLM gap at 4B (1.59×; 1.40× at 8B, `runs/3k3k_2026-11-12/`)
within 0.3 %, so the profiled 3k→1k task is a faithful stand-in for dissecting it.
(vLLM nsys-step wall is the progress-bar figure; its `result.json` predates the
bench's wall-recording-in-all-modes fix. A second torch-profile run measured
11,297.2 ms, +0.05 % vs 11,291.1 — profiler runs are themselves reproducible.)

Determinism: DS outputs bit-identical across all three instrumentation modes
(`response_sha256 = 59383bb3…`). vLLM outputs are likewise bit-identical
across the plain and both torch-profile runs, including across independent
engine restarts (`a8d2cf…`, `--ignore-eos` + seeded Gumbel path). (The
nsys-step `result.json` predates response-recording-in-all-modes: its
`response_ids` is null and its sha is the digest of `null`, not of an output.)

## 2. GPU picture under the same (torch) profiler

| | DS F 4B | vLLM 4B |
|---|---|---|
| kernel/op events | **3,021,734** | **629,656** (4.8× fewer) |
| total kernel time | 16,788.8 ms | 10,734.9 ms |
| trace span | 20,870.2 ms | 11,281.8 ms |
| **GPU busy** | **80.3 %** | **94.8 %** |
| GPU idle in wall | 19.8 % (4.13 s) | 5.2 % (0.59 s) |

Kernel-time ratio 1.564×; the wall ratio under the same profiler is 1.85×
because profiler overhead is asymmetric (DS +22.6 % vs vLLM +5.5 % — kineto's
per-event CPU cost scales with event count, §6).

### Bucket decomposition (measured iteration only; Δ = DS − vLLM)

| bucket | DS (ms, % of kernels) | vLLM (ms, %) | Δ ms | Δ ms/decode step |
|---|---|---|---|---|
| GEMM | 7,986.8 (47.6 %) | 7,348.8 (68.5 %) | **+638.0** | +0.62 |
| elementwise / move | 6,090.5 (36.3 %) | 854.2 (8.0 %) | **+5,236.4** | **+5.11** |
| attention | 2,098.2 (12.5 %) | 2,385.9 (22.2 %) | **−287.7** | −0.28 |
| other (mostly sampling-side) | 594.9 (3.5 %) | 113.0 (1.1 %) | **+481.9** | +0.47 |
| sampler | 18.5 (0.1 %) | 33.0 (0.3 %) | −14.5 | −0.01 |
| **total** | **16,788.8** | **10,734.9** | **+6,053.9** | **+5.92** |

The 6.05 s kernel-time gap (≈ 96 % of the 6.32 s plain wall gap) is:

1. **~87 % unfused per-layer elementwise glue (+5.24 s)** — §3, the headline
2. **~8 % sampling-side GPU work (+0.48 s)** — DS Gumbel path is eager, §4
3. **~8 % GEMM mix (+0.64 s)** — near tie, §5
4. offset by **−0.29 s decode attention** — the one bucket DS wins, §5b

The remaining ~0.3 s of the plain wall gap is host/scheduling (§6).

## 3. Headline: unfused elementwise — ~450 small kernels/step vs 5 fused

DS runs the layer glue (RMSNorm, RoPE, dtype conversion, residual adds, KV
scatter) as individual PyTorch kernels; vLLM's inductor/torch.compile pass
fuses the same work into a handful of Triton kernels.

DS top elementwise kernels (measured iteration):

| kernel | ms | calls | ~calls/step |
|---|---|---|---|
| `elementwise Binary` (mul/fma glue) | 1,399.0 | 444,691 | ~434 |
| `reduce MeanOps` (RMSNorm mean) | 557.7 | 148,915 | ~145 |
| `elementwise direct` (copies) | 425.5 | 147,888 | ~144 |
| `CUDAFunctor_add bf16` (residuals) | 406.2 | 221,832 | ~217 |
| `elementwise Binary` (2nd variant) | 373.7 | 149,942 | ~146 |
| `index_elementwise` (KV paged scatter) | 330.2 | 73,944 | ~72 |
| float↔bf16 `converting_copy` ×2 | 534.4 | 300,911 | ~294 |
| `rsqrt` / `pow` / `add` (RMSNorm tail) | 781.1 | 447,772 | ~437 |
| ⇒ top-11 of 36.3 % bucket ≈ 4,808 ms; ~450 elementwise kernels per decode step, 3–7 µs each |

vLLM elementwise (854.2 ms total):

| kernel | ms | calls |
|---|---|---|
| `triton_per_fused_add_cat_index_select_mul_rms_norm…` ×2 (QKV+RoPE+KV-cache+norm fused) | 264.8 | 36,680 |
| `triton_red_fused_fused_add_rms_norm` ×2 | 216.1 | 37,728 |
| `triton_poi_fused_mul_silu_slice` (MLP act) | 99.2 | 37,728 |
| `elementwise direct` (remaining copies) | 225.7 | 73,224 |
| ⇒ ~5 fused kernels cover the layer glue; ~0.83 ms/step vs DS 5.95 ms/step |

**+5.11 ms/step (≈ 87 % of the kernel gap), i.e. ~5.2 s of the 17 s wall.**
The E-level "Fused" ablation closed part of this; at F the per-layer
norm/rope/cast chain is still unfused — this is the #1 optimization target
(fuse RMSNorm+RoPE+casts into one Triton kernel or a cuBLASLt epilogue;
reaching vLLM's 0.83 ms/step elementwise floor would alone take DS to
≈ 11.8 s wall, within ~10 % of vLLM).

## 4. Sampling side: DS pays ~0.6 ms/step of eager GPU + a per-step sync

DS's compiled Gumbel sampler is *not* inside the CUDA graph: the nsys
session cross-check shows it runs eagerly every step —
`tensor_kernel_scan` (cumsum) 555 ms × 2,046 (~1/step over both runs),
`radix_sort` 156 ms × 8,216 (~4/step), `cunn_SoftMaxForward` 260 ms × 4,108
(~2/step) — plus gumbel 18.5 ms in the trace. vLLM's FlashInfer
top-p/top-k + Gumbel chain is ~146 ms total for the whole iteration
(`_topp_sb_*` ~78 ms, `_gumbel_sample` 13.8 ms, `_temperature`/`argmax`/`gather`
rest) with no per-step sync outside the graph. Net **+0.46 ms/step** GPU
plus the host round-trip every step.

## 5. GEMM: near tie, different kernel mix

7,987 vs 7,349 ms (+0.62 ms/step). The shared top kernel
(`ampere_bf16_64x64_sliced1x2_stages5`, ~74k calls each) runs 48.9 µs/call
under DS and 60.1 µs/call under vLLM (same cuBLAS kernel; clock/graph-context
effect — noted, not a driver). DS additionally executes `64x64_stages6`
(757 ms × 36,828), `cutlass_64x64` (595 ms × 73,656) and a splitKreduce pair
(330 ms × 147,312) that vLLM does not, while vLLM runs its split-K GEMM as a
single `splitKreduce` (176 ms × 73,260). Prefill GEMM (256×128 tiles) is
comparable: DS 342 ms × 720 vs vLLM 344 ms × 2,592.

### 5b. Attention: the one bucket DS wins

| engine | decode kernel | ms | calls | µs/call |
|---|---|---|---|---|
| DS | flashinfer `BatchPrefillWithPagedKVCache` (paged decode, single kernel) | 1,882.0 | 36,828 | 51.1 |
| vLLM | FA2 `flash_fwd_splitkv` + `flash_fwd_splitkv_combine` | 1,879.2 + 243.1 | 36,828 + 36,396 | 51.0 + 6.7 |
| vLLM (prefill) | FA2 `varlen_fwd` (chunked, 900 chunks) | 145.8 | 900 | 162.0 |

Decode attention: 1.88 s (DS) vs 2.12 s (vLLM) → **−0.23 ms/step for DS**;
bucket total 2.10 s vs 2.39 s. Consistent with the CB-ablation finding
(FlashInfer up to 2.3× the paged decode kernels).

## 6. Host side: vLLM stays async; DS stalls ~1/5 of the time under the profiler

Under identical torch profiling: vLLM burns 4.92 s of CPU in
`cudaEventSynchronize` (74 % of CPU time) yet the GPU idles only 5.2 % —
16,034 `cuLaunchKernelEx` + 1,023 `cudaGraphLaunch` keep the queue full. DS
idles 19.8 % (4.13 s): even with graph capture, ~450 eager ops/step plus the
eager sampler path create per-step stalls, and kineto's per-event cost
(3.02 M events vs 630 k) inflates the measurement. In plain mode the
host-side residual is ~0.3 s of the 6.32 s wall gap (6.05 s is GPU kernel
work) — real, but secondary.

## 7. Profiler-overhead asymmetry (itself a finding)

| | plain → nsys | plain → torch |
|---|---|---|
| DS | +1.06 s (+6.2 %) | **+3.85 s (+22.6 %)** |
| vLLM | +0.03 s (+0.3 %) | +0.62 s (+5.8 %) |

Kineto's CPU cost scales with traced events: DS emits 3.02 M kernel/op events,
vLLM 630 k (CUDA-graph replay keeps its event count an order of magnitude
downer). Any future head-to-head must profile both engines and normalize.

## 8. Caveats

- `nsys cuda_gpu_kern_sum` does **not** expand CUDA-graph-replayed kernels;
  its histograms (DS session: 25.3 % `256x128` GEMM, 20.6 % cumsum, 9.6 %
  softmax; vLLM session: 38.2 % `64x64` GEMM, 25.4 % `256x128`, 10.5 %
  splitkv) only see *eager* work (prefill, warmup, per-step sampler) and are
  used as a cross-check, not the per-kernel source of truth. The torch
  traces expand graph-replayed kernels and are authoritative. (Future:
  `nsys stats --report cuda_gpu_graph_sum`.)
- nsys sessions include model load + engine init + warmup (DS GPU activity
  span 40.7 s, vLLM 122.9 s incl. ~90 s first-run JIT), so nsys fractions are
  session-wide. HtoD totals (DS 5,551 ms, vLLM 4,674 ms) are dominated by
  weight loading (single ops up to 694/483 ms), not decode.
- GEMM per-call timing on the shared kernel differs 23 % between traces
  (clock state / graph context) — direction noted, not relied on.
- 8B was not profiled: at 8B the production gap is smaller (1.40× vs 1.59×),
  so 4B is the worst case and the right place to find the gap.

## 9. Slide one-liners

- **The 1.59× gap at 4B is 87 % unfused per-layer elementwise:** DS runs
  ~450 small kernels/decode step (5.95 ms) where vLLM's compiled Triton
  runs ~5 fused kernels (0.83 ms).
- **Attention is DS's win:** FlashInfer paged decode beats FA2 splitkv+combine
  by 0.23 ms/step (1.88 s vs 2.12 s for 36 layers × 1024 steps).
- **GEMM is a near tie** (7.99 s vs 7.35 s); **sampling adds +0.46 ms/step**
  (DS cumsum+sort+softmax runs eagerly outside the graph, with a per-step
  sync).
- **Host side:** vLLM keeps the GPU at 95 % busy on 630 k traced events
  (1k graph launches); DS sits at 80 % on 3.02 M events — even the
  profiler-overhead asymmetry (22.6 % vs 5.5 %) is a symptom of the same
  per-op fan-out.
- **Fix order:** (1) fuse RMSNorm+RoPE+casts into one kernel/epilogue
  (~5.2 s off 17 s → ≈ 11.8 s, within ~10 % of vLLM), (2) move the Gumbel
  sampler fully in-graph (~0.5 s), (3) trim GEMM mix + per-step eager
  host ops (~0.3–0.4 s) → lands ≈ vLLM within noise.

## 10. Artifacts & regeneration

```
runs/profiling_2026-11-12/
├── ds_f4b/            result_plain|_|torch.json (walls, sha 59383bb3…),
│                      torch_ds_f4b.json (4.4 GB chrome trace),
│                      nsys_ds_f4b.nsys-rep + .sqlite + 2 CSVs
├── vllm4b/            result.json (nsys step; iterations empty — predates
│                      all-modes recording), result_plain.json (wall
│                      10,673.0 ms), result_torch.json (wall 11,291.1 ms),
│                      result_torch2.json (wall 11,297.2 ms, determinism
│                      check), torch/ (88 MB gz trace +
│                      rank0.profiler_out.txt = CUDA-time table), torch2/
│                      (same, second run), nsys_vllm4b.nsys-rep + .sqlite + 2 CSVs
└── gaps.md, ds_rerun.log, vllm_rerun.log, vllm_plain_rerun.log

training/opsd/prof_ds_rerun.sh, prof_vllm_rerun.sh — rerun chains + env recipe:
    CUDA_HOME=HPC-SDK 13.2 + CPATH=<13.2 include>:<flashinfer-bundled CCCL>;
    keeps JIT caches warm — do not deviate, see script comments
```

Trace analysis: `python -c "import sys; sys.path.insert(0,'scripts');
import analyze_prof; analyze_prof.report('<label>', '<trace>', <wall_ms>)"`.
nsys CSVs: `nsys stats --report cuda_gpu_kern_sum --report
cuda_gpu_mem_time_sum [--force-export=true] -o <base> <base>.nsys-rep`.
