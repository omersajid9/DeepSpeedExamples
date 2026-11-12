# Rollout Generation Optimization — Meeting Summary

**System:** A100-SXM4-40GB (gpua021) · Qwen3-4B/8B · bf16 · torch 2.15.0+cu132, flashinfer 0.7.4 · vLLM 0.33.0 (oracle)
**Data:** LongBench-style prompts; 6k task = 6144 prompt / 16 new (Gumbel, T=0.7); 20k task = 20480 prompt / 8 new (greedy). Seed 0, 1 warmup + 1 measured iteration per cell. CB study runs at C=4 (6k) / C=2 (20k); vLLM oracles run at the same concurrency (batch 4 / batch 2, util 0.85, KV auto-sized).
**Figures:** `runs/meeting_figs_2026-11-12/` (PDF + PNG, 300 dpi). **Raw data:** `runs/a100_2026-11-11/`, `runs/cb_ablation_2026-11-12/` (+ `vllm_oracles/`).

---

## Slide 1 — TL;DR

- **C=1 (single request):** new-best config closes most of the gap to vLLM — **1.24× vLLM wall at 6k, 1.09× at 20k** (was 2.69× / 1.59× for dense eager).
- **Continuous batching (C≥2), vs vLLM at the same concurrency:** new-best **E is within 14–35% of vLLM wall at all four workloads (1.14–1.35×)**; eager+CB is 1.6–3.3× away. Paged KV is what gets us there.
- **Decode forward time: 4.0× faster at 6k, 6.2× at 20k** vs dense eager — FlashInfer paged decode + compiled sampler are the largest single levers.
- **FlashInfer decode wins at every CB width tested** (incl. C=4) — the "FI decode only good at batch 1" concern does not hold under real graph replay.
- **Chunked prefill: a pure peak-memory knob, not a speed knob** — latency ≈ no-chunk from c≥1024; peak memory −12–15% at c128.

## Slide 2 — Optimization ladder (C=1, 4B, wall ms)

| Step | 6k wall | Δ | 20k wall | Δ | What it is |
|---|---|---|---|---|---|
| Dense eager | 1080.7 | — | 2074.5 | — | baseline, no graphs |
| + CUDA graph | 825.4 | −24% | 1990.5 | −4% | single full-width graph |
| + paged decode | 702.0 | −15% | 1804.8 | −9% | paged KV, native kernels |
| + paged prefill | 679.9 | −3% | 1707.9 | −5% | paged prefill path |
| + torch.compile | 619.0 | −9% | 1589.5 | −7% | fused attention/sampler |
| **+ FI decode + compiled sampler** | **497.6** | **−20%** | **1429.3** | **−10%** | **new best: 2.17× / 1.45× total** |
| vLLM 0.33 | 401.1 | | 1307.4 | | oracle (wall only) |

- **Decode forward:** 704.5 → **175.6 ms** (4.0×) at 6k; 567.2 → **92.1 ms** (6.2×) at 20k.
- **Prefill forward:** 355.9 → 305.1 ms (−14%) at 6k; 1490.8 → 1325.1 ms (−11%) at 20k — prefill is now the residual gap.
- Gumbel vs greedy sampling: **no measurable overhead** (497.6 vs 502.7 ms; identical tokens).

## Slide 3 — Continuous batching (post-fix rerun)

Wall / decode ms, ratio to A (eager + CB):

| Workload (C) | A eager | B graph | C ladder | D paged | **E fused** | F FI+sampler |
|---|---|---|---|---|---|---|
| 4B 6k (C=4) wall | 2410 (1.00) | 2241 (.93) | 2238 (.93) | 1819 (.75) | **1668 (.69)** | 1729 (.72) |
| 4B 6k (C=4) decode | 1002 (1.00) | 829 (.83) | 830 (.83) | 404 (.40) | 380 (.38) | **238 (.24)** |
| 4B 20k (C=2) wall | 8822 (1.00) | 8757 (.99) | 8746 (.99) | 3232 (.37) | **3033 (.34)** | 3174 (.36) |
| 4B 20k (C=2) decode | 734 (1.00) | 651 (.89) | 652 (.89) | 278 (.38) | 269 (.37) | **121 (.17)** |
| 8B 6k (C=4) wall | 3204 (1.00) | 3031 (.95) | 3033 (.95) | 2622 (.82) | **2398 (.75)** | 2531 (.79) |
| 8B 6k (C=4) decode | 1077 (1.00) | 903 (.84) | 903 (.84) | 479 (.44) | 453 (.42) | **311 (.29)** |
| 8B 20k (C=2) wall | 10068 (1.00) | 9994 (.99) | 9993 (.99) | 4386 (.44) | **4217 (.42)** | 4357 (.43) |
| 8B 20k (C=2) decode | 770 (1.00) | 687 (.89) | 688 (.89) | 312 (.41) | 302 (.39) | **157 (.20)** |

**vs vLLM at the same concurrency** (wall ratio; vLLM oracles 1235.1 / 2646.4 / 1960.3 / 3709.8 ms for 4B-6k-C4 / 4B-20k-C2 / 8B-6k-C4 / 8B-20k-C2):

| Workload (C) | A eager | D paged | **E fused** | F FI+sampler |
|---|---|---|---|---|
| 4B 6k (C=4) | 1.95 | 1.47 | **1.35** | 1.40 |
| 4B 20k (C=2) | 3.33 | 1.22 | **1.15** | 1.20 |
| 8B 6k (C=4) | 1.63 | 1.34 | **1.22** | 1.29 |
| 8B 20k (C=2) | 2.71 | 1.18 | **1.14** | 1.17 |

**Takeaways**
- CUDA graphs alone under CB are worth only **~1–9%** — the dense-graph path does not exploit CB.
- **Paged KV is the real lever at width:** ~2.4–2.9× wall at C=2 (20k), 1.4–1.5× at C=4 (6k).
- **E (paged + torch.compile) is the wall champion at all 4 workloads.**
- **F (FlashInfer decode + compiled sampler) has the best decode everywhere (0.17–0.29× of eager)** and even wins wall vs D at 4B 6k (1729 vs 1819 ms).

## Slide 4 — FlashInfer decode at width

- Pre-fix data (all cells silently eager) suggested FI decode regressed at C≥2. After the lazy-init fix (below), **that prediction is overturned: FI decode leads at every width** — e.g. 4B 6k C=4: 238 ms decode vs 404 ms native paged (−41%); 8B 20k C=2: 157 vs 312 ms (−50%).
- FI prefill is a small cost (prefill +3–5% vs native paged at C=4/C=2) — decode gain dominates at C=4, roughly cancels at C=2.
- Memory: F ≈ D (±2%); both well below dense+CB.

## Slide 5 — Chunked prefill: latency vs memory (new-best config, C=1)

| Chunk | 6k wall (ms) | 6k peak (MiB) | 20k wall (ms) | 20k peak (MiB) |
|---|---|---|---|---|
| 128 | 2121.5 | 10429 | 6160.5 | 14190 |
| 512 | 682.6 | 10545 | 2203.5 | 13665 |
| 1024 | 556.4 | 10698 | 1765.7 | 13969 |
| 2048 | 524.0 | 11404 | 1549.5 | 15059 |
| 8192 | 492.7 | 13223 | 1454.9 | 19122 |
| **no chunk** | **497.6** | **12359** | **1429.3** | **16460** |

**Takeaway:** no wall-time benefit at any chunk size (c8192 ≈ no-chunk); chunking trades **12–15% peak memory at c128** for **+330% latency at c128**, flattening out from c≥1024. Use chunking only as an OOM escape hatch (it is the memory knob that makes 8B 20k E feasible — that cell needed `expandable_segments` even so).

## Slide 6 — Regime guidance

| Regime | Recommendation |
|---|---|
| Single request (C=1) | new-best config (paged prefill+decode, torch.compile, FI decode, compiled sampler): 1.24×/1.09× vLLM wall |
| Continuous batching C≥2 | paged KV mandatory; **E (fused/compiled)** for lowest wall; **F (FI decode)** when decode-bound (best decode, memory ≈ D) |
| Tight memory / OOM risk | chunk prefill at c128–c512 (−12–15% peak at 6k/20k, C=1); re-enable at c≥1024 if latency matters |
| Sampling | Gumbel at no cost; greedy for reproducibility |
| Do not bother | CUDA graphs under dense CB (≤9%), chunking for speed (none) |

## Figure guide (one line each)

| Figure | Takeaway |
|---|---|
| `cb_levels_vs_eager_latency` | Paged+CB cuts wall to 0.34–0.75× of eager; B/C (graphs) are near-flat |
| `cb_levels_vs_eager_prefill` | Prefill barely moves with graphs (0.99–1.00); FI prefill costs +3–5% |
| `cb_levels_vs_eager_decode` | Largest lever: F = 0.17–0.29× of eager decode at every workload |
| `cb_levels_vs_eager_tok_s` | Mirrors wall (throughput ratios) |
| `cb_levels_vs_eager_peak` | Paged cache is memory-efficient (0.65–0.76× dense+CB); E adds inductor buffers at 20k (~2×) |
| `cb_vs_vllm_total` | Your 6 CB steps vs vLLM at the same concurrency: new-best E within 14–35% of vLLM at every workload; eager+CB 1.6–3.3× away |
| `cb_vs_vllm_tok_s` | Same in throughput: E ≈ 0.74–0.87× of vLLM total tok/s |
| `cb_vs_vllm_prefill_vs_A` | Prefill barely moves through the ladder (vLLM has no phase data, so ratio to A) |
| `cb_vs_vllm_decode_vs_A` | Decode collapses 2.4–5.9× through the ladder — the FI-decode step is the last big drop |
| `ablation_vs_vllm_{6k,20k}_total` | C=1 ladder closes 2.69× → 1.24× (6k) and 1.59× → 1.09× (20k) vs vLLM (single-request study, batch 1) |
| `ablation_vs_vllm_{6k,20k}_tok_s` | Same in throughput: 0.80× / 0.92× of vLLM |
| `ablation_vs_dense_eager_{6k,20k}_prefill` | Residual gap: prefill only reaches 0.89× of dense eager (6k/20k) |
| `ablation_vs_dense_eager_{6k,20k}_decode` | Decode down to 0.25× / 0.16× of dense eager — where the wins live |
| `ablation_vs_dense_eager_{6k,20k}_peak` | Memory flat until paged prefill (−8–9%), +10–13% with torch.compile |
| `chunk_sweep_latency_memory` | Latency plateaus by c1024; memory grows with chunk — pure tradeoff curve |

## Caveats

- **Bug found & fixed in this study:** `fork/optimization-merged`'s paged capture paths check `layer.is_initialized` / call `layer.lazy_initialization()`, which the up-front-allocated `paged_cache.py` never defined → every paged graph capture raised `AttributeError` and **silently fell back to eager**. All 12 CB D/E/F cells were re-run on `fix/paged-layer-lazy-init` (`00d4ff3b7`); pre-fix numbers preserved in `runs/cb_ablation_2026-11-12/pre_lazy_init_fix/`. A/B/C (dense) are unaffected.
- **vLLM oracles.** The CB-width vLLM cells (`vllm_oracles/`, run 2026-11-12 via `r_vllm_cb_oracles.sh`, vLLM 0.33.0, util 0.85, KV auto-sized) carry total wall + derivable total throughput only (throughput = batch × new tokens / wall, same definition as the bench). The C=1 A100 oracles likewise have no phase or memory data, so all prefill/decode/peak figures are normalized to the dense-eager/A rung instead. Token SHAs can differ between engines at batch ≥ 2 (batch-dependent numerics / sampling streams) — expected, not a correctness issue.
- **E_fused** cells hit the dynamo recompile limit on the paged write path (consistent across all E cells; identical code path, so comparisons stay fair).
- **8B 20k E_fused** required `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (5.8 GiB inductor logit buffer vs 5.2 GiB free) — genuine A100-40GB memory wall; 4B 20k E runs without it.
- One measured iteration per cell (after 1 warmup); medians not applicable.
