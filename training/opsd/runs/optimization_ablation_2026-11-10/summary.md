# Rollout optimization ablation - Qwen3-4B

Tasks: 6k prompt + 6k decode (batch 1, C=1, 1 measured iteration), ReTool (batch 16, C=3, 512 new, 3 iterations), LongBench MFQA (batch 16, C=3, 8192 prompt, 256 new, 3 iterations). Sampling arms use temperature=0.7, top_p=0.9, top_k=-1. Chunked prefill is disabled in every arm.

| Task | Arm | Wall (s) | Prefill (s) | Decode (s) | Tok/s | Peak (MiB) | Graph | Sampling |
|---|---|---:|---:|---:|---:|---:|---|---|
| retool | all_greedy | 86.94 | 0.93 | 83.84 | 94.23 | 8479 | yes | 0.0 |

