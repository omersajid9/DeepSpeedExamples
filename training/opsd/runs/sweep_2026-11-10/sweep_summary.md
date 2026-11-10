# OPSD rollout optimization sweep — results

Environment: A40 46,068 MiB; torch 2.14.1+cu126; transformers 5.22.2; DeepSpeed local checkout. Baseline is continuous eager (no graph, no chunk). Optimized is decode CUDA graphs + chunked prefill. Qwen3.5 prefill graphs are run eagerly because hybrid recurrent-state capture is not token-stable; its decode graphs and chunked prefill remain active.

## Per-cell results

| Model | Workload | Arm | C | Wall (s) | Prefill (s) | Decode (s) | Wall tok/s | Prefill tok/s | Decode tok/s | Peak (MiB) | Match | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3-4b | 10k | baseline | 3 | 49.31 | 21.42 | 27.30 | 10.38 | 7649.68 | 18.76 | 18722 | False | token_mismatch |
| qwen3-4b | 10k | optimized | 3 | 68.64 | 48.24 | 19.88 | 7.46 | 3396.70 | 25.76 | 16958 | False | token_mismatch |
| qwen3-4b | 20k | baseline | 3 | 152.31 | 109.48 | 42.38 | 3.36 | 2992.98 | 12.08 | 31491 | False | token_mismatch |
| qwen3-4b | 20k | optimized | 3 | 204.10 | 162.70 | 40.36 | 2.51 | 2014.03 | 12.68 | 26111 | False | token_mismatch |
| qwen3-4b | longbench | optimized | 3 | 162.30 | 34.31 | 126.81 | 22.80 | 3820.57 | 29.18 | 15258 | None | missing |
| qwen3-4b | retool | baseline | 3 | 159.67 | 0.91 | 156.38 | 51.31 | 35977.50 | 52.39 | 8454 | False | token_mismatch |
| qwen3-4b | retool | optimized | 3 | 86.87 | 0.92 | 83.80 | 94.30 | 35613.20 | 97.75 | 8675 | False | token_mismatch |
| qwen3-8b | 10k | baseline | 3 | 56.10 | 31.54 | 24.26 | 9.13 | 5194.86 | 21.10 | 27430 | False | token_mismatch |
| qwen3-8b | 10k | optimized | 3 | 81.02 | 58.39 | 22.25 | 6.32 | 2806.02 | 23.01 | 24868 | False | token_mismatch |

## Optimization deltas

| Model | Workload | Wall Δ | Prefill Δ | Decode Δ | Wall tok/s Δ | Peak Δ | Match |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3-4b | 10k | -28.2 | -55.6 | 37.3 | -28.2 | -9.4 | False |
| qwen3-4b | 20k | -25.4 | -32.7 | 5.0 | -25.4 | -17.1 | False |
| qwen3-4b | retool | 83.8 | -1.0 | 86.6 | 83.8 | 2.6 | False |
| qwen3-8b | 10k | -30.8 | -46.0 | 9.0 | -30.8 | -9.3 | False |

