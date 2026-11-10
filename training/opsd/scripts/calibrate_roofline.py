# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure achieved DRAM bandwidth of one eager batch-1 decode forward."""

import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL = "/dev/shm/hf_cache/models/Qwen3-4B"
CONTEXTS = {"retool": 2048, "6k6k": 6144, "longbench": 8192}
PEAK_GBPS = 696.0


def main():
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    model.eval()
    cfg = model.config
    dtype_bytes = 2
    kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * dtype_bytes
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    results = {}
    for name, context in CONTEXTS.items():
        input_ids = torch.randint(0, cfg.vocab_size, (1, context + 1), device=device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
            past = out.past_key_values
            next_ids = input_ids[:, -1:]
            next_attn = attn
            times = []
            for _ in range(6):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(input_ids=next_ids, attention_mask=next_attn, past_key_values=past,
                      use_cache=True, cache_position=torch.tensor([context], device=device))
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        ms = statistics.median(times[1:])
        kv_bytes = kv_per_token * (context + 1)
        total_bytes = weight_bytes + kv_bytes
        achieved = total_bytes / (ms / 1000.0) / 1e9
        results[name] = {
            "context": context,
            "ms_per_decode": ms,
            "weight_gib": weight_bytes / 2**30,
            "kv_mib": kv_bytes / 2**20,
            "achieved_gbps": achieved,
            "pct_of_peak": achieved / PEAK_GBPS * 100.0,
            "roofline_ms": total_bytes / (PEAK_GBPS * 1e9) * 1000.0,
        }
        print(f"{name}: {ms:.2f} ms, {achieved:.1f} GB/s, {achieved/PEAK_GBPS*100:.1f}% of peak", flush=True)
    Path("runs/roofline_calibration.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
