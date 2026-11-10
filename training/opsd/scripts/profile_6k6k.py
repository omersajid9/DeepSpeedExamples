# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Profile one eager batch-1 decode step at 6k context."""

import json
import shutil
import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM

MODEL = "/dev/shm/hf_cache/models/Qwen3-4B"
CONTEXT = 6144


def bucket(name):
    low = name.lower()
    if "attention" in low or "softmax" in low or "sdpa" in low or "flash" in low or "bmm" in low:
        return "attention"
    if "layer_norm" in low or "rms" in low or "norm" in low:
        return "norm"
    if "lm_head" in low or "logits" in low:
        return "lm_head"
    if "gemm" in low or "addmm" in low or "mm" in low or "silu" in low or "gate" in low:
        return "mlp"
    return "other"


def main():
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, CONTEXT + 1), device=device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
        past = out.past_key_values
        next_ids = input_ids[:, -1:]
        pos = torch.tensor([CONTEXT], device=device)
        for _ in range(2):
            model(input_ids=next_ids, attention_mask=attn, past_key_values=past, use_cache=True, cache_position=pos)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=True) as prof:
            model(input_ids=next_ids, attention_mask=attn, past_key_values=past, use_cache=True, cache_position=pos)
        torch.cuda.synchronize()

    events = []
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_cuda_time_total", 0.0)
        if cuda_us <= 0:
            continue
        events.append({"name": evt.key, "cuda_us": cuda_us, "bucket": bucket(evt.key)})
    events.sort(key=lambda item: item["cuda_us"], reverse=True)
    by_bucket = {}
    for evt in events:
        by_bucket[evt["bucket"]] = by_bucket.get(evt["bucket"], 0.0) + evt["cuda_us"]
    total = sum(by_bucket.values()) or 1.0
    result = {
        "context": CONTEXT,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "top5_kernels": events[:5],
        "bucket_pct": {key: value / total * 100.0 for key, value in sorted(by_bucket.items(), key=lambda x: -x[1])},
        "mask_kernels": [evt for evt in events if "mask" in evt["name"].lower() or "causal" in evt["name"].lower()][:10],
        "ncu_available": shutil.which("ncu") is not None,
    }
    Path("runs/profile_6k6k.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
