# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Continuous-batching + graph-capture rollout baseline for one GPU."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)


def _load_request(path, batch_size, tokenizer, max_prompt_length, device):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) == batch_size:
                break
    if len(prompts) != batch_size:
        raise ValueError(f"need {batch_size} prompts, found {len(prompts)}")

    encoded = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded.append(
            tokenizer(rendered, add_special_tokens=False, truncation=True,
                      max_length=max_prompt_length)["input_ids"])

    width = max(len(ids) for ids in encoded)
    pad = tokenizer.pad_token_id
    prompt_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    prompt_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        n = len(ids)
        prompt_ids[row, width - n:] = torch.tensor(ids, dtype=torch.long)
        prompt_mask[row, width - n:] = 1

    return RolloutRequest(prompt_ids.to(device), prompt_mask.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--continuous-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default="runs/continuous_graph_baseline.json")
    args = parser.parse_args()

    device = torch.device(get_accelerator().current_device_name())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    request = _load_request(args.data, args.batch_size, tokenizer,
                            args.max_prompt_length, device)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True),
    )
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n_samples_per_prompt=1,
        continuous_batch_size=args.continuous_batch_size,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            rollout.generate(request, sampling)
        get_accelerator().synchronize()
        get_accelerator().reset_peak_memory_stats()

        iterations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            rollout.generate(request, sampling)
            get_accelerator().synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            iterations.append({
                "wall_ms": wall_ms,
                "profile": dict(rollout.get_last_profile() or {}),
                "continuous_stats": dict(rollout.get_last_continuous_stats() or {}),
            })

    result = {
        "model": args.model,
        "data": args.data,
        "batch_size": args.batch_size,
        "continuous_batch_size": args.continuous_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "graph_capture": True,
        "peak_memory_mb": get_accelerator().max_memory_allocated() / (1024**2),
        "iterations": iterations,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
