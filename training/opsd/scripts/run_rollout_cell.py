# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Run one rollout arm/task cell against the checked-out DeepSpeed tree."""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from benchmarks.bench_continuous_graph import _load_model, _load_request, _response_ids

MODEL = "/dev/shm/hf_cache/models/Qwen3-4B"
OPSD = Path(__file__).resolve().parent.parent

ARMS = {
    # name: (use_graph_capture, temperature, use_continuous_batching)
    "single_graph": (True, 0.0, True),
    "eager_sampling": (False, 0.7, False),
    "multiple_graphs": (True, 0.0, True),
    "device_sampler": (False, 0.7, True),
    "static_cache": (False, 0.7, True),
    "all_greedy": (True, 0.0, True),
    "all_sampling": (True, 0.7, True),
}

TASKS = {
    "6k6k": dict(data="data/longbench_20k.jsonl", batch=1, c_batch=1,
                 max_prompt=6144, max_new=6144, warmup=1, iterations=1),
    "retool": dict(data="data/retool_prompts.jsonl", batch=16, c_batch=3,
                   max_prompt=2048, max_new=512, warmup=1, iterations=3),
    "longbench": dict(data="data/longbench_mfqa.jsonl", batch=16, c_batch=3,
                      max_prompt=8192, max_new=256, warmup=1, iterations=3),
}


def build_config(use_graph):
    fields = {field.name for field in dataclasses.fields(HybridEngineRolloutConfig)}
    kwargs = {"use_graph_capture": use_graph, "enable_profiling": True}
    if "enable_chunked_prefill" in fields:
        kwargs["enable_chunked_prefill"] = False
    if "prefill_chunk_size" in fields:
        kwargs["prefill_chunk_size"] = 512
    if "graph_capture_batch_sizes" in fields:
        kwargs["graph_capture_batch_sizes"] = None
    return HybridEngineRolloutConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    use_graph, temperature, continuous = ARMS[args.arm]
    task = TASKS[args.task]
    device = torch.device(get_accelerator().current_device_name())

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    module, is_hybrid = _load_model(MODEL, "auto")
    module.to(device)
    request = _load_request(task["data"], task["batch"], tokenizer, task["max_prompt"], device)
    rollout = HybridEngineRollout(
        SimpleNamespace(module=module),
        tokenizer,
        cfg=build_config(use_graph),
    )
    sampling = SamplingConfig(
        max_new_tokens=task["max_new"],
        temperature=temperature,
        top_p=0.9,
        top_k=-1,
        n_samples_per_prompt=1,
        continuous_batch_size=task["c_batch"] if continuous else None,
    )

    with torch.no_grad():
        for _ in range(task["warmup"]):
            rollout.generate(request, sampling)
            warmup_stats = dict(rollout.get_last_continuous_stats() or {})
        get_accelerator().synchronize()
        warmup_peak = get_accelerator().max_memory_allocated() / (1024**2)
        get_accelerator().reset_peak_memory_stats()

        iterations = []
        response_rows = None
        for _ in range(task["iterations"]):
            start = time.perf_counter()
            batch = rollout.generate(request, sampling)
            get_accelerator().synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            rows = _response_ids(batch)
            if response_rows is None:
                response_rows = rows
            iterations.append({
                "wall_ms": wall_ms,
                "profile": dict(rollout.get_last_profile() or {}),
                "continuous_stats": dict(rollout.get_last_continuous_stats() or {}),
            })

    peak_mb = get_accelerator().max_memory_allocated() / (1024**2)
    peak_reserved = get_accelerator().max_memory_reserved() / (1024**2)
    result = {
        "arm": args.arm,
        "task": args.task,
        "model": MODEL,
        "model_type": getattr(module.config, "model_type", None),
        "is_hybrid": is_hybrid,
        "data": task["data"],
        "batch_size": task["batch"],
        "continuous_batch_size": task["c_batch"] if continuous else None,
        "max_prompt_length": task["max_prompt"],
        "max_new_tokens": task["max_new"],
        "temperature": temperature,
        "top_p": 0.9,
        "graph_capture": use_graph,
        "enable_chunked_prefill": False,
        "prefill_chunk_size": 0,
        "warmup_peak_memory_mb": warmup_peak,
        "warmup_stats": warmup_stats,
        "peak_memory_mb": peak_mb,
        "peak_reserved_mb": peak_reserved,
        "iterations": iterations,
        "response_ids": response_rows,
        "response_sha256": hashlib.sha256(json.dumps(response_rows).encode()).hexdigest(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {args.task}/{args.arm} -> {out}", flush=True)


if __name__ == "__main__":
    main()
