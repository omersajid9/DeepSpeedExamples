# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""One rollout ablation cell with per-phase metrics and path proof."""

import argparse
import dataclasses
import json
import math
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
    # non-CB 16 rows
    "noncb_greedy": (False, 0.0, False, False),
    "noncb_greedy_graph": (True, 0.0, False, False),
    "noncb_sampling": (False, 0.7, False, False),
    "noncb_sampling_graph": (True, 0.7, False, False),
    # CB=16
    "cb_greedy": (False, 0.0, True, False),
    "cb_greedy_graph": (True, 0.0, True, False),
    "cb_greedy_graph_buckets": (True, 0.0, True, True),
    "cb_sampling": (False, 0.7, True, False),
    "cb_sampling_graph": (True, 0.7, True, False),
    # CB Pareto, graph+buckets+greedy
    "cb3_greedy_graph_buckets": (True, 0.0, True, True),
    "cb6_greedy_graph_buckets": (True, 0.0, True, True),
    "cb12_greedy_graph_buckets": (True, 0.0, True, True),
    "cb16_greedy_graph_buckets": (True, 0.0, True, True),
    # gates
    "gate_topk1_sampling": (False, 0.7, False, False),
    "gate_t1_sampling": (False, 1.0, False, False),
    # legacy reference only
    "legacy_single_graph": (True, 0.0, True, False),
}

C_BATCH = {
    "cb3_greedy_graph_buckets": 3,
    "cb6_greedy_graph_buckets": 6,
    "cb12_greedy_graph_buckets": 12,
    "cb16_greedy_graph_buckets": 16,
}

TASKS = {
    "6k6k": dict(data="data/longbench_20k.jsonl", batch=1, c_batch=1,
                 max_prompt=6144, max_new=6144, warmup=1, iterations=1),
    "retool": dict(data="data/retool_prompts.jsonl", batch=16, c_batch=16,
                   max_prompt=2048, max_new=512, warmup=1, iterations=3),
    "longbench": dict(data="data/longbench_mfqa.jsonl", batch=16, c_batch=16,
                      max_prompt=8192, max_new=256, warmup=1, iterations=3),
}


def build_config(use_graph, buckets, c_batch):
    fields = {field.name for field in dataclasses.fields(HybridEngineRolloutConfig)}
    kwargs = {"use_graph_capture": use_graph, "enable_profiling": True}
    if "enable_chunked_prefill" in fields:
        kwargs["enable_chunked_prefill"] = False
    if "prefill_chunk_size" in fields:
        kwargs["prefill_chunk_size"] = 512
    if "graph_capture_batch_sizes" in fields:
        if use_graph and buckets:
            kwargs["graph_capture_batch_sizes"] = None  # default ladder
        elif use_graph:
            kwargs["graph_capture_batch_sizes"] = (c_batch,)
        else:
            kwargs["graph_capture_batch_sizes"] = None
    return HybridEngineRolloutConfig(**kwargs)


def row_stop_reason(row, eos_id, budget):
    if not row:
        return "empty"
    if eos_id is not None and row[-1] == eos_id:
        return "EOS"
    if len(row) >= budget:
        return "budget"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    use_graph, temperature, continuous, buckets = ARMS[args.arm]
    task = TASKS[args.task]
    c_batch = C_BATCH.get(args.arm, task["c_batch"])
    if not continuous:
        c_batch = None
    top_k = 1 if args.arm == "gate_topk1_sampling" else -1
    top_p = 1.0 if args.arm == "gate_t1_sampling" else 0.9
    device = torch.device(get_accelerator().current_device_name())

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    module, is_hybrid = _load_model(MODEL, "auto")
    module.to(device)
    request = _load_request(task["data"], task["batch"], tokenizer, task["max_prompt"], device)
    effective_prompt_width = int(request.prompt_ids.shape[1])
    rollout = HybridEngineRollout(SimpleNamespace(module=module), tokenizer, cfg=build_config(use_graph, buckets, c_batch or task["c_batch"]))
    sampling = SamplingConfig(
        max_new_tokens=task["max_new"],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n_samples_per_prompt=1,
        continuous_batch_size=c_batch,
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
        response_logprobs = None
        for _ in range(task["iterations"]):
            start = time.perf_counter()
            batch = rollout.generate(request, sampling)
            get_accelerator().synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            rows = _response_ids(batch)
            if response_rows is None:
                response_rows = rows
                response_logprobs = getattr(batch, "response_logprobs", None)
                if response_logprobs is not None:
                    response_logprobs = response_logprobs.detach().cpu()
            iterations.append({
                "wall_ms": wall_ms,
                "profile": dict(rollout.get_last_profile() or {}),
                "continuous_stats": dict(rollout.get_last_continuous_stats() or {}),
            })

    peak_mb = get_accelerator().max_memory_allocated() / (1024**2)
    peak_reserved = get_accelerator().max_memory_reserved() / (1024**2)
    eos_id = tokenizer.eos_token_id
    output_lengths = [len(row) for row in response_rows] if response_rows else []
    stop_reasons = [row_stop_reason(row, eos_id, task["max_new"]) for row in response_rows] if response_rows else []
    logs = response_logprobs
    logprobs_present = logs is not None
    logprobs_finite = bool(logprobs_present and torch.isfinite(logs).all().item())
    logprobs_max = float(logs.max().item()) if logprobs_present and logs.numel() else None
    logprobs_min = float(logs.min().item()) if logprobs_present and logs.numel() else None
    paths = []
    for it in iterations:
        cs = it["continuous_stats"]
        paths.append({
            "graph_captures": cs.get("graph_captures"),
            "decode_bucket_widths": cs.get("decode_bucket_widths"),
            "prefill_padded_rows": cs.get("prefill_padded_rows"),
            "keys": sorted(cs.keys()),
        })
    result = {
        "arm": args.arm,
        "task": args.task,
        "model": MODEL,
        "model_type": getattr(module.config, "model_type", None),
        "attn_implementation": getattr(module.config, "_attn_implementation", None),
        "is_hybrid": is_hybrid,
        "data": task["data"],
        "batch_size": task["batch"],
        "continuous_batch_size": c_batch,
        "n_samples_per_prompt": 1,
        "equal_padded_prompt_width": True,
        "effective_prompt_width": effective_prompt_width,
        "max_prompt_length": task["max_prompt"],
        "max_new_tokens": task["max_new"],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "graph_capture": use_graph,
        "buckets": buckets,
        "enable_chunked_prefill": False,
        "prefill_chunk_size": 0,
        "warmup_peak_memory_mb": warmup_peak,
        "peak_memory_mb": peak_mb,
        "peak_reserved_mb": peak_reserved,
        "iterations": iterations,
        "response_ids": response_rows,
        "output_lengths": output_lengths,
        "stop_reasons": stop_reasons,
        "response_sha256": __import__("hashlib").sha256(json.dumps(response_rows).encode()).hexdigest(),
        "logprobs_present": logprobs_present,
        "logprobs_finite": logprobs_finite,
        "logprobs_max": logprobs_max,
        "logprobs_min": logprobs_min,
        "path_proof": paths,
        "warmup_stats": warmup_stats,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {args.task}/{args.arm} -> {out}", flush=True)


if __name__ == "__main__":
    main()
