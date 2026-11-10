# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Continuous-batching + graph-capture rollout baseline for one GPU."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def _load_model(model_name, backend):
    config = AutoConfig.from_pretrained(model_name)
    is_hybrid = getattr(config, "model_type", "") == "qwen3_5" or backend == "qwen3_5"
    if is_hybrid:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        # Text-only rollout: the vision tower is dead weight and never called
        # when pixel_values are absent.
        model.model.visual = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    # The rollout engine checks this flag to decide whether it may pass a cache
    # class directly. Both model families accept one.
    model._supports_cache_class = True
    return model.eval(), is_hybrid


def _response_ids(batch):
    rows = []
    for row in range(batch.input_ids.shape[0]):
        start = int(batch.response_start_idx[row].item())
        mask = batch.attention_mask[row, start:].bool()
        rows.append(batch.input_ids[row, start:][mask].tolist())
    return rows


def _response_digest(rows):
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-backend", choices=("auto", "qwen3_5", "causal_lm"), default="auto")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--continuous-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default="runs/continuous_graph_baseline.json")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--graph-capture", dest="graph_capture", action="store_true", default=True)
    parser.add_argument("--no-graph-capture", dest="graph_capture", action="store_false")
    parser.add_argument("--graph-capture-batch-sizes", default="")
    parser.add_argument("--use-paged-cache", action="store_true")
    parser.add_argument("--paged-block-size", type=int, default=0)
    parser.add_argument("--save-response-ids", action="store_true")
    args = parser.parse_args()

    device = torch.device(get_accelerator().current_device_name())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, is_hybrid = _load_model(args.model, args.model_backend)
    model.to(device)

    request = _load_request(args.data, args.batch_size, tokenizer,
                            args.max_prompt_length, device)

    ladder = tuple(int(value) for value in args.graph_capture_batch_sizes.split(",") if value.strip())
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(
            use_graph_capture=args.graph_capture and not args.use_paged_cache,
            enable_profiling=True,
            enable_chunked_prefill=args.enable_chunked_prefill,
            prefill_chunk_size=args.prefill_chunk_size,
            graph_capture_batch_sizes=ladder or None,
            use_paged_cache=args.use_paged_cache,
            paged_block_size=args.paged_block_size or None,
        ),
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
        warmup_stats = None
        for _ in range(args.warmup):
            rollout.generate(request, sampling)
            warmup_stats = dict(rollout.get_last_continuous_stats() or {})
        get_accelerator().synchronize()
        warmup_peak_memory_mb = get_accelerator().max_memory_allocated() / (1024**2)
        get_accelerator().reset_peak_memory_stats()

        iterations = []
        response_rows = None
        for _ in range(args.iterations):
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
                "response_sha256": _response_digest(rows),
            })

    result = {
        "model": args.model,
        "model_type": getattr(model.config, "model_type", None),
        "is_hybrid": is_hybrid,
        "data": args.data,
        "batch_size": args.batch_size,
        "continuous_batch_size": args.continuous_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "prefill_chunk_size": args.prefill_chunk_size,
        "graph_capture": args.graph_capture and not args.use_paged_cache,
        "graph_capture_batch_sizes": list(ladder),
        "use_paged_cache": args.use_paged_cache,
        "paged_block_size": args.paged_block_size or None,
        "warmup_peak_memory_mb": warmup_peak_memory_mb,
        "warmup_stats": warmup_stats,
        "peak_memory_mb": get_accelerator().max_memory_allocated() / (1024**2),
        "peak_reserved_mb": get_accelerator().max_memory_reserved() / (1024**2),
        "iterations": iterations,
    }
    if args.save_response_ids:
        result["response_ids"] = response_rows
        result["response_sha256"] = _response_digest(response_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
