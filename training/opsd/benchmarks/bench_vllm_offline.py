# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Offline vLLM comparison for the OPSD rollout workloads.

Uses the same tokenizer, chat template, truncation length, batch size, chunk budget,
sampler settings, and generated-token count as ``bench_continuous_graph.py``.
Run this from a vLLM environment, not the DeepSpeed environment.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _load_prompts(path, batch_size, tokenizer, max_prompt_length):
    texts = []
    with open(path) as handle:
        for line in handle:
            if len(texts) == batch_size:
                break
            payload = json.loads(line)
            text = payload.get("prompt") or payload.get("input") or payload.get("question")
            if text is None:
                raise ValueError(f"no prompt field in {path}")
            texts.append(text)
    if len(texts) != batch_size:
        raise ValueError(f"need {batch_size} prompts in {path}, found {len(texts)}")

    prompt_token_ids = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(rendered,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_prompt_length)["input_ids"]
        prompt_token_ids.append(ids)
    return prompt_token_ids


def _digest(rows):
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = _load_prompts(args.data, args.batch_size, tokenizer, args.max_prompt_length)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    max_model_len = args.max_prompt_length + args.max_new_tokens

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.prefill_chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        ignore_eos=True,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            llm.generate(prompts, sampling)
        torch.cuda.synchronize()
        warmup_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()

        iterations = []
        response_rows = None
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            rows = [list(output.outputs[0].token_ids) for output in outputs]
            if response_rows is None:
                response_rows = rows
            iterations.append({"wall_ms": wall_ms, "response_ids": rows})

    result = {
        "engine": "vllm",
        "model": args.model,
        "data": args.data,
        "batch_size": args.batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "enable_prefix_caching": args.enable_prefix_caching,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_peak_memory_mb": warmup_peak_mb,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "response_ids": response_rows,
        "response_sha256": _digest(response_rows),
        "iterations": iterations,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.output}")
    for iteration in iterations:
        print(f"wall {iteration['wall_ms']:.1f} ms; tokens/s "
              f"{args.batch_size * args.max_new_tokens / (iteration['wall_ms'] / 1000.0):.2f}")


if __name__ == "__main__":
    main()