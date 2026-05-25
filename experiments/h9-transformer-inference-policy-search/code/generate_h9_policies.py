#!/usr/bin/env python
"""Generate the initial H9 vLLM inference policy grid."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def policy(
    name: str,
    description: str,
    *,
    dtype: str,
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
    enforce_eager: bool = False,
    gpu_memory_utilization: float,
    max_model_len: int,
    block_size: int,
    expected_role: str,
) -> dict[str, Any]:
    llm_kwargs = clean_kwargs(
        {
            "dtype": dtype,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "enforce_eager": enforce_eager,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "block_size": block_size,
            "trust_remote_code": True,
        }
    )
    return {
        "policy_name": name,
        "description": description,
        "expected_role": expected_role,
        "llm_kwargs": llm_kwargs,
        "search_tags": {
            "dtype": dtype,
            "quantization": quantization or "none",
            "kv_cache_dtype": kv_cache_dtype or "auto",
            "runtime_mode": "eager" if enforce_eager else "default",
        },
    }


def workloads() -> list[dict[str, Any]]:
    long_prompt = (
        "Summarize the following technical context for a systems researcher. "
        "Transformer inference has separate prefill and decode phases. Prefill "
        "is dominated by processing the prompt through attention and MLP blocks, "
        "while decode repeatedly reads and updates the KV cache. Precision choices "
        "for model weights, activations, attention kernels, and KV cache can change "
        "latency, memory, and output quality. "
    )
    return [
        {
            "name": "prefill_heavy",
            "max_tokens": 16,
            "prompts": [long_prompt * 16],
            "description": "Long prompt and short generation; prefill-dominated latency proxy.",
        },
        {
            "name": "decode_heavy",
            "max_tokens": 128,
            "prompts": ["Explain hardware-aware mixed precision for LLM serving in one paragraph."],
            "description": "Short prompt and long generation; decode-throughput proxy.",
        },
        {
            "name": "mixed",
            "max_tokens": 64,
            "prompts": [
                "Define KV cache quantization.",
                long_prompt * 4,
                "List two risks of low-precision attention.",
            ],
            "description": "Small varied batch mixing prompt and generation lengths.",
        },
    ]


def main() -> None:
    args = parse_args()
    common = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
    }
    policies = [
        policy(
            "bf16_default",
            "Default bf16 vLLM baseline with automatic KV-cache dtype.",
            dtype="bfloat16",
            expected_role="baseline",
            **common,
        ),
        policy(
            "fp16_default",
            "Default fp16 vLLM baseline with automatic KV-cache dtype.",
            dtype="float16",
            expected_role="baseline",
            **common,
        ),
        policy(
            "bf16_kv_fp8_e4m3",
            "bf16 model weights with FP8 E4M3 KV cache if supported.",
            dtype="bfloat16",
            kv_cache_dtype="fp8_e4m3",
            expected_role="kv_cache_memory_candidate",
            **common,
        ),
        policy(
            "fp16_kv_fp8_e4m3",
            "fp16 model weights with FP8 E4M3 KV cache if supported.",
            dtype="float16",
            kv_cache_dtype="fp8_e4m3",
            expected_role="kv_cache_memory_candidate",
            **common,
        ),
        policy(
            "bf16_kv_fp8",
            "bf16 model weights with vLLM generic FP8 KV cache if supported.",
            dtype="bfloat16",
            kv_cache_dtype="fp8",
            expected_role="kv_cache_memory_candidate",
            **common,
        ),
        policy(
            "fp16_kv_fp8",
            "fp16 model weights with vLLM generic FP8 KV cache if supported.",
            dtype="float16",
            kv_cache_dtype="fp8",
            expected_role="kv_cache_memory_candidate",
            **common,
        ),
        policy(
            "fp16_bitsandbytes",
            "vLLM bitsandbytes quantization path with fp16 compute dtype.",
            dtype="float16",
            quantization="bitsandbytes",
            expected_role="weight_quantization_candidate",
            **common,
        ),
        policy(
            "fp16_torchao",
            "vLLM torchao quantization path; requires local torchao compatibility.",
            dtype="float16",
            quantization="torchao",
            expected_role="weight_quantization_candidate",
            **common,
        ),
        policy(
            "bf16_eager",
            "bf16 eager-mode control for separating compile/runtime effects.",
            dtype="bfloat16",
            enforce_eager=True,
            expected_role="diagnostic_control",
            **common,
        ),
        policy(
            "fp16_eager",
            "fp16 eager-mode control for separating compile/runtime effects.",
            dtype="float16",
            enforce_eager=True,
            expected_role="diagnostic_control",
            **common,
        ),
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "runtime": "vllm",
        "search_objective": "pareto_latency_memory_quality",
        "candidate_policies": policies,
        "workloads": workloads(),
        "notes": [
            "Policies are launch configurations, not claims of support.",
            "Run inspect_h9_backend_inventory.py before benchmarking.",
            "A policy is reportable only after vLLM instantiates it on the target hardware.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"policies: {len(policies)}")
    print(f"workloads: {len(payload['workloads'])}")


if __name__ == "__main__":
    main()
