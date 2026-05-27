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
    parser.add_argument(
        "--profile",
        choices=["h9_1_default", "h9_2_long_context"],
        default="h9_1_default",
        help="Policy/workload profile to generate.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--artifact-policies",
        type=Path,
        action="append",
        default=[],
        help="JSON file(s) with artifact-backed quantized policies to append.",
    )
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


def artifact_policy(
    spec: dict[str, Any],
    *,
    gpu_memory_utilization: float,
    max_model_len: int,
    block_size: int,
) -> dict[str, Any]:
    missing = [key for key in ("policy_name", "model_name", "quantization") if not spec.get(key)]
    if missing:
        raise SystemExit(f"Artifact policy is missing required keys {missing}: {spec}")
    dtype = str(spec.get("dtype", "float16"))
    quantization = str(spec["quantization"])
    llm_kwargs = clean_kwargs(
        {
            "dtype": dtype,
            "quantization": quantization,
            "kv_cache_dtype": spec.get("kv_cache_dtype"),
            "enforce_eager": bool(spec.get("enforce_eager", False)),
            "gpu_memory_utilization": float(spec.get("gpu_memory_utilization", gpu_memory_utilization)),
            "max_model_len": int(spec.get("max_model_len", max_model_len)),
            "block_size": int(spec.get("block_size", block_size)),
            "trust_remote_code": bool(spec.get("trust_remote_code", True)),
            "revision": spec.get("revision"),
            "tokenizer_revision": spec.get("tokenizer_revision"),
            "hf_overrides": spec.get("hf_overrides"),
        }
    )
    return {
        "policy_name": str(spec["policy_name"]),
        "model_name": str(spec["model_name"]),
        "description": str(
            spec.get(
                "description",
                f"Artifact-backed {quantization} policy for {spec['model_name']}.",
            )
        ),
        "expected_role": str(spec.get("expected_role", "artifact_backed_quantization_candidate")),
        "llm_kwargs": llm_kwargs,
        "search_tags": {
            "dtype": dtype,
            "quantization": quantization,
            "kv_cache_dtype": spec.get("kv_cache_dtype") or "auto",
            "runtime_mode": "eager" if spec.get("enforce_eager", False) else "default",
            "artifact_backed": "true",
        },
        "artifact_notes": str(spec.get("notes", "")),
    }


def load_artifact_policies(paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    policies = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Artifact policy file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        specs = payload.get("artifact_policies", payload if isinstance(payload, list) else [])
        if not isinstance(specs, list):
            raise SystemExit(f"Expected list or artifact_policies list in {path}")
        for spec in specs:
            if not isinstance(spec, dict):
                raise SystemExit(f"Expected artifact policy objects in {path}: {spec}")
            policies.append(
                artifact_policy(
                    spec,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    block_size=args.block_size,
                )
            )
    return policies


TORCHAO_INT8_WEIGHT_ONLY = {
    "_type": "Int8WeightOnlyConfig",
    "_version": 1,
    "_data": {
        "group_size": None,
        "granularity": {"_type": "PerRow", "_version": 1, "_data": {"dim": -1}},
        "set_inductor_config": True,
    },
}

TORCHAO_INT8_DYNAMIC_INT8_WEIGHT = {
    "_type": "Int8DynamicActivationInt8WeightConfig",
    "_version": 1,
    "_data": {
        "layout": {"_type": "PlainLayout", "_version": 1, "_data": {}},
        "act_mapping_type": {"_type": "MappingType", "_data": "SYMMETRIC"},
        "weight_only_decode": False,
        "granularity": {"_type": "PerRow", "_version": 1, "_data": {"dim": -1}},
        "set_inductor_config": True,
    },
}

TORCHAO_INT4_WEIGHT_ONLY_GROUP128 = {
    "_type": "Int4WeightOnlyConfig",
    "_version": 2,
    "_data": {
        "group_size": 128,
        "set_inductor_config": True,
        "int4_packing_format": {"_type": "Int4PackingFormat", "_data": "PLAIN"},
        "int4_choose_qparams_algorithm": {"_type": "Int4ChooseQParamsAlgorithm", "_data": "TINYGEMM"},
    },
}


def torchao_policy(
    name: str,
    description: str,
    *,
    torchao_config: dict[str, Any],
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    block_size: int,
    expected_role: str = "weight_quantization_candidate",
) -> dict[str, Any]:
    return policy(
        name,
        description,
        dtype=dtype,
        quantization="torchao",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        expected_role=expected_role,
    ) | {
        "llm_kwargs": clean_kwargs(
            {
                "dtype": dtype,
                "quantization": "torchao",
                "enforce_eager": False,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "block_size": block_size,
                "trust_remote_code": True,
                "hf_overrides": {
                    "quantization_config_dict_json": json.dumps(torchao_config),
                },
            }
        )
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


def long_context_workloads() -> list[dict[str, Any]]:
    base = (
        "Transformer serving systems must manage prompt prefill, autoregressive "
        "decode, attention kernels, model-weight layout, and KV-cache storage. "
        "Hardware-aware mixed precision can choose different formats for model "
        "weights and cached keys and values, but the benefit depends on whether "
        "the runtime actually uses efficient kernels and whether quality remains "
        "stable. "
    )
    section = (
        "In a long-context workload, KV-cache memory grows with sequence length, "
        "number of layers, hidden size, number of KV heads, and dtype width. "
        "A lower-precision KV cache should matter most when contexts are long "
        "enough that cache allocation and bandwidth become visible in measured "
        "latency and memory. "
    )
    long_prompt_4kish = (base + section) * 30
    long_prompt_2kish = (base + section) * 24
    mixed_prompt_1kish = (base + section) * 12
    return [
        {
            "name": "prefill_4k",
            "max_tokens": 16,
            "prompts": [long_prompt_4kish],
            "description": "Near-4k prompt with short generation; stresses prefill and large KV allocation.",
        },
        {
            "name": "decode_2k_context",
            "max_tokens": 128,
            "prompts": [long_prompt_2kish],
            "description": "Long prompt plus longer decode; stresses decode after a large KV cache exists.",
        },
        {
            "name": "batch_mixed_long",
            "max_tokens": 64,
            "prompts": [
                mixed_prompt_1kish,
                mixed_prompt_1kish + " Compare FP8 KV cache with default KV cache.",
                mixed_prompt_1kish + " Explain why quality gates are needed.",
                mixed_prompt_1kish + " Identify when a decode-only speedup is misleading.",
            ],
            "description": "Batch of long prompts with moderate generation; stresses memory pressure and batching.",
        },
    ]


def candidate_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
    }
    all_policies = [
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
            "vLLM torchao quantization path placeholder; requires an explicit torchao_config before benchmarking.",
            dtype="float16",
            quantization="torchao",
            expected_role="requires_backend_config",
            **common,
        ),
        torchao_policy(
            "fp16_torchao_int8wo",
            "vLLM TorchAO int8 weight-only online quantization with fp16 activation dtype.",
            dtype="float16",
            torchao_config=TORCHAO_INT8_WEIGHT_ONLY,
            **common,
        ),
        torchao_policy(
            "fp16_torchao_int8dyn_int8w",
            "vLLM TorchAO int8 dynamic-activation/int8-weight online quantization with fp16 dtype.",
            dtype="float16",
            torchao_config=TORCHAO_INT8_DYNAMIC_INT8_WEIGHT,
            **common,
        ),
        torchao_policy(
            "fp16_torchao_int4wo_g128",
            "vLLM TorchAO int4 weight-only online quantization with group size 128 and fp16 activation dtype.",
            dtype="float16",
            torchao_config=TORCHAO_INT4_WEIGHT_ONLY_GROUP128,
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
    if args.profile == "h9_2_long_context":
        keep = {
            "bf16_default",
            "fp16_default",
            "bf16_kv_fp8_e4m3",
            "fp16_kv_fp8_e4m3",
            "bf16_kv_fp8",
            "fp16_kv_fp8",
            "fp16_torchao_int8wo",
            "fp16_torchao_int8dyn_int8w",
            "fp16_torchao_int4wo_g128",
        }
        all_policies = [candidate for candidate in all_policies if candidate["policy_name"] in keep]
    all_policies.extend(load_artifact_policies(args.artifact_policies, args))
    return all_policies


def main() -> None:
    args = parse_args()
    policies = candidate_policies(args)
    selected_workloads = long_context_workloads() if args.profile == "h9_2_long_context" else workloads()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "runtime": "vllm",
        "profile": args.profile,
        "search_objective": "pareto_latency_memory_quality",
        "candidate_policies": policies,
        "workloads": selected_workloads,
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
