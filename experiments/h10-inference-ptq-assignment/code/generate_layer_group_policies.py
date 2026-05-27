#!/usr/bin/env python3
"""Generate H10 layer/group mixed-precision backend policies for Llama."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT = Path("experiments/h10-inference-ptq-assignment/results/layer_group_policy_candidates.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def workloads() -> list[dict[str, Any]]:
    long_prompt = (
        "Transformer inference has separate prefill and decode phases. Prefill "
        "processes the prompt through attention and MLP blocks, while decode "
        "repeatedly reads and updates the KV cache. Layer-wise mixed precision "
        "must preserve quality while mapping to real kernels. "
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
                "Define layer-wise post-training quantization.",
                long_prompt * 4,
                "List two risks of quantizing MLP projections.",
            ],
            "description": "Small varied batch mixing prompt and generation lengths.",
        },
    ]


def baseline_policy(name: str, dtype: str) -> dict[str, Any]:
    return {
        "policy_name": name,
        "description": f"Transformers {dtype} baseline for matched layer/group backend comparisons.",
        "expected_role": "baseline",
        "llm_kwargs": {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        },
        "search_tags": {
            "dtype": dtype,
            "quantization": "none",
            "runtime_mode": "transformers",
            "layer_group_backend": "none",
        },
    }


def layer_group_policy(
    *,
    name: str,
    description: str,
    dtype: str,
    config: dict[str, Any],
    module_regex: str,
    group_name: str,
    selection_source: str,
) -> dict[str, Any]:
    return {
        "policy_name": name,
        "description": description,
        "expected_role": "layer_group_mixed_precision_candidate",
        "llm_kwargs": {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        },
        "layer_group_policy": {
            "backend": "torchao_fqn_to_config",
            "default_precision": dtype,
            "selection_source": selection_source,
            "groups": [
                {
                    "group_name": group_name,
                    "module_regex": module_regex,
                    "config": config,
                }
            ],
        },
        "search_tags": {
            "dtype": dtype,
            "quantization": config["type"],
            "runtime_mode": "transformers",
            "layer_group_backend": "torchao_fqn_to_config",
        },
    }


def candidate_policies() -> list[dict[str, Any]]:
    late_mlp_regex = r"model\.layers\.(2[4-9]|3[0-1])\.mlp\.(gate_proj|up_proj|down_proj)"
    late_gate_up_regex = r"model\.layers\.(2[4-9]|3[0-1])\.mlp\.(gate_proj|up_proj)"
    return [
        baseline_policy("bf16_transformers", "bfloat16"),
        baseline_policy("fp16_transformers", "float16"),
        layer_group_policy(
            name="h10_lg_late_mlp_int8wo",
            description=(
                "TorchAO FqnToConfig policy that keeps the Llama model in fp16 but "
                "applies int8 weight-only quantization to late-layer MLP projections."
            ),
            dtype="float16",
            config={"type": "int8_weight_only", "group_size": None},
            module_regex=late_mlp_regex,
            group_name="late_mlp_proj_layers_24_31",
            selection_source="starter_backend_policy_from_h6_h7_late_mlp_low_risk_pattern",
        ),
        layer_group_policy(
            name="h10_lg_late_gate_up_int8wo",
            description=(
                "TorchAO FqnToConfig policy that quantizes only late-layer MLP gate/up "
                "projections, leaving down projections and attention in fp16."
            ),
            dtype="float16",
            config={"type": "int8_weight_only", "group_size": None},
            module_regex=late_gate_up_regex,
            group_name="late_mlp_gate_up_layers_24_31",
            selection_source="starter_backend_policy_from_h6_h7_gate_up_low_risk_pattern",
        ),
        layer_group_policy(
            name="h10_lg_late_gate_up_int4wo_g128",
            description=(
                "TorchAO FqnToConfig policy that applies int4 weight-only group-128 "
                "quantization to late-layer MLP gate/up projections."
            ),
            dtype="float16",
            config={"type": "int4_weight_only", "group_size": 128},
            module_regex=late_gate_up_regex,
            group_name="late_mlp_gate_up_layers_24_31",
            selection_source="starter_backend_policy_from_h6_h7_gate_up_low_risk_pattern",
        ),
    ]


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "runtime": "transformers_torchao_layer_group",
        "profile": "h10_layer_group_backend",
        "search_objective": "pareto_latency_memory_quality_layer_group",
        "max_model_len": args.max_model_len,
        "candidate_policies": candidate_policies(),
        "workloads": workloads(),
        "notes": [
            "These are real backend policies for Transformers plus TorchAO FqnToConfig.",
            "Only rows that complete benchmark and prompt-NLL quality runs should be used as H10 evidence.",
            "The starter Llama groups intentionally quantize late MLP projections before adding broader layer search.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"policies: {len(payload['candidate_policies'])}")
    print(f"workloads: {len(payload['workloads'])}")


if __name__ == "__main__":
    main()
