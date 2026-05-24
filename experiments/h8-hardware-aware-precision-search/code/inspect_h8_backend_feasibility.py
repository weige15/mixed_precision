#!/usr/bin/env python
"""Inspect whether H8 selective-rescue targets are expressible under QLoRA.

This is a no-training probe. It mirrors the H8 runner's QLoRA load path, wraps
the model with LoRA, resolves candidate rescue modules, and reports whether
each target is already high precision or backed by a bitsandbytes quantized
linear module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = Path("experiments/h8-hardware-aware-precision-search/results/h8_policy_candidates.json")
DEFAULT_OUTPUT_DIR = Path("experiments/h8-hardware-aware-precision-search/results/backend_feasibility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Exact model name from h8_policy_candidates.json.",
    )
    parser.add_argument(
        "--policy-name",
        default="h8_rescue_norm_logits",
        help="Candidate policy to inspect for the selected model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-lora-wrap",
        action="store_true",
        help="Inspect the base quantized model before PEFT wrapping. Default mirrors the training runner.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for metadata debugging. Real QLoRA feasibility requires CUDA.",
    )
    return parser.parse_args()


def load_candidate(path: Path, model_name: str, policy_name: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    models = data.get("models")
    if models is None:
        models = [
            {
                "model_name": data.get("model_filter"),
                "candidate_policies": data.get("candidate_policies", []),
            }
        ]
    for model in models:
        if model.get("model_name") != model_name:
            continue
        for policy in model.get("candidate_policies", []):
            if policy.get("policy_name") == policy_name:
                return policy
        available = sorted(p.get("policy_name", "") for p in model.get("candidate_policies", []))
        raise SystemExit(f"Policy not found for {model_name}: {policy_name}. Available: {available}")
    available_models = sorted(str(m.get("model_name")) for m in models)
    raise SystemExit(f"Model not found in {path}: {model_name}. Available: {available_models}")


def require_packages() -> None:
    missing = []
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit("Missing required package(s): " + ", ".join(missing))


def infer_lora_targets(model: Any) -> list[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and is_linear_like(module):
            found.add(leaf)
    if not found:
        raise SystemExit("Could not infer LoRA target modules for this model.")
    return sorted(found)


def is_linear_like(module: Any) -> bool:
    return "linear" in module.__class__.__name__.lower()


def resolve_module(model: Any, target: str) -> tuple[str | None, Any | None, list[str]]:
    named_modules = dict(model.named_modules())
    if target in named_modules:
        return target, named_modules[target], []
    matches = [name for name in named_modules if name.endswith(target)]
    if len(matches) == 1:
        return matches[0], named_modules[matches[0]], []
    return None, None, sorted(matches)


def tensor_dtype_name(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def local_tensors(module: Any) -> list[dict[str, str]]:
    tensors = []
    for name, param in module.named_parameters(recurse=False):
        tensors.append({"kind": "parameter", "name": name, "dtype": tensor_dtype_name(param)})
    for name, buffer in module.named_buffers(recurse=False):
        tensors.append({"kind": "buffer", "name": name, "dtype": tensor_dtype_name(buffer)})
    return tensors


def child_summary(module: Any) -> dict[str, Any] | None:
    base_layer = getattr(module, "base_layer", None)
    if base_layer is None:
        return None
    return {
        "class_name": base_layer.__class__.__name__,
        "class_module": base_layer.__class__.__module__,
        "is_quantized_linear": is_quantized_linear(base_layer),
        "local_tensors": local_tensors(base_layer),
    }


def is_quantized_linear(module: Any) -> bool:
    class_name = module.__class__.__name__.lower()
    class_module = module.__class__.__module__.lower()
    weight = getattr(module, "weight", None)
    return (
        "linear4bit" in class_name
        or "linear8bit" in class_name
        or "bitsandbytes" in class_module
        or hasattr(weight, "quant_state")
        or weight.__class__.__name__.lower() in {"params4bit", "int8params"}
    )


def classify_feasibility(module: Any) -> tuple[str, str]:
    base_layer = getattr(module, "base_layer", None)
    quantized = is_quantized_linear(module) or (base_layer is not None and is_quantized_linear(base_layer))
    if quantized:
        return (
            "quantized_target",
            "Target is backed by bitsandbytes quantization; rescue likely requires loading or replacing this module in higher precision, not just casting it.",
        )
    if is_linear_like(module):
        return (
            "non_quantized_linear",
            "Target is a normal linear-like module in the loaded graph; check dtype and memory impact before counting it as rescued.",
        )
    return (
        "already_non_quantized",
        "Target is not a quantized linear module in this backend path; it may already run in regular precision.",
    )


def inspect_target(model: Any, target: str) -> dict[str, Any]:
    resolved_name, module, ambiguous_matches = resolve_module(model, target)
    if module is None:
        return {
            "requested_name": target,
            "resolved_name": None,
            "status": "missing" if not ambiguous_matches else "ambiguous",
            "ambiguous_matches": ambiguous_matches,
        }
    status, interpretation = classify_feasibility(module)
    return {
        "requested_name": target,
        "resolved_name": resolved_name,
        "status": status,
        "interpretation": interpretation,
        "class_name": module.__class__.__name__,
        "class_module": module.__class__.__module__,
        "is_quantized_linear": is_quantized_linear(module),
        "local_tensors": local_tensors(module),
        "base_layer": child_summary(module),
    }


def main() -> None:
    args = parse_args()
    require_packages()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Re-run on the target GPU host, or pass --allow-cpu for metadata debugging.")

    policy = load_candidate(args.candidates, args.model_name, args.policy_name)
    local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_targets: list[str] = []
    if not args.no_lora_wrap:
        lora_targets = infer_lora_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
            ),
        )

    target_reports = [inspect_target(model, target) for target in policy.get("rescue_modules", [])]
    status_counts: dict[str, int] = {}
    for report in target_reports:
        status_counts[report["status"]] = status_counts.get(report["status"], 0) + 1

    payload = {
        "model_name": args.model_name,
        "policy_name": args.policy_name,
        "candidate_source": str(args.candidates),
        "base_backend": policy.get("base_backend"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_compute": use_bf16,
        "lora_wrapped": not args.no_lora_wrap,
        "lora_targets": lora_targets,
        "status_counts": status_counts,
        "targets": target_reports,
        "summary": (
            "Selective rescue is implementation-feasible only for targets that can be reloaded or replaced in higher precision "
            "while preserving most QLoRA memory savings. This probe identifies which targets are quantized versus already regular precision."
        ),
    }

    safe_model = args.model_name.replace("/", "_").replace(".", "").replace("-", "_")
    output = args.output_dir / f"{safe_model}_{args.policy_name}_backend_feasibility.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    summary_keys = ["model_name", "policy_name", "device", "cuda_device_name", "status_counts"]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
