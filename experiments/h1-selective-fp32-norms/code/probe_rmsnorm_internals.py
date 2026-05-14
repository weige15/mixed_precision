#!/usr/bin/env python
"""Probe internal dtype behavior for a Qwen-style RMSNorm module."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from contextlib import nullcontext
from typing import Any


def require_packages() -> None:
    missing = []
    for name in ("torch", "transformers"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + ". Install project dependencies with: pip install -r requirements.txt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--module-name", default="", help="Optional exact module name to probe.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not contact the Hugging Face Hub.")
    parser.add_argument(
        "--output-json",
        default="experiments/h1-selective-fp32-norms/results/h5_rmsnorm_internal_dtype.json",
    )
    return parser.parse_args()


def dtype_from_arg(torch: Any, name: str) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def choose_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return requested


def is_rmsnorm_candidate(name: str, module: Any) -> bool:
    haystack = f"{name} {module.__class__.__name__}".lower()
    return "rmsnorm" in haystack


def first_tensor(obj: Any) -> Any | None:
    import torch

    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = first_tensor(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = first_tensor(value)
            if found is not None:
                return found
    return None


def tensor_summary(tensor: Any | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
    }


def reference_rmsnorm_trace(torch: Any, module: Any, hidden_states: Any) -> tuple[Any, list[dict[str, str]]]:
    trace: list[dict[str, str]] = []

    def record(name: str, tensor: Any) -> Any:
        trace.append({"op": name, "dtype": str(tensor.dtype), "shape": str(tuple(tensor.shape))})
        return tensor

    input_dtype = hidden_states.dtype
    record("input", hidden_states)
    fp32_hidden = record("hidden_states.float()", hidden_states.float())
    squared = record("pow_2", fp32_hidden.pow(2))
    variance = record("mean_last_dim", squared.mean(-1, keepdim=True))
    rsqrt = record("rsqrt_variance_plus_eps", torch.rsqrt(variance + module.variance_epsilon))
    normalized = record("multiply_by_rsqrt", fp32_hidden * rsqrt)
    restored = record("restore_input_dtype", normalized.to(input_dtype))
    weighted = record("weight_multiply", module.weight * restored)
    return weighted, trace


def main() -> None:
    require_packages()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = choose_device(torch, args.device)
    requested_dtype = dtype_from_arg(torch, args.dtype)
    load_dtype = requested_dtype if device == "cuda" and requested_dtype != torch.float32 else torch.float32

    local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    target_name = ""
    target_module = None
    for name, module in model.named_modules():
        if args.module_name:
            if name == args.module_name:
                target_name = name
                target_module = module
                break
        elif name and is_rmsnorm_candidate(name, module):
            target_name = name
            target_module = module
            break
    if target_module is None:
        raise SystemExit("No RMSNorm-like module found to probe.")

    captured: dict[str, Any] = {}

    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        captured["input"] = first_tensor(inputs).detach()
        captured["output"] = first_tensor(output).detach()

    handle = target_module.register_forward_hook(hook)

    text = "This is an RMSNorm internal dtype probe for mixed precision fine-tuning."
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    if encoded["input_ids"].shape[1] < args.seq_len:
        pad_len = args.seq_len - encoded["input_ids"].shape[1]
        encoded["input_ids"] = torch.nn.functional.pad(encoded["input_ids"], (0, pad_len), value=tokenizer.eos_token_id)
        encoded["attention_mask"] = torch.nn.functional.pad(encoded["attention_mask"], (0, pad_len), value=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    use_autocast = device == "cuda" and requested_dtype in (torch.bfloat16, torch.float16)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=requested_dtype) if use_autocast else nullcontext()
    with torch.no_grad():
        with autocast_ctx:
            _ = model(**encoded)
    handle.remove()

    captured_input = captured.get("input")
    captured_output = captured.get("output")
    if captured_input is None or captured_output is None:
        raise SystemExit("Target RMSNorm module did not execute during probe.")

    with torch.no_grad():
        reference_output, reference_trace = reference_rmsnorm_trace(torch, target_module, captured_input)

    diff = (captured_output.float() - reference_output.float()).abs()
    try:
        forward_source = inspect.getsource(target_module.__class__.forward)
    except (OSError, TypeError):
        forward_source = None

    payload = {
        "model_name": args.model_name,
        "device": device,
        "requested_dtype": str(requested_dtype),
        "autocast_enabled": use_autocast,
        "seq_len": args.seq_len,
        "module": target_name,
        "class": target_module.__class__.__name__,
        "boundary": {
            "input": tensor_summary(captured_input),
            "output": tensor_summary(captured_output),
            "weight": tensor_summary(getattr(target_module, "weight", None)),
        },
        "reference_trace": reference_trace,
        "reference_matches_actual": {
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        },
        "forward_source": forward_source,
        "interpretation": (
            "If the forward source and reference trace cast hidden_states to float32 before pow/mean/rsqrt, "
            "then baseline boundary bf16 tensors do not imply bf16 RMSNorm reduction arithmetic."
        ),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Probed {target_name} ({target_module.__class__.__name__})")
    print(f"Boundary input dtype: {captured_input.dtype}")
    print(f"Boundary output dtype: {captured_output.dtype}")
    for row in reference_trace:
        print(f"{row['op']}: {row['dtype']}")
    print(f"Max abs diff vs reference: {payload['reference_matches_actual']['max_abs_diff']}")
    print(f"Saved RMSNorm internal dtype probe to {args.output_json}")


if __name__ == "__main__":
    main()
