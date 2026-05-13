#!/usr/bin/env python
"""Probe module dtypes under autocast without training."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-json", default="experiments/h1-selective-fp32-norms/results/dtype_probe.json")
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


def first_tensor_dtype(obj: Any) -> str | None:
    import torch

    if torch.is_tensor(obj):
        return str(obj.dtype)
    if isinstance(obj, dict):
        for value in obj.values():
            found = first_tensor_dtype(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = first_tensor_dtype(value)
            if found is not None:
                return found
    return None


def module_matches(name: str, module: Any) -> bool:
    haystack = f"{name} {module.__class__.__name__}".lower()
    return any(token in haystack for token in ("rmsnorm", "layernorm", "norm", "softmax", "lm_head"))


def print_table(rows: list[dict[str, str | None]]) -> None:
    if not rows:
        print("No matching modules were observed.")
        return
    headers = ["module", "class", "input", "output", "param"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        widths["module"] = min(max(widths["module"], len(row["module"] or "")), 72)
        widths["class"] = max(widths["class"], len(row["class"] or ""))
        widths["input"] = max(widths["input"], len(row["input_dtype"] or "None"))
        widths["output"] = max(widths["output"], len(row["output_dtype"] or "None"))
        widths["param"] = max(widths["param"], len(row["parameter_dtype"] or "None"))

    fmt = f"{{module:<{widths['module']}}}  {{class_:<{widths['class']}}}  {{input:<{widths['input']}}}  {{output:<{widths['output']}}}  {{param:<{widths['param']}}}"
    print(fmt.format(module="module", class_="class", input="input", output="output", param="param"))
    print("-" * (sum(widths.values()) + 8))
    for row in rows:
        module_name = row["module"] or ""
        if len(module_name) > widths["module"]:
            module_name = "..." + module_name[-(widths["module"] - 3) :]
        print(
            fmt.format(
                module=module_name,
                class_=row["class"] or "",
                input=row["input_dtype"] or "None",
                output=row["output_dtype"] or "None",
                param=row["parameter_dtype"] or "None",
            )
        )


def main() -> None:
    require_packages()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = choose_device(torch, args.device)
    requested_dtype = dtype_from_arg(torch, args.dtype)
    load_dtype = requested_dtype if device == "cuda" and requested_dtype != torch.float32 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    text = "This is a dtype probe for mixed precision LoRA fine-tuning."
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    if encoded["input_ids"].shape[1] < args.seq_len:
        pad_len = args.seq_len - encoded["input_ids"].shape[1]
        encoded["input_ids"] = torch.nn.functional.pad(encoded["input_ids"], (0, pad_len), value=tokenizer.eos_token_id)
        encoded["attention_mask"] = torch.nn.functional.pad(encoded["attention_mask"], (0, pad_len), value=0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    rows: list[dict[str, str | None]] = []
    hooks = []

    def make_hook(module_name: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            param_dtype = None
            for param in module.parameters(recurse=False):
                param_dtype = str(param.dtype)
                break
            rows.append(
                {
                    "module": module_name,
                    "class": module.__class__.__name__,
                    "input_dtype": first_tensor_dtype(inputs),
                    "output_dtype": first_tensor_dtype(output),
                    "parameter_dtype": param_dtype,
                }
            )

        return hook

    for name, module in model.named_modules():
        if name and module_matches(name, module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    use_autocast = device == "cuda" and requested_dtype in (torch.bfloat16, torch.float16)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=requested_dtype) if use_autocast else nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            _ = model(**encoded)

    for hook in hooks:
        hook.remove()

    payload = {
        "model_name": args.model_name,
        "device": device,
        "requested_dtype": str(requested_dtype),
        "autocast_enabled": use_autocast,
        "seq_len": args.seq_len,
        "modules": rows,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_table(rows)
    print(f"\nSaved dtype probe to {args.output_json}")


if __name__ == "__main__":
    main()
