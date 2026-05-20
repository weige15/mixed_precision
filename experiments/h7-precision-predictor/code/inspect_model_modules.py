#!/usr/bin/env python
"""Inspect PEFT-wrapped model module names and emit a target probe panel."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output", default="experiments/h7-precision-predictor/results/llama31_8b_module_inventory.json")
    parser.add_argument("--panel-output", default="experiments/h7-precision-predictor/results/llama31_8b_probe_modules.txt")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def dtype_from_arg(torch: Any, name: str) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def infer_lora_targets(model: Any) -> list[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and module.__class__.__name__.lower() == "linear":
            found.add(leaf)
    return sorted(found)


def module_role(name: str, module: Any) -> str | None:
    class_name = module.__class__.__name__.lower()
    leaf = name.rsplit(".", 1)[-1].lower()
    haystack = f"{name} {class_name}".lower()
    if "lm_head" in haystack:
        return "logits"
    if "rmsnorm" in haystack or "layernorm" in haystack or leaf == "norm" or leaf.endswith("_norm"):
        return "norm"
    if class_name == "linear" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention_projection"
    if class_name == "linear" and leaf in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp_projection"
    return None


def layer_idx(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def by_layer_leaf(rows: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    out = {}
    for row in rows:
        idx = row.get("layer_idx")
        leaf = row["name"].rsplit(".", 1)[-1]
        if idx is not None:
            out[(idx, leaf)] = row["name"]
    return out


def choose_existing(index: dict[tuple[int, str], str], layer: int, leaf: str) -> str | None:
    return index.get((layer, leaf))


def build_probe_panel(rows: list[dict[str, Any]]) -> list[str]:
    layer_indices = sorted({row["layer_idx"] for row in rows if row.get("layer_idx") is not None})
    if not layer_indices:
        return []
    max_layer = max(layer_indices)
    index = by_layer_leaf(rows)
    candidates: list[tuple[int, str]] = [
        (2, "down_proj"),
        (3, "down_proj"),
        (max_layer - 3, "down_proj"),
        (max_layer - 1, "gate_proj"),
        (max_layer - 1, "up_proj"),
        (max_layer, "gate_proj"),
        (max_layer, "up_proj"),
        (4, "input_layernorm"),
        (4, "post_attention_layernorm"),
        (2, "o_proj"),
        (max_layer - 1, "q_proj"),
        (max_layer - 1, "o_proj"),
    ]
    panel = []
    for layer, leaf in candidates:
        name = choose_existing(index, layer, leaf)
        if name is not None:
            panel.append(name)
    for row in rows:
        if row["name"].endswith(".norm") and row["role"] == "norm":
            panel.append(row["name"])
    for row in rows:
        if row["role"] == "logits":
            panel.append(row["name"])
    deduped = []
    seen = set()
    for name in panel:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.to(device)
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

    rows = []
    counts = defaultdict(int)
    for name, module in model.named_modules():
        role = module_role(name, module)
        if role is None:
            continue
        row = {
            "name": name,
            "role": role,
            "class": module.__class__.__name__,
            "layer_idx": layer_idx(name),
            "leaf": name.rsplit(".", 1)[-1],
        }
        rows.append(row)
        counts[role] += 1

    panel = build_probe_panel(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "device": device,
        "dtype": args.dtype,
        "lora_targets": lora_targets,
        "role_counts": dict(counts),
        "modules": rows,
        "probe_panel": panel,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.panel_output).write_text("\n".join(panel) + "\n", encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ["model_name", "device", "dtype", "lora_targets", "role_counts"]}, indent=2))
    print("Probe panel:")
    for name in panel:
        print(name)
    print(f"Wrote inventory to {output}")
    print(f"Wrote panel to {args.panel_output}")


if __name__ == "__main__":
    main()
