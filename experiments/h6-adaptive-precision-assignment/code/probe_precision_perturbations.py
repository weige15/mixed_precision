#!/usr/bin/env python
"""Measure one-module precision perturbation loss deltas for H6.

This is Stage 2 of the calibration-guided precision assignment study. It
keeps model weights fixed, measures baseline loss on fixed calibration batches,
then fake-quantizes one selected module output at a time and records the local
loss delta. The goal is to test whether Stage 1 signal-only rankings predict
actual perturbation sensitivity before freezing any training policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any


DEFAULT_MODULES = [
    "base_model.model.model.layers.2.mlp.down_proj",
    "base_model.model.model.layers.3.mlp.down_proj",
    "base_model.model.model.layers.21.mlp.down_proj",
    "base_model.model.model.layers.23.mlp.gate_proj",
    "base_model.model.model.layers.23.mlp.up_proj",
    "base_model.model.model.layers.22.mlp.gate_proj",
    "base_model.model.model.layers.22.mlp.up_proj",
    "base_model.model.model.layers.4.input_layernorm",
    "base_model.model.model.layers.4.post_attention_layernorm",
    "base_model.model.lm_head",
]


def require_packages() -> None:
    missing = []
    for name in ("torch", "transformers", "datasets", "peft", "numpy", "tqdm"):
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
    parser.add_argument("--dataset-name", default="tatsu-lab/alpaca")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--bits", type=int, nargs="+", default=[8], choices=[4, 8])
    parser.add_argument("--candidate-policy", default="")
    parser.add_argument("--modules", nargs="*", default=None, help="Exact module names to perturb.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(torch: Any, name: str) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def format_example(example: dict[str, Any]) -> str:
    if {"instruction", "input", "output"}.issubset(example):
        instruction = str(example.get("instruction") or "").strip()
        inp = str(example.get("input") or "").strip()
        output = str(example.get("output") or "").strip()
        if inp:
            return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:\n{output}"
        return f"Instruction:\n{instruction}\n\nResponse:\n{output}"
    for key in ("text", "content", "prompt"):
        if key in example and example[key] is not None:
            return str(example[key])
    values = [str(value) for value in example.values() if isinstance(value, (str, int, float))]
    return "\n".join(values)


def load_dataset_sample(dataset_name: str, seed: int, dataset_size: int):
    from datasets import load_dataset

    if dataset_size <= 0:
        raise SystemExit("--dataset-size must be positive.")
    dataset = load_dataset(dataset_name)
    split = dataset["train"] if "train" in dataset else dataset[next(iter(dataset.keys()))]
    if len(split) < dataset_size:
        raise SystemExit(f"Dataset split is too small: need {dataset_size}, found {len(split)}.")
    return split.shuffle(seed=seed).select(range(dataset_size))


def tokenize_dataset(dataset: Any, tokenizer: Any, seq_len: int):
    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            format_example(example),
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        encoded["labels"] = [
            token_id if mask else -100
            for token_id, mask in zip(encoded["input_ids"], encoded["attention_mask"])
        ]
        return encoded

    keep_columns = ["input_ids", "attention_mask", "labels"]
    return dataset.map(tokenize, remove_columns=dataset.column_names).select_columns(keep_columns)


def infer_lora_targets(model: Any) -> list[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and module.__class__.__name__.lower() == "linear":
            found.add(leaf)
    if found:
        return sorted(found)
    fallback = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if module.__class__.__name__.lower() == "linear" and "lm_head" not in name:
            fallback.add(leaf)
    if not fallback:
        raise SystemExit("Could not infer LoRA target linear modules for this model.")
    return sorted(fallback)


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


def fake_quant_dequant(tensor: Any, bits: int) -> Any:
    import torch

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    original_dtype = tensor.dtype
    x = tensor.float()
    finite_mask = torch.isfinite(x)
    if not bool(finite_mask.all()):
        return tensor
    qmax = float(2 ** (bits - 1) - 1)
    max_abs = torch.max(torch.abs(x))
    if float(max_abs.item()) == 0.0:
        return tensor
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return (q * scale).to(original_dtype)


def perturb_output(obj: Any, bits: int) -> Any:
    import torch

    if torch.is_tensor(obj):
        return fake_quant_dequant(obj, bits)
    if isinstance(obj, tuple):
        return tuple(perturb_output(value, bits) for value in obj)
    if isinstance(obj, list):
        return [perturb_output(value, bits) for value in obj]
    if isinstance(obj, dict):
        return {key: perturb_output(value, bits) for key, value in obj.items()}
    return obj


def run_losses(model: Any, batches: list[dict[str, Any]], device: str, autocast_ctx: Any) -> list[float]:
    import torch

    losses = []
    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_ctx:
                loss = model(**batch).loss
            losses.append(float(loss.detach().float().item()) if torch.isfinite(loss.detach()) else math.nan)
    return losses


def mean_finite(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else None


def load_policy(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["module"]: row for row in rows}


def main() -> None:
    require_packages()
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_dtype = dtype_from_arg(torch, args.dtype)
    use_autocast = device == "cuda" and requested_dtype in (torch.bfloat16, torch.float16)
    load_dtype = requested_dtype if use_autocast else torch.float32
    local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

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
    model.eval()

    raw = load_dataset_sample(args.dataset_name, args.seed, args.dataset_size)
    tokenized = tokenize_dataset(raw, tokenizer, args.seq_len)
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.calibration_batches:
            break
        batches.append(batch)
    if not batches:
        raise SystemExit("No calibration batches were produced.")

    named_modules = dict(model.named_modules())
    policy_by_module = load_policy(args.candidate_policy)
    target_modules = args.modules if args.modules is not None and len(args.modules) > 0 else DEFAULT_MODULES
    missing = [name for name in target_modules if name not in named_modules]
    if missing:
        raise SystemExit("Requested module(s) not found:\n" + "\n".join(missing))

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=requested_dtype)
        if use_autocast
        else nullcontext()
    )

    start = time.time()
    baseline_losses = run_losses(model, batches, device, autocast_ctx)
    baseline_mean = mean_finite(baseline_losses)
    if baseline_mean is None:
        raise SystemExit("Baseline losses were all non-finite.")

    rows = []
    for module_name in tqdm(target_modules, desc="h6-perturb-modules"):
        module = named_modules[module_name]
        role = module_role(module_name, module)
        for bits in args.bits:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, perturb_bits=bits: perturb_output(output, perturb_bits)
            )
            try:
                perturbed_losses = run_losses(model, batches, device, autocast_ctx)
            finally:
                handle.remove()
            perturbed_mean = mean_finite(perturbed_losses)
            loss_delta = None if perturbed_mean is None else perturbed_mean - baseline_mean
            per_batch_delta = [
                None if not (math.isfinite(p) and math.isfinite(b)) else p - b
                for p, b in zip(perturbed_losses, baseline_losses)
            ]
            finite_deltas = [value for value in per_batch_delta if value is not None]
            policy_row = policy_by_module.get(module_name, {})
            signal_row = policy_row.get("signals", {})
            rows.append(
                {
                    "module": module_name,
                    "role": role,
                    "class": module.__class__.__name__,
                    "bits": bits,
                    "baseline_loss_mean": baseline_mean,
                    "perturbed_loss_mean": perturbed_mean,
                    "loss_delta": loss_delta,
                    "loss_delta_abs": None if loss_delta is None else abs(loss_delta),
                    "max_batch_loss_delta_abs": max((abs(value) for value in finite_deltas), default=None),
                    "baseline_losses": baseline_losses,
                    "perturbed_losses": perturbed_losses,
                    "per_batch_loss_delta": per_batch_delta,
                    "stage1_assignment": policy_row.get("assigned_precision"),
                    "stage1_reason": policy_row.get("reason"),
                    "stage1_signals": {
                        "input_outlier_score_max": signal_row.get("input_outlier_score_max"),
                        "output_outlier_score_max": signal_row.get("output_outlier_score_max"),
                        "input_int8_rel_mse_mean": signal_row.get("input_int8_rel_mse_mean"),
                        "output_int8_rel_mse_mean": signal_row.get("output_int8_rel_mse_mean"),
                        "output_int4_rel_mse_mean": signal_row.get("output_int4_rel_mse_mean"),
                    },
                }
            )

    rows_sorted = sorted(rows, key=lambda row: row["loss_delta_abs"] if row["loss_delta_abs"] is not None else -1, reverse=True)
    payload = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "calibration_batches": len(batches),
        "dataset_size": args.dataset_size,
        "dtype": args.dtype,
        "device": device,
        "autocast_enabled": use_autocast,
        "bits": args.bits,
        "candidate_policy": args.candidate_policy,
        "lora_targets": lora_targets,
        "baseline_loss_mean": baseline_mean,
        "baseline_losses": baseline_losses,
        "elapsed_sec": time.time() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else None,
        "results": rows_sorted,
    }

    output_path = os.path.join(args.output_dir, "perturbation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        json.dumps(
            {
                "device": payload["device"],
                "autocast_enabled": payload["autocast_enabled"],
                "baseline_loss_mean": payload["baseline_loss_mean"],
                "elapsed_sec": payload["elapsed_sec"],
                "peak_cuda_memory_gib": payload["peak_cuda_memory_gib"],
                "top_loss_deltas": [
                    {
                        "module": row["module"],
                        "bits": row["bits"],
                        "loss_delta": row["loss_delta"],
                        "stage1_assignment": row["stage1_assignment"],
                    }
                    for row in rows_sorted[:10]
                ],
            },
            indent=2,
        )
    )
    print(f"Saved perturbation results to {output_path}")


if __name__ == "__main__":
    main()
