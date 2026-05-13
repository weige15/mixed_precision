#!/usr/bin/env python
"""Collect per-module stability signals for adaptive precision assignment.

This is a non-invasive calibration probe: it runs fixed batches through the
LoRA model, records activation statistics and fake-quantization error, and
writes a frozen candidate precision policy. It does not train or update model
weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any


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
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-modules", type=int, default=0, help="0 means probe all candidate modules.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-name", default="h6_probe_policy")
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


def fake_quant_stats(tensor: Any, bits: int) -> dict[str, float]:
    import torch

    x = tensor.detach().float()
    finite_mask = torch.isfinite(x)
    if not bool(finite_mask.all()):
        x = x[finite_mask]
    if x.numel() == 0:
        return {"mse": math.nan, "rel_mse": math.nan, "max_abs_error": math.nan, "saturation_rate": math.nan}

    qmax = float(2 ** (bits - 1) - 1)
    max_abs = torch.max(torch.abs(x))
    if float(max_abs.item()) == 0.0:
        return {"mse": 0.0, "rel_mse": 0.0, "max_abs_error": 0.0, "saturation_rate": 0.0}

    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    dequant = q * scale
    err = dequant - x
    mse = torch.mean(err * err)
    denom = torch.mean(x * x).clamp_min(1e-12)
    saturation_rate = torch.mean((torch.abs(q) >= qmax).float())
    return {
        "mse": float(mse.item()),
        "rel_mse": float((mse / denom).item()),
        "max_abs_error": float(torch.max(torch.abs(err)).item()),
        "saturation_rate": float(saturation_rate.item()),
    }


def tensor_stats(tensor: Any) -> dict[str, float | int | str]:
    import torch

    x = tensor.detach().float()
    finite_mask = torch.isfinite(x)
    finite_fraction = float(finite_mask.float().mean().item()) if x.numel() else 1.0
    if not bool(finite_mask.all()):
        x = x[finite_mask]
    if x.numel() == 0:
        return {
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "finite_fraction": finite_fraction,
            "mean": math.nan,
            "std": math.nan,
            "rms": math.nan,
            "abs_max": math.nan,
            "abs_p99": math.nan,
            "outlier_score": math.nan,
        }

    abs_x = torch.abs(x)
    rms = torch.sqrt(torch.mean(x * x)).clamp_min(1e-12)
    abs_max = torch.max(abs_x)
    return {
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "finite_fraction": finite_fraction,
        "mean": float(torch.mean(x).item()),
        "std": float(torch.std(x, unbiased=False).item()),
        "rms": float(rms.item()),
        "abs_max": float(abs_max.item()),
        "abs_p99": float(torch.quantile(abs_x, 0.99).item()),
        "outlier_score": float((abs_max / rms).item()),
    }


@dataclass
class SignalAccumulator:
    module: str
    role: str
    class_name: str
    observations: int = 0
    input_outlier_scores: list[float] = field(default_factory=list)
    output_outlier_scores: list[float] = field(default_factory=list)
    input_int8_rel_mse: list[float] = field(default_factory=list)
    output_int8_rel_mse: list[float] = field(default_factory=list)
    output_int4_rel_mse: list[float] = field(default_factory=list)
    output_int8_saturation: list[float] = field(default_factory=list)
    output_int4_saturation: list[float] = field(default_factory=list)
    finite_fractions: list[float] = field(default_factory=list)
    dtypes: set[str] = field(default_factory=set)

    def add(self, inputs: Any, output: Any) -> None:
        input_tensor = first_tensor(inputs)
        output_tensor = first_tensor(output)
        if input_tensor is None or output_tensor is None:
            return
        input_summary = tensor_stats(input_tensor)
        output_summary = tensor_stats(output_tensor)
        input_q8 = fake_quant_stats(input_tensor, 8)
        output_q8 = fake_quant_stats(output_tensor, 8)
        output_q4 = fake_quant_stats(output_tensor, 4)

        self.observations += 1
        self.input_outlier_scores.append(float(input_summary["outlier_score"]))
        self.output_outlier_scores.append(float(output_summary["outlier_score"]))
        self.input_int8_rel_mse.append(float(input_q8["rel_mse"]))
        self.output_int8_rel_mse.append(float(output_q8["rel_mse"]))
        self.output_int4_rel_mse.append(float(output_q4["rel_mse"]))
        self.output_int8_saturation.append(float(output_q8["saturation_rate"]))
        self.output_int4_saturation.append(float(output_q4["saturation_rate"]))
        self.finite_fractions.append(float(min(input_summary["finite_fraction"], output_summary["finite_fraction"])))
        self.dtypes.add(str(input_tensor.dtype))
        self.dtypes.add(str(output_tensor.dtype))

    def summary(self) -> dict[str, Any]:
        def avg(values: list[float]) -> float | None:
            clean = [value for value in values if math.isfinite(value)]
            return float(sum(clean) / len(clean)) if clean else None

        def mx(values: list[float]) -> float | None:
            clean = [value for value in values if math.isfinite(value)]
            return float(max(clean)) if clean else None

        return {
            "module": self.module,
            "role": self.role,
            "class": self.class_name,
            "observations": self.observations,
            "dtypes": sorted(self.dtypes),
            "input_outlier_score_mean": avg(self.input_outlier_scores),
            "input_outlier_score_max": mx(self.input_outlier_scores),
            "output_outlier_score_mean": avg(self.output_outlier_scores),
            "output_outlier_score_max": mx(self.output_outlier_scores),
            "input_int8_rel_mse_mean": avg(self.input_int8_rel_mse),
            "output_int8_rel_mse_mean": avg(self.output_int8_rel_mse),
            "output_int4_rel_mse_mean": avg(self.output_int4_rel_mse),
            "output_int8_saturation_mean": avg(self.output_int8_saturation),
            "output_int4_saturation_mean": avg(self.output_int4_saturation),
            "finite_fraction_min": min(self.finite_fractions) if self.finite_fractions else None,
        }


def decide_precision(row: dict[str, Any]) -> tuple[str, str]:
    finite_min = row.get("finite_fraction_min")
    outlier = max(row.get("input_outlier_score_max") or 0.0, row.get("output_outlier_score_max") or 0.0)
    int8_error = max(row.get("input_int8_rel_mse_mean") or 0.0, row.get("output_int8_rel_mse_mean") or 0.0)
    int4_error = row.get("output_int4_rel_mse_mean") or 0.0
    role = row["role"]

    if finite_min is not None and finite_min < 1.0:
        return "fp32", "non-finite activation observed"
    if role in {"norm", "logits"}:
        if outlier >= 12.0 or int8_error >= 1e-3:
            return "fp32", "high-sensitivity reduction/output path"
        return "bf16", "stable reduction/output path under calibration"
    if role in {"attention_projection", "mlp_projection"}:
        if outlier <= 12.0 and int8_error <= 1e-3:
            if int4_error <= 5e-3 and role == "mlp_projection":
                return "int4_candidate", "low fake-quant error; verify with perturbation before use"
            return "int8_candidate", "low int8 fake-quant error and bounded outlier score"
        return "bf16", "projection appears sensitive under activation probe"
    return "bf16", "default conservative assignment"


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
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype,
        trust_remote_code=True,
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
    model.train()

    raw = load_dataset_sample(args.dataset_name, args.seed, args.dataset_size)
    tokenized = tokenize_dataset(raw, tokenizer, args.seq_len)
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    accumulators: dict[str, SignalAccumulator] = {}
    hooks = []
    for name, module in model.named_modules():
        role = module_role(name, module)
        if role is None:
            continue
        if args.max_modules > 0 and len(accumulators) >= args.max_modules:
            break
        accumulators[name] = SignalAccumulator(name, role, module.__class__.__name__)

        def make_hook(module_name: str):
            def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                accumulators[module_name].add(inputs, output)

            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=requested_dtype)
        if use_autocast
        else nullcontext()
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    losses = []
    nan_or_inf_count = 0
    start = time.time()
    for batch_idx, batch in enumerate(tqdm(loader, desc="h6-signal-probe")):
        if batch_idx >= args.calibration_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_ctx:
            loss = model(**batch).loss
        if not torch.isfinite(loss.detach()):
            nan_or_inf_count += 1
            continue
        loss.backward()
        losses.append(float(loss.detach().float().item()))

    for hook in hooks:
        hook.remove()

    module_rows = [acc.summary() for acc in accumulators.values() if acc.observations > 0]
    policy_rows = []
    for row in module_rows:
        precision, reason = decide_precision(row)
        policy_rows.append(
            {
                "module": row["module"],
                "role": row["role"],
                "assigned_precision": precision,
                "reason": reason,
                "signals": row,
            }
        )

    role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in policy_rows:
        role_counts[row["role"]][row["assigned_precision"]] += 1

    payload = {
        "policy_name": args.policy_name,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "calibration_batches": args.calibration_batches,
        "dtype": args.dtype,
        "device": device,
        "autocast_enabled": use_autocast,
        "lora_targets": lora_targets,
        "mean_calibration_loss": float(sum(losses) / len(losses)) if losses else None,
        "nan_or_inf_count": nan_or_inf_count,
        "elapsed_sec": time.time() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else None,
        "decision_thresholds": {
            "outlier_score_fp32_threshold": 12.0,
            "int8_rel_mse_candidate_threshold": 1e-3,
            "int4_rel_mse_candidate_threshold": 5e-3,
        },
        "role_precision_counts": {role: dict(counts) for role, counts in role_counts.items()},
        "modules": module_rows,
        "policy_trace": policy_rows,
    }

    signals_path = os.path.join(args.output_dir, "stability_signals.json")
    policy_path = os.path.join(args.output_dir, "policy_trace.json")
    with open(signals_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy_rows, f, indent=2)

    print(json.dumps({k: payload[k] for k in payload if k not in {"modules", "policy_trace"}}, indent=2))
    print(f"Saved stability signals to {signals_path}")
    print(f"Saved frozen policy trace to {policy_path}")


if __name__ == "__main__":
    main()
