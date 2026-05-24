#!/usr/bin/env python
"""Minimal LoRA fine-tuning runner for H1 precision-policy experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

H6_LATE_MLP_INT8_MODULES = {
    "base_model.model.model.layers.22.mlp.gate_proj",
    "base_model.model.model.layers.22.mlp.up_proj",
    "base_model.model.model.layers.23.mlp.gate_proj",
    "base_model.model.model.layers.23.mlp.up_proj",
}

H8_DEFAULT_CANDIDATES = Path("experiments/h8-hardware-aware-precision-search/results/h8_policy_candidates.json")
H8_SELECTIVE_RESCUE_POLICY = "h8_qlora_nf4_rescue_projection_top4"


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
    parser.add_argument(
        "--precision-policy",
        required=True,
        choices=[
            "bf16_baseline",
            "fp32_norms",
            "h6_late_mlp_int8_candidate",
            "h6_custom_int8",
            "qlora_4bit_nf4",
            H8_SELECTIVE_RESCUE_POLICY,
            "lora_8bit_int8",
        ],
    )
    parser.add_argument(
        "--fake-int8-modules",
        nargs="*",
        default=None,
        help="Exact module names or unique suffixes to fake-int8 when --precision-policy=h6_custom_int8.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=0,
        help="Maximum validation batches per eval; 0 means full validation set.",
    )
    parser.add_argument(
        "--hardware-label",
        default=os.environ.get("HARDWARE_LABEL", ""),
        help="Optional run context label such as rtx4050-local or rtx3090-lab.",
    )
    parser.add_argument(
        "--llm-int8-threshold",
        type=float,
        default=6.0,
        help="bitsandbytes LLM.int8 outlier threshold for --precision-policy=lora_8bit_int8.",
    )
    parser.add_argument(
        "--h8-candidates",
        type=Path,
        default=H8_DEFAULT_CANDIDATES,
        help="H8 policy candidate JSON used by h8 selective-rescue policies.",
    )
    parser.add_argument(
        "--h8-policy-name",
        default="h8_rescue_projection_top4",
        help="Candidate policy name inside --h8-candidates for h8 selective rescue.",
    )
    parser.add_argument(
        "--h8-rescue-precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Precision used for rescued H8 projection modules.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Load the model, apply precision policy/LoRA wrapping, write setup_summary.json, and exit before data/training.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_dataset_split(dataset_name: str, seed: int, train_size: int, eval_size: int):
    from datasets import load_dataset

    if train_size <= 0 or eval_size <= 0:
        raise SystemExit("--train-size and --eval-size must both be positive.")

    dataset = load_dataset(dataset_name)
    if "train" not in dataset:
        first_split = next(iter(dataset.keys()))
        train = dataset[first_split]
    else:
        train = dataset["train"]

    train = train.shuffle(seed=seed)
    required = train_size + eval_size
    if len(train) < required:
        raise SystemExit(
            f"Dataset split is too small for requested train/eval sizes: "
            f"need {required}, found {len(train)}."
        )
    return train.select(range(train_size)), train.select(range(train_size, required))


def tokenize_dataset(dataset: Any, tokenizer: Any, seq_len: int):
    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = format_example(example)
        encoded = tokenizer(
            text,
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


class Fp32NormWrapper:
    """Callable wrapper that runs a norm module in fp32 and restores incoming dtype."""

    def __init__(self, module: Any):
        self.module = module
        self.original_forward = module.forward

    def __call__(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        if not torch.is_tensor(hidden_states):
            return self.original_forward(hidden_states, *args, **kwargs)

        incoming_dtype = hidden_states.dtype
        # Disable autocast inside the norm so the reduction/statistics happen in fp32.
        autocast_disabled = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if hidden_states.is_cuda
            else nullcontext()
        )
        with autocast_disabled:
            output = self.original_forward(hidden_states.float(), *args, **kwargs)
        if incoming_dtype is not None and torch.is_tensor(output):
            return output.to(incoming_dtype)
        return output


def is_norm_module(name: str, module: Any) -> bool:
    haystack = f"{name} {module.__class__.__name__}".lower()
    return "rmsnorm" in haystack or "layernorm" in haystack or ".norm" in haystack or " norm" in haystack


def is_linear_like(module: Any) -> bool:
    return "linear" in module.__class__.__name__.lower()


def apply_fp32_norms(model: Any) -> list[str]:
    wrapped = []
    for name, module in model.named_modules():
        if name and is_norm_module(name, module):
            module.forward = Fp32NormWrapper(module)  # type: ignore[method-assign]
            wrapped.append(name)
    return wrapped


def fake_quant_dequant_ste(tensor: Any, bits: int = 8) -> Any:
    import torch

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    qmax = float((2 ** (bits - 1)) - 1)
    max_abs = tensor.detach().abs().amax()
    if float(max_abs.item()) == 0.0:
        return tensor
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale
    return tensor + (quantized - tensor).detach()


def fake_quant_output(output: Any, bits: int = 8) -> Any:
    if isinstance(output, tuple):
        return tuple(fake_quant_output(item, bits) for item in output)
    if isinstance(output, list):
        return [fake_quant_output(item, bits) for item in output]
    if isinstance(output, dict):
        return {key: fake_quant_output(value, bits) for key, value in output.items()}
    return fake_quant_dequant_ste(output, bits)


def resolve_module_targets(model: Any, requested: list[str]) -> list[str]:
    named_modules = dict(model.named_modules())
    resolved = []
    for target in requested:
        if target in named_modules:
            resolved.append(target)
            continue
        matches = [name for name in named_modules if name.endswith(target)]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if not matches:
            raise SystemExit(f"Requested fake-int8 module not found: {target}")
        raise SystemExit(
            "Requested fake-int8 module suffix is ambiguous: "
            + target
            + "\nMatches:\n"
            + "\n".join(sorted(matches))
        )
    return sorted(set(resolved))


def apply_fake_int8_modules(model: Any, requested: list[str]) -> list[str]:
    hooked = []
    targets = set(resolve_module_targets(model, requested))
    for name, module in model.named_modules():
        if name not in targets:
            continue
        if not is_linear_like(module):
            raise SystemExit(f"H6 candidate target is not a Linear module: {name} ({module.__class__.__name__})")
        module.register_forward_hook(lambda _module, _inputs, output: fake_quant_output(output, bits=8))
        hooked.append(name)

    missing = sorted(targets - set(hooked))
    if missing:
        raise SystemExit("H6 candidate module(s) not found:\n" + "\n".join(missing))
    return sorted(hooked)


def apply_h6_late_mlp_int8_candidate(model: Any) -> list[str]:
    return apply_fake_int8_modules(model, sorted(H6_LATE_MLP_INT8_MODULES))


def infer_lora_targets(model: Any) -> list[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and is_linear_like(module):
            found.add(leaf)
    if found:
        return sorted(found)
    fallback = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if is_linear_like(module) and "lm_head" not in name:
            fallback.add(leaf)
    if not fallback:
        raise SystemExit("Could not infer LoRA target linear modules for this model.")
    return sorted(fallback)


def load_h8_candidate_policy(candidates_path: Path, model_name: str, policy_name: str) -> dict[str, Any]:
    data = json.loads(candidates_path.read_text())
    models = data.get("models")
    if models is None:
        models = [
            {
                "model_name": data.get("model_filter"),
                "candidate_policies": data.get("candidate_policies", []),
            }
        ]

    for model_entry in models:
        if model_entry.get("model_name") != model_name:
            continue
        for policy in model_entry.get("candidate_policies", []):
            if policy.get("policy_name") == policy_name:
                return policy
        available = sorted(str(policy.get("policy_name")) for policy in model_entry.get("candidate_policies", []))
        raise SystemExit(f"H8 policy not found for {model_name}: {policy_name}. Available: {available}")

    available_models = sorted(str(model_entry.get("model_name")) for model_entry in models)
    raise SystemExit(f"H8 model not found in {candidates_path}: {model_name}. Available: {available_models}")


def h8_runtime_module_name(candidate_name: str) -> str:
    """Map PEFT-wrapped candidate names back to the pre-PEFT base model graph."""

    for prefix in ("base_model.model.", "base_model."):
        if candidate_name.startswith(prefix):
            return candidate_name[len(prefix) :]
    return candidate_name


def resolve_h8_model_snapshot(model_name: str) -> Path:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("H8 selective rescue requires huggingface_hub to locate checkpoint shards.") from exc

    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    try:
        snapshot = snapshot_download(
            repo_id=model_name,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "pytorch_model*.bin.index.json",
            ],
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise SystemExit(f"Could not locate checkpoint files for H8 selective rescue: {model_name}") from exc
    return Path(snapshot)


def find_checkpoint_shard(snapshot_dir: Path, tensor_name: str) -> tuple[Path, str]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot_dir / index_name
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_name = weight_map.get(tensor_name)
        if shard_name is not None:
            return snapshot_dir / shard_name, "safetensors" if shard_name.endswith(".safetensors") else "torch"

    safetensors_path = snapshot_dir / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_path, "safetensors"

    torch_path = snapshot_dir / "pytorch_model.bin"
    if torch_path.exists():
        return torch_path, "torch"

    raise SystemExit(f"Could not find a checkpoint shard containing tensor: {tensor_name}")


def load_checkpoint_tensor(snapshot_dir: Path, tensor_name: str) -> Any:
    shard_path, shard_type = find_checkpoint_shard(snapshot_dir, tensor_name)
    if shard_type == "safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise SystemExit("H8 selective rescue requires safetensors to read checkpoint shards.") from exc
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if tensor_name not in handle.keys():
                raise SystemExit(f"Tensor {tensor_name} was not found in {shard_path}.")
            return handle.get_tensor(tensor_name)

    import torch

    shard = torch.load(shard_path, map_location="cpu")
    if tensor_name not in shard:
        raise SystemExit(f"Tensor {tensor_name} was not found in {shard_path}.")
    return shard[tensor_name]


def replace_module(model: Any, module_name: str, replacement: Any) -> None:
    parent_name, leaf_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf_name, replacement)


def apply_h8_selective_rescue(
    model: Any,
    model_name: str,
    candidates_path: Path,
    policy_name: str,
    rescue_precision: str,
    device: str,
    bf16_ok: bool,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn

    if rescue_precision == "bf16":
        if not bf16_ok:
            raise SystemExit("H8 bf16 rescue requested, but torch.cuda.is_bf16_supported() is false.")
        rescue_dtype = torch.bfloat16
    else:
        rescue_dtype = torch.float32

    policy = load_h8_candidate_policy(candidates_path, model_name, policy_name)
    rescue_modules = policy.get("rescue_modules", [])
    if not rescue_modules:
        raise SystemExit(f"H8 policy has no rescue_modules: {policy_name}")

    snapshot_dir = resolve_h8_model_snapshot(model_name)
    reports = []
    for candidate_name in rescue_modules:
        module_name = h8_runtime_module_name(candidate_name)
        try:
            original_module = model.get_submodule(module_name)
        except AttributeError as exc:
            raise SystemExit(f"H8 rescue target not found in loaded model: {candidate_name} -> {module_name}") from exc
        if not is_linear_like(original_module):
            raise SystemExit(f"H8 rescue target is not linear-like: {module_name} ({original_module.__class__.__name__})")

        in_features = getattr(original_module, "in_features", None)
        out_features = getattr(original_module, "out_features", None)
        if in_features is None or out_features is None:
            raise SystemExit(f"H8 rescue target lacks in/out feature metadata: {module_name}")

        weight_key = f"{module_name}.weight"
        weight = load_checkpoint_tensor(snapshot_dir, weight_key).to(dtype=rescue_dtype)
        has_bias = getattr(original_module, "bias", None) is not None
        replacement = nn.Linear(
            int(in_features),
            int(out_features),
            bias=has_bias,
            device=device,
            dtype=rescue_dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(weight.to(device=device))
            if has_bias:
                bias_key = f"{module_name}.bias"
                bias = load_checkpoint_tensor(snapshot_dir, bias_key).to(device=device, dtype=rescue_dtype)
                replacement.bias.copy_(bias)
        replacement.requires_grad_(False)
        replace_module(model, module_name, replacement)

        reports.append(
            {
                "candidate_name": candidate_name,
                "runtime_name": module_name,
                "checkpoint_weight": weight_key,
                "original_class": original_module.__class__.__name__,
                "replacement_class": replacement.__class__.__name__,
                "replacement_dtype": str(rescue_dtype),
                "in_features": int(in_features),
                "out_features": int(out_features),
                "bias": bool(has_bias),
            }
        )
        del weight

    return reports


def grad_norm(parameters: Any) -> float:
    import torch

    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def evaluate(model: Any, loader: Any, device: str, autocast_ctx: Any, max_batches: int | None = None) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_ctx:
                loss = model(**batch).loss
            losses.append(float(loss.detach().float().item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    require_packages()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_ok
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.precision_policy in {"qlora_4bit_nf4", H8_SELECTIVE_RESCUE_POLICY, "lora_8bit_int8"} and device != "cuda":
        raise SystemExit(f"precision-policy {args.precision_policy} requires CUDA for bitsandbytes k-bit training.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    qlora_config: dict[str, Any] | None = None
    if args.precision_policy in {"qlora_4bit_nf4", H8_SELECTIVE_RESCUE_POLICY}:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        qlora_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16" if use_bf16 else "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    elif args.precision_policy == "lora_8bit_int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_has_fp16_weight=False,
        )
        qlora_config = {
            "load_in_8bit": True,
            "llm_int8_threshold": args.llm_int8_threshold,
            "llm_int8_has_fp16_weight": False,
        }

    from_pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": load_dtype,
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        from_pretrained_kwargs["quantization_config"] = quantization_config
        from_pretrained_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
    model.config.use_cache = False
    if quantization_config is None:
        model.to(device)
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    h8_rescued_modules: list[dict[str, Any]] = []
    if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY:
        h8_rescued_modules = apply_h8_selective_rescue(
            model=model,
            model_name=args.model_name,
            candidates_path=args.h8_candidates,
            policy_name=args.h8_policy_name,
            rescue_precision=args.h8_rescue_precision,
            device=device,
            bf16_ok=bf16_ok,
        )
        if not h8_rescued_modules:
            raise SystemExit("H8 selective rescue did not replace any modules.")

    lora_targets = infer_lora_targets(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    wrapped_norms: list[str] = []
    h6_int8_modules: list[str] = []
    if args.precision_policy == "fp32_norms":
        wrapped_norms = apply_fp32_norms(model)
        if not wrapped_norms:
            raise SystemExit("precision-policy fp32_norms requested, but no RMSNorm/LayerNorm-like modules were found.")
    elif args.precision_policy == "h6_late_mlp_int8_candidate":
        h6_int8_modules = apply_h6_late_mlp_int8_candidate(model)
    elif args.precision_policy == "h6_custom_int8":
        if not args.fake_int8_modules:
            raise SystemExit("--precision-policy h6_custom_int8 requires at least one --fake-int8-modules target.")
        h6_int8_modules = apply_fake_int8_modules(model, args.fake_int8_modules)

    if args.setup_only:
        trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_count = sum(param.numel() for param in model.parameters())
        setup_summary = {
            "model_name": args.model_name,
            "precision_policy": args.precision_policy,
            "device": device,
            "hardware_label": args.hardware_label,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "bf16_autocast": use_bf16,
            "qlora_config": qlora_config,
            "lora_targets": lora_targets,
            "trainable_params": trainable_count,
            "total_params": total_count,
            "fp32_norm_wrapped_modules": wrapped_norms,
            "h6_fake_int8_output_modules": h6_int8_modules,
            "h8_policy_name": args.h8_policy_name if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
            "h8_candidates": str(args.h8_candidates) if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
            "h8_rescue_precision": args.h8_rescue_precision if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
            "h8_rescued_modules": h8_rescued_modules,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else None,
        }
        with open(os.path.join(args.output_dir, "setup_summary.json"), "w", encoding="utf-8") as f:
            json.dump(setup_summary, f, indent=2)
        print(json.dumps(setup_summary, indent=2))
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    train_raw, eval_raw = load_dataset_split(args.dataset_name, args.seed, args.train_size, args.eval_size)
    train_ds = tokenize_dataset(train_raw, tokenizer, args.seq_len)
    eval_ds = tokenize_dataset(eval_raw, tokenizer, args.seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=data_generator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    train_iter = iter(train_loader)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    nan_or_inf_count = 0
    loss_history: list[float] = []
    spike_count = 0
    total_tokens = 0
    timed_train_tokens = 0
    timed_train_sec = 0.0
    timed_train_tokens_excluding_first = 0
    timed_train_sec_excluding_first = 0.0
    max_grad_norm = 0.0
    final_eval_loss = None
    eval_max_batches = args.eval_max_batches if args.eval_max_batches > 0 else None
    start_time = time.time()

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        progress = tqdm(range(1, args.max_steps + 1), desc=args.precision_policy)
        for step in progress:
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_tokens = 0

            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                batch = {key: value.to(device) for key, value in batch.items()}
                step_tokens += int(batch["attention_mask"].sum().item())
                with autocast_ctx:
                    loss = model(**batch).loss / args.gradient_accumulation_steps
                if not torch.isfinite(loss.detach()):
                    nan_or_inf_count += 1
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
                loss.backward()
                accum_loss += float(loss.detach().float().item())

            current_grad_norm = grad_norm(trainable_params)
            if not math.isfinite(current_grad_norm):
                nan_or_inf_count += 1
                raise FloatingPointError(f"Non-finite gradient norm at step {step}: {current_grad_norm}")
            max_grad_norm = max(max_grad_norm, current_grad_norm)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss = accum_loss
            loss_history.append(train_loss)
            if len(loss_history) > 50:
                previous = sorted(loss_history[-51:-1])
                rolling_median = previous[len(previous) // 2]
                if rolling_median > 0 and train_loss > 2.0 * rolling_median:
                    spike_count += 1

            if device == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - step_start
            total_tokens += step_tokens
            timed_train_tokens += step_tokens
            timed_train_sec += step_time
            if step > 1:
                timed_train_tokens_excluding_first += step_tokens
                timed_train_sec_excluding_first += step_time
            record = {
                "step": step,
                "train_loss": train_loss,
                "eval_loss": None,
                "grad_norm": current_grad_norm,
                "step_time_sec": step_time,
                "tokens": step_tokens,
                "tokens_per_sec": step_tokens / step_time if step_time > 0 else None,
                "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else None,
                "nan_or_inf_count": nan_or_inf_count,
                "loss_spike_count": spike_count,
            }
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.max_steps):
                final_eval_loss = evaluate(model, eval_loader, device, autocast_ctx, eval_max_batches)
                record["eval_loss"] = final_eval_loss
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            progress.set_postfix(loss=f"{train_loss:.3f}", grad=f"{current_grad_norm:.2f}")

    elapsed = time.time() - start_time
    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "precision_policy": args.precision_policy,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "seq_len": args.seq_len,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_sequences": args.per_device_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "eval_max_batches": args.eval_max_batches,
        "device": device,
        "hardware_label": args.hardware_label,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "bf16_autocast": use_bf16,
        "qlora_config": qlora_config,
        "lora_targets": lora_targets,
        "fp32_norm_wrapped_modules": wrapped_norms,
        "h6_fake_int8_output_modules": h6_int8_modules,
        "h6_fake_int8_bits": 8 if h6_int8_modules else None,
        "h6_fake_int8_gradient": "straight_through_estimator" if h6_int8_modules else None,
        "h8_policy_name": args.h8_policy_name if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
        "h8_candidates": str(args.h8_candidates) if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
        "h8_rescue_precision": args.h8_rescue_precision if args.precision_policy == H8_SELECTIVE_RESCUE_POLICY else None,
        "h8_rescued_modules": h8_rescued_modules,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "final_eval_loss": final_eval_loss,
        "max_grad_norm": max_grad_norm,
        "loss_spike_count": spike_count,
        "nan_or_inf_count": nan_or_inf_count,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else None,
        "elapsed_sec": elapsed,
        "train_time_sec": timed_train_sec,
        "tokens_per_sec_overall": total_tokens / elapsed if elapsed > 0 else None,
        "tokens_per_sec_train": timed_train_tokens / timed_train_sec if timed_train_sec > 0 else None,
        "tokens_per_sec_train_excluding_first_step": (
            timed_train_tokens_excluding_first / timed_train_sec_excluding_first
            if timed_train_sec_excluding_first > 0
            else None
        ),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
