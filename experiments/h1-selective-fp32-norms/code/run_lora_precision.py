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
from typing import Any

H6_LATE_MLP_INT8_MODULES = {
    "base_model.model.model.layers.22.mlp.gate_proj",
    "base_model.model.model.layers.22.mlp.up_proj",
    "base_model.model.model.layers.23.mlp.gate_proj",
    "base_model.model.model.layers.23.mlp.up_proj",
}


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
        choices=["bf16_baseline", "fp32_norms", "h6_late_mlp_int8_candidate"],
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


def apply_h6_late_mlp_int8_candidate(model: Any) -> list[str]:
    hooked = []
    for name, module in model.named_modules():
        if name not in H6_LATE_MLP_INT8_MODULES:
            continue
        if module.__class__.__name__.lower() != "linear":
            raise SystemExit(f"H6 candidate target is not a Linear module: {name} ({module.__class__.__name__})")
        module.register_forward_hook(lambda _module, _inputs, output: fake_quant_output(output, bits=8))
        hooked.append(name)

    missing = sorted(H6_LATE_MLP_INT8_MODULES - set(hooked))
    if missing:
        raise SystemExit("H6 candidate module(s) not found:\n" + "\n".join(missing))
    return sorted(hooked)


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
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_ok
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32
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
        "bf16_autocast": use_bf16,
        "lora_targets": lora_targets,
        "fp32_norm_wrapped_modules": wrapped_norms,
        "h6_fake_int8_output_modules": h6_int8_modules,
        "h6_fake_int8_bits": 8 if h6_int8_modules else None,
        "h6_fake_int8_gradient": "straight_through_estimator" if h6_int8_modules else None,
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
