#!/usr/bin/env python3
"""Static block/group-wise progressive residual mixed-bit experiment.

The experiment targets Llama projection groups and evaluates prompt NLL after
reconstructing selected weights from a single progressive 2-bit residual stack:

0 bit = zero reconstruction
2 bit = slice 0
4 bit = slice 0 + slice 1
6 bit = slice 0 + slice 1 + slice 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

DEFAULT_OUTPUT_DIR = Path("experiments/h10-soft-residual-mixed-bit/results/static_llama31_8b_instruct")
DEFAULT_H10_GPTQ_SUMMARY = Path(
    "experiments/h9-transformer-inference-policy-search/results/h9_instruct_gptq_marlin_summary.json"
)

BUILTIN_PROMPTS = [
    "Hardware-aware mixed precision assigns lower precision only where the model and backend can tolerate it.",
    "The KV cache stores keys and values from previous tokens so autoregressive decoding can avoid recomputing the full prefix.",
    "A useful inference benchmark should report prefill latency, decode throughput, peak memory, and quality change.",
    "Explain hardware-aware mixed precision for LLM serving in one paragraph.",
    "Define KV cache quantization.",
    "List two risks of low-precision attention.",
]


@dataclass(frozen=True)
class BlockSpec:
    key: str
    module_name: str
    leaf: str
    col_start: int
    col_end: int
    num_params: int
    errors: dict[int, float]


@dataclass
class MethodResult:
    method: str
    average_bits_per_parameter: float
    metadata_bits_per_parameter: float
    effective_bits_per_parameter: float
    prompt_nll: float | None
    prompt_nll_delta_vs_bf16: float | None
    storage_redundancy: float | None
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--calibration-prompts", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--group-cols", type=int, default=128)
    parser.add_argument("--matched-budget-bits", type=float, default=3.0)
    parser.add_argument("--metadata-scale-bits", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-modules", type=int, default=0, help="Debug/smoke limit. 0 means all target modules.")
    parser.add_argument("--skip-eval", action="store_true", help="Build policies/storage table without prompt NLL.")
    parser.add_argument("--include-h10-gptq", action="store_true")
    parser.add_argument("--h10-gptq-summary", type=Path, default=DEFAULT_H10_GPTQ_SUMMARY)
    parser.add_argument("--toy-smoke", action="store_true", help="Run tensor-only policy/quantization smoke, no HF model.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned config and exit before model loading.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(torch: Any, name: str) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def choose_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false.")
    return requested


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_name(torch, args.dtype)
    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return model, tokenizer, device


def cleanup_model(model: Any | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def encode_prompts(tokenizer: Any, prompts: list[str], seq_len: int) -> list[dict[str, Any]]:
    encoded = []
    for prompt in prompts:
        item = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )
        labels = item["input_ids"].clone()
        labels[item["attention_mask"] == 0] = -100
        item["labels"] = labels
        encoded.append(item)
    return encoded


def target_linear_modules(model: Any, max_modules: int = 0) -> list[tuple[str, Any]]:
    modules = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_LEAVES and module.__class__.__name__.lower() == "linear":
            modules.append((name, module))
    modules.sort()
    if max_modules > 0:
        modules = modules[:max_modules]
    if not modules:
        raise SystemExit("No Llama target projection Linear modules were found.")
    return modules


def collect_input_second_moments(
    model: Any,
    modules: list[tuple[str, Any]],
    batches: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    import torch

    stats: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(module_name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            sums = torch.sum(x * x, dim=0).cpu()
            if module_name not in stats:
                stats[module_name] = {"sum_sq": sums, "count": x.shape[0]}
            else:
                stats[module_name]["sum_sq"] += sums
                stats[module_name]["count"] += x.shape[0]

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))
    first_device = next(model.parameters()).device if device == "cuda" else torch.device("cpu")
    try:
        with torch.inference_mode():
            for batch in batches:
                inputs = {key: value.to(first_device) for key, value in batch.items() if key != "labels"}
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {
        name: (entry["sum_sq"] / max(int(entry["count"]), 1)).clamp_min(1e-12)
        for name, entry in stats.items()
    }


def quantize_affine_2bit(x: Any) -> Any:
    import torch

    x = x.float()
    min_val = torch.min(x)
    max_val = torch.max(x)
    if torch.isclose(min_val, max_val):
        return torch.zeros_like(x) + min_val
    scale = (max_val - min_val) / 3.0
    q = torch.clamp(torch.round((x - min_val) / scale), 0, 3)
    return q * scale + min_val


def progressive_reconstructions(block: Any) -> dict[int, Any]:
    import torch

    block = block.float()
    recon = torch.zeros_like(block)
    out = {0: recon.clone()}
    for bits in (2, 4, 6):
        residual = block - recon
        recon = recon + quantize_affine_2bit(residual)
        out[bits] = recon.clone()
    return out


def weighted_sse(block: Any, recon: Any, col_energy: Any) -> float:
    import torch

    err = (block.float() - recon.float()) ** 2
    weighted = err * col_energy.float().reshape(1, -1)
    return float(torch.sum(weighted).item())


def build_block_specs(
    model: Any,
    modules: list[tuple[str, Any]],
    input_moments: dict[str, Any],
    group_cols: int,
) -> list[BlockSpec]:
    specs = []
    for module_name, module in modules:
        weight = module.weight.detach().cpu().float()
        leaf = module_name.rsplit(".", 1)[-1]
        in_features = weight.shape[1]
        moments = input_moments.get(module_name)
        if moments is None:
            moments = weight.new_ones(in_features)
        for col_start in range(0, in_features, group_cols):
            col_end = min(col_start + group_cols, in_features)
            block = weight[:, col_start:col_end]
            col_energy = moments[col_start:col_end]
            recons = progressive_reconstructions(block)
            errors = {bits: weighted_sse(block, recon, col_energy) for bits, recon in recons.items()}
            key = f"{module_name}:cols{col_start}-{col_end}"
            specs.append(
                BlockSpec(
                    key=key,
                    module_name=module_name,
                    leaf=leaf,
                    col_start=col_start,
                    col_end=col_end,
                    num_params=int(block.numel()),
                    errors=errors,
                )
            )
    return specs


def assign_uniform(specs: list[BlockSpec], bits: int) -> dict[str, int]:
    return {spec.key: bits for spec in specs}


def assign_hard_0_4(specs: list[BlockSpec], target_bits: float) -> dict[str, int]:
    total_params = sum(spec.num_params for spec in specs)
    budget = target_bits * total_params
    assignment = {spec.key: 0 for spec in specs}
    ranked = sorted(
        specs,
        key=lambda spec: (spec.errors[0] - spec.errors[4]) / max(4 * spec.num_params, 1),
        reverse=True,
    )
    spent = 0.0
    for spec in ranked:
        cost = 4 * spec.num_params
        if spent + cost <= budget + 1e-9:
            assignment[spec.key] = 4
            spent += cost
    return assignment


def assign_soft_progressive(specs: list[BlockSpec], target_bits: float) -> dict[str, int]:
    total_params = sum(spec.num_params for spec in specs)
    budget = target_bits * total_params
    assignment = {spec.key: 0 for spec in specs}
    spent = 0.0
    spec_by_key = {spec.key: spec for spec in specs}
    while True:
        best: tuple[float, BlockSpec, int, float] | None = None
        for spec in specs:
            current = assignment[spec.key]
            if current >= 6:
                continue
            nxt = current + 2
            cost = 2 * spec.num_params
            if spent + cost > budget + 1e-9:
                continue
            benefit = spec.errors[current] - spec.errors[nxt]
            score = benefit / max(cost, 1)
            if best is None or score > best[0]:
                best = (score, spec, nxt, cost)
        if best is None or best[0] <= 0.0:
            break
        _, spec, nxt, cost = best
        if spec.key not in spec_by_key:
            raise AssertionError("Internal assignment error.")
        assignment[spec.key] = nxt
        spent += cost
    return assignment


def assignment_stats(
    specs: list[BlockSpec],
    assignment: dict[str, int],
    metadata_scale_bits: int,
    assignment_code_bits: int,
) -> tuple[float, float, float]:
    total_params = sum(spec.num_params for spec in specs)
    raw_bits = sum(assignment[spec.key] * spec.num_params for spec in specs) / total_params
    metadata_bits = 0.0
    for spec in specs:
        bits = assignment[spec.key]
        active_slices = bits // 2
        metadata_bits += active_slices * metadata_scale_bits
        metadata_bits += assignment_code_bits
    metadata_bpp = metadata_bits / total_params
    return raw_bits, metadata_bpp, raw_bits + metadata_bpp


def reconstruct_weight(weight: Any, assignments: dict[str, int], module_name: str, group_cols: int) -> Any:
    import torch

    if getattr(weight, "is_meta", False):
        raise RuntimeError(
            f"Target weight is on the meta device for {module_name}. "
            "This usually means the model was partially offloaded during reload. "
            "Use a larger GPU, reduce other GPU memory pressure, or run with a "
            "Transformers/Accelerate loading setup that materializes target projection weights."
        )
    original_dtype = weight.dtype
    weight_f = weight.detach().cpu().float()
    out = torch.empty_like(weight_f)
    in_features = weight_f.shape[1]
    for col_start in range(0, in_features, group_cols):
        col_end = min(col_start + group_cols, in_features)
        key = f"{module_name}:cols{col_start}-{col_end}"
        bits = assignments[key]
        block = weight_f[:, col_start:col_end]
        out[:, col_start:col_end] = progressive_reconstructions(block)[bits]
    return out.to(dtype=original_dtype)


def apply_assignment(model: Any, assignments: dict[str, int], group_cols: int, max_modules: int) -> dict[str, Any]:
    modules = target_linear_modules(model, max_modules=max_modules)
    applied = []
    for module_name, module in modules:
        reconstructed = reconstruct_weight(module.weight, assignments, module_name, group_cols)
        module.weight.data.copy_(reconstructed.to(module.weight.device))
        applied.append(module_name)
    return {"num_modules": len(applied), "modules": applied[:10], "truncated_modules": max(len(applied) - 10, 0)}


def prompt_nll(model: Any, batches: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch

    total_loss = 0.0
    total_tokens = 0
    first_device = next(model.parameters()).device if device == "cuda" else torch.device("cpu")
    with torch.inference_mode():
        for batch in batches:
            inputs = {key: value.to(first_device) for key, value in batch.items()}
            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = outputs.loss
            tokens = int((labels != -100).sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
    return {"mean_prompt_nll": total_loss / max(total_tokens, 1), "tokens_scored": total_tokens}


def write_table(path: Path, results: list[MethodResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "average bits per parameter",
        "metadata bits per parameter",
        "effective bits per parameter",
        "prompt NLL",
        "prompt NLL delta versus bf16",
        "storage redundancy",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "average bits per parameter": fmt(result.average_bits_per_parameter),
                    "metadata bits per parameter": fmt(result.metadata_bits_per_parameter),
                    "effective bits per parameter": fmt(result.effective_bits_per_parameter),
                    "prompt NLL": fmt(result.prompt_nll),
                    "prompt NLL delta versus bf16": fmt(result.prompt_nll_delta_vs_bf16),
                    "storage redundancy": fmt(result.storage_redundancy),
                    "notes": result.notes,
                }
            )


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.6f}"


def storage_redundancy(effective_bpp: float) -> float:
    separate_2_4_6_raw_bits = 12.0
    return separate_2_4_6_raw_bits / effective_bpp if effective_bpp > 0 else math.inf


def load_gptq_result(summary_path: Path, bf16_nll: float | None) -> MethodResult | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in summary.get("completed", [])
        if row.get("policy_name") == "llama31_8b_instruct_gptq_marlin_artifact"
    ]
    if not rows:
        return None
    quality = rows[0].get("quality") or {}
    nll = quality.get("mean_prompt_nll")
    delta = (float(nll) - bf16_nll) if nll is not None and bf16_nll is not None else None
    return MethodResult(
        method="existing GPTQ-Marlin 4-bit baseline",
        average_bits_per_parameter=4.0,
        metadata_bits_per_parameter=float("nan"),
        effective_bits_per_parameter=float("nan"),
        prompt_nll=float(nll) if nll is not None else None,
        prompt_nll_delta_vs_bf16=delta,
        storage_redundancy=None,
        notes="Imported from current H10/H9 Instruct GPTQ-Marlin artifact summary; metadata not audited here.",
    )


def make_policy_payload(specs: list[BlockSpec], policies: dict[str, dict[str, int]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "target_leaves": sorted(TARGET_LEAVES),
        "group_cols": args.group_cols,
        "matched_budget_bits": args.matched_budget_bits,
        "num_blocks": len(specs),
        "num_target_params": sum(spec.num_params for spec in specs),
        "policies": {
            name: {
                "bit_counts": {str(bits): list(assignments.values()).count(bits) for bits in (0, 2, 4, 6)},
                "assignments": assignments,
            }
            for name, assignments in policies.items()
        },
    }


def run_toy_smoke(args: argparse.Namespace) -> None:
    import torch

    set_seed(args.seed)
    weight = torch.randn(32, 64)
    moments = torch.linspace(0.25, 2.0, steps=64)
    class Module:
        pass

    module = Module()
    module.weight = weight
    specs = []
    for col_start in range(0, 64, args.group_cols):
        col_end = min(col_start + args.group_cols, 64)
        block = weight[:, col_start:col_end]
        recons = progressive_reconstructions(block)
        errors = {bits: weighted_sse(block, recon, moments[col_start:col_end]) for bits, recon in recons.items()}
        specs.append(BlockSpec(f"toy.linear:cols{col_start}-{col_end}", "toy.linear", "toy", col_start, col_end, block.numel(), errors))
    policies = {
        "uniform_2bit_residual": assign_uniform(specs, 2),
        "uniform_4bit_residual": assign_uniform(specs, 4),
        f"hard_0_4_budget_{args.matched_budget_bits:g}bit": assign_hard_0_4(specs, args.matched_budget_bits),
        f"soft_0_2_4_6_budget_{args.matched_budget_bits:g}bit": assign_soft_progressive(specs, args.matched_budget_bits),
    }
    payload = make_policy_payload(specs, policies, args)
    payload["status"] = "toy_smoke_completed"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "toy_smoke.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"toy_smoke_completed -> {args.output_dir / 'toy_smoke.json'}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if args.toy_smoke:
        run_toy_smoke(args)
        return
    if args.dry_run:
        print(f"dry_run -> {args.output_dir / 'run_config.json'}")
        return

    set_seed(args.seed)
    start = time.perf_counter()
    try:
        model, tokenizer, device = load_model_and_tokenizer(args)
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure artifact.
        payload = {
            "schema_version": 1,
            "status": "failed",
            "stage": "load_model_and_tokenizer",
            "elapsed_sec": time.perf_counter() - start,
            "error": f"{exc.__class__.__name__}: {exc}",
            "model_name": args.model_name,
            "local_files_only": args.local_files_only,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(payload["error"]) from exc
    prompts = BUILTIN_PROMPTS[: max(args.calibration_prompts, args.eval_prompts)]
    calibration_batches = encode_prompts(tokenizer, prompts[: args.calibration_prompts], args.seq_len)
    eval_batches = encode_prompts(tokenizer, prompts[: args.eval_prompts], args.seq_len)
    modules = target_linear_modules(model, max_modules=args.max_modules)
    input_moments = collect_input_second_moments(model, modules, calibration_batches, device)
    specs = build_block_specs(model, modules, input_moments, args.group_cols)
    policies = {
        "uniform 2-bit residual": assign_uniform(specs, 2),
        "uniform 4-bit residual": assign_uniform(specs, 4),
        f"hard 0/4-bit pruning @ {args.matched_budget_bits:g} avg bits": assign_hard_0_4(specs, args.matched_budget_bits),
        f"soft 0/2/4/6-bit residual @ {args.matched_budget_bits:g} avg bits": assign_soft_progressive(
            specs, args.matched_budget_bits
        ),
    }
    (args.output_dir / "policies.json").write_text(
        json.dumps(make_policy_payload(specs, policies, args), indent=2) + "\n",
        encoding="utf-8",
    )

    results: list[MethodResult] = []
    bf16_nll = None
    if not args.skip_eval:
        bf16_stats = prompt_nll(model, eval_batches, device)
        bf16_nll = float(bf16_stats["mean_prompt_nll"])
    results.append(
        MethodResult(
            method="bf16 baseline",
            average_bits_per_parameter=16.0,
            metadata_bits_per_parameter=0.0,
            effective_bits_per_parameter=16.0,
            prompt_nll=bf16_nll,
            prompt_nll_delta_vs_bf16=0.0 if bf16_nll is not None else None,
            storage_redundancy=None,
            notes="Matched Transformers baseline for this runner.",
        )
    )
    cleanup_model(model)

    if args.include_h10_gptq:
        gptq = load_gptq_result(args.h10_gptq_summary, bf16_nll)
        if gptq is not None:
            results.append(gptq)

    for method, assignment in policies.items():
        raw_bpp, meta_bpp, effective_bpp = assignment_stats(
            specs,
            assignment,
            metadata_scale_bits=args.metadata_scale_bits,
            assignment_code_bits=0 if method.startswith("uniform") else 2,
        )
        nll = None
        if not args.skip_eval:
            model, _, device = load_model_and_tokenizer(args)
            try:
                apply_assignment(model, assignment, args.group_cols, args.max_modules)
                stats = prompt_nll(model, eval_batches, device)
                nll = float(stats["mean_prompt_nll"])
            finally:
                cleanup_model(model)
        results.append(
            MethodResult(
                method=method,
                average_bits_per_parameter=raw_bpp,
                metadata_bits_per_parameter=meta_bpp,
                effective_bits_per_parameter=effective_bpp,
                prompt_nll=nll,
                prompt_nll_delta_vs_bf16=(nll - bf16_nll) if nll is not None and bf16_nll is not None else None,
                storage_redundancy=storage_redundancy(effective_bpp),
                notes="Single progressive residual stack; no separate 2/4/6 checkpoints.",
            )
        )

    write_table(args.output_dir / "results_table.csv", results)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_sec": time.perf_counter() - start,
        "results": [result.__dict__ for result in results],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"completed -> {args.output_dir / 'results_table.csv'}")


if __name__ == "__main__":
    main()
