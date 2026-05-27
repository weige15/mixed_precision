#!/usr/bin/env python3
"""Run H10 layer/group mixed-precision policies with Transformers + TorchAO."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_POLICIES = Path("experiments/h10-inference-ptq-assignment/results/layer_group_policy_candidates.json")
DEFAULT_BENCHMARK_DIR = Path("experiments/h10-inference-ptq-assignment/results/layer_group_benchmarks")
DEFAULT_QUALITY_DIR = Path("experiments/h10-inference-ptq-assignment/results/layer_group_quality")
DEFAULT_RUNTIME_CACHE_DIR = Path("tmp/h10_layer_group_backend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--policy-name", action="append", default=[])
    parser.add_argument("--benchmark-output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--quality-output-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    parser.add_argument("--hardware-label", default=os.environ.get("HARDWARE_LABEL", "unknown"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--quality-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--runtime-cache-dir", type=Path, default=DEFAULT_RUNTIME_CACHE_DIR)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_runtime_cache(runtime_cache_dir: Path | None) -> None:
    if runtime_cache_dir is None:
        return
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    runtime_cache = runtime_cache_dir.expanduser().resolve()
    runtime_cache.mkdir(parents=True, exist_ok=True)
    for child in ["tmp", "torchinductor", "triton", "cuda", "xdg"]:
        (runtime_cache / child).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_cache / "tmp")
    os.environ["TEMP"] = str(runtime_cache / "tmp")
    os.environ["TMP"] = str(runtime_cache / "tmp")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime_cache / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(runtime_cache / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(runtime_cache / "cuda")
    os.environ["XDG_CACHE_HOME"] = str(runtime_cache / "xdg")
    os.environ.setdefault("HF_HOME", hf_home)


def load_policy_grid(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Policy file does not exist: {path}. Run generate_layer_group_policies.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def select_policies(grid: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    policies = grid.get("candidate_policies", [])
    if not names:
        return policies
    by_name = {policy["policy_name"]: policy for policy in policies}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown policy name(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def quality_prompts(grid: dict[str, Any]) -> list[str]:
    prompts = [
        "Hardware-aware mixed precision assigns lower precision only where the model and backend can tolerate it.",
        "Layer-wise PTQ should be benchmarked with the same runtime that will serve the model.",
        "A useful inference benchmark should report prefill latency, decode throughput, peak memory, and quality change.",
    ]
    for workload in grid.get("workloads", []):
        prompts.extend(str(prompt) for prompt in workload.get("prompts", [])[:1])
    return prompts


def prompt_batches(prompts: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise SystemExit("--quality-batch-size must be >= 1")
    return [prompts[idx : idx + batch_size] for idx in range(0, len(prompts), batch_size)]


def package_snapshot() -> dict[str, dict[str, Any]]:
    packages = {}
    for name in ["torch", "transformers", "accelerate", "torchao"]:
        try:
            module = importlib.import_module(name)
            imported = True
            error = None
        except Exception as exc:  # noqa: BLE001
            module = None
            imported = False
            error = f"{exc.__class__.__name__}: {exc}"
        try:
            version = importlib.metadata.version(name)
        except Exception:
            version = getattr(module, "__version__", None) if module is not None else None
        packages[name] = {"imported": imported, "version": version, "error": error}
    return packages


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_import_error": f"{exc.__class__.__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
        "used_gib_from_mem_get_info": (total_bytes - free_bytes) / 1024**3,
    }


def reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def sync_if_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def dtype_from_name(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise SystemExit(f"Unsupported torch dtype name: {name}")
    return mapping[name]


def torchao_config(spec: dict[str, Any]) -> Any:
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
    )

    config_type = spec.get("type")
    if config_type == "int8_weight_only":
        return Int8WeightOnlyConfig(group_size=spec.get("group_size"), version=int(spec.get("version", 2)))
    if config_type == "int4_weight_only":
        return Int4WeightOnlyConfig(group_size=int(spec.get("group_size", 128)))
    if config_type == "int8_dynamic_activation_int8_weight":
        return Int8DynamicActivationInt8WeightConfig()
    raise SystemExit(f"Unsupported TorchAO layer/group config type: {config_type}")


def layer_group_uses_config(policy: dict[str, Any], config_type: str) -> bool:
    layer_group_policy = policy.get("layer_group_policy") or {}
    return any(
        (group.get("config") or {}).get("type") == config_type
        for group in layer_group_policy.get("groups", [])
    )


def validate_layer_group_backend(policy: dict[str, Any], device: str) -> None:
    layer_group_policy = policy.get("layer_group_policy")
    if not layer_group_policy or layer_group_policy.get("backend") != "torchao_fqn_to_config":
        return
    if not layer_group_uses_config(policy, "int4_weight_only") or device != "cuda":
        return

    import torch

    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    if major < 9:
        raise RuntimeError(
            "backend_infeasible: TorchAO int4_weight_only uses the MSLK/FlashInfer "
            f"SM90 TMA kernel path, but {device_name} has capability sm{major}{minor}. "
            "Run this policy on Hopper-class hardware or exclude it on RTX 3090."
        )


def regex_matches(pattern: str, module_name: str) -> bool:
    return re.fullmatch(pattern, module_name) is not None


def matched_modules(model: Any, layer_group_policy: dict[str, Any]) -> list[dict[str, Any]]:
    matches = []
    groups = layer_group_policy.get("groups", [])
    for group in groups:
        pattern = str(group["module_regex"])
        group_name = str(group.get("group_name", pattern))
        config = group.get("config", {})
        for module_name, module in model.named_modules():
            if regex_matches(pattern, module_name):
                matches.append(
                    {
                        "group_name": group_name,
                        "module_name": module_name,
                        "module_type": module.__class__.__name__,
                        "config_type": config.get("type"),
                    }
                )
    return matches


def apply_layer_group_policy(model: Any, policy: dict[str, Any], device: str) -> dict[str, Any]:
    layer_group_policy = policy.get("layer_group_policy")
    if not layer_group_policy:
        return {"backend": "none", "matched_modules": [], "num_matched_modules": 0}
    if layer_group_policy.get("backend") != "torchao_fqn_to_config":
        raise SystemExit(f"Unsupported layer/group backend: {layer_group_policy.get('backend')}")

    from torchao.quantization import FqnToConfig, quantize_

    matches = matched_modules(model, layer_group_policy)
    if not matches:
        raise RuntimeError("Layer/group policy matched zero modules; refusing to run an empty mixed-precision policy.")

    fqn_to_config = OrderedDict()
    for group in layer_group_policy.get("groups", []):
        fqn_to_config[f"re:{group['module_regex']}"] = torchao_config(group["config"])
    quantize_(model, FqnToConfig(fqn_to_config), filter_fn=None, device=device)
    return {
        "backend": layer_group_policy.get("backend"),
        "selection_source": layer_group_policy.get("selection_source"),
        "matched_modules": matches,
        "num_matched_modules": len(matches),
        "fqn_patterns": list(fqn_to_config.keys()),
    }


def load_model_and_tokenizer(grid: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(policy.get("model_name") or grid["model_name"])
    llm_kwargs = policy.get("llm_kwargs", {})
    torch_dtype = dtype_from_name(str(llm_kwargs.get("torch_dtype", "float16")))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")
    validate_layer_group_backend(policy, args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(llm_kwargs.get("trust_remote_code", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": bool(llm_kwargs.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model.to(args.device)
    layer_group_report = apply_layer_group_policy(model, policy, args.device)
    return model, tokenizer, layer_group_report


def sample_text(tokenizer: Any, sequences: Any, limit: int = 3) -> list[str]:
    texts = tokenizer.batch_decode(sequences[:limit], skip_special_tokens=True)
    return [text[:400] for text in texts]


def run_workload(model: Any, tokenizer: Any, workload: dict[str, Any], device: str, seed: int, repeat: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed + max(repeat, 0))
    prompts = workload["prompts"]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    prompt_tokens = int(encoded["attention_mask"].sum().item())
    reset_cuda_peak()
    before = cuda_snapshot()
    sync_if_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=int(workload["max_tokens"]),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    sync_if_cuda()
    latency = time.perf_counter() - start
    after = cuda_snapshot()
    generated = max(0, int(outputs.shape[1] - encoded["input_ids"].shape[1])) * int(outputs.shape[0])
    total_tokens = prompt_tokens + generated
    return {
        "workload_name": workload["name"],
        "repeat": repeat,
        "num_prompts": len(prompts),
        "max_tokens": int(workload["max_tokens"]),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "total_tokens": total_tokens,
        "latency_sec": latency,
        "output_tokens_per_sec": generated / latency if latency > 0 else None,
        "total_tokens_per_sec": total_tokens / latency if latency > 0 else None,
        "cuda_before": before,
        "cuda_after": after,
        "sample_outputs": sample_text(tokenizer, outputs),
    }


def score_prompt_batch(model: Any, tokenizer: Any, prompts: list[str], device: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    with torch.inference_mode():
        logits = model(**encoded).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = encoded["input_ids"][:, 1:]
    shift_mask = encoded["attention_mask"][:, 1:].bool()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    token_losses = losses[shift_mask]
    return {
        "mean_prompt_nll": float(token_losses.mean().item()) if token_losses.numel() else None,
        "tokens_scored": int(token_losses.numel()),
        "tokens_missing_logprob": 0,
    }


def write_benchmark(output_dir: Path, policy_name: str, payload: dict[str, Any]) -> Path:
    path = output_dir / policy_name / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_quality(output_dir: Path, policy_name: str, payload: dict[str, Any]) -> Path:
    path = output_dir / policy_name / "quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def base_payload(grid: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "policy_name": policy["policy_name"],
        "policy": policy,
        "model_name": str(policy.get("model_name") or grid["model_name"]),
        "baseline_model_name": grid["model_name"],
        "runtime": grid.get("runtime", "transformers_torchao_layer_group"),
        "hardware_label": args.hardware_label,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_snapshot": package_snapshot(),
    }


def run_policy(grid: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_payload = base_payload(grid, policy, args) | {
        "smoke": False,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "runs": [],
    }
    quality_payload = base_payload(grid, policy, args) | {
        "num_prompts": len(quality_prompts(grid)),
        "quality_batch_size": args.quality_batch_size,
    }
    if args.dry_run:
        benchmark_payload["status"] = "dry_run"
        benchmark_payload["workloads"] = grid.get("workloads", [])
        quality_payload["status"] = "dry_run"
        quality_payload["prompts"] = [{"chars": len(prompt), "text": prompt} for prompt in quality_prompts(grid)]
        return benchmark_payload, quality_payload

    try:
        benchmark_payload["pre_load_cuda"] = cuda_snapshot()
        load_start = time.perf_counter()
        model, tokenizer, layer_group_report = load_model_and_tokenizer(grid, policy, args)
        load_time = time.perf_counter() - load_start
        benchmark_payload["load_time_sec"] = load_time
        benchmark_payload["post_load_cuda"] = cuda_snapshot()
        benchmark_payload["layer_group_report"] = layer_group_report
        quality_payload["load_time_sec"] = load_time
        quality_payload["layer_group_report"] = layer_group_report

        if not args.skip_benchmark:
            for workload in grid.get("workloads", []):
                for warmup_idx in range(args.warmup_runs):
                    _ = run_workload(model, tokenizer, workload, args.device, args.seed, repeat=-(warmup_idx + 1))
                for repeat_idx in range(args.repeats):
                    benchmark_payload["runs"].append(
                        run_workload(model, tokenizer, workload, args.device, args.seed, repeat=repeat_idx)
                    )
            benchmark_payload["post_run_cuda"] = cuda_snapshot()
            benchmark_payload["status"] = "completed"
        else:
            benchmark_payload["status"] = "skipped"

        if not args.skip_quality:
            prompts = quality_prompts(grid)
            total_nll = 0.0
            total_scored = 0
            prompt_results = []
            quality_start = time.perf_counter()
            for batch in prompt_batches(prompts, args.quality_batch_size):
                batch_start = time.perf_counter()
                stats = score_prompt_batch(model, tokenizer, batch, args.device)
                scored = int(stats["tokens_scored"])
                if scored and stats["mean_prompt_nll"] is not None:
                    total_nll += float(stats["mean_prompt_nll"]) * scored
                total_scored += scored
                prompt_results.append(
                    {
                        "prompt_chars": [len(prompt) for prompt in batch],
                        "quality_time_sec": time.perf_counter() - batch_start,
                        **stats,
                    }
                )
            quality_payload["quality_time_sec"] = time.perf_counter() - quality_start
            quality_payload["prompt_results"] = prompt_results
            quality_payload["mean_prompt_nll"] = total_nll / total_scored if total_scored else None
            quality_payload["tokens_scored"] = total_scored
            quality_payload["tokens_missing_logprob"] = 0
            quality_payload["status"] = "completed"
        else:
            quality_payload["status"] = "skipped"

        del model
        cleanup_cuda()
    except Exception as exc:  # noqa: BLE001 - failed policies are research artifacts.
        error = f"{exc.__class__.__name__}: {exc}"
        benchmark_payload["status"] = "failed"
        benchmark_payload["error"] = error
        benchmark_payload["failure_cuda"] = cuda_snapshot()
        quality_payload["status"] = "failed"
        quality_payload["error"] = error
        cleanup_cuda()
    return benchmark_payload, quality_payload


def main() -> None:
    args = parse_args()
    configure_runtime_cache(args.runtime_cache_dir)
    grid = load_policy_grid(args.policies)
    policies = select_policies(grid, args.policy_name)
    for policy in policies:
        benchmark_payload, quality_payload = run_policy(grid, policy, args)
        if not args.skip_benchmark or args.dry_run:
            benchmark_path = write_benchmark(args.benchmark_output_dir, policy["policy_name"], benchmark_payload)
            print(f"{policy['policy_name']}: benchmark {benchmark_payload['status']} -> {benchmark_path}")
        if not args.skip_quality or args.dry_run:
            quality_path = write_quality(args.quality_output_dir, policy["policy_name"], quality_payload)
            print(f"{policy['policy_name']}: quality {quality_payload['status']} -> {quality_path}")


if __name__ == "__main__":
    main()
