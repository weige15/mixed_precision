#!/usr/bin/env python
"""Measure H9 policy quality with vLLM prompt logprobs when available."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_POLICIES = Path("experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json")
DEFAULT_OUTPUT_DIR = Path("experiments/h9-transformer-inference-policy-search/results/quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--policy-name", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hardware-label", default=os.environ.get("HARDWARE_LABEL", "unknown"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality-batch-size",
        type=int,
        default=1,
        help="Number of prompts to score per vLLM generate call. Use 1 for long-context prompt-logprob scoring.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override the policy gpu_memory_utilization for quality scoring.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_policy_grid(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Policy file does not exist: {path}. Run generate_h9_policies.py first.")
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
        "The KV cache stores keys and values from previous tokens so autoregressive decoding can avoid recomputing the full prefix.",
        "A useful inference benchmark should report prefill latency, decode throughput, peak memory, and quality change.",
    ]
    for workload in grid.get("workloads", []):
        prompts.extend(str(prompt) for prompt in workload.get("prompts", [])[:1])
    return prompts


def token_logprob_value(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get(token_id) or entry.get(str(token_id))
        if value is None:
            return None
        logprob = getattr(value, "logprob", None)
        return float(logprob if logprob is not None else value)
    return None


def prompt_nll(request_outputs: list[Any]) -> dict[str, Any]:
    token_logprobs = []
    missing = 0
    for request_output in request_outputs:
        token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
        prompt_logprobs = list(getattr(request_output, "prompt_logprobs", []) or [])
        for idx, token_id in enumerate(token_ids):
            if idx == 0:
                continue
            entry = prompt_logprobs[idx] if idx < len(prompt_logprobs) else None
            logprob = token_logprob_value(entry, int(token_id))
            if logprob is None:
                missing += 1
                continue
            token_logprobs.append(logprob)
    nll = -mean(token_logprobs) if token_logprobs else None
    return {
        "mean_prompt_nll": nll,
        "tokens_scored": len(token_logprobs),
        "tokens_missing_logprob": missing,
    }


def prompt_batches(prompts: list[str], batch_size: int) -> list[list[tuple[int, str]]]:
    if batch_size < 1:
        raise SystemExit("--quality-batch-size must be >= 1")
    indexed = list(enumerate(prompts))
    return [indexed[idx : idx + batch_size] for idx in range(0, len(indexed), batch_size)]


def cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
    gc.collect()


def package_snapshot() -> dict[str, dict[str, Any]]:
    packages = {}
    for name in ["torch", "vllm", "bitsandbytes", "torchao", "flash_attn"]:
        try:
            module = importlib.import_module(name)
            imported = True
            error = None
        except Exception as exc:  # noqa: BLE001 - diagnostic path.
            module = None
            imported = False
            error = f"{exc.__class__.__name__}: {exc}"
        try:
            version = importlib.metadata.version("flash-attn" if name == "flash_attn" else name)
        except Exception:
            version = getattr(module, "__version__", None) if module is not None else None
        packages[name] = {"imported": imported, "version": version, "error": error}
    return packages


def write_result(output_dir: Path, policy_name: str, payload: dict[str, Any]) -> Path:
    path = output_dir / policy_name / "quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_policy(grid: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompts = quality_prompts(grid)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "policy_name": policy["policy_name"],
        "policy": policy,
        "model_name": grid["model_name"],
        "runtime": grid.get("runtime", "vllm"),
        "hardware_label": args.hardware_label,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "quality_batch_size": args.quality_batch_size,
        "gpu_memory_utilization_override": args.gpu_memory_utilization,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.dry_run:
        payload["status"] = "dry_run"
        payload["prompts"] = [{"index": idx, "chars": len(prompt), "text": prompt} for idx, prompt in enumerate(prompts)]
        return payload
    if policy.get("llm_kwargs", {}).get("quantization") == "torchao" and "torchao_config" not in policy.get("llm_kwargs", {}):
        payload["status"] = "failed"
        payload["error"] = "Invalid H9 policy: vLLM torchao quantization requires an explicit torchao_config; skip fp16_torchao until a concrete torchao_config is added."
        payload["package_snapshot"] = package_snapshot()
        return payload
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"vLLM import failed: {exc.__class__.__name__}: {exc}"
        payload["package_snapshot"] = package_snapshot()
        return payload
    try:
        llm_kwargs = dict(policy.get("llm_kwargs", {}))
        llm_kwargs["seed"] = args.seed
        if args.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        load_start = time.perf_counter()
        llm = LLM(model=grid["model_name"], **llm_kwargs)
        payload["load_time_sec"] = time.perf_counter() - load_start
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=args.seed)
        start = time.perf_counter()
        total_nll = 0.0
        total_scored = 0
        total_missing = 0
        prompt_results = []
        for batch in prompt_batches(prompts, args.quality_batch_size):
            batch_indices = [idx for idx, _ in batch]
            batch_prompts = [prompt for _, prompt in batch]
            batch_start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            batch_stats = prompt_nll(outputs)
            scored = int(batch_stats["tokens_scored"])
            if scored and batch_stats["mean_prompt_nll"] is not None:
                total_nll += float(batch_stats["mean_prompt_nll"]) * scored
            total_scored += scored
            total_missing += int(batch_stats["tokens_missing_logprob"])
            prompt_results.append(
                {
                    "prompt_indices": batch_indices,
                    "prompt_chars": [len(prompt) for prompt in batch_prompts],
                    "quality_time_sec": time.perf_counter() - batch_start,
                    **batch_stats,
                }
            )
        payload["quality_time_sec"] = time.perf_counter() - start
        payload["prompt_results"] = prompt_results
        payload["mean_prompt_nll"] = total_nll / total_scored if total_scored else None
        payload["tokens_scored"] = total_scored
        payload["tokens_missing_logprob"] = total_missing
        payload["status"] = "completed"
        del llm
        cleanup_cuda()
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        cleanup_cuda()
    return payload


def main() -> None:
    args = parse_args()
    grid = load_policy_grid(args.policies)
    policies = select_policies(grid, args.policy_name)
    for policy in policies:
        payload = run_policy(grid, policy, args)
        path = write_result(args.output_dir, policy["policy_name"], payload)
        print(f"{policy['policy_name']}: {payload['status']} -> {path}")


if __name__ == "__main__":
    main()
