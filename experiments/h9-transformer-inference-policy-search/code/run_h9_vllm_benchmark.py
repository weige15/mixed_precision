#!/usr/bin/env python
"""Run concrete H9 vLLM inference policy benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLICIES = Path("experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json")
DEFAULT_OUTPUT_DIR = Path("experiments/h9-transformer-inference-policy-search/results/benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--policy-name", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hardware-label", default=os.environ.get("HARDWARE_LABEL", "unknown"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Use one tiny prompt/workload for fast policy instantiation checks.")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned benchmark configs without loading vLLM.")
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


def smoke_workloads() -> list[dict[str, Any]]:
    return [
        {
            "name": "smoke",
            "max_tokens": 4,
            "prompts": ["Briefly define KV cache."],
            "description": "Tiny policy-instantiation smoke workload.",
        }
    ]


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_import_error": f"{exc.__class__.__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "free_gib": free_gib,
        "total_gib": total_gib,
        "used_gib_from_mem_get_info": total_gib - free_gib,
    }


def reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
    gc.collect()


def output_token_count(request_outputs: list[Any]) -> int:
    total = 0
    for request_output in request_outputs:
        for completion in getattr(request_output, "outputs", []) or []:
            total += len(getattr(completion, "token_ids", []) or [])
    return total


def prompt_token_count(request_outputs: list[Any]) -> int:
    total = 0
    for request_output in request_outputs:
        total += len(getattr(request_output, "prompt_token_ids", []) or [])
    return total


def sample_text(request_outputs: list[Any], limit: int = 3) -> list[str]:
    samples = []
    for request_output in request_outputs:
        for completion in getattr(request_output, "outputs", []) or []:
            samples.append(str(getattr(completion, "text", ""))[:400])
            if len(samples) >= limit:
                return samples
    return samples


def run_workload(llm: Any, sampling_params_cls: Any, workload: dict[str, Any], repeat: int) -> dict[str, Any]:
    prompts = workload["prompts"]
    sampling_seed = repeat if repeat >= 0 else 0
    sampling_params = sampling_params_cls(
        temperature=0.0,
        max_tokens=int(workload["max_tokens"]),
        seed=sampling_seed,
    )
    reset_cuda_peak()
    before = cuda_snapshot()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    after = cuda_snapshot()
    latency = end - start
    generated = output_token_count(outputs)
    prompt_tokens = prompt_token_count(outputs)
    total_tokens = generated + prompt_tokens
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
        "sample_outputs": sample_text(outputs),
    }


def write_result(output_dir: Path, policy_name: str, payload: dict[str, Any]) -> Path:
    path = output_dir / policy_name / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_policy(
    *,
    grid: dict[str, Any],
    policy: dict[str, Any],
    workloads: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy_name = policy["policy_name"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "policy_name": policy_name,
        "policy": policy,
        "model_name": grid["model_name"],
        "runtime": grid.get("runtime", "vllm"),
        "hardware_label": args.hardware_label,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runs": [],
    }
    if args.dry_run:
        payload["status"] = "dry_run"
        payload["workloads"] = workloads
        return payload

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"vLLM import failed: {exc.__class__.__name__}: {exc}"
        return payload

    try:
        llm_kwargs = dict(policy.get("llm_kwargs", {}))
        llm_kwargs["seed"] = args.seed
        payload["pre_load_cuda"] = cuda_snapshot()
        load_start = time.perf_counter()
        llm = LLM(model=grid["model_name"], **llm_kwargs)
        payload["load_time_sec"] = time.perf_counter() - load_start
        payload["post_load_cuda"] = cuda_snapshot()

        for workload in workloads:
            for warmup_idx in range(args.warmup_runs):
                _ = run_workload(llm, SamplingParams, workload, repeat=-(warmup_idx + 1))
            for repeat_idx in range(args.repeats):
                payload["runs"].append(run_workload(llm, SamplingParams, workload, repeat=repeat_idx))
        payload["status"] = "completed"
        payload["post_run_cuda"] = cuda_snapshot()
        del llm
        cleanup_cuda()
    except Exception as exc:  # noqa: BLE001 - failed policies are research artifacts.
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        payload["failure_cuda"] = cuda_snapshot()
        cleanup_cuda()
    return payload


def main() -> None:
    args = parse_args()
    grid = load_policy_grid(args.policies)
    policies = select_policies(grid, args.policy_name)
    workloads = smoke_workloads() if args.smoke else grid.get("workloads", [])
    if not workloads:
        raise SystemExit("No workloads found in policy grid.")

    dry_runs = []
    for policy in policies:
        payload = run_policy(grid=grid, policy=policy, workloads=workloads, args=args)
        path = write_result(args.output_dir, policy["policy_name"], payload)
        print(f"{policy['policy_name']}: {payload['status']} -> {path}")
        if args.dry_run:
            dry_runs.append(str(path))
    if args.dry_run:
        print(f"dry-run benchmark plans written: {len(dry_runs)}")


if __name__ == "__main__":
    main()
