#!/usr/bin/env python
"""Measure H9 policy quality with vLLM prompt logprobs when available."""

from __future__ import annotations

import argparse
import gc
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


def cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
    gc.collect()


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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.dry_run:
        payload["status"] = "dry_run"
        payload["prompts"] = prompts
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
        load_start = time.perf_counter()
        llm = LLM(model=grid["model_name"], **llm_kwargs)
        payload["load_time_sec"] = time.perf_counter() - load_start
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=args.seed)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        payload["quality_time_sec"] = time.perf_counter() - start
        payload.update(prompt_nll(outputs))
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
