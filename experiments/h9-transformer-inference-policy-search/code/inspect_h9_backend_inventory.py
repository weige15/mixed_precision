#!/usr/bin/env python
"""Inspect local H9 vLLM backend support without loading the full model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLICIES = Path("experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json")
DEFAULT_OUTPUT = Path("experiments/h9-transformer-inference-policy-search/results/backend_inventory.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def package_info(import_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    distribution_name = distribution_name or import_name
    try:
        module = importlib.import_module(import_name)
        imported = True
        import_error = None
    except Exception as exc:  # noqa: BLE001 - inventory should record import failures.
        module = None
        imported = False
        import_error = f"{exc.__class__.__name__}: {exc}"
    try:
        version = importlib.metadata.version(distribution_name)
    except Exception:
        version = getattr(module, "__version__", None) if module is not None else None
    return {
        "import_name": import_name,
        "distribution_name": distribution_name,
        "imported": imported,
        "version": version,
        "import_error": import_error,
    }


def cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_import_error": f"{exc.__class__.__name__}: {exc}"}
    available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "cuda_available": available,
        "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if available else False,
    }
    if available:
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gib": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        info["devices"] = devices
    return info


def vllm_info() -> dict[str, Any]:
    try:
        import vllm
        from vllm import LLM
        from vllm.config import CacheConfig
    except Exception as exc:  # noqa: BLE001
        return {"imported": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return {
        "imported": True,
        "version": getattr(vllm, "__version__", None),
        "llm_signature": str(inspect.signature(LLM)),
        "cache_config_signature": str(inspect.signature(CacheConfig)),
        "known_quantization_values": [
            "awq",
            "gptq",
            "gptq_marlin",
            "awq_marlin",
            "bitblas",
            "bitsandbytes",
            "torchao",
            "fp8",
            "compressed-tensors",
            "mxfp4",
        ],
        "known_kv_cache_dtypes": ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"],
    }


def classify_policy(policy: dict[str, Any], packages: dict[str, dict[str, Any]], cuda: dict[str, Any]) -> dict[str, Any]:
    kwargs = policy.get("llm_kwargs", {})
    quantization = kwargs.get("quantization")
    kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
    hf_overrides = kwargs.get("hf_overrides") or {}
    has_torchao_config = bool(
        hf_overrides.get("quantization_config_dict_json") or hf_overrides.get("quantization_config_file")
    )
    status = "supported_unverified"
    reasons = []
    if not cuda.get("cuda_available"):
        status = "not_tested"
        reasons.append("CUDA is not available in this shell; instantiate on target GPU host.")
    if quantization == "bitsandbytes" and not packages["bitsandbytes"]["imported"]:
        status = "unsupported_backend"
        reasons.append("bitsandbytes is not importable.")
    if quantization == "torchao":
        if not packages["torchao"]["imported"]:
            status = "unsupported_backend"
            reasons.append("torchao is not importable.")
        elif not has_torchao_config:
            status = "missing_config"
            reasons.append("vLLM torchao quantization requires an explicit torchao_config; quantization='torchao' alone is not a runnable policy.")
        else:
            reasons.append("TorchAO is importable and a quantization config is supplied through hf_overrides; run vLLM smoke benchmark to confirm.")
    if quantization in {"awq", "gptq", "gptq_marlin", "awq_marlin", "bitblas"}:
        status = "missing_artifact"
        reasons.append("This policy requires a compatible quantized model artifact; H9.1 does not assume one.")
    if kv_cache_dtype not in {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"}:
        status = "unsupported_backend"
        reasons.append(f"Unknown KV-cache dtype for this inventory: {kv_cache_dtype}.")
    if not reasons:
        reasons.append("No preflight blocker found; run vLLM smoke benchmark to confirm.")
    return {
        "policy_name": policy.get("policy_name"),
        "status": status,
        "reasons": reasons,
        "llm_kwargs": kwargs,
    }


def main() -> None:
    args = parse_args()
    policies = json.loads(args.policies.read_text(encoding="utf-8")) if args.policies.exists() else {"candidate_policies": []}
    packages = {
        "torch": package_info("torch"),
        "transformers": package_info("transformers"),
        "vllm": package_info("vllm"),
        "bitsandbytes": package_info("bitsandbytes"),
        "flash_attn": package_info("flash_attn", "flash-attn"),
        "torchao": package_info("torchao"),
        "auto_gptq": package_info("auto_gptq"),
        "awq": package_info("awq"),
    }
    cuda = cuda_info()
    policy_reports = [classify_policy(policy, packages, cuda) for policy in policies.get("candidate_policies", [])]
    status_counts: dict[str, int] = {}
    for report in policy_reports:
        status_counts[report["status"]] = status_counts.get(report["status"], 0) + 1
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
        "policy_source": str(args.policies),
        "model_name": policies.get("model_name"),
        "packages": packages,
        "cuda": cuda,
        "vllm": vllm_info(),
        "policy_status_counts": status_counts,
        "policies": policy_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"policy_status_counts: {status_counts}")


if __name__ == "__main__":
    main()
