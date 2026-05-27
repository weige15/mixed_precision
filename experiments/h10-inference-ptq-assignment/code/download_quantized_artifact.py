#!/usr/bin/env python3
"""Download a quantized HF artifact and optionally run the H10 PTQ pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DOWNLOAD_ROOT = Path("models/hf_quantized")
PIPELINE = Path("experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py")


CANDIDATES: dict[str, dict[str, str]] = {
    "llama31_8b_base_gptq": {
        "repo_id": "shuyuej/Meta-Llama-3.1-8B-GPTQ",
        "policy_name": "llama31_8b_base_gptq_artifact",
        "quantization": "gptq",
        "model_family": "base",
        "notes": "Best first candidate because it matches the current base-model H9 baseline.",
    },
    "llama31_8b_instruct_awq": {
        "repo_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "policy_name": "llama31_8b_instruct_awq_artifact",
        "quantization": "awq",
        "model_family": "instruct",
        "notes": "Good AWQ/Marlin candidate, but needs matched Instruct bf16/fp16 baselines for quality claims.",
    },
    "llama31_8b_instruct_gptq": {
        "repo_id": "shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ",
        "policy_name": "llama31_8b_instruct_gptq_artifact",
        "quantization": "gptq",
        "model_family": "instruct",
        "notes": "GPTQ Instruct candidate; compare only against regenerated Instruct baselines.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="llama31_8b_base_gptq")
    parser.add_argument("--repo-id", help="Override the selected candidate with any HF repo id.")
    parser.add_argument("--policy-name", help="Override policy name.")
    parser.add_argument("--quantization", help="Override vLLM quantization value.")
    parser.add_argument("--revision")
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--local-dir", type=Path, help="Exact local output directory.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token; defaults to HF_TOKEN env var.")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--hardware-label", default=os.environ.get("HARDWARE_LABEL", "rtx3090-lab"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return parser.parse_args()


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "__", value.strip())
    return clean.strip("._-") or "artifact"


def selected_spec(args: argparse.Namespace) -> dict[str, str]:
    spec = dict(CANDIDATES[args.candidate])
    if args.repo_id:
        spec["repo_id"] = args.repo_id
        spec.setdefault("model_family", "custom")
        spec.setdefault("notes", "Custom user-supplied artifact.")
    if args.policy_name:
        spec["policy_name"] = args.policy_name
    elif args.repo_id:
        spec["policy_name"] = f"{slug(args.repo_id).lower()}_artifact"
    if args.quantization:
        spec["quantization"] = args.quantization
    if not spec.get("policy_name") or not spec.get("repo_id") or not spec.get("quantization"):
        raise SystemExit("Need policy_name, repo_id, and quantization.")
    return spec


def local_dir_for(args: argparse.Namespace, spec: dict[str, str]) -> Path:
    if args.local_dir:
        return args.local_dir
    return args.download_root / slug(spec["repo_id"])


def print_candidates() -> None:
    for name, spec in CANDIDATES.items():
        print(f"{name}")
        print(f"  repo_id:      {spec['repo_id']}")
        print(f"  quantization: {spec['quantization']}")
        print(f"  family:       {spec['model_family']}")
        print(f"  notes:        {spec['notes']}")


def download(args: argparse.Namespace, spec: dict[str, str], local_dir: Path) -> None:
    print(f"repo_id:      {spec['repo_id']}")
    print(f"quantization: {spec['quantization']}")
    print(f"policy_name:  {spec['policy_name']}")
    print(f"local_dir:    {local_dir}")
    if spec.get("model_family") == "instruct":
        print("warning: this is an Instruct artifact; regenerate matched Instruct baselines before quality claims.")
    if args.dry_run:
        print("dry-run: not downloading")
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "huggingface_hub is required. Install it or use `huggingface-cli download` manually."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": spec["repo_id"],
        "local_dir": str(local_dir),
        "token": args.token,
        "revision": args.revision,
    }
    if args.allow_pattern:
        kwargs["allow_patterns"] = args.allow_pattern
    if args.ignore_pattern:
        kwargs["ignore_patterns"] = args.ignore_pattern
    # Remove None values because snapshot_download uses None differently for some args.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    path = snapshot_download(**kwargs)
    print(f"downloaded: {path}")


def run_pipeline(args: argparse.Namespace, spec: dict[str, str], local_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--policy-name",
        spec["policy_name"],
        "--model-name",
        str(local_dir),
        "--quantization",
        spec["quantization"],
        "--hardware-label",
        args.hardware_label,
    ]
    if args.run_smoke and not args.run_full:
        cmd.append("--smoke-only")
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
    printable = " ".join(cmd)
    if args.cuda_visible_devices is not None:
        printable = f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} {printable}"
    print(f"\n$ {printable}")
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.list_candidates:
        print_candidates()
        return
    spec = selected_spec(args)
    local_dir = local_dir_for(args, spec)
    download(args, spec, local_dir)
    if args.run_smoke or args.run_full:
        run_pipeline(args, spec, local_dir)
    else:
        print("\nNext commands:")
        print(
            "CUDA_VISIBLE_DEVICES=0 "
            "python experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py "
            f"--policy-name {spec['policy_name']} "
            f"--model-name {local_dir} "
            f"--quantization {spec['quantization']} "
            "--smoke-only "
            f"--hardware-label {args.hardware_label}"
        )
        print(
            "CUDA_VISIBLE_DEVICES=0 "
            "python experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py "
            f"--policy-name {spec['policy_name']} "
            f"--model-name {local_dir} "
            f"--quantization {spec['quantization']} "
            f"--hardware-label {args.hardware_label}"
        )


if __name__ == "__main__":
    main()
