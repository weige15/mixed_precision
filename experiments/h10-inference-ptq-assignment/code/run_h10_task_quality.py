#!/usr/bin/env python
"""Run a small exact-match task-quality screen for H10 inference PTQ policies."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import re
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLICIES = Path(
    "experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_gptq_marlin.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/h10-inference-ptq-assignment/results/task_quality")
DEFAULT_RUNTIME_CACHE_DIR = Path("tmp/h10_task_quality")

TASKS = [
    {
        "task_id": "arithmetic_addition",
        "prompt": "Answer with only the final answer.\nQuestion: What is 17 + 28?\nAnswer:",
        "answers": ["45"],
    },
    {
        "task_id": "numeric_comparison",
        "prompt": "Answer with only the final answer.\nQuestion: Which number is larger, 0.9 or 0.11?\nAnswer:",
        "answers": ["0.9"],
    },
    {
        "task_id": "sentiment_positive",
        "prompt": (
            "Classify the sentiment as positive or negative. Answer with one word.\n"
            "Text: I loved the careful benchmark report.\nSentiment:"
        ),
        "answers": ["positive"],
    },
    {
        "task_id": "letter_sequence",
        "prompt": "Answer with only the next letter.\nSequence: A B C D\nNext:",
        "answers": ["E"],
    },
    {
        "task_id": "inference_domain_yesno",
        "prompt": (
            "Answer yes or no.\n"
            "Question: Is a GPU normally used to accelerate large language model inference?\n"
            "Answer:"
        ),
        "answers": ["yes"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--policy-name", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional subdirectory label, e.g. a100-lab, to keep hardware-specific task-quality outputs separate.",
    )
    parser.add_argument("--hardware-label", default=os.environ.get("HARDWARE_LABEL", "unknown"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--runtime-cache-dir", type=Path, default=DEFAULT_RUNTIME_CACHE_DIR)
    parser.add_argument(
        "--runtime-backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Execution backend for task quality. Use transformers on Colab if vLLM is incompatible.",
    )
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        metavar="POLICY=MODEL",
        help="Override one policy's model path, e.g. llama31_8b_instruct_gptq_marlin_artifact=repo/name.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_runtime_cache(runtime_cache_dir: Path | None) -> None:
    if runtime_cache_dir is None:
        return
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    runtime_cache = runtime_cache_dir.expanduser().resolve()
    runtime_cache.mkdir(parents=True, exist_ok=True)
    for child in ["tmp", "torchinductor", "triton", "cuda", "xdg", "vllm"]:
        (runtime_cache / child).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_cache / "tmp")
    os.environ["TEMP"] = str(runtime_cache / "tmp")
    os.environ["TMP"] = str(runtime_cache / "tmp")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime_cache / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(runtime_cache / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(runtime_cache / "cuda")
    os.environ["XDG_CACHE_HOME"] = str(runtime_cache / "xdg")
    os.environ["VLLM_CACHE_ROOT"] = str(runtime_cache / "vllm")
    os.environ.setdefault("HF_HOME", hf_home)


def load_policy_grid(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Policy file does not exist: {path}")
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


def parse_model_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--model-override must be POLICY=MODEL, got: {value}")
        policy_name, model_name = value.split("=", 1)
        policy_name = policy_name.strip()
        model_name = model_name.strip()
        if not policy_name or not model_name:
            raise SystemExit(f"--model-override must be POLICY=MODEL, got: {value}")
        overrides[policy_name] = model_name
    return overrides


def policy_model_name(grid: dict[str, Any], policy: dict[str, Any], overrides: dict[str, str]) -> str:
    return str(overrides.get(policy["policy_name"]) or policy.get("model_name") or grid["model_name"])


def normalize_answer(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.whitespace + string.punctuation)


def score_prediction(prediction: str, answers: list[str]) -> bool:
    normalized = normalize_answer(prediction)
    for answer in answers:
        target = normalize_answer(answer)
        if normalized == target or normalized.startswith(f"{target} "):
            return True
    return False


def cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def package_snapshot() -> dict[str, dict[str, Any]]:
    packages = {}
    for name in ["torch", "vllm", "transformers", "accelerate", "bitsandbytes", "torchao", "flash_attn"]:
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


def write_result(output_dir: Path, policy_name: str, payload: dict[str, Any], run_label: str | None) -> Path:
    root = output_dir / run_label if run_label else output_dir
    path = root / policy_name / "task_quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def torchao_policy_error(policy: dict[str, Any]) -> str | None:
    llm_kwargs = policy.get("llm_kwargs", {})
    hf_overrides = llm_kwargs.get("hf_overrides") or {}
    has_torchao_config = bool(
        hf_overrides.get("quantization_config_dict_json") or hf_overrides.get("quantization_config_file")
    )
    if llm_kwargs.get("quantization") == "torchao" and not has_torchao_config:
        return (
            "Invalid H10 task-quality policy: vLLM torchao quantization requires "
            "hf_overrides.quantization_config_dict_json or hf_overrides.quantization_config_file."
        )
    return None


def base_payload(
    grid: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "policy_name": policy["policy_name"],
        "policy": policy,
        "model_name": model_name,
        "baseline_model_name": grid["model_name"],
        "runtime": args.runtime_backend,
        "hardware_label": args.hardware_label,
        "run_label": args.run_label,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "num_tasks": len(TASKS),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return payload


def run_policy_vllm(
    grid: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, Any]:
    payload = base_payload(grid, policy, args, model_name)
    if args.dry_run:
        payload["status"] = "dry_run"
        payload["tasks"] = TASKS
        return payload
    policy_error = torchao_policy_error(policy)
    if policy_error is not None:
        payload["status"] = "failed"
        payload["error"] = policy_error
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
        llm = LLM(model=model_name, **llm_kwargs)
        payload["load_time_sec"] = time.perf_counter() - load_start
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            stop=["\n"],
            seed=args.seed,
        )
        prompts = [task["prompt"] for task in TASKS]
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        payload["task_quality_time_sec"] = time.perf_counter() - start
        task_results = []
        correct = 0
        for task, output in zip(TASKS, outputs, strict=True):
            text = output.outputs[0].text if output.outputs else ""
            is_correct = score_prediction(text, task["answers"])
            correct += int(is_correct)
            task_results.append(
                {
                    "task_id": task["task_id"],
                    "answers": task["answers"],
                    "raw_prediction": text,
                    "normalized_prediction": normalize_answer(text),
                    "correct": is_correct,
                }
            )
        payload["task_results"] = task_results
        payload["accuracy"] = correct / len(TASKS)
        payload["status"] = "completed"
        del llm
        cleanup_cuda()
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        payload["package_snapshot"] = package_snapshot()
        cleanup_cuda()
    return payload


def torch_dtype_from_policy(policy: dict[str, Any]) -> Any:
    import torch

    dtype = str(policy.get("llm_kwargs", {}).get("dtype", "float16")).lower()
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


def run_policy_transformers(
    grid: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, Any]:
    payload = base_payload(grid, policy, args, model_name)
    if args.dry_run:
        payload["status"] = "dry_run"
        payload["tasks"] = TASKS
        return payload
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"Transformers import failed: {exc.__class__.__name__}: {exc}"
        payload["package_snapshot"] = package_snapshot()
        return payload
    try:
        torch.manual_seed(args.seed)
        dtype = torch_dtype_from_policy(policy)
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.eval()
        payload["load_time_sec"] = time.perf_counter() - load_start
        payload["torch_dtype"] = str(dtype).replace("torch.", "")
        task_results = []
        correct = 0
        start = time.perf_counter()
        for task in TASKS:
            inputs = tokenizer(task["prompt"], return_tensors="pt")
            first_device = next(model.parameters()).device
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            is_correct = score_prediction(text, task["answers"])
            correct += int(is_correct)
            task_results.append(
                {
                    "task_id": task["task_id"],
                    "answers": task["answers"],
                    "raw_prediction": text,
                    "normalized_prediction": normalize_answer(text),
                    "correct": is_correct,
                }
            )
        payload["task_quality_time_sec"] = time.perf_counter() - start
        payload["task_results"] = task_results
        payload["accuracy"] = correct / len(TASKS)
        payload["status"] = "completed"
        del model
        cleanup_cuda()
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        payload["package_snapshot"] = package_snapshot()
        cleanup_cuda()
    return payload


def run_policy(
    grid: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
    model_overrides: dict[str, str],
) -> dict[str, Any]:
    model_name = policy_model_name(grid, policy, model_overrides)
    if args.runtime_backend == "transformers":
        return run_policy_transformers(grid, policy, args, model_name)
    return run_policy_vllm(grid, policy, args, model_name)


def main() -> None:
    args = parse_args()
    configure_runtime_cache(args.runtime_cache_dir)
    grid = load_policy_grid(args.policies)
    policies = select_policies(grid, args.policy_name)
    model_overrides = parse_model_overrides(args.model_override)
    for policy in policies:
        payload = run_policy(grid, policy, args, model_overrides)
        path = write_result(args.output_dir, policy["policy_name"], payload, args.run_label)
        print(f"{policy['policy_name']}: {payload['status']} -> {path}")


if __name__ == "__main__":
    main()
