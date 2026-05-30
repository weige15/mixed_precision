"""QAQ evaluation harness and artifact helpers."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from statistics import mean
import time
from typing import Any

from qaq.code.model_adapter import InventoryRecord, discover_linear_module_inventory, inventory_from_dicts, inventory_to_dicts
from qaq.code.oracle_labels import select_oracle_label
from qaq.code.policies import PrecisionPolicy, builtin_policy, expand_policy, save_expanded_policy
from qaq.code.router import (
    FeatureRecord,
    build_router_trace,
    extract_token_features,
    load_router,
    save_router,
    train_router,
)


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
ALLOWED_WIDTHS = (2, 4, 6, 8)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require_runtime_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Real QAQ runs require installed dependencies from requirements.txt: {exc}") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, load_dataset


def dtype_from_arg(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer, _ = require_runtime_deps()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    return torch, model, tokenizer


def load_text_samples(args: argparse.Namespace) -> list[dict[str, str]]:
    _, _, _, load_dataset = require_runtime_deps()
    if args.text:
        return [{"sample_id": f"text-{idx}", "text": text} for idx, text in enumerate(args.text)]
    dataset_kwargs = {}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    dataset = load_dataset(
        args.dataset_name,
        **dataset_kwargs,
        split=args.dataset_split,
        streaming=args.streaming,
    )
    rows = []
    for idx, row in enumerate(dataset):
        text = str(row.get(args.text_field, "")).strip()
        if not text:
            continue
        rows.append({"sample_id": str(row.get("id", f"sample-{idx}")), "text": text})
        if len(rows) >= args.max_samples:
            break
    if not rows:
        raise SystemExit(f"No non-empty text rows found in dataset field {args.text_field!r}")
    return rows


def encode_sample(torch: Any, tokenizer: Any, text: str, seq_len: int, device: Any) -> dict[str, Any] | None:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encoded["input_ids"]
    if int(input_ids.shape[-1]) < 2:
        return None
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": encoded.get("attention_mask", torch.ones_like(input_ids)).to(device),
    }


def first_parameter_device(model: Any) -> Any:
    return next(model.parameters()).device


def bitplane_artifact_path(output_dir: Path) -> Path:
    return output_dir / "bitplanes.pt"


def load_bitplane_records(path: Path) -> dict[str, Any]:
    torch, _, _, _ = require_runtime_deps()
    return torch.load(path, map_location="cpu", weights_only=False)


def save_bitplane_records(path: Path, records: dict[str, Any]) -> None:
    torch, _, _, _ = require_runtime_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, path)


def load_inventory_file(path: Path) -> list[InventoryRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return inventory_from_dicts(payload.get("inventory", payload))


def prepare_records_from_model(model: Any, inventory: list[InventoryRecord]) -> dict[str, Any]:
    from qaq.code.bitplanes import BitPlaneConfig, quantize_tensor_to_bitplanes

    modules = dict(model.named_modules())
    config = BitPlaneConfig(max_bits=8, allowed_bit_widths=ALLOWED_WIDTHS)
    records = {}
    for row in inventory:
        module = modules[row.module_name]
        records[row.tensor_name] = quantize_tensor_to_bitplanes(
            module.weight,
            config,
            tensor_name=row.tensor_name,
            group_id=row.group_id,
        )
    return records


def load_prepared_artifacts(args: argparse.Namespace) -> tuple[list[InventoryRecord], dict[str, Any]]:
    artifact_dir = Path(args.bitplanes_dir)
    inventory = load_inventory_file(artifact_dir / "inventory.json")
    records = load_bitplane_records(artifact_dir / "bitplanes.pt")
    return inventory, records


def group_storage_bytes(records: dict[str, Any], group_bit_widths: dict[str, int]) -> int:
    from qaq.code.bitplanes import estimate_storage_bytes

    total = 0
    for record in records.values():
        if record.group_id is None:
            continue
        total += estimate_storage_bytes(record, int(group_bit_widths[record.group_id]))
    return int(total)


def materialized_weight_bytes(model: Any, inventory: list[InventoryRecord]) -> int:
    modules = dict(model.named_modules())
    total = 0
    for row in inventory:
        weight = modules[row.module_name].weight
        total += int(weight.numel() * weight.element_size())
    return total


def apply_reconstructed_weights(
    model: Any,
    inventory: list[InventoryRecord],
    records: dict[str, Any],
    group_bit_widths: dict[str, int],
) -> None:
    from qaq.code.bitplanes import reconstruct_tensor

    modules = dict(model.named_modules())
    for row in inventory:
        module = modules[row.module_name]
        record = records[row.tensor_name]
        width = int(group_bit_widths[row.group_id])
        reconstructed = reconstruct_tensor(
            record,
            width,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        module.weight.data.copy_(reconstructed)


@contextlib.contextmanager
def maybe_no_grad(torch: Any) -> Any:
    with torch.no_grad():
        yield


def evaluate_policy_on_samples(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    inventory: list[InventoryRecord],
    records: dict[str, Any],
    policy: PrecisionPolicy,
    samples: list[dict[str, str]],
    *,
    trace_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    torch = __import__("torch")
    expanded = expand_policy(policy, inventory)
    apply_reconstructed_weights(model, inventory, records, expanded)
    storage_bytes = group_storage_bytes(records, expanded)
    materialized_bytes = materialized_weight_bytes(model, inventory)
    device = first_parameter_device(model)
    metrics = []
    with maybe_no_grad(torch):
        for sample in samples:
            encoded = encode_sample(torch, tokenizer, sample["text"], args.seq_len, device)
            if encoded is None:
                continue
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=encoded["input_ids"],
            )
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0
            loss = float(output.loss.detach().cpu().item())
            input_tokens = int(encoded["attention_mask"].sum().detach().cpu().item())
            peak_gpu_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if torch.cuda.is_available() and device.type == "cuda"
                else None
            )
            metrics.append(
                {
                    "sample_id": f"{trace_prefix or policy.policy_name}:{sample['sample_id']}",
                    "raw_sample_id": sample["sample_id"],
                    "policy_name": policy.policy_name,
                    "seed": args.seed,
                    "hardware_label": args.hardware_label,
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "quality": -loss,
                    "loss": loss,
                    "perplexity": math.exp(loss) if loss < 20 else None,
                    "latency_ms": latency_ms,
                    "memory_bytes": storage_bytes,
                    "materialized_weight_bytes": materialized_bytes,
                    "peak_gpu_memory_bytes": peak_gpu_memory,
                    "selected_group_bit_widths": expanded,
                    "dynamic_loader_enabled": False,
                }
            )
    return metrics, expanded


def builtin_policy_catalog(inventory: list[InventoryRecord], names: list[str], seed: int) -> dict[str, PrecisionPolicy]:
    return {name: builtin_policy(name, inventory, seed=seed) for name in names}


def synthetic_inventory(group_granularity: str = "attention_mlp") -> list[InventoryRecord]:
    if group_granularity == "linear_module":
        attention_group = "linear:model.layers.0.self_attn.q_proj"
        mlp_group = "linear:model.layers.0.mlp.down_proj"
    elif group_granularity == "attention_mlp":
        attention_group = "layer_0000:attention"
        mlp_group = "layer_0000:mlp"
    else:
        attention_group = "layer_0000"
        mlp_group = "layer_0000"
    return [
        InventoryRecord(
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            module_name="model.layers.0.self_attn.q_proj",
            module_role="attention",
            layer_idx=0,
            group_id=attention_group,
            group_granularity=group_granularity,
            shape=(4, 4),
            source_dtype="torch.float32",
        ),
        InventoryRecord(
            tensor_name="model.layers.0.mlp.down_proj.weight",
            module_name="model.layers.0.mlp.down_proj",
            module_role="mlp",
            layer_idx=0,
            group_id=mlp_group,
            group_granularity=group_granularity,
            shape=(4, 4),
            source_dtype="torch.float32",
        ),
    ]


def selected_bit_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for width in (record.get("selected_group_bit_widths") or {}).values():
            key = str(width)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def aggregate_metrics(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    quality_values = [float(row["quality"]) for row in records if row.get("quality") is not None]
    latency_values = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
    memory_values = [float(row["memory_bytes"]) for row in records if row.get("memory_bytes") is not None]
    return {
        "model_name": config.get("model_name"),
        "dataset_name": config.get("dataset_name"),
        "dataset_split": config.get("dataset_split"),
        "seed": config.get("seed"),
        "hardware_label": config.get("hardware_label"),
        "group_granularity": config.get("group_granularity"),
        "allowed_bit_widths": config.get("allowed_bit_widths"),
        "policies": sorted({row.get("policy_name") for row in records if row.get("policy_name")}),
        "quality_metrics": {"mean_quality": mean(quality_values) if quality_values else None},
        "latency_metrics": {"mean_latency_ms": mean(latency_values) if latency_values else None},
        "memory_metrics": {"mean_memory_bytes": mean(memory_values) if memory_values else None},
        "selected_bit_distribution": selected_bit_distribution(records),
        "dynamic_loader_enabled": bool(config.get("dynamic_loader_enabled", False)),
        "errors": [],
    }


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload.pop("func", None)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def run_static(args: argparse.Namespace) -> None:
    if args.bitplanes_dir is not None:
        torch, model, tokenizer = load_model_and_tokenizer(args)
        output_dir = Path(args.output_dir)
        inventory, records = load_prepared_artifacts(args)
        samples = load_text_samples(args)
        config = serialize_config(args) | {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "allowed_bit_widths": list(ALLOWED_WIDTHS),
            "dynamic_loader_enabled": False,
        }
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "inventory.json", {"inventory": inventory_to_dicts(inventory)})

        metrics: list[dict[str, Any]] = []
        for policy_name in args.policy or ["static_8bit", "static_4bit", "mixed_attention_high"]:
            policy = builtin_policy(policy_name, inventory, seed=args.seed)
            policy_metrics, _ = evaluate_policy_on_samples(args, model, tokenizer, inventory, records, policy, samples)
            save_expanded_policy(policy, inventory, output_dir / f"{policy_name}_policy.json")
            append_jsonl(output_dir / "metrics.jsonl", policy_metrics)
            metrics.extend(policy_metrics)
        write_json(output_dir / "summary.json", aggregate_metrics(metrics, config))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    output_dir = Path(args.output_dir)
    inventory = synthetic_inventory(args.group_granularity)
    policy_names = args.policy or ["static_8bit", "static_4bit", "mixed_attention_high"]
    config = serialize_config(args) | {
        "model_name": args.model_name,
        "dataset_name": "synthetic",
        "dataset_split": "synthetic",
        "allowed_bit_widths": [2, 4, 6, 8],
        "dynamic_loader_enabled": False,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "inventory.json", {"inventory": inventory_to_dicts(inventory)})

    records: list[dict[str, Any]] = []
    for policy_name in policy_names:
        policy = builtin_policy(policy_name, inventory, seed=args.seed)
        expanded = expand_policy(policy, inventory)
        save_expanded_policy(policy, inventory, output_dir / f"{policy_name}_policy.json")
        avg_bits = mean(expanded.values())
        records.append(
            {
                "sample_id": "synthetic-0",
                "policy_name": policy_name,
                "seed": args.seed,
                "hardware_label": args.hardware_label,
                "input_tokens": 8,
                "output_tokens": 0,
                "quality": 1.0 / avg_bits,
                "latency_ms": avg_bits,
                "memory_bytes": int(avg_bits * 32),
                "selected_group_bit_widths": expanded,
                "dynamic_loader_enabled": False,
            }
        )
    append_jsonl(output_dir / "metrics.jsonl", records)
    write_json(output_dir / "summary.json", aggregate_metrics(records, config))


def build_oracle_labels(args: argparse.Namespace) -> None:
    if args.bitplanes_dir is not None:
        torch, model, tokenizer = load_model_and_tokenizer(args)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory, records = load_prepared_artifacts(args)
        samples = load_text_samples(args)
        policy_names = args.policy or ["static_4bit", "mixed_attention_high", "static_8bit"]
        config = serialize_config(args) | {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "allowed_bit_widths": list(ALLOWED_WIDTHS),
        }
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "inventory.json", {"inventory": inventory_to_dicts(inventory)})

        all_metrics: list[dict[str, Any]] = []
        by_sample: dict[str, list[dict[str, Any]]] = {}
        expanded_by_policy: dict[str, dict[str, int]] = {}
        for policy_name in policy_names:
            policy = builtin_policy(policy_name, inventory, seed=args.seed)
            policy_metrics, expanded = evaluate_policy_on_samples(args, model, tokenizer, inventory, records, policy, samples)
            expanded_by_policy[policy_name] = expanded
            all_metrics.extend(policy_metrics)
            append_jsonl(output_dir / "metrics.jsonl", policy_metrics)
            for row in policy_metrics:
                by_sample.setdefault(row["raw_sample_id"], []).append(row)

        labels = []
        score_rows = []
        feature_rows = []
        for sample in samples:
            sample_id = sample["sample_id"]
            encoded = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len,
            )
            feature = extract_token_features(
                sample_id,
                encoded["input_ids"],
                encoded.get("attention_mask"),
                {"dataset_name": args.dataset_name, "dataset_split": args.dataset_split},
            )
            feature_rows.append(feature.to_dict())
            scores = []
            for row in by_sample.get(sample_id, []):
                avg_bits = mean(expanded_by_policy[row["policy_name"]].values())
                scores.append(
                    {
                        "sample_id": sample_id,
                        "policy_id": row["policy_name"],
                        "quality": row["quality"],
                        "loss": row["loss"],
                        "expected_cost": avg_bits,
                    }
                )
            if scores:
                score_rows.extend(scores)
                labels.append(
                    select_oracle_label(
                        sample_id,
                        scores,
                        quality_tolerance=args.quality_tolerance,
                        reference_quality=max(score["quality"] for score in scores),
                        fallback_policy_id=args.fallback_policy,
                    ).to_dict()
                )
        append_jsonl(output_dir / "features.jsonl", feature_rows)
        append_jsonl(output_dir / "candidate_scores.jsonl", score_rows)
        append_jsonl(output_dir / "oracle_labels.jsonl", labels)
        write_json(
            output_dir / "summary.json",
            aggregate_metrics(all_metrics, config)
            | {
                "num_labels": len(labels),
                "tolerance_satisfied": sum(1 for row in labels if row["tolerance_satisfied"]),
            },
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = serialize_config(args) | {
        "model_name": args.model_name,
        "dataset_name": "synthetic",
        "dataset_split": "synthetic",
        "allowed_bit_widths": [2, 4, 6, 8],
    }
    write_json(output_dir / "config.json", config)
    labels = []
    score_rows = []
    for idx, length in enumerate([8, 32, 96]):
        sample_id = f"synthetic-{idx}"
        scores = [
            {"sample_id": sample_id, "policy_id": "static_4bit", "quality": 1.0 - length / 256.0, "expected_cost": 4.0},
            {"sample_id": sample_id, "policy_id": "static_8bit", "quality": 1.0, "expected_cost": 8.0},
        ]
        score_rows.extend(scores)
        labels.append(
            select_oracle_label(
                sample_id,
                scores,
                quality_tolerance=args.quality_tolerance,
                reference_quality=1.0,
                fallback_policy_id="static_8bit",
            ).to_dict()
        )
    append_jsonl(output_dir / "candidate_scores.jsonl", score_rows)
    append_jsonl(output_dir / "oracle_labels.jsonl", labels)
    write_json(
        output_dir / "summary.json",
        {
            "model_name": args.model_name,
            "seed": args.seed,
            "hardware_label": args.hardware_label,
            "num_labels": len(labels),
            "tolerance_satisfied": sum(1 for row in labels if row["tolerance_satisfied"]),
        },
    )


def train_router_cli(args: argparse.Namespace) -> None:
    if args.features_jsonl is not None and args.oracle_labels_jsonl is not None:
        output_dir = Path(args.output_dir)
        feature_payloads = read_jsonl(Path(args.features_jsonl))
        label_payloads = read_jsonl(Path(args.oracle_labels_jsonl))
        labels_by_sample = {str(row["sample_id"]): str(row["policy_id"]) for row in label_payloads}
        features = [
            FeatureRecord(
                sample_id=str(row["sample_id"]),
                prompt_metadata=dict(row.get("prompt_metadata", {})),
                feature_schema_version=str(row["feature_schema_version"]),
                features={str(key): float(value) for key, value in row["features"].items()},
            )
            for row in feature_payloads
            if str(row["sample_id"]) in labels_by_sample
        ]
        labels = [labels_by_sample[record.sample_id] for record in features]
        router = train_router(features, labels)
        write_json(output_dir / "config.json", serialize_config(args))
        save_router(router, output_dir / "router.json")
        append_jsonl(
            output_dir / "router_labels.jsonl",
            [{"sample_id": record.sample_id, "policy_id": label} for record, label in zip(features, labels)],
        )
        write_json(
            output_dir / "summary.json",
            {"status": "trained", "router_type": "nearest_centroid", "num_labels": len(labels)},
        )
        return

    output_dir = Path(args.output_dir)
    features = [
        extract_token_features("synthetic-short", list(range(8)), [1] * 8),
        extract_token_features("synthetic-medium", list(range(32)), [1] * 32),
        extract_token_features("synthetic-long", list(range(96)), [1] * 96),
    ]
    labels = ["static_4bit", "static_4bit", "static_8bit"]
    router = train_router(features, labels)
    write_json(output_dir / "config.json", serialize_config(args))
    append_jsonl(output_dir / "features.jsonl", [record.to_dict() for record in features])
    append_jsonl(output_dir / "router_labels.jsonl", [{"sample_id": record.sample_id, "policy_id": label} for record, label in zip(features, labels)])
    save_router(router, output_dir / "router.json")
    write_json(output_dir / "summary.json", {"status": "trained", "router_type": "nearest_centroid", "num_labels": len(labels)})


def run_router_cli(args: argparse.Namespace) -> None:
    if args.bitplanes_dir is not None and args.router_path is not None:
        torch, model, tokenizer = load_model_and_tokenizer(args)
        output_dir = Path(args.output_dir)
        inventory, records = load_prepared_artifacts(args)
        samples = load_text_samples(args)
        router = load_router(args.router_path)
        catalog_names = args.policy or ["static_4bit", "mixed_attention_high", "static_8bit"]
        catalog = builtin_policy_catalog(inventory, catalog_names, args.seed)
        traces = []
        metrics = []
        for sample in samples:
            encoded = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len,
            )
            feature = extract_token_features(
                sample["sample_id"],
                encoded["input_ids"],
                encoded.get("attention_mask"),
                {"dataset_name": args.dataset_name, "dataset_split": args.dataset_split},
            )
            trace = build_router_trace(router, feature, catalog, inventory)
            traces.append(trace)
            policy = catalog[trace["predicted_policy"]]
            sample_metrics, _ = evaluate_policy_on_samples(
                args,
                model,
                tokenizer,
                inventory,
                records,
                policy,
                [sample],
                trace_prefix="router",
            )
            metrics.extend(sample_metrics)
        config = serialize_config(args) | {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "allowed_bit_widths": list(ALLOWED_WIDTHS),
            "dynamic_loader_enabled": False,
        }
        write_json(output_dir / "config.json", config)
        append_jsonl(output_dir / "router_trace.jsonl", traces)
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        write_json(output_dir / "summary.json", aggregate_metrics(metrics, config))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    output_dir = Path(args.output_dir)
    inventory = synthetic_inventory(args.group_granularity)
    features = [
        extract_token_features("synthetic-short", list(range(8)), [1] * 8),
        extract_token_features("synthetic-long", list(range(96)), [1] * 96),
    ]
    router = train_router(features, ["static_4bit", "static_8bit"])
    catalog = {
        "static_4bit": builtin_policy("static_4bit", inventory, seed=args.seed),
        "static_8bit": builtin_policy("static_8bit", inventory, seed=args.seed),
    }
    traces = [build_router_trace(router, record, catalog, inventory) for record in features]
    records = []
    for trace in traces:
        avg_bits = mean(trace["selected_group_bit_widths"].values())
        records.append(
            {
                "sample_id": trace["sample_id"],
                "policy_name": trace["predicted_policy"],
                "seed": args.seed,
                "hardware_label": args.hardware_label,
                "input_tokens": int(trace["features"]["features"]["input_length"]),
                "output_tokens": 0,
                "quality": 1.0 / avg_bits,
                "latency_ms": avg_bits,
                "memory_bytes": int(avg_bits * 32),
                "selected_group_bit_widths": trace["selected_group_bit_widths"],
                "dynamic_loader_enabled": False,
            }
        )
    config = serialize_config(args) | {
        "model_name": args.model_name,
        "dataset_name": "synthetic",
        "dataset_split": "synthetic",
        "allowed_bit_widths": [2, 4, 6, 8],
        "dynamic_loader_enabled": False,
    }
    write_json(output_dir / "config.json", config)
    append_jsonl(output_dir / "router_trace.jsonl", traces)
    append_jsonl(output_dir / "metrics.jsonl", records)
    write_json(output_dir / "summary.json", aggregate_metrics(records, config))


def run_dynamic_loader_cli(args: argparse.Namespace) -> None:
    if args.bitplanes_dir is not None:
        torch, _, _, _ = require_runtime_deps()
        from qaq.code.dynamic_loader import LoaderConfig, materialize_policy

        output_dir = Path(args.output_dir)
        inventory, records = load_prepared_artifacts(args)
        policy = builtin_policy(args.policy[0] if args.policy else "static_4bit", inventory, seed=args.seed)
        expanded = expand_policy(policy, inventory)
        _, loader_metrics = materialize_policy(
            records,
            expanded,
            LoaderConfig(target_device=args.loader_device, dry_run_cpu=args.loader_dry_run_cpu),
        )
        rows = [
            row
            | {
                "sample_id": "dynamic-loader-materialization",
                "policy_name": policy.policy_name,
                "seed": args.seed,
                "hardware_label": args.hardware_label,
                "dynamic_loader_enabled": True,
            }
            for row in loader_metrics
        ]
        config = serialize_config(args) | {
            "model_name": args.model_name,
            "allowed_bit_widths": list(ALLOWED_WIDTHS),
            "dynamic_loader_enabled": True,
        }
        write_json(output_dir / "config.json", config)
        append_jsonl(output_dir / "metrics.jsonl", rows)
        write_json(
            output_dir / "summary.json",
            {
                "model_name": args.model_name,
                "seed": args.seed,
                "hardware_label": args.hardware_label,
                "dynamic_loader_enabled": True,
                "policy_name": policy.policy_name,
                "num_tensors": len(rows),
                "loader_metrics": {
                    "total_loader_ms": sum(float(row["total_loader_ms"]) for row in rows),
                    "transfer_ms": sum(float(row["transfer_ms"]) for row in rows),
                    "reconstruction_ms": sum(float(row["reconstruction_ms"]) for row in rows),
                    "cpu_plane_bytes": sum(int(row["cpu_plane_bytes"]) for row in rows),
                    "gpu_materialized_bytes": sum(int(row["gpu_materialized_bytes"]) for row in rows),
                },
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    try:
        import torch

        from qaq.code.bitplanes import BitPlaneConfig, quantize_tensor_to_bitplanes
        from qaq.code.dynamic_loader import LoaderConfig, materialize_tensor
    except ModuleNotFoundError as exc:
        raise SystemExit(f"run-dynamic-loader requires installed ML dependencies: {exc}") from exc

    output_dir = Path(args.output_dir)
    record = quantize_tensor_to_bitplanes(
        torch.tensor([[-1.0, 0.25], [0.5, 1.0]], dtype=torch.float32),
        BitPlaneConfig(),
        tensor_name="synthetic.weight",
        group_id="synthetic:group",
    )
    _, metrics = materialize_tensor(record, 4, LoaderConfig(target_device="cpu"))
    metrics.update(
        {
            "sample_id": "synthetic-0",
            "policy_name": "static_4bit",
            "seed": args.seed,
            "hardware_label": args.hardware_label,
            "dynamic_loader_enabled": True,
        }
    )
    config = serialize_config(args) | {
        "model_name": args.model_name,
        "dataset_name": "synthetic",
        "dataset_split": "synthetic",
        "allowed_bit_widths": [2, 4, 6, 8],
        "dynamic_loader_enabled": True,
    }
    write_json(output_dir / "config.json", config)
    append_jsonl(output_dir / "metrics.jsonl", [metrics])
    write_json(
        output_dir / "summary.json",
        {
            "model_name": args.model_name,
            "seed": args.seed,
            "hardware_label": args.hardware_label,
            "dynamic_loader_enabled": True,
            "loader_metrics": metrics,
        },
    )


def prepare_bitplanes(args: argparse.Namespace) -> None:
    if args.synthetic:
        output_dir = Path(args.output_dir)
        inventory = synthetic_inventory(args.group_granularity)
        write_json(output_dir / "config.json", serialize_config(args))
        write_json(output_dir / "inventory.json", {"inventory": inventory_to_dicts(inventory)})
        write_json(output_dir / "summary.json", {"status": "synthetic_inventory_only", "num_tensors": len(inventory)})
        return
    torch, model, _ = load_model_and_tokenizer(args)
    output_dir = Path(args.output_dir)
    inventory = discover_linear_module_inventory(
        model,
        group_granularity=args.group_granularity,
        module_name_allowlist=args.module_name_allowlist,
        include_lm_head=args.include_lm_head,
    )
    records = prepare_records_from_model(model, inventory)
    write_json(output_dir / "config.json", serialize_config(args) | {"allowed_bit_widths": list(ALLOWED_WIDTHS)})
    write_json(output_dir / "inventory.json", {"inventory": inventory_to_dicts(inventory)})
    save_bitplane_records(bitplane_artifact_path(output_dir), records)
    write_json(
        output_dir / "summary.json",
        {
            "status": "prepared",
            "model_name": args.model_name,
            "hardware_label": args.hardware_label,
            "num_tensors": len(records),
            "num_groups": len({row.group_id for row in inventory}),
            "bitplane_artifact": str(bitplane_artifact_path(output_dir)),
        },
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=Path("qaq/results/smoke"))
    common.add_argument("--model-name", default=DEFAULT_MODEL)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--hardware-label", default="unknown")
    common.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    common.add_argument("--device-map", default="auto")
    common.add_argument("--local-files-only", action="store_true")
    common.add_argument("--dataset-name", default=DEFAULT_DATASET)
    common.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    common.add_argument("--dataset-split", default="validation")
    common.add_argument("--text-field", default="text")
    common.add_argument("--max-samples", type=int, default=8)
    common.add_argument("--seq-len", type=int, default=512)
    common.add_argument("--streaming", action="store_true")
    common.add_argument("--text", action="append", default=[])
    common.add_argument(
        "--bitplanes-dir",
        type=Path,
        default=None,
        help="Directory containing inventory.json and bitplanes.pt from prepare-bitplanes.",
    )
    common.add_argument(
        "--group-granularity",
        choices=["transformer_layer", "attention_mlp", "linear_module"],
        default="attention_mlp",
    )

    prepare = subparsers.add_parser("prepare-bitplanes", parents=[common])
    prepare.add_argument("--synthetic", action="store_true")
    prepare.add_argument(
        "--module-name-allowlist",
        action="append",
        default=[],
        help="Optional full module name or leaf name to quantize; may be repeated.",
    )
    prepare.add_argument("--include-lm-head", action="store_true")
    prepare.set_defaults(func=prepare_bitplanes)

    static = subparsers.add_parser("run-static", parents=[common])
    static.add_argument("--policy", action="append", default=[])
    static.set_defaults(func=run_static)

    oracle = subparsers.add_parser("build-oracle-labels", parents=[common])
    oracle.add_argument("--policy", action="append", default=[])
    oracle.add_argument("--quality-tolerance", type=float, default=0.05)
    oracle.add_argument("--fallback-policy", default="static_8bit")
    oracle.set_defaults(func=build_oracle_labels)

    train = subparsers.add_parser("train-router", parents=[common])
    train.add_argument("--features-jsonl", type=Path, default=None)
    train.add_argument("--oracle-labels-jsonl", type=Path, default=None)
    train.set_defaults(func=train_router_cli)

    run_router = subparsers.add_parser("run-router", parents=[common])
    run_router.add_argument("--router-path", type=Path, default=None)
    run_router.add_argument("--policy", action="append", default=[])
    run_router.set_defaults(func=run_router_cli)

    dynamic = subparsers.add_parser("run-dynamic-loader", parents=[common])
    dynamic.add_argument("--policy", action="append", default=[])
    dynamic.add_argument("--loader-device", default="cpu")
    dynamic.add_argument("--loader-dry-run-cpu", action="store_true")
    dynamic.set_defaults(func=run_dynamic_loader_cli)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
