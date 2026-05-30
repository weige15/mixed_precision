"""QAQ evaluation harness and artifact helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from qaq.code.model_adapter import InventoryRecord, inventory_from_dicts, inventory_to_dicts
from qaq.code.oracle_labels import select_oracle_label
from qaq.code.policies import builtin_policy, expand_policy, save_expanded_policy
from qaq.code.router import build_router_trace, extract_token_features, save_router, train_router


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
    raise SystemExit(
        "prepare-bitplanes currently requires --synthetic in this environment. "
        "Install the declared ML dependencies and add a target-model run path before final evidence."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=Path("qaq/results/smoke"))
    common.add_argument("--model-name", default=DEFAULT_MODEL)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--hardware-label", default="unknown")
    common.add_argument(
        "--group-granularity",
        choices=["transformer_layer", "attention_mlp", "linear_module"],
        default="attention_mlp",
    )

    prepare = subparsers.add_parser("prepare-bitplanes", parents=[common])
    prepare.add_argument("--synthetic", action="store_true")
    prepare.set_defaults(func=prepare_bitplanes)

    static = subparsers.add_parser("run-static", parents=[common])
    static.add_argument("--policy", action="append", default=[])
    static.set_defaults(func=run_static)

    oracle = subparsers.add_parser("build-oracle-labels", parents=[common])
    oracle.add_argument("--quality-tolerance", type=float, default=0.05)
    oracle.set_defaults(func=build_oracle_labels)

    train = subparsers.add_parser("train-router", parents=[common])
    train.set_defaults(func=train_router_cli)

    run_router = subparsers.add_parser("run-router", parents=[common])
    run_router.set_defaults(func=run_router_cli)

    dynamic = subparsers.add_parser("run-dynamic-loader", parents=[common])
    dynamic.set_defaults(func=run_dynamic_loader_cli)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
