#!/usr/bin/env python3
"""Build an H10 HAQ-for-PEFT action table from existing H8 summaries.

This script converts matched bf16/QLoRA/selective-rescue comparisons into the
small grouped action table consumed by `solve_precision_assignment.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FIELDNAMES = [
    "model_name",
    "group_name",
    "module_names",
    "candidate_action",
    "backend",
    "hardware_label",
    "backend_feasible",
    "predicted_quality_risk",
    "predicted_instability_risk",
    "quality_recovery_vs_lowbit",
    "memory_delta_gib_vs_bf16",
    "throughput_delta_pct_vs_bf16",
    "source_artifact",
    "notes",
]


POLICY_TO_ACTION = {
    "qlora_nf4": ("blanket_qlora_nf4", "bitsandbytes", "all projections"),
    "h8_rescue": (
        "qlora_nf4_bf16_projection_rescue",
        "bitsandbytes+torch",
        "selected high-risk projection rescues",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h8-summary",
        type=Path,
        default=Path("experiments/h8-hardware-aware-precision-search/results/llama_h8_metrics_summary.json"),
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hardware-label", default="rtx3090-lab")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/h10-haq-peft-assignment/results/action_table.csv"),
    )
    return parser.parse_args()


def load_comparisons(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("comparisons", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Expected comparisons list in {path}")
    return rows


def matched_rows(rows: list[dict[str, Any]], hardware: str, max_steps: int, eval_batches: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("hardware_label") == hardware
        and int(row.get("max_steps", -1)) == max_steps
        and int(row.get("eval_max_batches", -1)) == eval_batches
        and row.get("policy") in POLICY_TO_ACTION
    ]


def aggregate_policy(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    return {
        "mean_eval_delta_risk": mean(float(row["eval_delta_percent"]) / 100.0 for row in rows),
        "max_instability": max(float(row.get("loss_spike_count", 0)) + float(row.get("nan_or_inf_count", 0)) for row in rows),
        "mean_memory_delta_gib": mean(
            float(row["policy_peak_memory_gib"]) - float(row["bf16_peak_memory_gib"]) for row in rows
        ),
        "mean_throughput_delta_pct": mean(float(row["tokens_per_sec_delta_percent"]) for row in rows),
        "mean_policy_eval_loss": mean(float(row["policy_eval_loss"]) for row in rows),
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    comparisons = matched_rows(
        load_comparisons(args.h8_summary),
        hardware=args.hardware_label,
        max_steps=args.max_steps,
        eval_batches=args.eval_max_batches,
    )
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        by_policy[str(row["policy"])].append(row)
    missing = sorted(set(POLICY_TO_ACTION) - set(by_policy))
    if missing:
        raise SystemExit(f"Missing required H8 policies for action-table generation: {missing}")

    aggregates = {policy: aggregate_policy(rows) for policy, rows in by_policy.items()}
    qlora_risk = aggregates["qlora_nf4"]["mean_eval_delta_risk"]

    action_rows: list[dict[str, str]] = []
    for policy in ("qlora_nf4", "h8_rescue"):
        candidate_action, backend, modules = POLICY_TO_ACTION[policy]
        agg = aggregates[policy]
        quality_recovery = max(0.0, qlora_risk - agg["mean_eval_delta_risk"])
        action_rows.append(
            {
                "model_name": args.model_name,
                "group_name": "projection_storage",
                "module_names": modules,
                "candidate_action": candidate_action,
                "backend": backend,
                "hardware_label": args.hardware_label,
                "backend_feasible": "true",
                "predicted_quality_risk": f"{agg['mean_eval_delta_risk']:.8f}",
                "predicted_instability_risk": f"{agg['max_instability']:.0f}",
                "quality_recovery_vs_lowbit": f"{quality_recovery:.8f}",
                "memory_delta_gib_vs_bf16": f"{agg['mean_memory_delta_gib']:.6f}",
                "throughput_delta_pct_vs_bf16": f"{agg['mean_throughput_delta_pct']:.6f}",
                "source_artifact": str(args.h8_summary),
                "notes": f"Generated from {len(by_policy[policy])} matched H8 {policy} rows.",
            }
        )

    action_rows.append(
        {
            "model_name": args.model_name,
            "group_name": "norm_logits",
            "module_names": "already non-quantized paths",
            "candidate_action": "backend_default",
            "backend": "bitsandbytes",
            "hardware_label": args.hardware_label,
            "backend_feasible": "true",
            "predicted_quality_risk": "0.00000000",
            "predicted_instability_risk": "0",
            "quality_recovery_vs_lowbit": "0.00000000",
            "memory_delta_gib_vs_bf16": "0.000000",
            "throughput_delta_pct_vs_bf16": "0.000000",
            "source_artifact": str(args.h8_summary),
            "notes": "H8 feasibility probes found norm/logit rescue mostly no-op under QLoRA/NF4.",
        }
    )
    return action_rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

