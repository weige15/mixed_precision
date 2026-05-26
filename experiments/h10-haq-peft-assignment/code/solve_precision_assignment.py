#!/usr/bin/env python3
"""Solve a small backend-aware PEFT precision assignment table.

The input is a CSV with one row per candidate `(group_name, candidate_action)`.
The solver chooses exactly one feasible row per group and searches grouped
combinations exhaustively. This is intentionally small and auditable: H10 is
about making the HAQ-for-PEFT abstraction executable before adding a heavier
optimizer.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "group_name",
    "candidate_action",
    "backend_feasible",
    "predicted_quality_risk",
    "predicted_instability_risk",
    "memory_delta_gib_vs_bf16",
    "throughput_delta_pct_vs_bf16",
}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "supported", "feasible"}


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return float(raw)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            parsed["backend_feasible"] = parse_bool(row["backend_feasible"])
            parsed["predicted_quality_risk"] = parse_float(row, "predicted_quality_risk")
            parsed["predicted_instability_risk"] = parse_float(row, "predicted_instability_risk")
            parsed["quality_recovery_vs_lowbit"] = parse_float(row, "quality_recovery_vs_lowbit")
            parsed["memory_delta_gib_vs_bf16"] = parse_float(row, "memory_delta_gib_vs_bf16")
            parsed["throughput_delta_pct_vs_bf16"] = parse_float(row, "throughput_delta_pct_vs_bf16")
            rows.append(parsed)
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_name"]), []).append(row)
    return grouped


def aggregate(policy: tuple[dict[str, Any], ...], alpha: float) -> dict[str, float]:
    quality_risk = sum(row["predicted_quality_risk"] for row in policy)
    instability_risk = max(row["predicted_instability_risk"] for row in policy)
    quality_recovery = sum(row["quality_recovery_vs_lowbit"] for row in policy)
    memory_delta = sum(row["memory_delta_gib_vs_bf16"] for row in policy)
    throughput_delta = sum(row["throughput_delta_pct_vs_bf16"] for row in policy)
    objective = quality_risk - alpha * quality_recovery + 0.001 * max(throughput_delta, 0.0)
    return {
        "objective": objective,
        "predicted_quality_risk": quality_risk,
        "predicted_instability_risk": instability_risk,
        "quality_recovery_vs_lowbit": quality_recovery,
        "memory_delta_gib_vs_bf16": memory_delta,
        "throughput_delta_pct_vs_bf16": throughput_delta,
    }


def solve(
    rows: list[dict[str, Any]],
    epsilon: float,
    tau: float,
    max_memory_delta_gib: float,
    alpha: float,
) -> tuple[tuple[dict[str, Any], ...], dict[str, float], list[dict[str, Any]]]:
    feasible_rows = [row for row in rows if row["backend_feasible"]]
    grouped = group_rows(feasible_rows)
    if not grouped:
        raise SystemExit("No feasible rows found.")

    best_policy: tuple[dict[str, Any], ...] | None = None
    best_metrics: dict[str, float] | None = None
    trace: list[dict[str, Any]] = []

    group_names = sorted(grouped)
    for combo in itertools.product(*(grouped[name] for name in group_names)):
        metrics = aggregate(combo, alpha=alpha)
        accepted = (
            metrics["predicted_quality_risk"] <= epsilon
            and metrics["predicted_instability_risk"] <= tau
            and metrics["memory_delta_gib_vs_bf16"] <= max_memory_delta_gib
        )
        trace.append(
            {
                "actions": {row["group_name"]: row["candidate_action"] for row in combo},
                "accepted": accepted,
                **metrics,
            }
        )
        if not accepted:
            continue
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_policy = combo
            best_metrics = metrics

    if best_policy is None or best_metrics is None:
        best_rejected = min(trace, key=lambda item: item["objective"], default=None)
        message = "No policy satisfied the constraints."
        if best_rejected:
            message += f" Best rejected candidate: {best_rejected}"
        raise SystemExit(message)

    return best_policy, best_metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument(
        "--max-memory-delta-gib",
        type=float,
        default=math.inf,
        help="Maximum allowed memory delta versus bf16. Use a negative value to require memory savings.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Reward weight for quality recovery versus blanket low-bit baseline.",
    )
    args = parser.parse_args()

    rows = load_rows(args.action_table)
    policy, metrics, trace = solve(
        rows,
        epsilon=args.epsilon,
        tau=args.tau,
        max_memory_delta_gib=args.max_memory_delta_gib,
        alpha=args.alpha,
    )

    output = {
        "constraints": {
            "epsilon": args.epsilon,
            "tau": args.tau,
            "max_memory_delta_gib": args.max_memory_delta_gib,
            "alpha": args.alpha,
        },
        "metrics": metrics,
        "selected_actions": [
            {
                key: row.get(key, "")
                for key in (
                    "model_name",
                    "group_name",
                    "candidate_action",
                    "backend",
                    "hardware_label",
                    "module_names",
                    "source_artifact",
                    "notes",
                )
            }
            for row in policy
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2) + "\n")


if __name__ == "__main__":
    main()

