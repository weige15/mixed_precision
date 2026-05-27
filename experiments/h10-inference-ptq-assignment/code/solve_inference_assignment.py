#!/usr/bin/env python3
"""Select backend-feasible H10 inference PTQ policies from an action table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ACTION_TABLE = Path("experiments/h10-inference-ptq-assignment/results/action_table.csv")
DEFAULT_OUTPUT = Path("experiments/h10-inference-ptq-assignment/results/selected_policy.json")
DEFAULT_TRACE = Path("experiments/h10-inference-ptq-assignment/results/solver_trace.json")


REQUIRED_COLUMNS = {
    "model_name",
    "group_name",
    "workload_name",
    "candidate_action",
    "backend",
    "hardware_label",
    "backend_feasible",
    "predicted_quality_risk",
    "latency_delta_pct_vs_bf16",
    "output_tokens_per_sec_delta_pct_vs_bf16",
    "memory_delta_gib_vs_bf16",
    "kv_cache_memory_delta_gib_vs_bf16",
    "source_artifact",
    "failure_reason",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-table", type=Path, default=DEFAULT_ACTION_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--quality-epsilon", type=float, default=0.01)
    parser.add_argument("--baseline-policy", default="bf16_default")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "supported", "feasible"}


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            row["backend_feasible"] = parse_bool(raw.get("backend_feasible", ""))
            for key in (
                "predicted_quality_risk",
                "prompt_nll_delta_pct_vs_bf16",
                "latency_delta_pct_vs_bf16",
                "prefill_latency_delta_pct_vs_bf16",
                "output_tokens_per_sec_delta_pct_vs_bf16",
                "decode_tokens_per_sec_delta_pct_vs_bf16",
                "total_tokens_per_sec_delta_pct_vs_bf16",
                "memory_delta_gib_vs_bf16",
                "memory_delta_pct_vs_bf16",
                "kv_cache_memory_delta_gib_vs_bf16",
            ):
                row[key] = parse_float(raw.get(key))
            rows.append(row)
    return rows


def value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    parsed = row.get(key)
    return float(parsed) if parsed is not None else default


def has_deployment_improvement(row: dict[str, Any]) -> bool:
    return any(
        [
            value(row, "latency_delta_pct_vs_bf16") < 0.0,
            value(row, "prefill_latency_delta_pct_vs_bf16") < 0.0,
            value(row, "output_tokens_per_sec_delta_pct_vs_bf16") > 0.0,
            value(row, "decode_tokens_per_sec_delta_pct_vs_bf16") > 0.0,
            value(row, "total_tokens_per_sec_delta_pct_vs_bf16") > 0.0,
            value(row, "memory_delta_gib_vs_bf16") < 0.0,
            value(row, "kv_cache_memory_delta_gib_vs_bf16") < 0.0,
        ]
    )


def rejection_reasons(row: dict[str, Any], quality_epsilon: float, baseline_policy: str) -> list[str]:
    reasons = []
    if not row["backend_feasible"]:
        reasons.append("backend_infeasible")
    if row.get("candidate_action") == baseline_policy:
        reasons.append("baseline_reference")
    quality = row.get("predicted_quality_risk")
    if quality is None:
        reasons.append("missing_quality")
    elif quality > quality_epsilon:
        reasons.append("quality_gate_failed")
    if not has_deployment_improvement(row):
        reasons.append("no_deployment_metric_improves")
    return reasons


def quality_passed(row: dict[str, Any], quality_epsilon: float) -> bool:
    quality = row.get("predicted_quality_risk")
    return quality is not None and quality <= quality_epsilon


def comparable_metrics(row: dict[str, Any]) -> list[tuple[float, bool]]:
    return [
        (value(row, "predicted_quality_risk"), False),
        (value(row, "latency_delta_pct_vs_bf16"), False),
        (value(row, "memory_delta_gib_vs_bf16"), False),
        (value(row, "output_tokens_per_sec_delta_pct_vs_bf16"), True),
    ]


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    comparisons = []
    for (a_value, higher_is_better), (b_value, _) in zip(comparable_metrics(a), comparable_metrics(b), strict=True):
        if higher_is_better:
            comparisons.append((a_value >= b_value, a_value > b_value))
        else:
            comparisons.append((a_value <= b_value, a_value < b_value))
    return all(ok for ok, _ in comparisons) and any(strict for _, strict in comparisons)


def mark_non_dominated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row.get("workload_name", ""))].append(row)

    marked = []
    for workload_rows in by_workload.values():
        for row in workload_rows:
            row = dict(row)
            row["pareto_non_dominated"] = not any(
                other is not row
                and other.get("candidate_action") != row.get("candidate_action")
                and dominates(other, row)
                for other in workload_rows
            )
            marked.append(row)
    return marked


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("group_name", "")),
        str(row.get("workload_name", "")),
        value(row, "predicted_quality_risk"),
        value(row, "latency_delta_pct_vs_bf16"),
        -value(row, "output_tokens_per_sec_delta_pct_vs_bf16"),
        value(row, "memory_delta_gib_vs_bf16"),
        str(row.get("candidate_action", "")),
    )


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model_name",
        "group_name",
        "workload_name",
        "candidate_action",
        "backend",
        "hardware_label",
        "predicted_quality_risk",
        "prompt_nll_delta_pct_vs_bf16",
        "latency_delta_pct_vs_bf16",
        "output_tokens_per_sec_delta_pct_vs_bf16",
        "total_tokens_per_sec_delta_pct_vs_bf16",
        "memory_delta_gib_vs_bf16",
        "kv_cache_memory_delta_gib_vs_bf16",
        "source_artifact",
        "notes",
    ]
    return {key: row.get(key) for key in keys}


def solve(rows: list[dict[str, Any]], quality_epsilon: float, baseline_policy: str) -> dict[str, Any]:
    trace = []
    candidates = []
    for row in rows:
        reasons = rejection_reasons(row, quality_epsilon, baseline_policy)
        accepted = not reasons
        trace.append(
            {
                "candidate_action": row.get("candidate_action"),
                "workload_name": row.get("workload_name"),
                "backend_feasible": row.get("backend_feasible"),
                "accepted_before_pareto": accepted,
                "rejection_reasons": reasons,
                "predicted_quality_risk": row.get("predicted_quality_risk"),
                "latency_delta_pct_vs_bf16": row.get("latency_delta_pct_vs_bf16"),
                "output_tokens_per_sec_delta_pct_vs_bf16": row.get("output_tokens_per_sec_delta_pct_vs_bf16"),
                "memory_delta_gib_vs_bf16": row.get("memory_delta_gib_vs_bf16"),
                "failure_reason": row.get("failure_reason"),
            }
        )
        if accepted:
            candidates.append(row)

    marked_candidates = mark_non_dominated(candidates)
    selected = sorted([row for row in marked_candidates if row["pareto_non_dominated"]], key=sort_key)
    rejected_after_pareto = [
        public_row(row) for row in sorted(marked_candidates, key=sort_key) if not row["pareto_non_dominated"]
    ]
    return {
        "constraints": {
            "quality_epsilon": quality_epsilon,
            "baseline_policy": baseline_policy,
            "requires_backend_feasible": True,
            "requires_one_deployment_metric_improves": True,
        },
        "n_input_rows": len(rows),
        "n_quality_passed_feasible_rows": sum(
            1 for row in rows if row["backend_feasible"] and quality_passed(row, quality_epsilon)
        ),
        "n_accepted_before_pareto": len(candidates),
        "n_selected_pareto_rows": len(selected),
        "selected_pareto_rows": [public_row(row) for row in selected],
        "accepted_but_dominated_rows": rejected_after_pareto,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.action_table)
    result = solve(rows, args.quality_epsilon, args.baseline_policy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_payload = {key: value for key, value in result.items() if key != "trace"}
    args.output.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(result["trace"], indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if args.trace_output:
        print(f"wrote {args.trace_output}")
    print(f"selected Pareto rows: {result['n_selected_pareto_rows']}")


if __name__ == "__main__":
    main()
