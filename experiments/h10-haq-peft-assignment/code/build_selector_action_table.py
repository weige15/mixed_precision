#!/usr/bin/env python3
"""Build a selector-aware H10 action table.

This is a planning bridge between the H10 risk-selector screen and the H8
backend-cost measurements. It does not create new empirical evidence. It asks:
if each selector proposed a top-k bf16 rescue set, and if top-k rescue costs
match the measured H8 top-4 rescue cost, which selector would the assignment
solver choose under the same memory/quality constraints?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-action-table",
        type=Path,
        default=Path("experiments/h10-haq-peft-assignment/results/action_table.csv"),
    )
    parser.add_argument(
        "--selector-evaluation",
        type=Path,
        default=Path("experiments/h10-peft-precision-risk/results/rescue_selector_evaluation_llama31_8b.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/h10-haq-peft-assignment/results/selector_action_table.csv"),
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Include the target perturbation upper-bound selector as a candidate action.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def read_base_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_action(rows: list[dict[str, str]], action: str) -> dict[str, str]:
    for row in rows:
        if row.get("candidate_action") == action:
            return row
    raise SystemExit(f"Could not find action {action!r} in base table")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    base_rows = read_base_rows(args.base_action_table)
    qlora = find_action(base_rows, "blanket_qlora_nf4")
    rescue = find_action(base_rows, "qlora_nf4_bf16_projection_rescue")
    selector_data = json.loads(args.selector_evaluation.read_text())

    qlora_risk = as_float(qlora, "predicted_quality_risk")
    measured_recovery = max(0.0, qlora_risk - as_float(rescue, "predicted_quality_risk"))
    rescue_memory_cost = as_float(rescue, "memory_delta_gib_vs_bf16") - as_float(qlora, "memory_delta_gib_vs_bf16")
    rescue_throughput_delta = as_float(rescue, "throughput_delta_pct_vs_bf16") - as_float(
        qlora, "throughput_delta_pct_vs_bf16"
    )

    rows: list[dict[str, str]] = []
    rows.append(qlora)

    for selector in selector_data.get("selectors", []):
        name = str(selector.get("selector", "unknown_selector"))
        is_oracle = name == "oracle_perturbation_upper_bound"
        if is_oracle and not args.include_oracle:
            continue
        recall = float(selector.get("unsafe_recall_at_k") or 0.0)
        expected_recovery = measured_recovery * recall
        module_names = ";".join(selector.get("module_names", []))
        action_name = f"qlora_nf4_bf16_rescue_{slugify(name)}"
        rows.append(
            {
                "model_name": selector_data.get("model_name", qlora.get("model_name", "")),
                "group_name": "projection_storage",
                "module_names": module_names,
                "candidate_action": action_name,
                "backend": rescue["backend"],
                "hardware_label": qlora["hardware_label"],
                "backend_feasible": "true",
                "predicted_quality_risk": f"{max(0.0, qlora_risk - expected_recovery):.8f}",
                "predicted_instability_risk": "0",
                "quality_recovery_vs_lowbit": f"{expected_recovery:.8f}",
                "memory_delta_gib_vs_bf16": f"{as_float(qlora, 'memory_delta_gib_vs_bf16') + rescue_memory_cost:.6f}",
                "throughput_delta_pct_vs_bf16": f"{as_float(qlora, 'throughput_delta_pct_vs_bf16') + rescue_throughput_delta:.6f}",
                "source_artifact": str(args.selector_evaluation),
                "notes": (
                    f"Planning row from selector {name}; expected recovery scales measured H8 top-4 "
                    f"recovery by unsafe_recall_at_k={recall:.3f}."
                ),
            }
        )

    for row in base_rows:
        if row.get("group_name") != "projection_storage":
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

