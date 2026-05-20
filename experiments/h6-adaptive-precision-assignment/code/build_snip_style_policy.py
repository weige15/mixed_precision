#!/usr/bin/env python
"""Build SNIP-style budgeted fake-int8 policies from H6 artifacts.

The script does not train. It aggregates existing Stage 1 calibration signals
and optional Stage 2 perturbation loss deltas, ranks eligible modules by a
conservative risk score, and writes frozen module lists for policy-width
experiments with the existing ``h6_custom_int8`` runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_BUDGETS = [4, 8, 16, 24]
DEFAULT_ANCHOR_MODULES = [
    "base_model.model.model.layers.22.mlp.gate_proj",
    "base_model.model.model.layers.22.mlp.up_proj",
    "base_model.model.model.layers.23.mlp.gate_proj",
    "base_model.model.model.layers.23.mlp.up_proj",
]


@dataclass
class ModuleAggregate:
    module: str
    role: str = ""
    calibration_rows: list[dict[str, Any]] = field(default_factory=list)
    perturbation_abs_deltas: list[float] = field(default_factory=list)
    perturbation_signed_deltas: list[float] = field(default_factory=list)

    def add_calibration(self, row: dict[str, Any]) -> None:
        self.role = row.get("role") or self.role
        self.calibration_rows.append(row.get("signals", row))

    def add_perturbation(self, row: dict[str, Any]) -> None:
        delta = row.get("loss_delta")
        if isinstance(delta, (int, float)) and math.isfinite(delta):
            self.perturbation_signed_deltas.append(float(delta))
            self.perturbation_abs_deltas.append(abs(float(delta)))

    def metric_max(self, key: str) -> float:
        values = []
        for row in self.calibration_rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        return max(values) if values else 0.0

    def metric_mean(self, key: str) -> float:
        values = []
        for row in self.calibration_rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        return sum(values) / len(values) if values else 0.0

    def finite_min(self) -> float:
        values = []
        for row in self.calibration_rows:
            value = row.get("finite_fraction_min")
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        return min(values) if values else 1.0

    def layer_index(self) -> int | None:
        match = re.search(r"\.layers\.(\d+)\.", self.module)
        return int(match.group(1)) if match else None

    def leaf_name(self) -> str:
        return self.module.rsplit(".", 1)[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/h6-adaptive-precision-assignment/results")
    parser.add_argument("--output-dir", default="experiments/h6-adaptive-precision-assignment/results/snip_style_policy")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--eligible-leaves",
        nargs="+",
        default=["gate_proj", "up_proj"],
        help="Projection leaf names eligible for demotion.",
    )
    parser.add_argument(
        "--include-attention",
        action="store_true",
        help="Allow attention projections. Off by default for the locked H6.1 protocol.",
    )
    parser.add_argument(
        "--max-perturbation-abs-delta",
        type=float,
        default=0.02,
        help="Exclude modules with observed mean absolute perturbation delta above this threshold.",
    )
    parser.add_argument(
        "--anchor-modules",
        nargs="*",
        default=DEFAULT_ANCHOR_MODULES,
        help="Already validated modules to include first. Default reproduces the current H6 narrow policy at k=4.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_create(modules: dict[str, ModuleAggregate], module: str) -> ModuleAggregate:
    if module not in modules:
        modules[module] = ModuleAggregate(module=module)
    return modules[module]


def load_calibration(results_dir: str, seeds: list[int]) -> dict[str, ModuleAggregate]:
    modules: dict[str, ModuleAggregate] = {}
    for seed in seeds:
        path = os.path.join(results_dir, f"calibration_bf16_seed{seed}", "policy_trace.json")
        if not os.path.exists(path):
            raise SystemExit(f"Missing calibration policy trace: {path}")
        for row in load_json(path):
            get_or_create(modules, row["module"]).add_calibration(row)
    return modules


def load_perturbations(results_dir: str, seeds: list[int], modules: dict[str, ModuleAggregate]) -> None:
    for seed in seeds:
        path = os.path.join(results_dir, f"perturbation_bf16_seed{seed}", "perturbation_results.json")
        if not os.path.exists(path):
            continue
        payload = load_json(path)
        for row in payload.get("results", []):
            if row.get("bits") != 8:
                continue
            get_or_create(modules, row["module"]).add_perturbation(row)


def eligible(module: ModuleAggregate, eligible_leaves: set[str], include_attention: bool) -> bool:
    if module.role == "attention_projection" and not include_attention:
        return False
    if module.role != "mlp_projection" and not (include_attention and module.role == "attention_projection"):
        return False
    return module.leaf_name() in eligible_leaves


def percentile_scales(rows: list[ModuleAggregate]) -> dict[str, float]:
    keys = [
        "input_outlier_score_max",
        "output_outlier_score_max",
        "input_int8_rel_mse_mean",
        "output_int8_rel_mse_mean",
        "output_int4_rel_mse_mean",
        "output_int8_saturation_mean",
    ]
    scales: dict[str, float] = {}
    for key in keys:
        values = sorted(max(row.metric_max(key), row.metric_mean(key)) for row in rows)
        values = [value for value in values if value > 0 and math.isfinite(value)]
        if not values:
            scales[key] = 1.0
            continue
        index = min(len(values) - 1, max(0, int(0.9 * (len(values) - 1))))
        scales[key] = max(values[index], 1e-12)
    return scales


def normalized(value: float, scale: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return min(max(value / max(scale, 1e-12), 0.0), 5.0)


def risk_row(
    module: ModuleAggregate,
    scales: dict[str, float],
    max_perturbation_abs_delta: float,
    anchor_modules: set[str],
) -> dict[str, Any]:
    outlier = max(module.metric_max("input_outlier_score_max"), module.metric_max("output_outlier_score_max"))
    int8_rel_mse = max(module.metric_mean("input_int8_rel_mse_mean"), module.metric_mean("output_int8_rel_mse_mean"))
    int4_rel_mse = module.metric_mean("output_int4_rel_mse_mean")
    saturation = module.metric_mean("output_int8_saturation_mean")
    finite_penalty = 0.0 if module.finite_min() >= 1.0 else 100.0
    perturbation_abs_mean = (
        sum(module.perturbation_abs_deltas) / len(module.perturbation_abs_deltas)
        if module.perturbation_abs_deltas
        else None
    )
    perturbation_abs_max = max(module.perturbation_abs_deltas) if module.perturbation_abs_deltas else None

    signal_risk = (
        0.35 * normalized(outlier, max(scales["input_outlier_score_max"], scales["output_outlier_score_max"]))
        + 0.30 * normalized(int8_rel_mse, max(scales["input_int8_rel_mse_mean"], scales["output_int8_rel_mse_mean"]))
        + 0.15 * normalized(int4_rel_mse, scales["output_int4_rel_mse_mean"])
        + 0.05 * normalized(saturation, scales["output_int8_saturation_mean"])
        + finite_penalty
    )
    if perturbation_abs_mean is None:
        perturbation_risk = 0.25
    else:
        perturbation_risk = 3.0 * normalized(perturbation_abs_mean, max_perturbation_abs_delta)

    risk = signal_risk + perturbation_risk
    excluded_reason = None
    if perturbation_abs_max is not None and perturbation_abs_max > max_perturbation_abs_delta:
        excluded_reason = "observed perturbation loss delta above safety threshold"
    if finite_penalty:
        excluded_reason = "non-finite activation observed"

    return {
        "module": module.module,
        "role": module.role,
        "leaf": module.leaf_name(),
        "layer": module.layer_index(),
        "anchor": module.module in anchor_modules,
        "risk_score": risk,
        "signal_risk": signal_risk,
        "perturbation_risk": perturbation_risk,
        "excluded_reason": excluded_reason,
        "metrics": {
            "outlier_score_max": outlier,
            "int8_rel_mse_mean_max_input_or_output": int8_rel_mse,
            "output_int4_rel_mse_mean": int4_rel_mse,
            "output_int8_saturation_mean": saturation,
            "finite_fraction_min": module.finite_min(),
            "perturbation_abs_delta_mean": perturbation_abs_mean,
            "perturbation_abs_delta_max": perturbation_abs_max,
            "perturbation_signed_deltas": module.perturbation_signed_deltas,
            "calibration_observation_seeds": len(module.calibration_rows),
            "perturbation_observation_seeds": len(module.perturbation_abs_deltas),
        },
    }


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    modules = load_calibration(args.results_dir, args.seeds)
    load_perturbations(args.results_dir, args.seeds, modules)

    candidates = [
        module
        for module in modules.values()
        if eligible(module, set(args.eligible_leaves), args.include_attention)
    ]
    if not candidates:
        raise SystemExit("No eligible modules found.")

    scales = percentile_scales(candidates)
    scored = [
        risk_row(module, scales, args.max_perturbation_abs_delta, set(args.anchor_modules))
        for module in candidates
    ]
    scored.sort(key=lambda row: (row["excluded_reason"] is not None, row["risk_score"], row["module"]))
    selectable = [row for row in scored if row["excluded_reason"] is None]
    selectable_by_module = {row["module"]: row for row in selectable}
    missing_anchors = [module for module in args.anchor_modules if module not in selectable_by_module]
    if missing_anchors:
        raise SystemExit("Anchor module(s) are not selectable:\n" + "\n".join(missing_anchors))
    anchor_rows = [selectable_by_module[module] for module in args.anchor_modules]
    expansion_rows = [row for row in selectable if row["module"] not in set(args.anchor_modules)]

    policies = {}
    for budget in args.budgets:
        if budget < len(anchor_rows):
            raise SystemExit(f"Budget {budget} is smaller than the {len(anchor_rows)} anchor modules.")
        selected = anchor_rows + expansion_rows[: budget - len(anchor_rows)]
        modules_path = os.path.join(args.output_dir, f"snip_style_k{budget}_modules.txt")
        policy_path = os.path.join(args.output_dir, f"snip_style_k{budget}_policy.json")
        with open(modules_path, "w", encoding="utf-8") as f:
            for row in selected:
                f.write(row["module"] + "\n")
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "policy_name": f"h6_snip_style_k{budget}",
                    "budget": budget,
                    "selected_modules": selected,
                    "module_list_path": modules_path,
                    "runner_args": ["--fake-int8-modules", *[row["module"] for row in selected]],
                },
                f,
                indent=2,
            )
        policies[f"k{budget}"] = {
            "module_count": len(selected),
            "module_list_path": modules_path,
            "policy_path": policy_path,
            "modules": [row["module"] for row in selected],
        }

    report = {
        "method": "SNIP-style conservative risk ranking over calibration and perturbation artifacts",
        "results_dir": args.results_dir,
        "seeds": args.seeds,
        "eligible_leaves": args.eligible_leaves,
        "include_attention": args.include_attention,
        "max_perturbation_abs_delta": args.max_perturbation_abs_delta,
        "anchor_modules": args.anchor_modules,
        "normalization_scales": scales,
        "candidate_count": len(candidates),
        "selectable_count": len(selectable),
        "excluded_count": len(scored) - len(selectable),
        "policies": policies,
        "ranked_modules": scored,
    }

    report_path = os.path.join(args.output_dir, "snip_style_policy_report.json")
    commands_path = os.path.join(args.output_dir, "run_commands_seed42.sh")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(commands_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for budget in args.budgets:
            modules_path = os.path.join(args.output_dir, f"snip_style_k{budget}_modules.txt")
            f.write(
                "python experiments/h1-selective-fp32-norms/code/run_lora_precision.py "
                "--precision-policy h6_custom_int8 "
                "--seed 42 --max-steps 500 --learning-rate 2e-4 --eval-every 100 "
                "--output-dir "
                + shell_quote(
                    f"experiments/h6-adaptive-precision-assignment/results/train_h6_snip_style_k{budget}_seed42_500"
                )
                + " --fake-int8-modules $(tr '\\n' ' ' < "
                + shell_quote(modules_path)
                + ")\n"
            )
    os.chmod(commands_path, 0o755)

    print(json.dumps({k: report[k] for k in report if k != "ranked_modules"}, indent=2))
    print(f"Saved SNIP-style policy report to {report_path}")
    print(f"Saved seed-42 command sketch to {commands_path}")


if __name__ == "__main__":
    main()
