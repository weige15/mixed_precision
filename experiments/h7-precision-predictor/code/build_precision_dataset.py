#!/usr/bin/env python
"""Build a module-level precision-risk dataset from H6 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "model_name",
    "model_size_b",
    "dataset_name",
    "seed",
    "source_stage",
    "source_dir",
    "module_name",
    "module_short",
    "module_role",
    "module_class",
    "module_leaf",
    "layer_idx",
    "num_layers_hint",
    "normalized_depth",
    "candidate_format",
    "stage1_assignment",
    "stage1_reason",
    "observations",
    "activation_outlier_score",
    "input_outlier_score",
    "output_outlier_score",
    "input_outlier_score_mean",
    "input_outlier_score_max",
    "output_outlier_score_mean",
    "output_outlier_score_max",
    "int8_rel_mse",
    "input_int8_rel_mse_mean",
    "output_int8_rel_mse_mean",
    "output_int4_rel_mse_mean",
    "output_int8_saturation_mean",
    "output_int4_saturation_mean",
    "finite_fraction_min",
    "baseline_loss_mean",
    "perturbed_loss_mean",
    "perturbation_delta",
    "abs_perturbation_delta",
    "max_batch_loss_delta_abs",
    "has_perturbation",
    "safe_label",
    "safe_threshold",
    "label_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h6-results", default="experiments/h6-adaptive-precision-assignment/results")
    parser.add_argument("--output", default="experiments/h7-precision-predictor/results/precision_dataset.csv")
    parser.add_argument("--safe-threshold", type=float, default=0.005)
    parser.add_argument(
        "--candidate-format",
        default="fake_int8_output",
        help="Candidate format represented by the perturbation probes.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.12g}"
        return ""
    return value


def model_size_b(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if not match:
        return None
    return float(match.group(1))


def layer_idx(module_name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    if not match:
        return None
    return int(match.group(1))


def module_leaf(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def module_short(module_name: str) -> str:
    marker = "base_model.model.model."
    return module_name.split(marker, 1)[-1] if marker in module_name else module_name


def infer_num_layers(model_name: str, modules: list[str]) -> int | None:
    indices = [idx for idx in (layer_idx(name) for name in modules) if idx is not None]
    if indices:
        return max(indices) + 1
    if "0.5B" in model_name:
        return 24
    if "7B" in model_name:
        return 28
    if "1.5B" in model_name:
        return 28
    return None


def normalized_depth(idx: int | None, num_layers: int | None) -> float | None:
    if idx is None or not num_layers or num_layers <= 1:
        return None
    return idx / float(num_layers - 1)


def signal_from_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    signals = dict(row.get("signals") or row)
    signals.setdefault("module", row.get("module"))
    signals.setdefault("role", row.get("role"))
    signals.setdefault("class", row.get("class"))
    return signals


def stage_name(path: Path) -> str:
    name = path.parent.name
    if name.startswith("h6_4_qwen7b_transfer"):
        return "h6_4_7b_transfer"
    if name.startswith("calibration_bf16"):
        return "h6_0p5b_calibration"
    if name.startswith("perturbation_bf16"):
        return "h6_0p5b_perturbation"
    return name


def make_base_row(
    payload: dict[str, Any],
    signals: dict[str, Any],
    source_stage: str,
    source_dir: str,
    candidate_format: str,
    safe_threshold: float,
    num_layers: int | None,
) -> dict[str, Any]:
    module = signals.get("module") or ""
    idx = layer_idx(module)
    input_outlier = finite_number(signals.get("input_outlier_score_max"))
    output_outlier = finite_number(signals.get("output_outlier_score_max"))
    input_mse = finite_number(signals.get("input_int8_rel_mse_mean"))
    output_mse = finite_number(signals.get("output_int8_rel_mse_mean"))
    return {
        "model_name": payload.get("model_name", ""),
        "model_size_b": model_size_b(str(payload.get("model_name", ""))),
        "dataset_name": payload.get("dataset_name", ""),
        "seed": payload.get("seed", ""),
        "source_stage": source_stage,
        "source_dir": source_dir,
        "module_name": module,
        "module_short": module_short(module),
        "module_role": signals.get("role", ""),
        "module_class": signals.get("class", ""),
        "module_leaf": module_leaf(module),
        "layer_idx": idx,
        "num_layers_hint": num_layers,
        "normalized_depth": normalized_depth(idx, num_layers),
        "candidate_format": candidate_format,
        "stage1_assignment": "",
        "stage1_reason": "",
        "observations": signals.get("observations", ""),
        "activation_outlier_score": max(input_outlier or 0.0, output_outlier or 0.0),
        "input_outlier_score": input_outlier,
        "output_outlier_score": output_outlier,
        "input_outlier_score_mean": signals.get("input_outlier_score_mean", ""),
        "input_outlier_score_max": signals.get("input_outlier_score_max", ""),
        "output_outlier_score_mean": signals.get("output_outlier_score_mean", ""),
        "output_outlier_score_max": signals.get("output_outlier_score_max", ""),
        "int8_rel_mse": max(input_mse or 0.0, output_mse or 0.0),
        "input_int8_rel_mse_mean": signals.get("input_int8_rel_mse_mean", ""),
        "output_int8_rel_mse_mean": signals.get("output_int8_rel_mse_mean", ""),
        "output_int4_rel_mse_mean": signals.get("output_int4_rel_mse_mean", ""),
        "output_int8_saturation_mean": signals.get("output_int8_saturation_mean", ""),
        "output_int4_saturation_mean": signals.get("output_int4_saturation_mean", ""),
        "finite_fraction_min": signals.get("finite_fraction_min", ""),
        "baseline_loss_mean": "",
        "perturbed_loss_mean": "",
        "perturbation_delta": "",
        "abs_perturbation_delta": "",
        "max_batch_loss_delta_abs": "",
        "has_perturbation": 0,
        "safe_label": "",
        "safe_threshold": safe_threshold,
        "label_source": "",
    }


def load_calibration_rows(results_dir: Path, candidate_format: str, safe_threshold: float) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/stability_signals.json")):
        payload = load_json(path)
        modules = [row.get("module", "") for row in payload.get("modules", [])]
        num_layers = infer_num_layers(str(payload.get("model_name", "")), modules)
        source_stage = stage_name(path)
        for module_row in payload.get("modules", []):
            signals = signal_from_policy_row(module_row)
            row = make_base_row(
                payload=payload,
                signals=signals,
                source_stage=source_stage,
                source_dir=path.parent.name,
                candidate_format=candidate_format,
                safe_threshold=safe_threshold,
                num_layers=num_layers,
            )
            key = (str(row["model_name"]), int(row["seed"]), str(row["module_name"]))
            rows[key] = row

    for path in sorted(results_dir.glob("*/policy_trace.json")):
        payload_list = load_json(path)
        stability_path = path.parent / "stability_signals.json"
        payload = load_json(stability_path) if stability_path.exists() else {}
        modules = [row.get("module", "") for row in payload_list]
        num_layers = infer_num_layers(str(payload.get("model_name", "")), modules)
        for policy_row in payload_list:
            signals = signal_from_policy_row(policy_row)
            row = make_base_row(
                payload=payload,
                signals=signals,
                source_stage=stage_name(path),
                source_dir=path.parent.name,
                candidate_format=candidate_format,
                safe_threshold=safe_threshold,
                num_layers=num_layers,
            )
            row["stage1_assignment"] = policy_row.get("assigned_precision", "")
            row["stage1_reason"] = policy_row.get("reason", "")
            key = (str(row["model_name"]), int(row["seed"]), str(row["module_name"]))
            prior = rows.get(key)
            if prior:
                prior.update({"stage1_assignment": row["stage1_assignment"], "stage1_reason": row["stage1_reason"]})
            else:
                rows[key] = row
    return rows


def merge_perturbations(
    rows: dict[tuple[str, int, str], dict[str, Any]],
    results_dir: Path,
    candidate_format: str,
    safe_threshold: float,
) -> None:
    for path in sorted(results_dir.glob("*/perturbation_results.json")):
        payload = load_json(path)
        modules = [row.get("module", "") for row in payload.get("results", [])]
        num_layers = infer_num_layers(str(payload.get("model_name", "")), modules)
        for result in payload.get("results", []):
            if int(result.get("bits", 0) or 0) != 8:
                continue
            signals = dict(result.get("stage1_signals") or {})
            signals.setdefault("module", result.get("module", ""))
            signals.setdefault("role", result.get("role", ""))
            signals.setdefault("class", result.get("class", ""))
            key = (str(payload.get("model_name", "")), int(payload.get("seed")), str(result.get("module", "")))
            row = rows.get(key)
            if not row:
                row = make_base_row(
                    payload=payload,
                    signals=signals,
                    source_stage=stage_name(path),
                    source_dir=path.parent.name,
                    candidate_format=candidate_format,
                    safe_threshold=safe_threshold,
                    num_layers=num_layers,
                )
                rows[key] = row
            row["source_stage"] = stage_name(path)
            row["source_dir"] = path.parent.name
            row["stage1_assignment"] = result.get("stage1_assignment", row.get("stage1_assignment", ""))
            row["stage1_reason"] = result.get("stage1_reason", row.get("stage1_reason", ""))
            row["baseline_loss_mean"] = result.get("baseline_loss_mean", payload.get("baseline_loss_mean", ""))
            row["perturbed_loss_mean"] = result.get("perturbed_loss_mean", "")
            row["perturbation_delta"] = result.get("loss_delta", "")
            row["abs_perturbation_delta"] = result.get("loss_delta_abs", "")
            row["max_batch_loss_delta_abs"] = result.get("max_batch_loss_delta_abs", "")
            row["has_perturbation"] = 1
            delta_abs = finite_number(result.get("loss_delta_abs"))
            if delta_abs is not None:
                row["safe_label"] = int(delta_abs <= safe_threshold)
                row["label_source"] = "perturbation_abs_threshold"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.h6_results)
    rows = load_calibration_rows(results_dir, args.candidate_format, args.safe_threshold)
    merge_perturbations(rows, results_dir, args.candidate_format, args.safe_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: (str(r["model_name"]), int(r["seed"]), str(r["module_name"]))):
            writer.writerow({key: fmt(row.get(key, "")) for key in FIELDNAMES})

    labeled = sum(1 for row in rows.values() if row.get("has_perturbation") == 1)
    safe = sum(1 for row in rows.values() if row.get("safe_label") == 1)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Labeled perturbation rows: {labeled}; safe rows at threshold {args.safe_threshold:g}: {safe}")


if __name__ == "__main__":
    main()
