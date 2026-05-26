#!/usr/bin/env python
"""Aggregate seed-level H7 precision rows into module-level H10 labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


GROUP_COLUMNS = [
    "model_name",
    "model_size_b",
    "dataset_name",
    "module_name",
    "module_short",
    "module_role",
    "module_class",
    "module_leaf",
    "layer_idx",
    "num_layers_hint",
    "normalized_depth",
    "candidate_format",
]

SIGNAL_COLUMNS = [
    "activation_outlier_score",
    "input_outlier_score",
    "output_outlier_score",
    "int8_rel_mse",
    "input_int8_rel_mse_mean",
    "output_int8_rel_mse_mean",
    "output_int4_rel_mse_mean",
    "output_int8_saturation_mean",
    "output_int4_saturation_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="experiments/h7-precision-predictor/results/precision_dataset_with_llama31_8b.csv",
    )
    parser.add_argument(
        "--output",
        default="experiments/h10-peft-precision-risk/results/seed_aggregated_precision_dataset.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="experiments/h10-peft-precision-risk/results/seed_aggregated_precision_summary.json",
    )
    parser.add_argument("--safe-threshold", type=float, default=0.005)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_labeled_rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "model_size_b",
        "seed",
        "layer_idx",
        "num_layers_hint",
        "normalized_depth",
        "baseline_loss_mean",
        "perturbed_loss_mean",
        "perturbation_delta",
        "abs_perturbation_delta",
        "max_batch_loss_delta_abs",
        "has_perturbation",
        "safe_label",
        *SIGNAL_COLUMNS,
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["has_perturbation"] == 1].copy()
    df = df.dropna(subset=["abs_perturbation_delta", "safe_label"])
    df["safe_label"] = df["safe_label"].astype(int)
    return df


def aggregate(df: pd.DataFrame, safe_threshold: float) -> pd.DataFrame:
    aggregations: dict[str, Any] = {
        "seed": [lambda values: ",".join(str(int(v)) for v in sorted(values.dropna().unique())), "nunique"],
        "baseline_loss_mean": "mean",
        "perturbed_loss_mean": "mean",
        "perturbation_delta": ["mean", "min", "max"],
        "abs_perturbation_delta": ["mean", "max", "std"],
        "max_batch_loss_delta_abs": "max",
        "safe_label": ["mean", "sum", "count", "min"],
    }
    for column in SIGNAL_COLUMNS:
        if column in df.columns:
            aggregations[column] = ["mean", "max"]

    grouped = df.groupby(GROUP_COLUMNS, dropna=False).agg(aggregations)
    grouped.columns = ["_".join(part for part in column if part) for column in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    rename = {
        "seed_<lambda_0>": "seeds",
        "seed_nunique": "n_labeled_seeds",
        "baseline_loss_mean_mean": "baseline_loss_mean",
        "perturbed_loss_mean_mean": "perturbed_loss_mean",
        "perturbation_delta_mean": "perturbation_delta_mean",
        "perturbation_delta_min": "perturbation_delta_min",
        "perturbation_delta_max": "perturbation_delta_max",
        "abs_perturbation_delta_mean": "mean_abs_delta",
        "abs_perturbation_delta_max": "max_abs_delta",
        "abs_perturbation_delta_std": "std_abs_delta",
        "max_batch_loss_delta_abs_max": "max_batch_loss_delta_abs",
        "safe_label_mean": "safe_rate",
        "safe_label_sum": "safe_seed_count",
        "safe_label_count": "label_count",
        "safe_label_min": "safe_all_seeds_int",
    }
    grouped = grouped.rename(columns=rename)
    grouped["safe_threshold"] = safe_threshold
    grouped["safe_all_seeds"] = (grouped["max_abs_delta"] <= safe_threshold).astype(int)
    grouped["safe_majority"] = (grouped["safe_rate"] >= 0.5).astype(int)
    grouped["unsafe_all_seeds"] = (grouped["safe_rate"] <= 0.0).astype(int)
    grouped["unsafe_any_seed"] = (grouped["safe_all_seeds"] == 0).astype(int)
    grouped["conservative_rescue_target"] = grouped["unsafe_any_seed"]
    grouped["activation_outlier_score_agg"] = grouped.get(
        "activation_outlier_score_max",
        grouped.get("activation_outlier_score_mean", 0.0),
    )
    grouped["int8_rel_mse_agg"] = grouped.get("int8_rel_mse_max", grouped.get("int8_rel_mse_mean", 0.0))
    return grouped.sort_values(["model_name", "module_name"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame, output: pd.DataFrame, source: str, safe_threshold: float) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_name, model_df in output.groupby("model_name"):
        models.append(
            {
                "model_name": model_name,
                "module_rows": int(len(model_df)),
                "projection_rows": int(model_df["module_role"].astype(str).str.contains("projection").sum()),
                "safe_all_seed_rows": int(model_df["safe_all_seeds"].sum()),
                "unsafe_any_seed_rows": int((model_df["safe_all_seeds"] == 0).sum()),
                "mean_labeled_seeds": finite_float(model_df["n_labeled_seeds"].mean()),
            }
        )
    return {
        "source": source,
        "safe_threshold": safe_threshold,
        "seed_level_labeled_rows": int(len(df)),
        "module_level_rows": int(len(output)),
        "models": models,
    }


def main() -> None:
    args = parse_args()
    df = read_labeled_rows(args.input)
    if df.empty:
        raise SystemExit("No perturbation-labeled rows found.")
    output = aggregate(df, args.safe_threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    summary = build_summary(df, output, args.input, args.safe_threshold)
    summary_path = Path(args.summary_output)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote aggregated dataset to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Seed rows: {len(df)}; module rows: {len(output)}")


if __name__ == "__main__":
    main()

