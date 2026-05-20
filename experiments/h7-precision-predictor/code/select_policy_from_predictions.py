#!/usr/bin/env python
"""Select a low-risk module policy from H7 predictor outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="experiments/h7-precision-predictor/results/predictions.csv")
    parser.add_argument("--dataset", default="experiments/h7-precision-predictor/results/precision_dataset.csv")
    parser.add_argument("--output", default="experiments/h7-precision-predictor/results/selected_policy.json")
    parser.add_argument("--modules-output", default="experiments/h7-precision-predictor/results/selected_modules.txt")
    parser.add_argument("--split", default="cross_scale_0p5b_to_7b")
    parser.add_argument("--model-size-min", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-predicted-risk", type=float, default=None)
    parser.add_argument("--roles", nargs="+", default=["mlp_projection", "attention_projection"])
    parser.add_argument("--leaves", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    pred = pred[pred["split"] == args.split].copy()
    if args.model_size_min is not None:
        pred = pred[pd.to_numeric(pred["model_size_b"], errors="coerce") >= args.model_size_min]
    if args.roles:
        pred = pred[pred["module_role"].isin(args.roles)]
    if args.leaves:
        pred = pred[pred["module_leaf"].isin(args.leaves)]
    if args.max_predicted_risk is not None:
        pred = pred[pd.to_numeric(pred["predicted_risk"], errors="coerce") <= args.max_predicted_risk]
    if pred.empty:
        raise SystemExit("No prediction rows left after filtering.")

    grouped = (
        pred.groupby(["module_name", "module_short", "module_role", "module_leaf"], as_index=False)
        .agg(
            predicted_risk_mean=("predicted_risk", "mean"),
            predicted_risk_max=("predicted_risk", "max"),
            predicted_safe_probability_mean=("predicted_safe_probability", "mean"),
            observed_abs_delta_mean=("abs_perturbation_delta", "mean"),
            observed_safe_rate=("safe_label", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["predicted_risk_mean", "predicted_risk_max", "module_name"])
    )
    selected = grouped.head(args.top_k).copy()
    payload = {
        "source_predictions": args.predictions,
        "split": args.split,
        "top_k": args.top_k,
        "filters": {
            "model_size_min": args.model_size_min,
            "roles": args.roles,
            "leaves": args.leaves,
            "max_predicted_risk": args.max_predicted_risk,
        },
        "modules": selected.to_dict(orient="records"),
        "module_names": selected["module_name"].tolist(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    modules_output = Path(args.modules_output)
    modules_output.write_text("\n".join(payload["module_names"]) + "\n", encoding="utf-8")
    print(f"Wrote policy to {output}")
    print(f"Wrote module list to {modules_output}")
    print("Selected modules:")
    for module in payload["module_names"]:
        print(module)


if __name__ == "__main__":
    main()

