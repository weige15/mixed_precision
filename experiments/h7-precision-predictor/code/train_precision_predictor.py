#!/usr/bin/env python
"""Train and evaluate a small precision-risk predictor from H7 dataset rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "model_size_b",
    "layer_idx",
    "normalized_depth",
    "log_activation_outlier_score",
    "log_int8_rel_mse",
    "log_output_int4_rel_mse",
    "log_output_int8_saturation_mean",
]

CATEGORICAL_FEATURES = [
    "module_role",
    "module_leaf",
    "candidate_format",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="experiments/h7-precision-predictor/results/precision_dataset.csv")
    parser.add_argument("--output", default="experiments/h7-precision-predictor/results/predictor_metrics.json")
    parser.add_argument("--predictions-output", default="experiments/h7-precision-predictor/results/predictions.csv")
    parser.add_argument(
        "--eval-mode",
        choices=["leave_one_seed_out", "cross_scale", "both"],
        default="both",
    )
    parser.add_argument("--safe-threshold", type=float, default=0.005)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "model_size_b",
        "seed",
        "layer_idx",
        "normalized_depth",
        "activation_outlier_score",
        "int8_rel_mse",
        "output_int4_rel_mse_mean",
        "output_int8_saturation_mean",
        "abs_perturbation_delta",
        "safe_label",
        "has_perturbation",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["has_perturbation"] == 1].copy()
    df = df.dropna(subset=["abs_perturbation_delta", "safe_label"])
    df["safe_label"] = df["safe_label"].astype(int)
    for source, target in [
        ("activation_outlier_score", "log_activation_outlier_score"),
        ("int8_rel_mse", "log_int8_rel_mse"),
        ("output_int4_rel_mse_mean", "log_output_int4_rel_mse"),
        ("output_int8_saturation_mean", "log_output_int8_saturation_mean"),
    ]:
        df[target] = np.log1p(df[source].fillna(0.0).clip(lower=0.0))
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("unknown").astype(str)
    return df.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", make_preprocessor()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def as_records(
    df: pd.DataFrame,
    split: str,
    pred_abs_delta: np.ndarray,
    pred_risk: np.ndarray,
    pred_safe_prob: np.ndarray | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append(
            {
                "split": split,
                "model_name": row.get("model_name", ""),
                "model_size_b": row.get("model_size_b", ""),
                "seed": row.get("seed", ""),
                "module_name": row.get("module_name", ""),
                "module_short": row.get("module_short", ""),
                "module_role": row.get("module_role", ""),
                "module_leaf": row.get("module_leaf", ""),
                "abs_perturbation_delta": row.get("abs_perturbation_delta", ""),
                "safe_label": row.get("safe_label", ""),
                "predicted_abs_delta": float(pred_abs_delta[i]),
                "predicted_risk": float(pred_risk[i]),
                "predicted_safe_probability": "" if pred_safe_prob is None else float(pred_safe_prob[i]),
                "activation_outlier_score": row.get("activation_outlier_score", ""),
                "int8_rel_mse": row.get("int8_rel_mse", ""),
            }
        )
    return records


def precision_at_k_safe(y_safe: np.ndarray, risk: np.ndarray, k: int) -> float | None:
    if len(y_safe) == 0:
        return None
    k = min(k, len(y_safe))
    order = np.argsort(risk)[:k]
    return float(np.mean(y_safe[order]))


def recall_at_k_unsafe(y_safe: np.ndarray, risk: np.ndarray, k: int) -> float | None:
    unsafe = 1 - y_safe
    total_unsafe = int(np.sum(unsafe))
    if total_unsafe == 0:
        return None
    k = min(k, len(y_safe))
    order = np.argsort(-risk)[:k]
    return float(np.sum(unsafe[order]) / total_unsafe)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def metric_bundle(y_abs: np.ndarray, y_safe: np.ndarray, risk: np.ndarray) -> dict[str, Any]:
    risk = np.asarray(risk, dtype=float)
    out: dict[str, Any] = {
        "n": int(len(y_abs)),
        "safe_count": int(np.sum(y_safe)),
        "unsafe_count": int(len(y_safe) - np.sum(y_safe)),
        "precision_at_4_lowest_risk": precision_at_k_safe(y_safe, risk, 4),
        "precision_at_8_lowest_risk": precision_at_k_safe(y_safe, risk, 8),
        "unsafe_recall_at_4_highest_risk": recall_at_k_unsafe(y_safe, risk, 4),
        "unsafe_recall_at_8_highest_risk": recall_at_k_unsafe(y_safe, risk, 8),
    }
    if len(y_abs) >= 2 and np.std(risk) > 0 and np.std(y_abs) > 0:
        corr = spearmanr(risk, y_abs)
        out["spearman_abs_delta"] = safe_float(corr.statistic)
        out["spearman_pvalue"] = safe_float(corr.pvalue)
    else:
        out["spearman_abs_delta"] = None
        out["spearman_pvalue"] = None
    if len(np.unique(y_safe)) == 2 and np.std(risk) > 0:
        out["auroc_unsafe"] = safe_float(roc_auc_score(1 - y_safe, risk))
    else:
        out["auroc_unsafe"] = None
    return out


def regression_bundle(y_abs: np.ndarray, pred_abs: np.ndarray, pred_risk: np.ndarray) -> dict[str, Any]:
    out = metric_bundle(y_abs, (y_abs <= 0.005).astype(int), pred_risk)
    out["mae_abs_delta"] = safe_float(mean_absolute_error(y_abs, pred_abs))
    out["rmse_abs_delta"] = safe_float(math.sqrt(mean_squared_error(y_abs, pred_abs)))
    out["r2_abs_delta"] = safe_float(r2_score(y_abs, pred_abs)) if len(y_abs) >= 2 else None
    return out


def fixed_threshold_risk(df: pd.DataFrame) -> np.ndarray:
    outlier = df["activation_outlier_score"].fillna(0.0).to_numpy(dtype=float)
    mse = df["int8_rel_mse"].fillna(0.0).to_numpy(dtype=float)
    return ((outlier > 12.0).astype(float) + (mse > 0.001).astype(float))


def train_eval_split(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    split_name: str,
    ridge_alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    if len(train) < 2 or len(test) == 0:
        return {"error": "not enough train/test rows", "train_n": len(train), "test_n": len(test)}, []

    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train_abs = train["abs_perturbation_delta"].to_numpy(dtype=float)
    y_test_abs = test["abs_perturbation_delta"].to_numpy(dtype=float)
    y_test_safe = test["safe_label"].to_numpy(dtype=int)

    ridge = make_ridge(alpha=ridge_alpha)
    ridge.fit(x_train, y_train_abs)
    pred_risk = ridge.predict(x_test)
    pred_abs = np.clip(pred_risk, 0.0, None)

    pred_safe_prob = None
    y_train_safe = train["safe_label"].to_numpy(dtype=int)
    if len(np.unique(y_train_safe)) == 2:
        logistic = make_logistic()
        logistic.fit(x_train, y_train_safe)
        pred_safe_prob = logistic.predict_proba(x_test)[:, 1]

    outlier_risk = test["activation_outlier_score"].fillna(0.0).to_numpy(dtype=float)
    int8_mse_risk = test["int8_rel_mse"].fillna(0.0).to_numpy(dtype=float)
    fixed_risk = fixed_threshold_risk(test)

    metrics = {
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "ridge": regression_bundle(y_test_abs, pred_abs, pred_risk),
        "baseline_outlier_rank": metric_bundle(y_test_abs, y_test_safe, outlier_risk),
        "baseline_int8_mse_rank": metric_bundle(y_test_abs, y_test_safe, int8_mse_risk),
        "baseline_fixed_threshold": metric_bundle(y_test_abs, y_test_safe, fixed_risk),
    }
    if pred_safe_prob is not None:
        metrics["logistic_safe_probability"] = metric_bundle(y_test_abs, y_test_safe, 1.0 - pred_safe_prob)

    return metrics, as_records(test, split_name, pred_abs, pred_risk, pred_safe_prob)


def evaluate_leave_one_seed_out(df: pd.DataFrame, ridge_alpha: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for seed in sorted(df["seed"].dropna().unique()):
        split_name = f"leave_seed_{int(seed)}"
        split_metrics, split_records = train_eval_split(
            df=df,
            train_mask=df["seed"] != seed,
            test_mask=df["seed"] == seed,
            split_name=split_name,
            ridge_alpha=ridge_alpha,
        )
        metrics[split_name] = split_metrics
        records.extend(split_records)
    return metrics, records


def evaluate_cross_scale(df: pd.DataFrame, ridge_alpha: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_mask = df["model_size_b"] < 7.0
    test_mask = df["model_size_b"] >= 7.0
    return train_eval_split(
        df=df,
        train_mask=train_mask,
        test_mask=test_mask,
        split_name="cross_scale_0p5b_to_7b",
        ridge_alpha=ridge_alpha,
    )


def main() -> None:
    args = parse_args()
    df = read_dataset(args.input)
    if len(df) == 0:
        raise SystemExit("No labeled perturbation rows found in dataset.")

    metrics: dict[str, Any] = {
        "input": args.input,
        "row_count": int(len(df)),
        "safe_threshold": args.safe_threshold,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
    records: list[dict[str, Any]] = []

    if args.eval_mode in {"leave_one_seed_out", "both"}:
        split_metrics, split_records = evaluate_leave_one_seed_out(df, args.ridge_alpha)
        metrics["leave_one_seed_out"] = split_metrics
        records.extend(split_records)

    if args.eval_mode in {"cross_scale", "both"}:
        split_metrics, split_records = evaluate_cross_scale(df, args.ridge_alpha)
        metrics["cross_scale_0p5b_to_7b"] = split_metrics
        records.extend(split_records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_output = Path(args.predictions_output)
    if records:
        pd.DataFrame.from_records(records).to_csv(predictions_output, index=False)
    print(f"Wrote metrics to {output}")
    if records:
        print(f"Wrote predictions to {predictions_output}")
    print(f"Labeled rows: {len(df)}; safe rows: {int(df['safe_label'].sum())}; unsafe rows: {int(len(df) - df['safe_label'].sum())}")


if __name__ == "__main__":
    main()
