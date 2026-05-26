#!/usr/bin/env python
"""Compare equal-budget high-precision rescue selectors on H10 labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "model_size_b",
    "layer_idx",
    "normalized_depth",
    "log_activation_outlier_score",
    "log_int8_rel_mse",
    "log_output_int4_rel_mse",
    "log_output_int8_saturation",
]

CATEGORICAL_FEATURES = [
    "module_role",
    "module_leaf",
    "candidate_format",
]

ROLE_PRIOR = {
    "down_proj": 0,
    "o_proj": 1,
    "q_proj": 2,
    "k_proj": 3,
    "v_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="experiments/h10-peft-precision-risk/results/seed_aggregated_precision_dataset.csv",
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--output",
        default="experiments/h10-peft-precision-risk/results/rescue_selector_evaluation_llama31_8b.json",
    )
    parser.add_argument(
        "--policies-output",
        default="experiments/h10-peft-precision-risk/results/h10_rescue_policy_candidates_llama31_8b.json",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "model_size_b",
        "layer_idx",
        "normalized_depth",
        "mean_abs_delta",
        "max_abs_delta",
        "safe_rate",
        "safe_all_seeds",
        "conservative_rescue_target",
        "activation_outlier_score_agg",
        "int8_rel_mse_agg",
        "output_int4_rel_mse_mean_mean",
        "output_int8_saturation_mean_mean",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["module_role"] = df["module_role"].fillna("unknown").astype(str)
    df["module_leaf"] = df["module_leaf"].fillna("unknown").astype(str)
    df["candidate_format"] = df["candidate_format"].fillna("unknown").astype(str)
    for source, target in [
        ("activation_outlier_score_agg", "log_activation_outlier_score"),
        ("int8_rel_mse_agg", "log_int8_rel_mse"),
        ("output_int4_rel_mse_mean_mean", "log_output_int4_rel_mse"),
        ("output_int8_saturation_mean_mean", "log_output_int8_saturation"),
    ]:
        values = pd.to_numeric(df.get(source, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        df[target] = np.log1p(values)
    return df


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


def train_cross_model_predictors(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, np.ndarray]:
    if len(train) < 2:
        return {}
    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_target = target[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_abs = train["max_abs_delta"].to_numpy(dtype=float)

    ridge = Pipeline(steps=[("features", make_preprocessor()), ("model", Ridge(alpha=1.0))])
    ridge.fit(x_train, y_abs)
    scores = {"cross_model_ridge": ridge.predict(x_target)}

    y_unsafe = train["conservative_rescue_target"].astype(int).to_numpy()
    if len(np.unique(y_unsafe)) == 2:
        logistic = Pipeline(
            steps=[
                ("features", make_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        logistic.fit(x_train, y_unsafe)
        scores["cross_model_logistic_unsafe"] = logistic.predict_proba(x_target)[:, 1]
    return scores


def projection_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["module_role"].astype(str).str.contains("projection", na=False)].copy()


def selected_payload(name: str, selected: pd.DataFrame, score_column: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "module_name": row["module_name"],
                "module_short": row.get("module_short", ""),
                "module_role": row.get("module_role", ""),
                "module_leaf": row.get("module_leaf", ""),
                "layer_idx": finite_float(row.get("layer_idx")),
                "selector_score": finite_float(row.get(score_column)),
                "mean_abs_delta": finite_float(row.get("mean_abs_delta")),
                "max_abs_delta": finite_float(row.get("max_abs_delta")),
                "safe_all_seeds": int(row.get("safe_all_seeds", 0)),
                "safe_rate": finite_float(row.get("safe_rate")),
            }
        )
    return {
        "selector": name,
        "score_column": score_column,
        "module_names": [row["module_name"] for row in rows],
        "modules": rows,
    }


def evaluate_selection(name: str, candidates: pd.DataFrame, selected: pd.DataFrame, score_column: str) -> dict[str, Any]:
    selected_unsafe = 1 - selected["safe_all_seeds"].astype(int)
    all_unsafe = 1 - candidates["safe_all_seeds"].astype(int)
    total_unsafe = int(all_unsafe.sum())
    captured_unsafe = int(selected_unsafe.sum())
    return {
        **selected_payload(name, selected, score_column),
        "k": int(len(selected)),
        "captured_unsafe_count": captured_unsafe,
        "total_unsafe_projection_count": total_unsafe,
        "unsafe_precision_at_k": finite_float(captured_unsafe / len(selected)) if len(selected) else None,
        "unsafe_recall_at_k": finite_float(captured_unsafe / total_unsafe) if total_unsafe else None,
        "captured_max_abs_delta_sum": finite_float(selected["max_abs_delta"].sum()),
        "captured_mean_abs_delta_sum": finite_float(selected["mean_abs_delta"].sum()),
        "selected_max_abs_delta_max": finite_float(selected["max_abs_delta"].max()) if len(selected) else None,
    }


def top_by_score(candidates: pd.DataFrame, score_column: str, top_k: int) -> pd.DataFrame:
    return candidates.sort_values([score_column, "module_name"], ascending=[False, True]).head(top_k).copy()


def role_prior_selection(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["_role_rank"] = ranked["module_leaf"].map(ROLE_PRIOR).fillna(99)
    ranked["_depth_rank"] = pd.to_numeric(ranked["normalized_depth"], errors="coerce").fillna(0.5)
    ranked["_role_prior_score"] = -ranked["_role_rank"] - 0.01 * ranked["_depth_rank"]
    return ranked.sort_values(["_role_rank", "_depth_rank", "module_name"]).head(top_k).copy()


def random_summary(candidates: pd.DataFrame, top_k: int, trials: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if len(candidates) == 0:
        return {"selector": "random_rescue", "error": "no candidates"}
    k = min(top_k, len(candidates))
    unsafe = (1 - candidates["safe_all_seeds"].astype(int)).to_numpy()
    max_abs = candidates["max_abs_delta"].to_numpy(dtype=float)
    captured_unsafe: list[float] = []
    captured_delta: list[float] = []
    for _ in range(trials):
        idx = rng.choice(len(candidates), size=k, replace=False)
        captured_unsafe.append(float(unsafe[idx].sum()))
        captured_delta.append(float(max_abs[idx].sum()))
    return {
        "selector": "random_rescue",
        "k": k,
        "trials": trials,
        "captured_unsafe_count_mean": finite_float(np.mean(captured_unsafe)),
        "captured_unsafe_count_p05": finite_float(np.quantile(captured_unsafe, 0.05)),
        "captured_unsafe_count_p95": finite_float(np.quantile(captured_unsafe, 0.95)),
        "captured_max_abs_delta_sum_mean": finite_float(np.mean(captured_delta)),
        "captured_max_abs_delta_sum_p05": finite_float(np.quantile(captured_delta, 0.05)),
        "captured_max_abs_delta_sum_p95": finite_float(np.quantile(captured_delta, 0.95)),
    }


def main() -> None:
    args = parse_args()
    df = read_dataset(args.input)
    target_all = df[df["model_name"] == args.model_name].copy()
    if target_all.empty:
        raise SystemExit(f"No rows found for model {args.model_name!r}")
    candidates = projection_candidates(target_all)
    if candidates.empty:
        raise SystemExit(f"No projection candidates found for model {args.model_name!r}")
    top_k = min(args.top_k, len(candidates))

    train = df[df["model_name"] != args.model_name].copy()
    predictor_scores = train_cross_model_predictors(train, candidates)
    for name, scores in predictor_scores.items():
        candidates[f"score_{name}"] = scores

    candidates["score_activation_outlier"] = candidates["activation_outlier_score_agg"].fillna(0.0)
    candidates["score_int8_mse"] = candidates["int8_rel_mse_agg"].fillna(0.0)
    candidates["score_oracle_max_delta"] = candidates["max_abs_delta"].fillna(0.0)

    evaluations: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []

    selectors = [
        ("oracle_perturbation_upper_bound", "score_oracle_max_delta"),
        ("activation_outlier_rescue", "score_activation_outlier"),
        ("int8_mse_rescue", "score_int8_mse"),
    ]
    for selector_name, score_column in selectors:
        selected = top_by_score(candidates, score_column, top_k)
        evaluations.append(evaluate_selection(selector_name, candidates, selected, score_column))
        policies.append(selected_payload(selector_name, selected, score_column))

    role_selected = role_prior_selection(candidates, top_k)
    evaluations.append(evaluate_selection("role_prior_rescue", candidates, role_selected, "_role_prior_score"))
    policies.append(selected_payload("role_prior_rescue", role_selected, "_role_prior_score"))

    for score_name in sorted(name for name in candidates.columns if name.startswith("score_cross_model_")):
        selector_name = score_name.removeprefix("score_")
        selected = top_by_score(candidates, score_name, top_k)
        evaluations.append(evaluate_selection(selector_name, candidates, selected, score_name))
        policies.append(selected_payload(selector_name, selected, score_name))

    random_eval = random_summary(candidates, top_k, args.random_trials, args.random_seed)

    payload = {
        "input": args.input,
        "model_name": args.model_name,
        "top_k": top_k,
        "candidate_filter": "projection modules only",
        "train_models_for_cross_model_predictor": sorted(train["model_name"].dropna().unique().tolist()),
        "n_target_module_rows": int(len(target_all)),
        "n_projection_candidates": int(len(candidates)),
        "n_unsafe_projection_candidates": int((1 - candidates["safe_all_seeds"].astype(int)).sum()),
        "selectors": evaluations,
        "random_baseline": random_eval,
        "notes": [
            "Oracle perturbation uses target max_abs_delta and is an upper-bound diagnostic, not a deployable predictor.",
            "Cross-model predictors train only on non-target model rows and use target calibration features.",
            "Selector quality is a proxy; GPU training is still required for final PEFT quality and memory claims.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    policy_payload = {
        "source_evaluation": str(output),
        "model_name": args.model_name,
        "base_backend": "qlora_4bit_nf4",
        "policy_type": "selective_bf16_rescue_from_low_bit",
        "top_k": top_k,
        "policies": policies,
    }
    policies_output = Path(args.policies_output)
    policies_output.write_text(json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote selector evaluation to {output}")
    print(f"Wrote policy candidates to {policies_output}")
    print(f"Projection candidates: {len(candidates)}; unsafe-any-seed: {payload['n_unsafe_projection_candidates']}")


if __name__ == "__main__":
    main()

