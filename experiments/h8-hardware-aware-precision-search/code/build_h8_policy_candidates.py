#!/usr/bin/env python
"""Build H8 hardware-aware selective-rescue policy candidates.

This script is intentionally conservative. It does not assume that single-module
perturbation deltas are additive, and it does not claim backend feasibility.
It groups high-risk modules from the H7/H6 precision dataset and emits a small
set of candidate rescue policies for later manual/backend validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _float_or_none(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _module_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("model_name", ""), row.get("module_name", "")


def load_module_risks(path: Path, model_filter: str | None) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if model_filter and row.get("model_name") != model_filter:
                continue
            if row.get("label_source") != "perturbation_abs_threshold":
                continue
            grouped[_module_key(row)].append(row)

    modules: list[dict[str, object]] = []
    for (model_name, module_name), rows in grouped.items():
        abs_deltas = [
            value
            for value in (_float_or_none(row.get("abs_perturbation_delta", "")) for row in rows)
            if value is not None
        ]
        outliers = [
            value
            for value in (_float_or_none(row.get("activation_outlier_score", "")) for row in rows)
            if value is not None
        ]
        if not abs_deltas:
            continue
        first = rows[0]
        modules.append(
            {
                "model_name": model_name,
                "module_name": module_name,
                "module_role": first.get("module_role", ""),
                "module_leaf": first.get("module_leaf", ""),
                "layer_idx": _float_or_none(first.get("layer_idx", "")),
                "mean_abs_delta": mean(abs_deltas),
                "max_abs_delta": max(abs_deltas),
                "mean_outlier_score": mean(outliers) if outliers else None,
                "n_labeled_seeds": len(abs_deltas),
            }
        )
    return sorted(modules, key=lambda item: (-float(item["max_abs_delta"]), str(item["module_name"])))


def build_candidates(modules: list[dict[str, object]], safe_threshold: float, topk: int) -> list[dict[str, object]]:
    high_risk = [m for m in modules if float(m["max_abs_delta"]) > safe_threshold]
    norms_logits = [m for m in high_risk if m["module_role"] in {"norm", "logits"}]
    down_proj = [m for m in high_risk if m["module_leaf"] == "down_proj"]
    projections = [m for m in high_risk if "projection" in str(m["module_role"])]

    def rescue(mods: list[dict[str, object]], name: str, rationale: str) -> dict[str, object]:
        return {
            "policy_name": name,
            "base_backend": "qlora_4bit_nf4",
            "policy_type": "selective_rescue_from_low_bit",
            "rescue_precision": "bf16_or_fp32_backend_dependent",
            "rescue_modules": [str(m["module_name"]) for m in mods],
            "rationale": rationale,
            "backend_feasibility": "unverified",
            "expected_effect": "quality rescue with possible memory/dispatch overhead",
        }

    return [
        rescue(
            norms_logits,
            "h8_rescue_norm_logits",
            "Rescue high-risk norm/logit paths first because reductions and logits are conservative precision targets.",
        ),
        rescue(
            norms_logits + down_proj,
            "h8_rescue_norm_logits_highrisk_down",
            "Add high-risk MLP down projections identified by perturbation deltas.",
        ),
        rescue(
            projections[:topk],
            f"h8_rescue_projection_top{topk}",
            "Rescue top-k high-risk projection modules by max perturbation delta.",
        ),
    ]


def build_model_payload(
    model_name: str,
    modules: list[dict[str, object]],
    safe_threshold: float,
    topk: int,
) -> dict[str, object]:
    high_risk = [m for m in modules if float(m["max_abs_delta"]) > safe_threshold]
    candidates = build_candidates(modules, safe_threshold, topk) if modules else []
    for candidate in candidates:
        candidate["model_name"] = model_name
    return {
        "model_name": model_name,
        "status": "candidate_ready" if modules else "no_perturbation_labels",
        "n_modules_with_perturbation_labels": len(modules),
        "n_high_risk_modules": len(high_risk),
        "high_risk_modules": high_risk,
        "candidate_policies": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision-dataset",
        type=Path,
        default=Path("experiments/h7-precision-predictor/results/precision_dataset_with_llama31_8b.csv"),
    )
    parser.add_argument(
        "--model-filter",
        nargs="+",
        default=["Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B"],
        help="One or more model names to include. Each model is summarized independently.",
    )
    parser.add_argument("--safe-threshold", type=float, default=0.005)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/h8-hardware-aware-precision-search/results/h8_policy_candidates.json"),
    )
    args = parser.parse_args()

    model_payloads = [
        build_model_payload(
            model_name,
            load_module_risks(args.precision_dataset, model_name),
            args.safe_threshold,
            args.topk,
        )
        for model_name in args.model_filter
    ]
    flat_candidates = [
        candidate
        for model_payload in model_payloads
        for candidate in model_payload["candidate_policies"]
    ]
    payload = {
        "source": str(args.precision_dataset),
        "model_filters": args.model_filter,
        "safe_threshold": args.safe_threshold,
        "topk": args.topk,
        "models": model_payloads,
        "candidate_policies_flat": flat_candidates,
    }
    if len(model_payloads) == 1:
        payload.update(
            {
                "model_filter": args.model_filter[0],
                "n_modules_with_perturbation_labels": model_payloads[0]["n_modules_with_perturbation_labels"],
                "high_risk_modules": model_payloads[0]["high_risk_modules"],
                "candidate_policies": model_payloads[0]["candidate_policies"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
