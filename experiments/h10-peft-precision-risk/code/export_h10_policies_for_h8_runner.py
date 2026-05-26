#!/usr/bin/env python
"""Export H10 selector policies in the H8 candidate schema.

The current PEFT runner already knows how to execute H8-style selective rescue:
it reads a candidate JSON, finds a policy by name, and replaces the listed
QLoRA/NF4 projection modules with bf16/fp32 Linear modules before LoRA wrapping.
H10 selector artifacts use a different schema, so this script adapts them
without changing the runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SELECTOR_NAMES = [
    "oracle_perturbation_upper_bound",
    "activation_outlier_rescue",
    "role_prior_rescue",
    "cross_model_logistic_unsafe",
    "cross_model_ridge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h10-policies",
        default="experiments/h10-peft-precision-risk/results/h10_rescue_policy_candidates_llama31_8b.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/h10-peft-precision-risk/results/h10_h8_runner_candidates_llama31_8b.json",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=DEFAULT_SELECTOR_NAMES,
        help="H10 selector names to export. Names not present in the source are skipped.",
    )
    parser.add_argument(
        "--prefix",
        default="h10",
        help="Prefix for exported policy names.",
    )
    return parser.parse_args()


def policy_name(prefix: str, selector: str) -> str:
    clean = selector.replace("oracle_perturbation_upper_bound", "oracle_perturbation_top4")
    clean = clean.replace("_rescue", "_top4")
    return f"{prefix}_{clean}"


def main() -> None:
    args = parse_args()
    source_path = Path(args.h10_policies)
    source = json.loads(source_path.read_text())
    selector_filter = set(args.selectors)
    exported: list[dict[str, Any]] = []

    for policy in source.get("policies", []):
        selector = str(policy.get("selector", ""))
        if selector not in selector_filter:
            continue
        modules = [str(module) for module in policy.get("module_names", [])]
        if not modules:
            continue
        exported.append(
            {
                "policy_name": policy_name(args.prefix, selector),
                "source_selector": selector,
                "base_backend": source.get("base_backend", "qlora_4bit_nf4"),
                "policy_type": "selective_rescue_from_low_bit",
                "rescue_precision": "bf16_or_fp32_backend_dependent",
                "rescue_modules": modules,
                "rationale": (
                    "H10 selector policy exported for the existing H8 selective-rescue runner. "
                    "Use oracle policies only as perturbation-informed upper-bound controls."
                ),
                "backend_feasibility": "unverified_until_setup_only_smoke",
                "expected_effect": "quality rescue with possible memory/dispatch overhead",
                "selector_modules": policy.get("modules", []),
            }
        )

    if not exported:
        raise SystemExit(f"No matching policies exported from {source_path}")

    model_name = source.get("model_name")
    payload = {
        "source": str(source_path),
        "model_filters": [model_name],
        "topk": source.get("top_k"),
        "models": [
            {
                "model_name": model_name,
                "status": "candidate_ready",
                "n_candidate_policies": len(exported),
                "candidate_policies": exported,
            }
        ],
        "candidate_policies_flat": [{**policy, "model_name": model_name} for policy in exported],
        "notes": [
            "This file intentionally uses the H8 candidate schema so run_lora_precision.py can execute H10 policies.",
            "The runner ignores selector metadata and consumes only policy_name and rescue_modules.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote H8-compatible H10 candidates to {output}")
    print("Exported policies:")
    for policy in exported:
        print(f"  {policy['policy_name']}: {len(policy['rescue_modules'])} modules")


if __name__ == "__main__":
    main()

