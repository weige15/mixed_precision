#!/usr/bin/env python3
"""Build an H10 inference PTQ action table from H9 vLLM artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_H9_SUMMARY = Path("experiments/h9-transformer-inference-policy-search/results/h9_benchmark_summary.json")
DEFAULT_H9_LONG_SUMMARY = Path(
    "experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_summary.json"
)
DEFAULT_POLICY_CANDIDATES = Path(
    "experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json"
)
DEFAULT_BACKEND_INVENTORY = Path(
    "experiments/h9-transformer-inference-policy-search/results/backend_inventory.json"
)
DEFAULT_OUTPUT = Path("experiments/h10-inference-ptq-assignment/results/action_table.csv")


FIELDNAMES = [
    "model_name",
    "group_name",
    "workload_name",
    "candidate_action",
    "backend",
    "hardware_label",
    "backend_feasible",
    "calibration_signal",
    "perturbation_risk",
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
    "source_artifact",
    "failure_reason",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h9-summary", type=Path, default=DEFAULT_H9_SUMMARY)
    parser.add_argument("--h9-long-summary", type=Path, default=DEFAULT_H9_LONG_SUMMARY)
    parser.add_argument(
        "--skip-default-summaries",
        action="store_true",
        help="Only ingest summaries passed through --extra-h9-summary.",
    )
    parser.add_argument(
        "--extra-h9-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Additional H9 summary to ingest, for example "
            "h9_instruct_awq_marlin=experiments/.../h9_instruct_awq_marlin_summary.json. "
            "May be repeated."
        ),
    )
    parser.add_argument("--policy-candidates", type=Path, default=DEFAULT_POLICY_CANDIDATES)
    parser.add_argument(
        "--extra-policy-candidates",
        action="append",
        default=[],
        type=Path,
        help="Additional H9 policy-candidate JSON files needed by extra summaries. May be repeated.",
    )
    parser.add_argument("--backend-inventory", type=Path, default=DEFAULT_BACKEND_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return float(value) if value is not None else None


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return 100.0 * (candidate - baseline) / baseline


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_risk(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def policy_map(policy_candidates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(policy["policy_name"]): policy
        for policy in policy_candidates.get("candidate_policies", [])
        if policy.get("policy_name")
    }


def merge_policy_maps(policy_candidate_payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for payload in policy_candidate_payloads:
        policies.update(policy_map(payload))
    return policies


def parse_extra_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return (path.stem, path)
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --extra-h9-summary label in {value!r}")
    return (label, Path(raw_path))


def backend_for_policy(policy: dict[str, Any] | None) -> str:
    if not policy:
        return "unknown"
    tags = policy.get("search_tags") or {}
    quantization = tags.get("quantization") or "none"
    kv_cache = tags.get("kv_cache_dtype") or "auto"
    dtype = tags.get("dtype") or (policy.get("llm_kwargs") or {}).get("dtype") or "auto"
    runtime_mode = str(tags.get("runtime_mode") or "")
    runtime = "transformers" if runtime_mode == "transformers" else "vllm"
    pieces = [runtime, str(dtype)]
    layer_group_backend = tags.get("layer_group_backend")
    if layer_group_backend and layer_group_backend != "none":
        pieces.append(str(layer_group_backend))
    if quantization != "none":
        pieces.append(str(quantization))
    if kv_cache != "auto":
        pieces.append(f"kv_{kv_cache}")
    return "+".join(pieces)


def layer_group_policy_summary(policy: dict[str, Any] | None) -> str | None:
    if not policy:
        return None
    layer_group_policy = policy.get("layer_group_policy")
    if not layer_group_policy:
        return None
    groups = []
    for group in layer_group_policy.get("groups", []):
        config = group.get("config") or {}
        groups.append(
            "{group}:{config}:{pattern}".format(
                group=group.get("group_name", "unnamed_group"),
                config=config.get("type", "unknown_config"),
                pattern=group.get("module_regex", ""),
            )
        )
    return "; ".join(groups)


def calibration_signal_for_policy(policy: dict[str, Any] | None) -> str:
    if not policy or not policy.get("layer_group_policy"):
        return "not_applicable_global_vllm_policy"
    layer_group_policy = policy["layer_group_policy"]
    source = layer_group_policy.get("selection_source") or "unspecified_layer_group_selection"
    return f"layer_group_policy:{source}"


def perturbation_risk_for_policy(policy: dict[str, Any] | None) -> str:
    summary = layer_group_policy_summary(policy)
    if summary is None:
        return "not_applicable_global_vllm_policy"
    return f"backend_real_layer_group:{summary}"


def load_inventory_reasons(backend_inventory: dict[str, Any]) -> dict[str, str]:
    reasons = {}
    for policy in backend_inventory.get("policies", []):
        name = policy.get("policy_name")
        if not name:
            continue
        policy_reasons = policy.get("reasons") or []
        reasons[str(name)] = "; ".join(str(reason) for reason in policy_reasons)
    return reasons


def completed_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in summary.get("completed", []) if row.get("status") == "completed"]


def baseline_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    baseline_policy = str(summary.get("baseline_policy", "bf16_default"))
    return {
        str(row["workload_name"]): row
        for row in completed_rows(summary)
        if row.get("policy_name") == baseline_policy
    }


def workload_group(workload: str) -> str:
    if "prefill" in workload:
        return "prefill"
    if "decode" in workload:
        return "decode"
    if "batch" in workload or "mixed" in workload:
        return "mixed_batch"
    return workload


def is_prefill_workload(workload: str) -> bool:
    return "prefill" in workload


def is_decode_workload(workload: str) -> bool:
    return "decode" in workload


def build_completed_action_row(
    row: dict[str, Any],
    baseline: dict[str, Any],
    policy: dict[str, Any] | None,
    source_label: str,
) -> dict[str, str]:
    workload = str(row["workload_name"])
    deltas = row.get("delta_vs_baseline") or {}
    prompt_nll_delta = deltas.get("prompt_nll_percent")
    quality_risk = float(prompt_nll_delta) / 100.0 if prompt_nll_delta is not None else None
    latency_delta = pct_delta(mean_value(row, "latency_sec"), mean_value(baseline, "latency_sec"))
    output_tps_delta = pct_delta(
        mean_value(row, "output_tokens_per_sec"),
        mean_value(baseline, "output_tokens_per_sec"),
    )
    total_tps_delta = pct_delta(
        mean_value(row, "total_tokens_per_sec"),
        mean_value(baseline, "total_tokens_per_sec"),
    )
    memory_delta_gib = None
    candidate_memory = mean_value(row, "used_gib_from_mem_get_info")
    baseline_memory = mean_value(baseline, "used_gib_from_mem_get_info")
    if candidate_memory is not None and baseline_memory is not None:
        memory_delta_gib = candidate_memory - baseline_memory
    memory_delta_pct = pct_delta(candidate_memory, baseline_memory)

    return {
        "model_name": str(row.get("model_name", "")),
        "group_name": workload_group(workload),
        "workload_name": workload,
        "candidate_action": str(row.get("policy_name", "")),
        "backend": backend_for_policy(policy),
        "hardware_label": str(row.get("hardware_label", "")),
        "backend_feasible": "true",
        "calibration_signal": calibration_signal_for_policy(policy),
        "perturbation_risk": perturbation_risk_for_policy(policy),
        "predicted_quality_risk": fmt_risk(quality_risk),
        "prompt_nll_delta_pct_vs_bf16": fmt_float(prompt_nll_delta),
        "latency_delta_pct_vs_bf16": fmt_float(latency_delta),
        "prefill_latency_delta_pct_vs_bf16": fmt_float(latency_delta) if is_prefill_workload(workload) else "",
        "output_tokens_per_sec_delta_pct_vs_bf16": fmt_float(output_tps_delta),
        "decode_tokens_per_sec_delta_pct_vs_bf16": fmt_float(output_tps_delta) if is_decode_workload(workload) else "",
        "total_tokens_per_sec_delta_pct_vs_bf16": fmt_float(total_tps_delta),
        "memory_delta_gib_vs_bf16": fmt_float(memory_delta_gib),
        "memory_delta_pct_vs_bf16": fmt_float(memory_delta_pct),
        "kv_cache_memory_delta_gib_vs_bf16": fmt_float(memory_delta_gib),
        "source_artifact": str(row.get("artifact_path") or source_label),
        "failure_reason": "",
        "notes": (
            f"{source_label}; quality risk is prompt-NLL percent delta divided by 100. "
            "Memory and KV-cache deltas use total CUDA mem_get_info used GiB because "
            "backend worker allocation may not be captured by PyTorch reserved memory."
        ),
    }


def build_failure_row(
    failure: dict[str, Any],
    policy: dict[str, Any] | None,
    inventory_reasons: dict[str, str],
    source_label: str,
    default_model_name: str,
) -> dict[str, str]:
    policy_name = str(failure.get("policy_name", "unknown"))
    known_failure = failure.get("known_failure")
    interpretation = failure.get("failure_interpretation")
    error = failure.get("error") or inventory_reasons.get(policy_name) or "backend attempt did not complete"
    reason_parts = [str(error)]
    if known_failure:
        reason_parts.append(f"known_failure={known_failure}")
    if interpretation:
        reason_parts.append(str(interpretation))
    reason = " | ".join(reason_parts)
    model_name = ""
    if policy:
        model_name = str(policy.get("model_name") or "")
    if not model_name:
        model_name = default_model_name
    return {
        "model_name": model_name,
        "group_name": "backend_feasibility",
        "workload_name": "",
        "candidate_action": policy_name,
        "backend": backend_for_policy(policy),
        "hardware_label": str(failure.get("hardware_label", "")),
        "backend_feasible": "false",
        "calibration_signal": "not_applicable_backend_failure",
        "perturbation_risk": "not_applicable_backend_failure",
        "predicted_quality_risk": "",
        "prompt_nll_delta_pct_vs_bf16": "",
        "latency_delta_pct_vs_bf16": "",
        "prefill_latency_delta_pct_vs_bf16": "",
        "output_tokens_per_sec_delta_pct_vs_bf16": "",
        "decode_tokens_per_sec_delta_pct_vs_bf16": "",
        "total_tokens_per_sec_delta_pct_vs_bf16": "",
        "memory_delta_gib_vs_bf16": "",
        "memory_delta_pct_vs_bf16": "",
        "kv_cache_memory_delta_gib_vs_bf16": "",
        "source_artifact": str(failure.get("artifact_path") or source_label),
        "failure_reason": reason,
        "notes": f"{source_label}; infeasible row retained to make backend support limits explicit.",
    }


def build_rows(
    summaries: list[tuple[str, dict[str, Any]]],
    policies: dict[str, dict[str, Any]],
    inventory_reasons: dict[str, str],
    default_model_name: str,
) -> list[dict[str, str]]:
    action_rows: list[dict[str, str]] = []
    seen_failures: set[tuple[str, str, str]] = set()
    for source_label, summary in summaries:
        baselines = baseline_rows(summary)
        for row in completed_rows(summary):
            workload = str(row.get("workload_name", ""))
            baseline = baselines.get(workload)
            if baseline is None:
                continue
            policy = policies.get(str(row.get("policy_name", "")))
            action_rows.append(build_completed_action_row(row, baseline, policy, source_label))
        for failure in summary.get("failures", []):
            key = (
                str(failure.get("policy_name", "")),
                str(failure.get("hardware_label", "")),
                str(failure.get("artifact_path", "")),
            )
            if key in seen_failures:
                continue
            seen_failures.add(key)
            policy = policies.get(str(failure.get("policy_name", "")))
            action_rows.append(build_failure_row(failure, policy, inventory_reasons, source_label, default_model_name))
    return action_rows


def main() -> None:
    args = parse_args()
    policy_candidate_payloads = [load_json(args.policy_candidates)]
    policy_candidate_payloads.extend(load_json(path) for path in args.extra_policy_candidates)
    backend_inventory = load_json(args.backend_inventory)
    policies = merge_policy_maps(policy_candidate_payloads)
    inventory_reasons = load_inventory_reasons(backend_inventory)
    policy_candidates = policy_candidate_payloads[0]
    default_model_name = str(policy_candidates.get("model_name") or backend_inventory.get("model_name") or "")
    summaries = []
    if not args.skip_default_summaries:
        summaries.extend(
            [
                ("h9_1_default", load_json(args.h9_summary)),
                ("h9_2_long_context", load_json(args.h9_long_summary)),
            ]
        )
    for label, path in [parse_extra_summary(value) for value in args.extra_h9_summary]:
        summaries.append((label, load_json(path)))
    rows = build_rows(
        summaries,
        policies,
        inventory_reasons,
        default_model_name,
    )
    if not rows:
        raise SystemExit("No H10 action rows were built. Regenerate H9 summaries and retry.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    print(f"action rows: {len(rows)}")
    print(f"infeasible rows: {sum(1 for row in rows if row['backend_feasible'] != 'true')}")


if __name__ == "__main__":
    main()
