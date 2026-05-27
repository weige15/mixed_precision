#!/usr/bin/env python
"""Summarize H9 vLLM benchmark artifacts and mark Pareto candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_RESULTS = Path("experiments/h9-transformer-inference-policy-search/results/benchmarks")
DEFAULT_QUALITY = Path("experiments/h9-transformer-inference-policy-search/results/quality")
DEFAULT_OUTPUT = Path("experiments/h9-transformer-inference-policy-search/results/h9_benchmark_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-policy", default="bf16_default")
    return parser.parse_args()


def load_artifacts(results_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(results_dir.glob("*/benchmark.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        artifacts.append(data)
    return artifacts


def load_quality(quality_dir: Path) -> dict[str, dict[str, Any]]:
    quality = {}
    for path in sorted(quality_dir.glob("*/quality.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        quality[str(data.get("policy_name"))] = data
    return quality


def get_peak_reserved_gib(run: dict[str, Any]) -> float | None:
    after = run.get("cuda_after") or {}
    value = after.get("max_reserved_gib")
    return float(value) if value is not None else None


def get_used_gib(run: dict[str, Any]) -> float | None:
    after = run.get("cuda_after") or {}
    value = after.get("used_gib_from_mem_get_info")
    if value is not None:
        return float(value)
    free = after.get("free_gib")
    total = after.get("total_gib")
    if free is None or total is None:
        return None
    return float(total) - float(free)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "stdev": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_completed(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in artifact.get("runs", []):
        if int(run.get("repeat", 0)) < 0:
            continue
        grouped[str(run.get("workload_name"))].append(run)
    rows = []
    for workload, runs in sorted(grouped.items()):
        rows.append(
            {
                "policy_name": artifact["policy_name"],
                "workload_name": workload,
                "status": artifact["status"],
                "hardware_label": artifact.get("hardware_label"),
                "model_name": artifact.get("model_name"),
                "runtime": artifact.get("runtime"),
                "n_runs": len(runs),
                "latency_sec": summarize_values([float(run["latency_sec"]) for run in runs]),
                "output_tokens_per_sec": summarize_values(
                    [float(run["output_tokens_per_sec"]) for run in runs if run.get("output_tokens_per_sec") is not None]
                ),
                "total_tokens_per_sec": summarize_values(
                    [float(run["total_tokens_per_sec"]) for run in runs if run.get("total_tokens_per_sec") is not None]
                ),
                "peak_reserved_gib": summarize_values(
                    [value for run in runs for value in [get_peak_reserved_gib(run)] if value is not None]
                ),
                "used_gib_from_mem_get_info": summarize_values(
                    [value for run in runs for value in [get_used_gib(run)] if value is not None]
                ),
                "artifact_path": artifact["_path"],
            }
        )
    return rows


def rel_delta(candidate: float | None, baseline: float | None, higher_is_better: bool = False) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    raw = 100.0 * (candidate - baseline) / baseline
    return raw if not higher_is_better else -raw


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_latency = a["latency_sec"]["mean"]
    b_latency = b["latency_sec"]["mean"]
    a_tps = a["output_tokens_per_sec"]["mean"]
    b_tps = b["output_tokens_per_sec"]["mean"]
    a_mem = a["peak_reserved_gib"]["mean"]
    b_mem = b["peak_reserved_gib"]["mean"]
    if a_mem in (None, 0.0) or b_mem in (None, 0.0):
        a_mem = a["used_gib_from_mem_get_info"]["mean"]
        b_mem = b["used_gib_from_mem_get_info"]["mean"]
    comparisons = []
    if a_latency is not None and b_latency is not None:
        comparisons.append((a_latency <= b_latency, a_latency < b_latency))
    if a_tps is not None and b_tps is not None:
        comparisons.append((a_tps >= b_tps, a_tps > b_tps))
    if a_mem is not None and b_mem is not None:
        comparisons.append((a_mem <= b_mem, a_mem < b_mem))
    return bool(comparisons) and all(ok for ok, _ in comparisons) and any(strict for _, strict in comparisons)


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[row["workload_name"]].append(row)
    marked = []
    for workload_rows in by_workload.values():
        for row in workload_rows:
            row = dict(row)
            row["pareto_non_dominated"] = not any(
                other["policy_name"] != row["policy_name"] and dominates(other, row) for other in workload_rows
            )
            marked.append(row)
    return marked


def add_baseline_deltas(
    rows: list[dict[str, Any]],
    baseline_policy: str,
    quality: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    baselines = {
        row["workload_name"]: row
        for row in rows
        if row["policy_name"] == baseline_policy and row["status"] == "completed"
    }
    updated = []
    for row in rows:
        row = dict(row)
        baseline = baselines.get(row["workload_name"])
        policy_quality = quality.get(row["policy_name"], {})
        baseline_quality = quality.get(baseline_policy, {})
        policy_nll = policy_quality.get("mean_prompt_nll") if policy_quality.get("status") == "completed" else None
        baseline_nll = baseline_quality.get("mean_prompt_nll") if baseline_quality.get("status") == "completed" else None
        if baseline:
            row["delta_vs_baseline"] = {
                "latency_percent": rel_delta(row["latency_sec"]["mean"], baseline["latency_sec"]["mean"]),
                "output_tokens_per_sec_percent": rel_delta(
                    row["output_tokens_per_sec"]["mean"],
                    baseline["output_tokens_per_sec"]["mean"],
                    higher_is_better=True,
                ),
                "peak_reserved_gib_percent": rel_delta(
                    row["peak_reserved_gib"]["mean"],
                    baseline["peak_reserved_gib"]["mean"],
                ),
                "used_gib_from_mem_get_info_percent": rel_delta(
                    row["used_gib_from_mem_get_info"]["mean"],
                    baseline["used_gib_from_mem_get_info"]["mean"],
                ),
                "prompt_nll_percent": rel_delta(policy_nll, baseline_nll),
            }
        else:
            row["delta_vs_baseline"] = None
        row["quality"] = {
            "status": policy_quality.get("status"),
            "mean_prompt_nll": policy_nll,
            "tokens_scored": policy_quality.get("tokens_scored"),
            "artifact_path": policy_quality.get("_path"),
        }
        updated.append(row)
    return updated


def main() -> None:
    args = parse_args()
    artifacts = load_artifacts(args.results_dir)
    quality = load_quality(args.quality_dir)
    completed_rows = []
    failures = []
    for artifact in artifacts:
        if artifact.get("status") == "completed":
            completed_rows.extend(aggregate_completed(artifact))
        elif artifact.get("status") == "failed":
            failures.append(
                {
                    "policy_name": artifact.get("policy_name"),
                    "hardware_label": artifact.get("hardware_label"),
                    "error": artifact.get("error"),
                    "known_failure": artifact.get("known_failure"),
                    "failure_interpretation": artifact.get("failure_interpretation"),
                    "artifact_path": artifact.get("_path"),
                }
            )
    completed_rows = mark_pareto(add_baseline_deltas(completed_rows, args.baseline_policy, quality))
    payload = {
        "results_dir": str(args.results_dir),
        "baseline_policy": args.baseline_policy,
        "n_artifacts": len(artifacts),
        "n_completed_policy_workloads": len(completed_rows),
        "n_failed_policies": len(failures),
        "n_quality_artifacts": len(quality),
        "completed": completed_rows,
        "failures": failures,
        "note": "Quality metrics are included when quality artifacts exist. Pareto marking uses latency, output throughput, and memory; memory falls back to cuda mem_get_info used GiB when PyTorch allocator stats do not capture vLLM worker allocations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"completed policy-workload rows: {len(completed_rows)}")
    print(f"failed policies: {len(failures)}")


if __name__ == "__main__":
    main()
