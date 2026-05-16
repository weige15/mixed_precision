#!/usr/bin/env python
"""Summarize H6 resource results without mixing hardware contexts."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-glob",
        default="experiments/h6-adaptive-precision-assignment/results/resource_*/summary.json",
    )
    parser.add_argument(
        "--hardware-label",
        default="",
        help="Optional hardware group to print, e.g. rtx4050-local or rtx3090-lab.",
    )
    return parser.parse_args()


def hardware_key(summary: dict[str, Any], path: str) -> str:
    label = summary.get("hardware_label") or ""
    if label:
        return label
    dirname = os.path.basename(os.path.dirname(path))
    for marker in ("rtx4050", "4050", "rtx3090", "3090", "lab", "local", "basic1", "github"):
        if marker in dirname.lower():
            return f"legacy:{marker}"
    return "unlabeled"


def main() -> None:
    args = parse_args()
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(glob.glob(args.results_glob)):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        key = hardware_key(summary, path)
        groups[key].append((path, summary))

    if args.hardware_label:
        groups = {args.hardware_label: groups.get(args.hardware_label, [])}

    for key, rows in sorted(groups.items()):
        if not rows:
            continue
        print(f"\n## {key}")
        print("dir\tpolicy\tseed\tsteps\tlr\tbatch\teval\tmem_gib\ttok_s")
        by_run: dict[tuple[int, int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
        for path, summary in rows:
            run_dir = os.path.basename(os.path.dirname(path))
            policy = summary.get("precision_policy")
            seed = int(summary.get("seed"))
            steps = int(summary.get("max_steps"))
            lr = float(summary.get("learning_rate"))
            batch = summary.get("effective_batch_size_sequences")
            eval_loss = summary.get("final_eval_loss")
            mem = summary.get("peak_cuda_memory_gib")
            tok_s = summary.get("tokens_per_sec_train_excluding_first_step") or summary.get("tokens_per_sec_train")
            print(f"{run_dir}\t{policy}\t{seed}\t{steps}\t{lr:g}\t{batch}\t{eval_loss:.6f}\t{mem:.3f}\t{tok_s:.1f}")
            by_run[(seed, steps, lr)][policy] = summary

        print("\npaired policies vs bf16_baseline")
        print("policy\tseed\tsteps\tlr\teval_delta%\tmem_delta%\ttok_s_delta%")
        for (seed, steps, lr), policies in sorted(by_run.items()):
            base = policies.get("bf16_baseline")
            if not base:
                continue
            base_tps = base.get("tokens_per_sec_train_excluding_first_step") or base.get("tokens_per_sec_train")
            for policy, summary in sorted(policies.items()):
                if policy == "bf16_baseline":
                    continue
                policy_tps = summary.get("tokens_per_sec_train_excluding_first_step") or summary.get("tokens_per_sec_train")
                eval_delta = (summary["final_eval_loss"] - base["final_eval_loss"]) / base["final_eval_loss"] * 100
                mem_delta = (summary["peak_cuda_memory_gib"] - base["peak_cuda_memory_gib"]) / base["peak_cuda_memory_gib"] * 100
                tps_delta = (policy_tps - base_tps) / base_tps * 100
                print(f"{policy}\t{seed}\t{steps}\t{lr:g}\t{eval_delta:+.2f}\t{mem_delta:+.2f}\t{tps_delta:+.2f}")


if __name__ == "__main__":
    main()
