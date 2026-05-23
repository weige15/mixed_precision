#!/usr/bin/env python
"""Summarize matched Llama H8 metric comparisons."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


POLICY_HINTS = {
    "bf16": "bf16",
    "qlora_nf4": "qlora_nf4",
    "h8_rescue": "h8_rescue",
}


def load_summary(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def infer_policy(path: Path, data: dict) -> str:
    text = str(path)
    for marker, policy in POLICY_HINTS.items():
        if marker in text:
            return policy
    return str(data.get("precision_policy") or data.get("policy_name") or "unknown")


def rel_delta(treatment: float, baseline: float) -> float:
    return 100.0 * (treatment - baseline) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/h8-hardware-aware-precision-search/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/h8-hardware-aware-precision-search/results/llama_h8_metrics_summary.json"),
    )
    args = parser.parse_args()

    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for summary_path in sorted(args.results_root.glob("llama31_8b_*_*_*/summary.json")):
        data = load_summary(summary_path)
        seed = int(data.get("seed"))
        hardware = str(data.get("hardware_label") or "unknown")
        policy = infer_policy(summary_path, data)
        grouped[(seed, hardware)][policy] = data

    rows = []
    for (seed, hardware), policies in sorted(grouped.items()):
        bf16 = policies.get("bf16")
        if not bf16:
            continue
        bf16_eval = float(bf16["final_eval_loss"])
        bf16_mem = float(bf16["peak_cuda_memory_gib"])
        bf16_tok = float(bf16.get("train_tokens_per_sec_excl_first") or bf16.get("train_tokens_per_sec") or 0.0)
        for policy, data in sorted(policies.items()):
            if policy == "bf16":
                continue
            eval_loss = float(data["final_eval_loss"])
            mem = float(data["peak_cuda_memory_gib"])
            tok = float(data.get("train_tokens_per_sec_excl_first") or data.get("train_tokens_per_sec") or 0.0)
            rows.append(
                {
                    "seed": seed,
                    "hardware_label": hardware,
                    "policy": policy,
                    "bf16_eval_loss": bf16_eval,
                    "policy_eval_loss": eval_loss,
                    "eval_delta_percent": rel_delta(eval_loss, bf16_eval),
                    "bf16_peak_memory_gib": bf16_mem,
                    "policy_peak_memory_gib": mem,
                    "memory_delta_percent": rel_delta(mem, bf16_mem),
                    "bf16_tokens_per_sec": bf16_tok,
                    "policy_tokens_per_sec": tok,
                    "tokens_per_sec_delta_percent": rel_delta(tok, bf16_tok) if bf16_tok else None,
                    "loss_spike_count": data.get("loss_spike_count"),
                    "nan_or_inf_count": data.get("nan_or_inf_count"),
                    "summary_path": data["_path"],
                }
            )

    payload = {
        "results_root": str(args.results_root),
        "n_matched_comparisons": len(rows),
        "comparisons": rows,
        "note": "Only compare rows with the same seed and hardware_label. H8 selective rescue is optional and may be absent.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

