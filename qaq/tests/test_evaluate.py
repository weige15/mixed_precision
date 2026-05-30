from __future__ import annotations

import json

from qaq.code.evaluate import aggregate_metrics, main, parse_args


def test_parse_args_and_aggregate_metrics() -> None:
    args = parse_args(["run-static", "--output-dir", "qaq/results/test", "--hardware-label", "cpu"])
    assert args.command == "run-static"
    summary = aggregate_metrics(
        [
            {
                "policy_name": "p",
                "quality": 1.0,
                "latency_ms": 2.0,
                "memory_bytes": 3,
                "selected_group_bit_widths": {"g": 4},
            }
        ],
        {"model_name": "m", "seed": 1, "hardware_label": "cpu", "allowed_bit_widths": [2, 4, 6, 8]},
    )
    assert summary["quality_metrics"]["mean_quality"] == 1.0
    assert summary["selected_bit_distribution"] == {"4": 1}


def test_run_static_writes_artifacts(tmp_path) -> None:
    out = tmp_path / "static"
    main(["run-static", "--output-dir", str(out), "--hardware-label", "cpu", "--seed", "3"])
    assert (out / "metrics.jsonl").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["hardware_label"] == "cpu"
    assert "static_8bit" in summary["policies"]

