from __future__ import annotations

import json

import pytest

from qaq.code.model_adapter import InventoryRecord
from qaq.code.oracle_labels import select_oracle_label
from qaq.code.policies import builtin_policy
from qaq.code.router import (
    build_router_trace,
    extract_token_features,
    load_router,
    save_router,
    train_router,
)


def inventory() -> list[InventoryRecord]:
    return [
        InventoryRecord("a.weight", "a", "attention", 0, "layer_0000:attention", "attention_mlp", (2, 2), "fp32"),
        InventoryRecord("m.weight", "m", "mlp", 0, "layer_0000:mlp", "attention_mlp", (2, 2), "fp32"),
    ]


def test_feature_extraction() -> None:
    record = extract_token_features("s0", [1, 2, 3], [1, 1, 0], {"task": "toy"})
    assert record.features["input_length"] == 3
    assert record.features["attention_length"] == 2
    assert record.prompt_metadata["task"] == "toy"


def test_oracle_selection_and_unsatisfied_fallback() -> None:
    scores = [
        {"policy_id": "static_4bit", "quality": 0.7, "expected_cost": 4.0},
        {"policy_id": "static_8bit", "quality": 0.9, "expected_cost": 8.0},
    ]
    label = select_oracle_label("s0", scores, quality_tolerance=0.05)
    assert label.policy_id == "static_8bit"
    assert label.tolerance_satisfied
    failed = select_oracle_label(
        "s0",
        scores,
        quality_tolerance=0.01,
        reference_quality=1.0,
        fallback_policy_id="static_8bit",
    )
    assert failed.policy_id == "static_8bit"
    assert not failed.tolerance_satisfied


def test_router_training_prediction_trace_and_artifact(tmp_path) -> None:
    rows = inventory()
    features = [
        extract_token_features("short", [1, 2], [1, 1]),
        extract_token_features("long", list(range(20)), [1] * 20),
    ]
    router = train_router(features, ["static_4bit", "static_8bit"])
    path = tmp_path / "router.json"
    save_router(router, path)
    loaded = load_router(path)
    catalog = {
        "static_4bit": builtin_policy("static_4bit", rows),
        "static_8bit": builtin_policy("static_8bit", rows),
    }
    trace = build_router_trace(loaded, features[0], catalog, rows)
    assert trace["predicted_policy"] == "static_4bit"
    assert set(trace["selected_group_bit_widths"].values()) == {4}
    assert trace["confidence"] > 0.0


def test_schema_mismatch_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"feature_schema_version": "old"}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_router(path)

