from __future__ import annotations

import json

import pytest

from qaq.code.model_adapter import InventoryRecord
from qaq.code.policies import builtin_policy, expand_policy, load_policy, save_expanded_policy


def inventory() -> list[InventoryRecord]:
    return [
        InventoryRecord("a.weight", "a", "attention", 0, "layer_0000:attention", "attention_mlp", (2, 2), "fp32"),
        InventoryRecord("m.weight", "m", "mlp", 0, "layer_0000:mlp", "attention_mlp", (2, 2), "fp32"),
    ]


def test_fixed_and_mixed_policies_expand() -> None:
    rows = inventory()
    assert set(expand_policy(builtin_policy("static_8bit", rows), rows).values()) == {8}
    assert set(expand_policy(builtin_policy("static_4bit", rows), rows).values()) == {4}
    mixed = expand_policy(builtin_policy("mixed_attention_high", rows), rows)
    assert mixed["layer_0000:attention"] == 8
    assert mixed["layer_0000:mlp"] == 4


def test_random_policy_is_seeded() -> None:
    rows = inventory()
    a = expand_policy(builtin_policy("random_router_baseline", rows, seed=7), rows)
    b = expand_policy(builtin_policy("random_router_baseline", rows, seed=7), rows)
    assert a == b


def test_invalid_policy_rejection() -> None:
    rows = inventory()
    policy = builtin_policy("static_8bit", rows, allowed_bit_widths=(2, 4, 6))
    with pytest.raises(ValueError, match="unsupported"):
        expand_policy(policy, rows)
    policy.group_bit_widths = {"missing": 4}
    policy.default_bit_width = 4
    with pytest.raises(ValueError, match="unknown group"):
        expand_policy(policy, rows)


def test_policy_json_round_trip(tmp_path) -> None:
    rows = inventory()
    path = tmp_path / "policy.json"
    payload = save_expanded_policy(builtin_policy("mixed_attention_high", rows), rows, path)
    assert json.loads(path.read_text()) == payload
    loaded = load_policy(path)
    assert expand_policy(loaded, rows)["layer_0000:attention"] == 8

