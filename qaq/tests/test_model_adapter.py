from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qaq.code.model_adapter import (  # noqa: E402
    classify_module_role,
    discover_linear_module_inventory,
    extract_layer_idx,
    inventory_from_dicts,
    inventory_to_dicts,
)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


def test_layer_role_and_inventory_serialization() -> None:
    model = TinyModel()
    inventory = discover_linear_module_inventory(model, group_granularity="attention_mlp")
    names = {row.module_name for row in inventory}
    assert "model.layers.1.self_attn.q_proj" in names
    assert extract_layer_idx("model.layers.12.self_attn.q_proj") == 12
    assert classify_module_role("model.layers.0.mlp.down_proj") == "mlp"
    assert any(row.group_id == "layer_0000:attention" for row in inventory)
    assert any(row.shape == (4, 4) for row in inventory)
    round_trip = inventory_from_dicts(inventory_to_dicts(inventory))
    assert round_trip == inventory


def test_group_granularities_are_stable() -> None:
    model = TinyModel()
    by_layer = discover_linear_module_inventory(model, group_granularity="transformer_layer")
    by_linear = discover_linear_module_inventory(model, group_granularity="linear_module")
    assert "layer_0000" in {row.group_id for row in by_layer}
    assert "linear:model.layers.0.self_attn.q_proj" in {row.group_id for row in by_linear}


def test_empty_selection_fails() -> None:
    with pytest.raises(ValueError, match="No selected"):
        discover_linear_module_inventory(torch.nn.Module(), module_name_allowlist=["missing"])

