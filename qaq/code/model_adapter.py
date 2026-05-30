"""Model module inventory and stable QAQ group id generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError:  # Allows JSON/policy helpers to import before ML deps are installed.
    torch = None  # type: ignore[assignment]


SUPPORTED_GRANULARITIES = {"transformer_layer", "attention_mlp", "linear_module"}


@dataclass(frozen=True)
class InventoryRecord:
    tensor_name: str
    module_name: str
    module_role: str
    layer_idx: int | None
    group_id: str
    group_granularity: str
    shape: tuple[int, ...]
    source_dtype: str
    target_quantized: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


def extract_layer_idx(module_name: str) -> int | None:
    patterns = [
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
        r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    return None


def classify_module_role(module_name: str) -> str:
    lowered = module_name.lower()
    leaf = lowered.rsplit(".", 1)[-1]
    if leaf in {"q_proj", "k_proj", "v_proj", "o_proj"} or "self_attn" in lowered or ".attn." in lowered:
        return "attention"
    if leaf in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"} or ".mlp." in lowered:
        return "mlp"
    if leaf in {"lm_head", "embed_out"}:
        return "lm_head"
    if "embed" in lowered:
        return "embedding"
    return "unknown"


def stable_group_id(module_name: str, module_role: str, layer_idx: int | None, granularity: str) -> str:
    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported group granularity: {granularity}")
    if granularity == "linear_module":
        return f"linear:{module_name}"
    layer_part = f"layer_{layer_idx:04d}" if layer_idx is not None else f"layer_unknown:{module_name}"
    if granularity == "transformer_layer":
        return layer_part
    coarse_role = module_role if module_role in {"attention", "mlp"} else "other"
    return f"{layer_part}:{coarse_role}"


def discover_linear_module_inventory(
    model: Any,
    group_granularity: str = "attention_mlp",
    module_name_allowlist: Iterable[str] | None = None,
    include_lm_head: bool = True,
) -> list[InventoryRecord]:
    """Find selected Linear weights without mutating model parameters."""

    if torch is None:
        raise RuntimeError("PyTorch is required to discover model module inventories")
    if group_granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported group granularity: {group_granularity}")
    allowlist = set(module_name_allowlist or [])
    rows: list[InventoryRecord] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if allowlist and module_name not in allowlist and module_name.rsplit(".", 1)[-1] not in allowlist:
            continue
        role = classify_module_role(module_name)
        if role == "lm_head" and not include_lm_head:
            continue
        layer_idx = extract_layer_idx(module_name)
        group_id = stable_group_id(module_name, role, layer_idx, group_granularity)
        weight = module.weight
        rows.append(
            InventoryRecord(
                tensor_name=f"{module_name}.weight",
                module_name=module_name,
                module_role=role,
                layer_idx=layer_idx,
                group_id=group_id,
                group_granularity=group_granularity,
                shape=tuple(int(dim) for dim in weight.shape),
                source_dtype=str(weight.dtype),
                target_quantized=True,
            )
        )
    if not rows:
        raise ValueError("No selected torch.nn.Linear modules found for QAQ quantization")
    return rows


def inventory_to_dicts(inventory: Iterable[InventoryRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in inventory]


def inventory_from_dicts(rows: Iterable[dict[str, Any]]) -> list[InventoryRecord]:
    return [
        InventoryRecord(
            tensor_name=str(row["tensor_name"]),
            module_name=str(row["module_name"]),
            module_role=str(row.get("module_role", "unknown")),
            layer_idx=None if row.get("layer_idx") is None else int(row["layer_idx"]),
            group_id=str(row["group_id"]),
            group_granularity=str(row["group_granularity"]),
            shape=tuple(int(dim) for dim in row["shape"]),
            source_dtype=str(row["source_dtype"]),
            target_quantized=bool(row.get("target_quantized", True)),
        )
        for row in rows
    ]
