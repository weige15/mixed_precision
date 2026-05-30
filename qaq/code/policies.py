"""Static and artifact-backed precision policies for QAQ."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path
from typing import Any, Iterable

from qaq.code.model_adapter import InventoryRecord


DEFAULT_ALLOWED_WIDTHS = (2, 4, 6, 8)


@dataclass
class PrecisionPolicy:
    policy_name: str
    group_granularity: str
    allowed_bit_widths: tuple[int, ...] = DEFAULT_ALLOWED_WIDTHS
    default_bit_width: int | None = None
    group_bit_widths: dict[str, int] = field(default_factory=dict)
    source: str = "builtin"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_bit_widths"] = list(self.allowed_bit_widths)
        payload["group_bit_widths"] = dict(sorted(self.group_bit_widths.items()))
        return payload


def _group_roles(inventory: Iterable[InventoryRecord]) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    for record in inventory:
        roles.setdefault(record.group_id, set()).add(record.module_role)
    return roles


def validate_inventory(inventory: Iterable[InventoryRecord]) -> list[InventoryRecord]:
    rows = list(inventory)
    if not rows:
        raise ValueError("Cannot expand a precision policy over an empty inventory")
    return rows


def expand_policy(policy: PrecisionPolicy, inventory: Iterable[InventoryRecord]) -> dict[str, int]:
    rows = validate_inventory(inventory)
    groups = sorted({record.group_id for record in rows})
    known = set(groups)
    unknown = sorted(set(policy.group_bit_widths) - known)
    if unknown:
        raise ValueError(f"Policy {policy.policy_name} references unknown group ids: {unknown}")
    if policy.default_bit_width is None:
        missing = [group for group in groups if group not in policy.group_bit_widths]
        if missing:
            raise ValueError(f"Policy {policy.policy_name} is missing assignments for groups: {missing}")
    expanded = {
        group: int(policy.group_bit_widths.get(group, policy.default_bit_width))
        for group in groups
    }
    invalid = {
        group: width for group, width in expanded.items() if width not in set(policy.allowed_bit_widths)
    }
    if invalid:
        raise ValueError(
            f"Policy {policy.policy_name} uses unsupported bit widths {invalid}; "
            f"allowed={list(policy.allowed_bit_widths)}"
        )
    return expanded


def builtin_policy(
    name: str,
    inventory: Iterable[InventoryRecord],
    *,
    allowed_bit_widths: Iterable[int] = DEFAULT_ALLOWED_WIDTHS,
    seed: int = 0,
) -> PrecisionPolicy:
    rows = validate_inventory(inventory)
    allowed = tuple(int(width) for width in allowed_bit_widths)
    granularity = rows[0].group_granularity
    if name == "static_8bit":
        return PrecisionPolicy(name, granularity, allowed, 8, source="builtin", description="All groups at 8 bits.")
    if name == "static_4bit":
        return PrecisionPolicy(name, granularity, allowed, 4, source="builtin", description="All groups at 4 bits.")
    if name == "mixed_attention_high":
        roles = _group_roles(rows)
        mapping = {
            group: 8 if "attention" in group_roles else 4
            for group, group_roles in roles.items()
        }
        return PrecisionPolicy(
            name,
            granularity,
            allowed,
            None,
            mapping,
            source="builtin",
            description="Attention groups at 8 bits, other groups at 4 bits.",
        )
    if name == "random_router_baseline":
        rng = random.Random(seed)
        groups = sorted({record.group_id for record in rows})
        mapping = {group: int(rng.choice(allowed)) for group in groups}
        return PrecisionPolicy(
            name,
            granularity,
            allowed,
            None,
            mapping,
            source="random",
            description="Seeded random group bit-width ablation.",
            metadata={"seed": seed},
        )
    raise ValueError(f"Unknown built-in QAQ policy: {name}")


def policy_from_dict(payload: dict[str, Any]) -> PrecisionPolicy:
    return PrecisionPolicy(
        policy_name=str(payload["policy_name"]),
        group_granularity=str(payload.get("group_granularity", "unknown")),
        allowed_bit_widths=tuple(int(width) for width in payload.get("allowed_bit_widths", DEFAULT_ALLOWED_WIDTHS)),
        default_bit_width=None
        if payload.get("default_bit_width") is None
        else int(payload.get("default_bit_width")),
        group_bit_widths={str(group): int(width) for group, width in payload.get("group_bit_widths", {}).items()},
        source=str(payload.get("source", "json")),
        description=str(payload.get("description", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def load_policy(path: str | Path) -> PrecisionPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "policy" in payload:
        payload = payload["policy"]
    return policy_from_dict(payload)


def save_expanded_policy(
    policy: PrecisionPolicy,
    inventory: Iterable[InventoryRecord],
    path: str | Path,
) -> dict[str, Any]:
    expanded = expand_policy(policy, inventory)
    payload = {
        "policy": policy.to_dict(),
        "expanded_group_bit_widths": expanded,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

