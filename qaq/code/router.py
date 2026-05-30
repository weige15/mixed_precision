"""Token-feature extraction and lightweight deterministic policy routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from qaq.code.model_adapter import InventoryRecord
from qaq.code.policies import PrecisionPolicy, expand_policy


FEATURE_SCHEMA_VERSION = "qaq.token_features.v1"
FEATURE_NAMES = ("input_length", "attention_length")


@dataclass(frozen=True)
class FeatureRecord:
    sample_id: str
    prompt_metadata: dict[str, Any]
    feature_schema_version: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NearestCentroidRouter:
    feature_schema_version: str
    feature_names: tuple[str, ...]
    centroids: dict[str, list[float]]
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "router_type": "nearest_centroid",
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "centroids": self.centroids,
            "label_counts": self.label_counts,
        }


def _flat_length(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "numel") and hasattr(value, "shape"):
        shape = tuple(int(dim) for dim in value.shape)
        return shape[-1] if shape else int(value.numel())
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return len(value[0])
        return len(value)
    return 1


def _attention_sum(attention_mask: Any, fallback_length: int) -> int:
    if attention_mask is None:
        return fallback_length
    if hasattr(attention_mask, "sum"):
        return int(attention_mask.sum().item())
    if isinstance(attention_mask, (list, tuple)):
        if attention_mask and isinstance(attention_mask[0], (list, tuple)):
            return int(sum(attention_mask[0]))
        return int(sum(attention_mask))
    return fallback_length


def extract_token_features(
    sample_id: str,
    input_ids: Any,
    attention_mask: Any | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> FeatureRecord:
    input_length = _flat_length(input_ids)
    attention_length = _attention_sum(attention_mask, input_length)
    return FeatureRecord(
        sample_id=str(sample_id),
        prompt_metadata=dict(prompt_metadata or {}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features={
            "input_length": float(input_length),
            "attention_length": float(attention_length),
        },
    )


def _vector(record: FeatureRecord, feature_names: Iterable[str]) -> list[float]:
    if record.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema {record.feature_schema_version}; expected {FEATURE_SCHEMA_VERSION}"
        )
    return [float(record.features[name]) for name in feature_names]


def train_router(
    feature_records: Iterable[FeatureRecord],
    labels: Iterable[str],
    *,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> NearestCentroidRouter:
    records = list(feature_records)
    label_list = [str(label) for label in labels]
    names = tuple(feature_names)
    if len(records) != len(label_list) or not records:
        raise ValueError("Router training requires the same non-zero number of features and labels")
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for record, label in zip(records, label_list):
        vector = _vector(record, names)
        sums.setdefault(label, [0.0 for _ in names])
        counts[label] = counts.get(label, 0) + 1
        for idx, value in enumerate(vector):
            sums[label][idx] += value
    centroids = {
        label: [value / counts[label] for value in values]
        for label, values in sums.items()
    }
    return NearestCentroidRouter(FEATURE_SCHEMA_VERSION, names, centroids, counts)


def predict_policy_id(router: NearestCentroidRouter, feature_record: FeatureRecord) -> tuple[str, float]:
    if router.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Router artifact schema {router.feature_schema_version} is incompatible with {FEATURE_SCHEMA_VERSION}"
        )
    vector = _vector(feature_record, router.feature_names)
    distances = {
        label: math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, centroid)))
        for label, centroid in router.centroids.items()
    }
    if not distances:
        raise ValueError("Router artifact has no centroids")
    label = min(distances, key=distances.get)
    inv = {key: 1.0 / (1.0 + dist) for key, dist in distances.items()}
    denom = sum(inv.values()) or 1.0
    return label, float(inv[label] / denom)


def save_router(router: NearestCentroidRouter, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(router.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_router(path: str | Path) -> NearestCentroidRouter:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Router artifact schema {payload.get('feature_schema_version')} is incompatible with {FEATURE_SCHEMA_VERSION}"
        )
    return NearestCentroidRouter(
        feature_schema_version=str(payload["feature_schema_version"]),
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        centroids={str(label): [float(value) for value in values] for label, values in payload["centroids"].items()},
        label_counts={str(label): int(count) for label, count in payload.get("label_counts", {}).items()},
    )


def build_router_trace(
    router: NearestCentroidRouter,
    feature_record: FeatureRecord,
    policy_catalog: dict[str, PrecisionPolicy],
    inventory: Iterable[InventoryRecord],
    *,
    expected_costs: dict[str, float] | None = None,
    quality_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    policy_id, confidence = predict_policy_id(router, feature_record)
    if policy_id not in policy_catalog:
        raise ValueError(f"Router predicted unknown policy id: {policy_id}")
    group_bits = expand_policy(policy_catalog[policy_id], inventory)
    return {
        "sample_id": feature_record.sample_id,
        "features": feature_record.to_dict(),
        "predicted_policy": policy_id,
        "selected_group_bit_widths": group_bits,
        "expected_cost": None if expected_costs is None else expected_costs.get(policy_id),
        "quality_metric": None if quality_metrics is None else quality_metrics.get(policy_id),
        "confidence": confidence,
    }

