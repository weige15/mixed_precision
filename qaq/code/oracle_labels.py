"""Oracle policy-label selection from candidate policy scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OracleLabel:
    sample_id: str
    policy_id: str
    expected_cost: float
    quality_metric: float
    tolerance_satisfied: bool
    reference_quality: float
    quality_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_oracle_label(
    sample_id: str,
    candidate_scores: Iterable[dict[str, Any]],
    quality_tolerance: float,
    *,
    higher_is_better: bool = True,
    reference_quality: float | None = None,
    fallback_policy_id: str | None = None,
) -> OracleLabel:
    """Choose the cheapest candidate within tolerance, else emit explicit fallback."""

    scores = [dict(row) for row in candidate_scores]
    if not scores:
        raise ValueError("Cannot select an oracle label from an empty candidate score list")

    def policy_id(row: dict[str, Any]) -> str:
        return str(row.get("policy_id") or row.get("policy_name"))

    def quality(row: dict[str, Any]) -> float:
        if "quality_metric" in row:
            return float(row["quality_metric"])
        if "quality" in row:
            return float(row["quality"])
        raise ValueError(f"Candidate score lacks quality metric: {row}")

    def cost(row: dict[str, Any]) -> float:
        return float(row.get("expected_cost", row.get("cost", 0.0)))

    qualities = [quality(row) for row in scores]
    ref = float(reference_quality) if reference_quality is not None else (
        max(qualities) if higher_is_better else min(qualities)
    )
    if higher_is_better:
        threshold = ref - float(quality_tolerance)
        satisfying = [row for row in scores if quality(row) >= threshold]
        best = max(scores, key=quality)
    else:
        threshold = ref + float(quality_tolerance)
        satisfying = [row for row in scores if quality(row) <= threshold]
        best = min(scores, key=quality)

    if satisfying:
        chosen = min(satisfying, key=lambda row: (cost(row), -quality(row) if higher_is_better else quality(row)))
        satisfied = True
    else:
        by_id = {policy_id(row): row for row in scores}
        chosen = by_id.get(fallback_policy_id, best) if fallback_policy_id else best
        satisfied = False

    return OracleLabel(
        sample_id=str(sample_id),
        policy_id=policy_id(chosen),
        expected_cost=cost(chosen),
        quality_metric=quality(chosen),
        tolerance_satisfied=satisfied,
        reference_quality=ref,
        quality_tolerance=float(quality_tolerance),
    )

