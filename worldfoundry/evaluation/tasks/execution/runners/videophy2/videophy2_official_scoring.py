"""VideoPhy2 official result normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _rule_classification_score,
    _score_from_records,
)
from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_official_scoring import (
    official_scores_from_records as videophy_official_scores_from_records,
)

BENCHMARK_ID = "videophy2"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "videophysics/videophy2_test prompt and rule metadata",
        "generated videos for the official prompts",
        "human SA/PC/rule labels or VideoPhy-2-AutoEval outputs",
        "per-sample sa, pc, joint, and rule-classification result fields",
    ],
}


def official_scores_from_records(
    records: list[Mapping[str, Any]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    if not records:
        return {}
    explicit_average = _score_from_records(records, "videophy2_average", official_results_path)
    scores = videophy_official_scores_from_records(
        records,
        official_results_path,
        average_id="videophy2_average",
        rating_scale_max=5.0,
    )
    rule_score = _score_from_records(
        records,
        "rule_classification_accuracy",
        official_results_path,
        aliases=("physical_rule_accuracy", "rule_accuracy", "pr_accuracy"),
    )
    if rule_score is None:
        rule_score = _rule_classification_score(records, official_results_path)
    if rule_score is not None:
        scores["rule_classification_accuracy"] = rule_score
    if explicit_average is not None:
        scores["videophy2_average"] = explicit_average
    elif "joint_score" in scores:
        joint_score = scores["joint_score"]
        scores["videophy2_average"] = OfficialMetricScore(
            score=joint_score.score,
            raw_value=joint_score.raw_value,
            evidence={
                "source_path": None if official_results_path is None else str(official_results_path),
                "aggregation": "videophy2_primary_joint_performance",
                "components": ["joint_score"],
            },
        )
    return scores
