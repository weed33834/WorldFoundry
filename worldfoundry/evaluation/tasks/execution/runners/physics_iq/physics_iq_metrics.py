"""Physics-IQ metric formulas from official summary or per-scenario CSV rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

METRIC_ORDER = (
    "physics_iq_score",
    "solid_mechanics",
    "fluid_dynamics",
    "optics",
    "thermodynamics",
    "magnetism",
    "physics_iq_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "physics_iq_score": {
        "name": "Physics-IQ Score",
        "group": "physics",
        "higher_is_better": True,
        "description": "Physics-IQ Score normalized benchmark score.",
    },
    "solid_mechanics": {
        "name": "Solid Mechanics",
        "group": "physics",
        "higher_is_better": True,
        "description": "Solid Mechanics normalized benchmark score.",
    },
    "fluid_dynamics": {
        "name": "Fluid Dynamics",
        "group": "physics",
        "higher_is_better": True,
        "description": "Fluid Dynamics normalized benchmark score.",
    },
    "optics": {
        "name": "Optics",
        "group": "physics",
        "higher_is_better": True,
        "description": "Optics normalized benchmark score.",
    },
    "thermodynamics": {
        "name": "Thermodynamics",
        "group": "physics",
        "higher_is_better": True,
        "description": "Thermodynamics normalized benchmark score.",
    },
    "magnetism": {
        "name": "Magnetism",
        "group": "physics",
        "higher_is_better": True,
        "description": "Magnetism normalized benchmark score.",
    },
    "physics_iq_average": {
        "name": "Physics-IQ Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Physics-IQ Average normalized benchmark score.",
        "primary": True,
    },
}

CATEGORY_TO_METRIC = {
    "Solid Mechanics": "solid_mechanics",
    "Fluid Dynamics": "fluid_dynamics",
    "Optics": "optics",
    "Thermodynamics": "thermodynamics",
    "Magnetism": "magnetism",
}

VIEWS = ("perspective-left", "perspective-center", "perspective-right")


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_unit_score(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _is_summary_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(key in rows[0] for key in ("metric_id", "Metric", "metric"))


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        metric_id = str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "").strip()
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if not metric_id or score is None:
            continue
        metrics[metric_id] = _normalize_unit_score(score)
    if "physics_iq_average" not in metrics and "physics_iq_score" in metrics:
        metrics["physics_iq_average"] = metrics["physics_iq_score"]
    return metrics


def _scenario_key(row: Mapping[str, Any]) -> str:
    scenario = str(row.get("scenario") or row.get("Scenario") or "").strip()
    if not scenario:
        return ""
    return scenario.split("_take-")[0].split("_trimmed-")[-1]


def _scenario_category_map(description_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in description_rows:
        scenario = str(row.get("scenario") or "")
        category = str(row.get("category") or "")
        if not scenario or not category:
            continue
        base = scenario.split("_take-")[0]
        mapping[base] = category
        trimmed = base.split("_trimmed-")[-1]
        mapping[trimmed] = category
    return mapping


def _scenario_proxy_score(row: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for view in VIEWS:
        for prefix in ("weighted_spatial_iou_v1", "spatial_iou_v1"):
            score = _to_float(row.get(f"{prefix}_{view}"))
            if score is not None:
                values.append(score)
                break
    if not values:
        return None
    return sum(values) / len(values)


def _category_metrics(
    *,
    scenario_rows: Sequence[Mapping[str, Any]],
    description_rows: Sequence[Mapping[str, str]],
) -> dict[str, float]:
    category_map = _scenario_category_map(description_rows)
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in CATEGORY_TO_METRIC.values()}
    for row in scenario_rows:
        scenario = str(row.get("scenario") or "").strip()
        category = category_map.get(scenario) or category_map.get(_scenario_key(row), "")
        metric_id = CATEGORY_TO_METRIC.get(category)
        if metric_id is None:
            continue
        proxy = _scenario_proxy_score(row)
        if proxy is not None:
            buckets[metric_id].append(proxy)
    return {
        metric_id: (sum(values) / len(values) if values else None)
        for metric_id, values in buckets.items()
    }


def compute_physics_iq_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    description_rows: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics = _summary_metrics(rows)
        component_values = [
            value
            for key, value in direct_metrics.items()
            if key not in {"physics_iq_score", "physics_iq_average"} and value is not None
        ]
        if direct_metrics.get("physics_iq_average") is None and component_values:
            direct_metrics["physics_iq_average"] = sum(component_values) / len(component_values)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    category_metrics: dict[str, float | None] = {}
    if description_rows:
        category_metrics = _category_metrics(scenario_rows=rows, description_rows=description_rows)

    direct_metrics: dict[str, float | None] = {
        **category_metrics,
    }
    component_values = [value for value in direct_metrics.values() if value is not None]
    if component_values:
        direct_metrics["physics_iq_average"] = sum(component_values) / len(component_values)
        direct_metrics["physics_iq_score"] = direct_metrics["physics_iq_average"]
    else:
        direct_metrics["physics_iq_average"] = None
        direct_metrics["physics_iq_score"] = None

    return {
        "metrics": direct_metrics,
        "components": {
            "scenario_count": len(rows),
            "format": "per_scenario_csv",
            "category_bucket_counts": {
                metric_id: len([value for value in [direct_metrics.get(metric_id)] if value is not None])
                for metric_id in CATEGORY_TO_METRIC.values()
            },
        },
    }


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        import json

        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "rows", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
        raise ValueError(f"Unsupported Physics-IQ JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
