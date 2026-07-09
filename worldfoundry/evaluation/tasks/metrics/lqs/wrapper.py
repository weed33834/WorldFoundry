"""Layout Quality Score (LQS) from arXiv:2208.06162."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, permutations
from typing import Any

import numpy as np


def _bbox_center(box: Sequence[float]) -> tuple[float, float]:
    if len(box) == 4:
        x1, y1, x2, y2 = box
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
    if len(box) == 5:
        x, y, w, h = box
        return (float(x + w / 2.0), float(y + h / 2.0))
    raise ValueError(f"bbox must have 4 or 5 values, got {box!r}")


def _bbox_area(box: Sequence[float]) -> float:
    if len(box) == 4:
        x1, y1, x2, y2 = box
        return float(abs(x2 - x1) * abs(y2 - y1))
    if len(box) == 5:
        _, _, w, h = box
        return float(abs(w) * abs(h))
    raise ValueError(f"bbox must have 4 or 5 values, got {box!r}")


def _labels(layout: Sequence[dict[str, Any]]) -> set[str]:
    return {str(item["label"]) for item in layout}


def _match_boxes(
    groundtruth: Sequence[dict[str, Any]],
    predicted: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    gt_by_label: dict[str, list[dict[str, Any]]] = {}
    pred_by_label: dict[str, list[dict[str, Any]]] = {}
    for item in groundtruth:
        gt_by_label.setdefault(str(item["label"]), []).append(item)
    for item in predicted:
        pred_by_label.setdefault(str(item["label"]), []).append(item)
    shared = set(gt_by_label) & set(pred_by_label)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for label in shared:
        gt_items = gt_by_label[label]
        pred_items = pred_by_label[label]
        if len(gt_items) == 1 and len(pred_items) == 1:
            pairs.append((gt_items[0], pred_items[0]))
            continue
        best_perm: tuple[int, ...] | None = None
        best_alc = float("inf")
        for perm in permutations(range(len(pred_items))):
            if len(perm) != len(gt_items):
                continue
            alc = 0.0
            for gt_item, pred_index in zip(gt_items, perm, strict=False):
                g = _bbox_center(gt_item["bbox"])
                p = _bbox_center(pred_items[pred_index]["bbox"])
                alc += float(np.linalg.norm(np.array(g) - np.array(p)))
            alc /= max(1, len(gt_items))
            if alc < best_alc:
                best_alc = alc
                best_perm = perm
        if best_perm is not None:
            for gt_item, pred_index in zip(gt_items, best_perm, strict=False):
                pairs.append((gt_item, pred_items[pred_index]))
    return pairs


def compute_lqs(
    groundtruth_layout: Sequence[dict[str, Any]],
    predicted_layout: Sequence[dict[str, Any]],
    *,
    image_area: float = 80.0,
    sigma_l: float = 1.0,
    gamma_lc: float = 0.25,
    gamma_ac: float = 0.25,
) -> dict[str, float]:
    """Compute Layout Quality Score and components (LayoutTransformer paper)."""
    gt_labels = _labels(groundtruth_layout)
    pred_labels = _labels(predicted_layout)
    intersection = gt_labels & pred_labels
    if not groundtruth_layout:
        raise ValueError("groundtruth_layout must be non-empty")
    lr = len(intersection) / len(gt_labels)
    lp = len(intersection) / len(pred_labels) if pred_labels else 0.0
    pairs = _match_boxes(groundtruth_layout, predicted_layout)
    if not pairs:
        return {"lqs": lr + lp, "lr": lr, "lp": lp, "lc": 0.0, "ac": 0.0}
    alc_values = []
    rlc_values = []
    gt_centers = []
    pred_centers = []
    gt_areas = []
    pred_areas = []
    for gt_item, pred_item in pairs:
        gt_center = np.array(_bbox_center(gt_item["bbox"]))
        pred_center = np.array(_bbox_center(pred_item["bbox"]))
        gt_centers.append(gt_center)
        pred_centers.append(pred_center)
        gt_areas.append(_bbox_area(gt_item["bbox"]))
        pred_areas.append(_bbox_area(pred_item["bbox"]))
        alc_values.append(float(np.linalg.norm(gt_center - pred_center)))
    alc = float(np.mean(alc_values))
    rel_terms = []
    for i, j in combinations(range(len(pairs)), 2):
        gt_rel = gt_centers[i] - gt_centers[j]
        pred_rel = pred_centers[i] - pred_centers[j]
        rel_terms.append(float(np.linalg.norm(gt_rel - pred_rel)))
    rlc = float(np.mean(rel_terms)) if rel_terms else 0.0
    lc = gamma_lc * np.exp(-alc / (2.0 * sigma_l**2)) + (1.0 - gamma_lc) * np.exp(-rlc / (2.0 * sigma_l**2))
    aac = 1.0 - float(np.mean([abs(p - g) / image_area for g, p in zip(gt_areas, pred_areas)]))
    rac_terms = []
    for i, j in combinations(range(len(pairs)), 2):
        gt_cmp = gt_areas[i] > gt_areas[j]
        pred_cmp = pred_areas[i] > pred_areas[j]
        rac_terms.append(1.0 - abs(int(gt_cmp) - int(pred_cmp)))
    rac = float(np.mean(rac_terms)) if rac_terms else 1.0
    ac = gamma_ac * aac + (1.0 - gamma_ac) * rac
    lqs = lr + lp + float(lc) + float(ac)
    return {
        "lqs": float(lqs),
        "lr": float(lr),
        "lp": float(lp),
        "lc": float(lc),
        "ac": float(ac),
        "alc": alc,
        "rlc": rlc,
        "aac": float(aac),
        "rac": rac,
    }


__all__ = ["compute_lqs"]
