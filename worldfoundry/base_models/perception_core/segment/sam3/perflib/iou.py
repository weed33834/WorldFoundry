"""Module for base_models -> perception_core -> segment -> sam3 -> perflib -> iou.py functionality."""

import torch


def pairwise_iou(pred_masks, gt_masks, eps=1e-6):
    """Pairwise iou.

    Args:
        pred_masks: The pred masks.
        gt_masks: The gt masks.
        eps: The eps.
    """
    N, H, W = pred_masks.shape
    M = gt_masks.shape[0]
    # Flatten and convert to float for matmul
    pred_flat = pred_masks.reshape(N, -1).float()
    gt_flat = gt_masks.reshape(M, -1).float()
    # Intersection: (N, M)
    intersection = torch.matmul(pred_flat, gt_flat.t())
    # Areas
    area_pred = pred_flat.sum(dim=1, keepdim=True)  # (N, 1)
    area_gt = gt_flat.sum(dim=1, keepdim=True)  # (M, 1)
    # Union: (N, M)
    union = area_pred + area_gt.t() - intersection
    if eps is None:
        iou = intersection / union.clamp(min=1)
    else:
        iou = intersection / (union + eps)
    return iou  # shape: (N, M)


def pairwise_iom(pred_masks, gt_masks, eps=1e-8):
    """Pairwise iom.

    Args:
        pred_masks: The pred masks.
        gt_masks: The gt masks.
        eps: The eps.
    """
    N, H, W = pred_masks.shape
    M = gt_masks.shape[0]
    # Flatten and convert to float for matmul
    pred_flat = pred_masks.reshape(N, -1).float()
    gt_flat = gt_masks.reshape(M, -1).float()
    # Intersection: (N, M)
    intersection = torch.matmul(pred_flat, gt_flat.t())
    # Areas
    area_pred = pred_flat.sum(dim=1, keepdim=True)  # (N, 1)
    area_gt = gt_flat.sum(dim=1, keepdim=True)  # (M, 1)
    # Union: (N, M)
    min_area = torch.min(area_pred, area_gt)
    iou = intersection / (min_area + eps)
    return iou  # shape: (N, M)
