"""Module for base_models -> three_dimensions -> general_3d -> shape_of_motion -> shape_of_motion_runtime -> flow3d -> loss_utils.py functionality."""

import numpy as np
import torch
import torch.nn.functional as F


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    """Masked mse loss.

    Args:
        pred: The pred.
        gt: The gt.
        mask: The mask.
        normalize: The normalize.
        quantile: The quantile.
    """
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


# def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
#     if mask is None:
#         return trimmed_l1_loss(pred, gt, quantile)
#     else:
#         sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
#         quantile_mask = (
#             (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
#             if quantile < 1
#             else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
#         )
#         ndim = sum_loss.shape[-1]
#         if normalize:
#             return torch.sum((sum_loss * mask)[quantile_mask]) / (
#                 ndim * torch.sum(mask[quantile_mask]) + 1e-8
#             )
#         else:
#             return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    """Masked l1 loss.

    Args:
        pred: The pred.
        gt: The gt.
        mask: The mask.
        normalize: The normalize.
        quantile: The quantile.
    """
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        # sum_loss.shape 
        # block     [218255, 1]
        # apple     [36673, 475, 1]     17,419,675
        # creeper   [37587, 360, 1]     13,531,320
        # backpack  [37828, 180, 1]     6,809,040
        # quantile_mask = (
        #     (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
        #     if quantile < 1
        #     else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        # )
        # use torch.sort instead of torch.quantile when input too large
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])

def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    """Masked huber loss.

    Args:
        pred: The pred.
        gt: The gt.
        delta: The delta.
        mask: The mask.
        normalize: The normalize.
    """
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    """Trimmed mse loss.

    Args:
        pred: The pred.
        gt: The gt.
        quantile: The quantile.
    """
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    """Trimmed l1 loss.

    Args:
        pred: The pred.
        gt: The gt.
        quantile: The quantile.
    """
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Knn.

    Args:
        x: The x.
        k: The k.

    Returns:
        The return value.
    """
    if x.numel() == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    with torch.no_grad():
        values = x.detach().float()
        distances = torch.cdist(values, values, p=2)
        distances.fill_diagonal_(float("inf"))
        kth = min(k, max(1, values.shape[0] - 1))
        nn_distances, indices = distances.topk(kth, largest=False)
    return nn_distances.cpu().numpy().astype(np.float32), indices.cpu().numpy().astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None):
    """Get weights for procrustes.

    Args:
        clusters: The clusters.
        visibilities: The visibilities.
    """
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    """Compute accel loss.

    Args:
        transls: The transls.
    """
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss
