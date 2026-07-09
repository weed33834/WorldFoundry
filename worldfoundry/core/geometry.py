"""Geometry helpers shared by camera-conditioned visual generation runtimes."""

from __future__ import annotations

import torch


def torch_meshgrid_ij(*args):
    """Return an ``ij``-indexed meshgrid across supported torch versions."""

    try:
        return torch.meshgrid(*args, indexing="ij")
    except TypeError:
        return torch.meshgrid(*args)


def ray_condition(
    K,
    c2w,
    H: int,
    W: int,
    device,
    flip_flag=None,
    *,
    use_ray_o: bool = False,
):
    """Build per-pixel Plucker ray features from intrinsics and camera-to-world poses.

    Args:
        K: Camera intrinsics in ``[fx, fy, cx, cy]`` layout with shape ``[B,V,4]``.
            When ``None``, rays use a constant forward camera-space direction.
        c2w: Camera-to-world matrices with shape ``[B,V,4,4]``.
        H: Output grid height.
        W: Output grid width.
        device: Device for generated coordinate grids.
        flip_flag: Optional boolean mask selecting horizontally flipped views.
        use_ray_o: If true, concatenate ``[ray_origin, ray_direction]`` instead of
            the Plucker ``[ray_direction x ray_origin, ray_direction]`` form.
    """

    batch, views = c2w.shape[:2]
    j, i = torch_meshgrid_ij(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape(1, 1, H * W).expand(batch, views, H * W) + 0.5
    j = j.reshape(1, 1, H * W).expand(batch, views, H * W) + 0.5

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = torch_meshgrid_ij(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape(1, 1, H * W).expand(batch, 1, H * W) + 0.5
        j_flip = j_flip.reshape(1, 1, H * W).expand(batch, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    if K is None:
        directions = torch.zeros(batch, views, H * W, 3, device=device, dtype=c2w.dtype)
        directions[..., 2] = 1.0
    else:
        fx, fy, cx, cy = K.chunk(4, dim=-1)
        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)
        directions = torch.stack((xs, ys, zs), dim=-1)
        directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    if use_ray_o:
        plucker = torch.cat([rays_o, rays_d], dim=-1)
    else:
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(batch, views, H, W, 6)


__all__ = ["ray_condition", "torch_meshgrid_ij"]
