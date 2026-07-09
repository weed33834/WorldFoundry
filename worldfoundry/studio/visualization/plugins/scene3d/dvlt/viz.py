# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DVLT scene visualization: depth overlay, Plotly, and Rerun backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import rerun as rr
import torch
from torch import Tensor

from dvlt.common.io import normalize_depth, normalize_image, read_depth, read_image_cv2
from dvlt.common.numpy.rotation import mat_to_quat
from dvlt.common.pose import to4x4
from dvlt.struct.cameras import Cameras
from worldfoundry.studio.visualization.plugins.scene3d.dvlt.color import (
    ColormapOptions,
    apply_depth_colormap,
)

def view_matrix_from_string(convention: str) -> np.ndarray:
    """Return a 3×3 matrix mapping coordinates expressed in *convention* to the
    default RFU (Right-Forward-Up) convention used by our visualization libs.

    The *convention* string must have three characters.  Each denotes the
    positive direction of the x-, y- and z-axis respectively:

        R / L – Right (+X)  /  Left (−X)
        F / B – Forward (+Y) /  Back (−Y)
        U / D – Up (+Z)     /  Down (−Z)

    Example
    -------
    >>> view_matrix_from_string("RDF")
    array([[ 1,  0,  0],
           [ 0,  0,  1],
           [ 0, -1,  0]])

    The matrix *M* satisfies::

        coords_rfu = M @ coords_in_convention

    so when *convention* == "RFU" we simply return the identity.
    """

    if len(convention) != 3:
        raise ValueError("Coordinate convention string must contain exactly 3 characters, e.g. 'RFU'.")

    mapping = {
        "R": np.array([1, 0, 0]),  # +X
        "L": np.array([-1, 0, 0]),  # -X
        "F": np.array([0, 1, 0]),  # +Y
        "B": np.array([0, -1, 0]),  # -Y
        "U": np.array([0, 0, 1]),  # +Z
        "D": np.array([0, 0, -1]),  # -Z
    }

    cols = []
    for c in convention.upper():
        if c not in mapping:
            raise ValueError(f"Invalid axis specifier '{c}' in coordinate convention '{convention}'.")
        cols.append(mapping[c])
    return np.stack(cols, axis=1).astype(float)  # shape (3,3)

def calculate_auto_image_plane_distance(
    cameras,
    points: Optional[Tensor | np.ndarray] = None,
    radius_scale_factor: float = 0.05,
) -> float:
    """Calculate automatic image plane distance based on scene extent.

    Uses point cloud data when available, otherwise falls back to camera positions.
    For dense point clouds, the radius uses a **median** center and **99.5th
    percentile** distance so a few outlier points (bad depths) do not blow up
    Rerun's camera frustum / image-plane distance; sparse camera-only paths use
    mean + max as before.

    Args:
        cameras: Camera objects containing poses
        points: Optional 3D point positions
        radius_scale_factor: Scale factor to apply to scene radius

    Returns:
        Calculated image plane distance as percentage of scene radius
    """
    if points is not None:
        positions = points.cpu().numpy() if isinstance(points, Tensor) else points
    elif isinstance(cameras, dict):
        all_positions = []
        for _, cams in cameras.items():
            for camera in cams:
                all_positions.append(camera.camera_to_worlds[:3, 3].cpu().numpy())
        positions = np.array(all_positions)
    else:
        positions = np.array([camera.camera_to_worlds[:3, 3].cpu().numpy() for camera in cameras])

    assert len(positions) > 0, "No positions to calculate scene radius"

    # Dense point clouds: mean + max is dominated by a handful of bad depths (Rerun
    # image planes / framing use this scalar). Camera-only fallbacks stay mean + max.
    from_points = points is not None
    n = len(positions)
    if from_points and n >= 8:
        centroid = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        scene_radius = float(np.percentile(distances, 99.5))
    else:
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        scene_radius = float(np.max(distances))
    return scene_radius * radius_scale_factor

def overlay_depth_map(
    image: Union[Tensor, np.ndarray],
    depth_map: Union[Tensor, np.ndarray],
    alpha: float = 0.5,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: Optional[ColormapOptions] = None,
) -> np.ndarray:
    """Overlay a depth map on an image.

    Args:
        image (Union[torch.Tensor, np.ndarray]): image shape (H, W, 3)
        depth_map (Union[torch.Tensor, np.ndarray]): depth map shape (H, W) or (H, W, 1)
        alpha (float): alpha value for the overlay. 0 means no overlay, 1 means full overlay. Defaults to 0.5.
        colormap_options (ColormapOptions): colormap options to use. Defaults to ColormapOptions("inferno_r").
    Returns:
        np.ndarray: overlayed image as numpy array (H, W, 3), uint8 0-255
    """
    if colormap_options is None:
        colormap_options = ColormapOptions("default")

    # Convert to torch if needed
    is_tensor = isinstance(image, torch.Tensor)
    if not is_tensor:
        image = torch.from_numpy(image.copy())

    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.from_numpy(depth_map.copy())

    # Normalize image to 0-1 if needed (likely uint8)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0

    # Ensure depth_map is 3D (H, W, 1)
    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(-1)

    # Apply colormap to depth
    depth_map_colored = apply_depth_colormap(
        depth_map, near_plane=near_plane, far_plane=far_plane, colormap_options=colormap_options
    )

    # Perform the overlay
    overlayed = image * (1 - alpha) + depth_map_colored * alpha
    invalid_mask = (depth_map == 0).squeeze(-1)
    overlayed[invalid_mask] = image[invalid_mask]

    # Always convert to numpy
    return (overlayed * 255).cpu().numpy().astype(np.uint8)

def _frustum_lines_plotly(
    K: np.ndarray, extr_c2w: np.ndarray, base_scale: float = 0.2, reference_focal_length: float = 1000.0
) -> tuple[list[float], list[float], list[float]]:
    """Return x, y, z coordinates for frustum lines for Plotly.

    Frustum size is scaled based on focal length to visualize field of view differences.
    Longer focal length = smaller frustum (narrower FOV).
    Shorter focal length = larger frustum (wider FOV).

    Returns:
        tuple: (x_coords, y_coords, z_coords) where each is a list containing
               coordinates for all frustum lines with None separators
    """
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    img_w, img_h = 2 * cx, 2 * cy
    corners_pix = np.array([[0, 0, 1], [img_w, 0, 1], [img_w, img_h, 1], [0, img_h, 1]]).T  # (3,4)
    Kinv = np.linalg.inv(K)
    dirs = Kinv @ corners_pix  # (3,4)

    # Scale based on focal length - use average focal length and reference of 1000
    avg_focal_length = (fx + fy) / 2
    focal_scale = reference_focal_length / avg_focal_length
    scale = base_scale * focal_scale

    dirs = dirs / np.linalg.norm(dirs, axis=0, keepdims=True) * scale
    # Accept 3x4 [R|t] or 4x4 matrices
    if extr_c2w.shape == (3, 4):
        R = extr_c2w[:, :3]
        c = extr_c2w[:, 3]
    else:  # assume 4x4
        R = extr_c2w[:3, :3]
        c = extr_c2w[:3, 3]
    corners_w = (R @ dirs).T + c  # (4,3)

    # Create line coordinates (center to corners + connecting corners)
    x_coords, y_coords, z_coords = [], [], []

    # Lines from camera center to corners
    for corner in corners_w:
        x_coords.extend([c[0], corner[0], None])
        y_coords.extend([c[1], corner[1], None])
        z_coords.extend([c[2], corner[2], None])

    # Lines connecting corners (forming rectangle)
    for i in range(4):
        corner1 = corners_w[i]
        corner2 = corners_w[(i + 1) % 4]
        x_coords.extend([corner1[0], corner2[0], None])
        y_coords.extend([corner1[1], corner2[1], None])
        z_coords.extend([corner1[2], corner2[2], None])

    return x_coords, y_coords, z_coords

def scene_to_plotly(
    seq_name: str,
    pts_pred: np.ndarray,
    pred_rgb: np.ndarray,
    cameras_pred: Cameras,
    pts_gt: np.ndarray | None = None,
    gt_rgb: np.ndarray | None = None,
    cameras_gt: Cameras | None = None,
    view_coordinates: str = "RFU",
    alpha: float = 0.3,
) -> go.Figure:
    """Create Plotly 3D figure with point clouds + camera frustums.

    Args:
        seq_name: Name of the sequence.
        pts_pred: Predicted point cloud coordinates, shape (N, 3)
        pts_gt: Ground truth point cloud coordinates, shape (N, 3)
        pred_rgb: RGB uint8 colors for predicted points, shape (N, 3)
        gt_rgb: RGB uint8 colors for ground truth points, shape (N, 3)
        cameras_pred: Predicted Cameras object containing multiple cameras
        cameras_gt: Ground truth Cameras object containing multiple cameras
        view_coordinates: Coordinate convention string (e.g. "RFU", "RDF") that
            defines the positive directions of the displayed x, y, z axes.
        alpha: Alpha value for blending red/green with predicted/ground truth points.

    Returns:
        Plotly Figure object with 3D visualization
    """

    # Extract intrinsics (do not depend on coordinate system)
    intrinsics_pred = cameras_pred.get_intrinsics_matrices().detach().cpu().numpy()
    extrinsics_pred = to4x4(cameras_pred.camera_to_worlds).detach().cpu().numpy()

    # Convert to 4x4 matrices using helper function, then to numpy
    if cameras_gt is not None:
        extrinsics_gt = to4x4(cameras_gt.camera_to_worlds).detach().cpu().numpy()
        valid_cameras = extrinsics_gt[:3, :3].any()
        extrinsics_gt = extrinsics_gt if valid_cameras else []
        intrinsics_gt = cameras_gt.get_intrinsics_matrices().detach().cpu().numpy()
        intrinsics_gt = intrinsics_gt if valid_cameras else []
    else:
        extrinsics_gt = []
        intrinsics_gt = []

    # 2) Apply view coordinate transform to points and extrinsics
    M = view_matrix_from_string(view_coordinates)

    pts_pred = (M @ pts_pred.T).T
    if pts_gt is not None:
        pts_gt = (M @ pts_gt.T).T

    for i in range(len(extrinsics_pred)):
        extrinsics_pred[i, :3, :3] = M @ extrinsics_pred[i, :3, :3]
        extrinsics_pred[i, :3, 3] = M @ extrinsics_pred[i, :3, 3]

    for i in range(len(extrinsics_gt)):
        extrinsics_gt[i, :3, :3] = M @ extrinsics_gt[i, :3, :3]
        extrinsics_gt[i, :3, 3] = M @ extrinsics_gt[i, :3, 3]

    if pts_gt is not None:
        # Build coloured point arrays (blend with red / green)
        red = np.array([[255, 0, 0]], dtype=np.float32)
        green = np.array([[0, 255, 0]], dtype=np.float32)
        pred_rgb = (pred_rgb * (1 - alpha) + red * alpha).astype(np.uint8)
        gt_rgb = (gt_rgb * (1 - alpha) + green * alpha).astype(np.uint8)

    # Create figure
    fig = go.Figure()

    # Add predicted point cloud (already transformed)
    fig.add_trace(
        go.Scatter3d(
            x=pts_pred[:, 0],
            y=pts_pred[:, 1],
            z=pts_pred[:, 2],
            mode="markers",
            marker=dict(size=2, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in pred_rgb], opacity=0.8),
            name="Predicted Points",
            hovertemplate="<b>Predicted</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
        )
    )

    # Add ground truth point cloud
    if pts_gt is not None:
        fig.add_trace(
            go.Scatter3d(
                x=pts_gt[:, 0],
                y=pts_gt[:, 1],
                z=pts_gt[:, 2],
                mode="markers",
                marker=dict(size=2, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in gt_rgb], opacity=0.8),
                name="Ground Truth Points",
                hovertemplate="<b>Ground Truth</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        )

    # Add camera frustums
    mean_pred_x = intrinsics_pred[:, 0, 0].mean()
    mean_pred_y = intrinsics_pred[:, 1, 1].mean()
    mean_gt_x = intrinsics_gt[:, 0, 0].mean() if len(intrinsics_gt) > 0 else mean_pred_x
    mean_gt_y = intrinsics_gt[:, 1, 1].mean() if len(intrinsics_gt) > 0 else mean_pred_y
    avg_focal_length_x = (mean_pred_x + mean_gt_x) / 2
    avg_focal_length_y = (mean_pred_y + mean_gt_y) / 2
    avg_focal_length = (avg_focal_length_x + avg_focal_length_y) / 2

    # Predicted camera frustums (red)
    pred_x_all, pred_y_all, pred_z_all = [], [], []
    for K, E in zip(intrinsics_pred, extrinsics_pred, strict=False):
        x_coords, y_coords, z_coords = _frustum_lines_plotly(K, E, reference_focal_length=avg_focal_length)
        pred_x_all.extend(x_coords)
        pred_y_all.extend(y_coords)
        pred_z_all.extend(z_coords)

    if pred_x_all:  # Only add if there are cameras
        fig.add_trace(
            go.Scatter3d(
                x=pred_x_all,
                y=pred_y_all,
                z=pred_z_all,
                mode="lines",
                line=dict(color="red", width=3),
                name="Predicted Cameras",
                hoverinfo="skip",
            )
        )

    # Ground truth camera frustums (green)
    gt_x_all, gt_y_all, gt_z_all = [], [], []
    for K, E in zip(intrinsics_gt, extrinsics_gt, strict=False):
        x_coords, y_coords, z_coords = _frustum_lines_plotly(K, E, reference_focal_length=avg_focal_length)
        gt_x_all.extend(x_coords)
        gt_y_all.extend(y_coords)
        gt_z_all.extend(z_coords)

    if gt_x_all:  # Only add if there are cameras
        fig.add_trace(
            go.Scatter3d(
                x=gt_x_all,
                y=gt_y_all,
                z=gt_z_all,
                mode="lines",
                line=dict(color="green", width=3),
                name="Ground Truth Cameras",
                hoverinfo="skip",
            )
        )

    # Configure layout for better 3D visualization
    fig.update_layout(
        title=f"3D Scene: Point Clouds and Camera Poses - {seq_name}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255, 255, 255, 0.8)"),
    )
    return fig

def _compute_shared_pointcloud_indices(points_dict, max_num_points):
    """Pre-compute a single random subsample index set shared across all entries
    in ``points_dict`` so paired pointclouds (e.g. ``pred`` and ``gt``) end up
    with corresponding rows after subsampling.

    Returns ``None`` when there's only one entry, or when entries disagree in
    shape (in which case the caller falls back to independent per-entity
    subsampling, preserving legacy behavior), or when no entry exceeds
    ``max_num_points``.
    """
    values = list(points_dict.values())
    if len(values) <= 1:
        return None

    def _length(x):
        return x.shape[0] if hasattr(x, "shape") else len(x)

    n = _length(values[0])
    for v in values[1:]:
        if _length(v) != n:
            return None
    if n <= max_num_points:
        return None
    return np.random.choice(n, size=max_num_points, replace=False)

def visualize_scene(
    log_path: str,
    cameras: Cameras | dict[str, Cameras],
    points: Optional[Tensor | np.ndarray | list[np.ndarray] | dict[str, Tensor | np.ndarray | list[np.ndarray]]] = None,
    rgb: Optional[Tensor | np.ndarray | list[np.ndarray] | dict[str, Tensor | np.ndarray | list[np.ndarray]]] = None,
    images: Optional[List[str | np.ndarray | Tensor] | dict[str, List[str | np.ndarray | Tensor]]] = None,
    depths: Optional[List[str | np.ndarray | Tensor] | dict[str, List[str | np.ndarray | Tensor]]] = None,
    depth_scale_factor: float = 1.0,
    server_address: Optional[str] = None,
    image_plane_distance: float | str = "auto",
    image_max_size: int = 0,
    save_path: Optional[str] = None,
    view_coordinates: str = "RFU",
    max_num_points: int = 200_000,
    app_id: Optional[str] = None,
):
    """Visualize a 3D scene using Rerun with cameras, pointclouds, images, and 3D bounding boxes.

    This function creates a comprehensive 3D scene visualization that includes camera poses,
    RGB/depth images, 3D pointclouds, instance segmentation masks, and 3D bounding boxes.
    The visualization is displayed using the Rerun framework and can be viewed in real-time
    or saved to a file for later analysis.

    Args:
        log_path (str): The logging path/name for the Rerun visualization session.

        cameras (Cameras | dict[str, Cameras]): Camera objects containing intrinsics and poses.
            Can be a single Cameras object or a dictionary mapping split names to Cameras objects
            (e.g., {"train": cameras_train, "val": cameras_val}).

        points (Optional[Tensor | np.ndarray | list[np.ndarray] | dict[str, ...]], optional): 3D point positions
            for pointcloud visualization. Shape: (N, 3) for static scenes or (T, N, 3) for dynamic scenes.
            Can also be a dict mapping names to arrays, in which case each is logged under its own entity path
            (e.g., ``{"pred": pts_pred, "gt": pts_gt}``). Defaults to None.

        rgb (Optional[Tensor | np.ndarray | list[np.ndarray] | dict[str, ...]], optional): Colors for the
            pointcloud points. Shape should match points but with 3 color channels. Must be a dict when
            ``points`` is a dict, with matching keys. Defaults to None.

        images (Optional[List[str | np.ndarray | Tensor] | dict[str, List[str | np.ndarray | Tensor]]], optional):
            RGB images to display in camera views. Can be file paths, numpy arrays, or tensors.
            If cameras is a dict, this should also be a dict with matching keys. Defaults to None.

        depths (Optional[List[str | np.ndarray | Tensor] | dict[str, List[str | np.ndarray | Tensor]]], optional):
            Depth images corresponding to the RGB images. Can be file paths, numpy arrays, or tensors.
            If cameras is a dict, this should also be a dict with matching keys. Defaults to None.

        depth_scale_factor (float, optional): Scale factor to apply to depth values. Defaults to 1.0.

        server_address (Optional[str], optional): TCP address of Rerun server to connect to. Defaults to None.

        image_plane_distance (float | str, optional): Distance of image planes from camera centers
            for visualization. If "auto", automatically calculated based on scene size. Defaults to "auto".

        image_max_size (int, optional): Maximum size (width or height) for displayed images.
            If > 0, images will be resized if they exceed this size. Defaults to 0 (no resizing).

        save_path (Optional[str], optional): File path to save the Rerun recording.
            If provided, the visualization will be saved to this location. Defaults to None.

        view_coordinates (str, optional): View coordinates to use for the visualization.
            Defaults to "RFU" (Right-Forward-Up). See rerun.ViewCoordinates for available options.

        max_num_points (int, optional): Maximum number of points to visualize. Defaults to 100_000.

        app_id (Optional[str], optional): Rerun ``application_id`` to use when initializing
            the recording. If ``None`` (default), ``log_path`` is used, preserving the historical
            behavior. Set this to a method- or experiment-qualified name (e.g. ``"da3-G/dtu_scan1"``)
            to make multiple recordings of the same sequence distinguishable in the Rerun viewer's
            recordings sidebar.

    Note:
        - The function automatically handles different input formats (file paths vs arrays/tensors)
        - When using dict inputs, all dict parameters must have matching keys

    Example:
        ```python
        # Simple single-camera visualization
        visualize_scene(
            log_path="my_scene",
            cameras=camera_objects,
            points=pointcloud_positions,
            rgb=pointcloud_colors,
            images=["image1.jpg", "image2.jpg"],
            depths=["depth1.png", "depth2.png"]
        )

        # Multi-split visualization
        visualize_scene(
            log_path="training_data",
            cameras={"train": train_cams, "val": val_cams},
            images={"train": train_images, "val": val_images},
            save_path="scene_recording.rrd"
        )
        ```
    """
    rr.init(app_id if app_id is not None else log_path)
    if server_address is not None:
        if "://" in server_address:
            url = server_address
        else:
            url = f"rerun+http://{server_address}/proxy"
        rr.connect_grpc(url)
    if save_path is not None:
        if os.path.dirname(save_path) != "" and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        rr.save(save_path)
    rr.set_time_seconds("stable_time", 0)
    rr.log(log_path, getattr(rr.ViewCoordinates, view_coordinates), static=True)

    if image_plane_distance == "auto":
        first_points = next(iter(points.values())) if isinstance(points, dict) else points
        image_plane_distance = calculate_auto_image_plane_distance(cameras, first_points)

    def _add_cameras(cams, ims=None, deps=None, name="cameras"):
        for idx, camera in enumerate(cams):
            camera: Cameras
            width, height = int(camera.width), int(camera.height)
            if image_max_size > 0:
                scale_factor = image_max_size / max((width, height))
                if scale_factor < 1.0:
                    width = int(width * scale_factor)
                    height = int(height * scale_factor)
                    camera.rescale_output_resolution(scale_factor)

            intrinsic = camera.get_intrinsics_matrices().detach().cpu().numpy()
            pose = camera.camera_to_worlds.detach().cpu().numpy()
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Transform3D(
                    translation=pose[:3, 3],
                    quaternion=mat_to_quat(pose[:3, :3]),
                    from_parent=False,
                ),
            )
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Pinhole(
                    image_from_camera=intrinsic,
                    height=height,
                    width=width,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=image_plane_distance,
                ),
            )
            if ims is not None:
                # Handle path vs array/tensor
                if isinstance(ims[idx], (str, Path)):
                    img_data = read_image_cv2(ims[idx])
                    if img_data is None:
                        print(f"Skipping image at {ims[idx]} - could not be loaded")
                        continue
                else:
                    img_data = ims[idx]

                processed_img = normalize_image(img_data, height, width)

                if deps is not None:
                    # Handle path vs array/tensor
                    if isinstance(deps[idx], (str, Path)):
                        depth_data = read_depth(deps[idx], height, width, depth_scale_factor)
                    else:
                        depth_data = deps[idx]

                    depth = normalize_depth(depth_data, height, width, depth_scale_factor)
                    processed_img = overlay_depth_map(processed_img, depth)

                rr.log(
                    f"{log_path}/{name}/{idx}/rgb",
                    rr.Image(processed_img),
                )

    if isinstance(cameras, dict):
        if images is not None:
            assert isinstance(images, dict)
        if depths is not None:
            assert isinstance(depths, dict)
        for split, cams in cameras.items():
            ims = images.get(split) if images is not None else None
            deps = depths.get(split) if depths is not None else None
            _add_cameras(cams, ims, deps, name=split)
    else:
        _add_cameras(cameras, images, depths)

    def _log_pointcloud(pts, colors, entity_prefix, shared_indices=None):
        """Log a single pointcloud under entity_prefix.

        If ``shared_indices`` is provided it is used directly as the subsample,
        which lets paired pointclouds (e.g. ``pred`` and ``gt``) keep their
        per-row correspondence.
        """
        positions = pts.cpu().numpy() if isinstance(pts, Tensor) else pts
        col_np = colors.cpu().numpy() if isinstance(colors, Tensor) else colors

        if shared_indices is not None:
            positions = positions[shared_indices]
            col_np = col_np[shared_indices]
        elif len(positions) > max_num_points:
            indices = np.random.choice(len(positions), size=max_num_points, replace=False)
            positions = positions[indices]
            col_np = col_np[indices]

        rr.log(entity_prefix, rr.Points3D(positions=positions, colors=col_np))

    if points is not None:
        assert rgb is not None
        if isinstance(points, dict):
            assert isinstance(rgb, dict)
            shared_indices = _compute_shared_pointcloud_indices(points, max_num_points)
            for name, pts in points.items():
                _log_pointcloud(pts, rgb[name], f"{log_path}/{name}/pointcloud", shared_indices=shared_indices)
        else:
            _log_pointcloud(points, rgb, f"{log_path}/pointcloud")

    rr.disconnect()
