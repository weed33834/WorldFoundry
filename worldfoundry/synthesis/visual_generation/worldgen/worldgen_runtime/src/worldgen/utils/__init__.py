import importlib

from .general_utils import (
    pano_to_cube,
    cube_to_pano,
    resize_img,
    resize_img_and_rays,
    pano_unit_rays,
    batch_nearest_dot,
    fill_mask_from_contour,
    map_image_to_pano,
    depth_match
)

_SPLAT_EXPORTS = {
    "SplatFile",
    "convert_rgbd_to_gs",
    "mask_splat",
    "merge_splats",
}


def __getattr__(name: str):
    if name in _SPLAT_EXPORTS:
        module = importlib.import_module(f"{__name__}.splat_utils")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "pano_to_cube",
    "cube_to_pano",
    "resize_img",
    "resize_img_and_rays",
    "pano_unit_rays",
    "batch_nearest_dot",
    "fill_mask_from_contour",
    "map_image_to_pano",
    "depth_match",

    "SplatFile",
    "convert_rgbd_to_gs",
    "mask_splat",
    "merge_splats"
]
