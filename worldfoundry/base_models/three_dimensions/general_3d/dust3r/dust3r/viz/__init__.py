"""Dust3R visualization helpers used by Dust3R, MASt3R, and MonST3R runtimes."""

__all__ = [
    "CAM_COLORS",
    "OPENGL",
    "PairViewer",
    "SceneViz",
    "add_scene_cam",
    "auto_cam_size",
    "cat_meshes",
    "pts3d_to_trimesh",
    "segment_sky",
    "to_numpy",
]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name == "PairViewer":
        from worldfoundry.base_models.three_dimensions.general_3d.dust3r.dust3r.cloud_opt.pair_viewer import PairViewer

        return PairViewer
    if name in {
        "CAM_COLORS",
        "OPENGL",
        "SceneViz",
        "add_scene_cam",
        "auto_cam_size",
        "cat_meshes",
        "pts3d_to_trimesh",
        "segment_sky",
        "to_numpy",
    }:
        from . import scene3d_dust

        return getattr(scene3d_dust, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
