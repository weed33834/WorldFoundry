# Auto-generated implementation redirecting to numpy/torch implementations
"""Module for base_models -> three_dimensions -> general_3d -> eastern_journalist -> utils3d -> interface -> __init__.py functionality."""

import sys
from typing import TYPE_CHECKING
import utils3d
from ..helpers import suppress_traceback

__all__ = ["sliding_window", 
"pooling", 
"max_pool_2d", 
"lookup", 
"lookup_get", 
"lookup_set", 
"segment_roll", 
"segment_take", 
"segment_concatenate", 
"csr_matrix_from_dense_indices", 
"group", 
"group_as_segments", 
"perspective_from_fov", 
"perspective_from_window", 
"intrinsics_from_fov", 
"intrinsics_from_focal_center", 
"fov_to_focal", 
"focal_to_fov", 
"intrinsics_to_fov", 
"view_look_at", 
"extrinsics_look_at", 
"perspective_to_intrinsics", 
"perspective_to_near_far", 
"intrinsics_to_perspective", 
"extrinsics_to_view", 
"view_to_extrinsics", 
"normalize_intrinsics", 
"denormalize_intrinsics", 
"crop_intrinsics", 
"pixel_to_uv", 
"pixel_to_ndc", 
"uv_to_pixel", 
"depth_linear_to_buffer", 
"depth_buffer_to_linear", 
"unproject_cv", 
"unproject_gl", 
"project_cv", 
"project_gl", 
"project", 
"unproject", 
"screen_coord_to_view_coord", 
"quaternion_to_matrix", 
"axis_angle_to_matrix", 
"matrix_to_quaternion", 
"extrinsics_to_essential", 
"axis_angle_to_quaternion", 
"euler_axis_angle_rotation", 
"euler_angles_to_matrix", 
"matrix_to_axis_angle", 
"matrix_to_euler_angles", 
"quaternion_to_axis_angle", 
"skew_symmetric", 
"rotation_matrix_from_vectors", 
"ray_intersection", 
"make_affine_matrix", 
"random_rotation_matrix", 
"lerp", 
"slerp", 
"slerp_rotation_matrix", 
"interpolate_se3_matrix", 
"piecewise_lerp", 
"piecewise_interpolate_se3_matrix", 
"transform_points", 
"angle_between", 
"vector_outer", 
"procrustes", 
"solve_pose", 
"solve_poses_sequential", 
"triangulate_mesh", 
"compute_face_corner_angles", 
"compute_face_corner_normals", 
"compute_face_corner_tangents", 
"compute_face_normals", 
"compute_face_tangents", 
"compute_vertex_normals", 
"remove_corrupted_faces", 
"merge_duplicate_vertices", 
"remove_unused_vertices", 
"subdivide_mesh", 
"mesh_edges", 
"mesh_half_edges", 
"mesh_connected_components", 
"graph_connected_components", 
"mesh_adjacency_graph", 
"flatten_mesh_indices", 
"create_cube_mesh", 
"create_icosahedron_mesh", 
"create_square_mesh", 
"create_camera_frustum_mesh", 
"merge_meshes", 
"uv_map", 
"pixel_coord_map", 
"screen_coord_map", 
"build_mesh_from_map", 
"build_mesh_from_depth_map", 
"depth_map_edge", 
"depth_map_aliasing", 
"normal_map_edge", 
"point_map_to_normal_map", 
"depth_map_to_point_map", 
"depth_map_to_normal_map", 
"chessboard", 
"masked_nearest_resize", 
"masked_area_resize", 
"colorize_depth_map", 
"colorize_normal_map", 
"flood_fill", 
"perlin_noise", 
"perlin_noise_map", 
"fractal_perlin_noise_map", 
"RastContext", 
"rasterize_triangles", 
"rasterize_triangles_peeling", 
"rasterize_lines", 
"rasterize_point_cloud", 
"sample_texture", 
"test_rasterization", 
"read_extrinsics_from_colmap", 
"read_intrinsics_from_colmap", 
"write_extrinsics_as_colmap", 
"write_intrinsics_as_colmap", 
"read_obj", 
"write_obj", 
"masked_min", 
"masked_max", 
"csr_eliminate_zeros", 
"rotation_matrix_2d", 
"rotate_2d", 
"translate_2d", 
"scale_2d", 
"mesh_dual_graph", 
"compute_boundaries", 
"remove_isolated_pieces", 
"compute_mesh_laplacian", 
"laplacian_smooth_mesh", 
"taubin_smooth_mesh", 
"laplacian_hc_smooth_mesh", 
"bounding_rect_from_mask", 
"texture_composite"]

def _contains_tensor(obj):
    """Helper function to contains tensor.

    Args:
        obj: The obj.
    """
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_tensor(value) for value in obj.values())
    else:
        import torch
        return isinstance(obj, torch.Tensor)


@suppress_traceback
def _call_based_on_args(fname, args, kwargs):
    """Helper function to call based on args.

    Args:
        fname: The fname.
        args: The args.
        kwargs: The kwargs.
    """
    if 'torch' in sys.modules:
        if any(_contains_tensor(arg) for arg in args) or any(_contains_tensor(v) for v in kwargs.values()):
            fn = getattr(utils3d.torch, fname, None)
            if fn is None:
                raise NotImplementedError(f"Function {fname} has no torch implementation.")
            return fn(*args, **kwargs)
    fn = getattr(utils3d.numpy, fname, None)
    if fn is None:
        raise NotImplementedError(f"Function {fname} has no numpy implementation.") 
    return fn(*args, **kwargs)


@suppress_traceback
def sliding_window(*args, **kwargs):
    """Sliding window."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sliding_window, utils3d.torch.sliding_window
    return _call_based_on_args('sliding_window', args, kwargs)

@suppress_traceback
def pooling(*args, **kwargs):
    """Pooling."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pooling, None
    return _call_based_on_args('pooling', args, kwargs)

@suppress_traceback
def max_pool_2d(*args, **kwargs):
    """Max pool 2d."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.max_pool_2d, None
    return _call_based_on_args('max_pool_2d', args, kwargs)

@suppress_traceback
def lookup(*args, **kwargs):
    """Lookup."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup, utils3d.torch.lookup
    return _call_based_on_args('lookup', args, kwargs)

@suppress_traceback
def lookup_get(*args, **kwargs):
    """Lookup get."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup_get, utils3d.torch.lookup_get
    return _call_based_on_args('lookup_get', args, kwargs)

@suppress_traceback
def lookup_set(*args, **kwargs):
    """Lookup set."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup_set, utils3d.torch.lookup_set
    return _call_based_on_args('lookup_set', args, kwargs)

@suppress_traceback
def segment_roll(*args, **kwargs):
    """Segment roll."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_roll, utils3d.torch.segment_roll
    return _call_based_on_args('segment_roll', args, kwargs)

@suppress_traceback
def segment_take(*args, **kwargs):
    """Segment take."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_take, utils3d.torch.segment_take
    return _call_based_on_args('segment_take', args, kwargs)

@suppress_traceback
def segment_concatenate(*args, **kwargs):
    """Segment concatenate."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_concatenate, None
    return _call_based_on_args('segment_concatenate', args, kwargs)

@suppress_traceback
def csr_matrix_from_dense_indices(*args, **kwargs):
    """Csr matrix from dense indices."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.csr_matrix_from_dense_indices, utils3d.torch.csr_matrix_from_dense_indices
    return _call_based_on_args('csr_matrix_from_dense_indices', args, kwargs)

@suppress_traceback
def group(*args, **kwargs):
    """Group."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.group, utils3d.torch.group
    return _call_based_on_args('group', args, kwargs)

@suppress_traceback
def group_as_segments(*args, **kwargs):
    """Group as segments."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.group_as_segments, utils3d.torch.group_as_segments
    return _call_based_on_args('group_as_segments', args, kwargs)

@suppress_traceback
def perspective_from_fov(*args, **kwargs):
    """Perspective from fov."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_fov, utils3d.torch.perspective_from_fov
    return _call_based_on_args('perspective_from_fov', args, kwargs)

@suppress_traceback
def perspective_from_window(*args, **kwargs):
    """Perspective from window."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_window, utils3d.torch.perspective_from_window
    return _call_based_on_args('perspective_from_window', args, kwargs)

@suppress_traceback
def intrinsics_from_fov(*args, **kwargs):
    """Intrinsics from fov."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_fov, utils3d.torch.intrinsics_from_fov
    return _call_based_on_args('intrinsics_from_fov', args, kwargs)

@suppress_traceback
def intrinsics_from_focal_center(*args, **kwargs):
    """Intrinsics from focal center."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_focal_center, utils3d.torch.intrinsics_from_focal_center
    return _call_based_on_args('intrinsics_from_focal_center', args, kwargs)

@suppress_traceback
def fov_to_focal(*args, **kwargs):
    """Fov to focal."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.fov_to_focal, utils3d.torch.fov_to_focal
    return _call_based_on_args('fov_to_focal', args, kwargs)

@suppress_traceback
def focal_to_fov(*args, **kwargs):
    """Focal to fov."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.focal_to_fov, utils3d.torch.focal_to_fov
    return _call_based_on_args('focal_to_fov', args, kwargs)

@suppress_traceback
def intrinsics_to_fov(*args, **kwargs):
    """Intrinsics to fov."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_fov, utils3d.torch.intrinsics_to_fov
    return _call_based_on_args('intrinsics_to_fov', args, kwargs)

@suppress_traceback
def view_look_at(*args, **kwargs):
    """View look at."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_look_at, utils3d.torch.view_look_at
    return _call_based_on_args('view_look_at', args, kwargs)

@suppress_traceback
def extrinsics_look_at(*args, **kwargs):
    """Extrinsics look at."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_look_at, utils3d.torch.extrinsics_look_at
    return _call_based_on_args('extrinsics_look_at', args, kwargs)

@suppress_traceback
def perspective_to_intrinsics(*args, **kwargs):
    """Perspective to intrinsics."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_intrinsics, utils3d.torch.perspective_to_intrinsics
    return _call_based_on_args('perspective_to_intrinsics', args, kwargs)

@suppress_traceback
def perspective_to_near_far(*args, **kwargs):
    """Perspective to near far."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_near_far, None
    return _call_based_on_args('perspective_to_near_far', args, kwargs)

@suppress_traceback
def intrinsics_to_perspective(*args, **kwargs):
    """Intrinsics to perspective."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_perspective, utils3d.torch.intrinsics_to_perspective
    return _call_based_on_args('intrinsics_to_perspective', args, kwargs)

@suppress_traceback
def extrinsics_to_view(*args, **kwargs):
    """Extrinsics to view."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_view, utils3d.torch.extrinsics_to_view
    return _call_based_on_args('extrinsics_to_view', args, kwargs)

@suppress_traceback
def view_to_extrinsics(*args, **kwargs):
    """View to extrinsics."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_to_extrinsics, utils3d.torch.view_to_extrinsics
    return _call_based_on_args('view_to_extrinsics', args, kwargs)

@suppress_traceback
def normalize_intrinsics(*args, **kwargs):
    """Normalize intrinsics."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normalize_intrinsics, utils3d.torch.normalize_intrinsics
    return _call_based_on_args('normalize_intrinsics', args, kwargs)

@suppress_traceback
def denormalize_intrinsics(*args, **kwargs):
    """Denormalize intrinsics."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.denormalize_intrinsics, utils3d.torch.denormalize_intrinsics
    return _call_based_on_args('denormalize_intrinsics', args, kwargs)

@suppress_traceback
def crop_intrinsics(*args, **kwargs):
    """Crop intrinsics."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.crop_intrinsics, utils3d.torch.crop_intrinsics
    return _call_based_on_args('crop_intrinsics', args, kwargs)

@suppress_traceback
def pixel_to_uv(*args, **kwargs):
    """Pixel to uv."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_uv, utils3d.torch.pixel_to_uv
    return _call_based_on_args('pixel_to_uv', args, kwargs)

@suppress_traceback
def pixel_to_ndc(*args, **kwargs):
    """Pixel to ndc."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_ndc, utils3d.torch.pixel_to_ndc
    return _call_based_on_args('pixel_to_ndc', args, kwargs)

@suppress_traceback
def uv_to_pixel(*args, **kwargs):
    """Uv to pixel."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.uv_to_pixel, utils3d.torch.uv_to_pixel
    return _call_based_on_args('uv_to_pixel', args, kwargs)

@suppress_traceback
def depth_linear_to_buffer(*args, **kwargs):
    """Depth linear to buffer."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_linear_to_buffer, utils3d.torch.depth_linear_to_buffer
    return _call_based_on_args('depth_linear_to_buffer', args, kwargs)

@suppress_traceback
def depth_buffer_to_linear(*args, **kwargs):
    """Depth buffer to linear."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_buffer_to_linear, utils3d.torch.depth_buffer_to_linear
    return _call_based_on_args('depth_buffer_to_linear', args, kwargs)

@suppress_traceback
def unproject_cv(*args, **kwargs):
    """Unproject cv."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_cv, utils3d.torch.unproject_cv
    return _call_based_on_args('unproject_cv', args, kwargs)

@suppress_traceback
def unproject_gl(*args, **kwargs):
    """Unproject gl."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_gl, utils3d.torch.unproject_gl
    return _call_based_on_args('unproject_gl', args, kwargs)

@suppress_traceback
def project_cv(*args, **kwargs):
    """Project cv."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_cv, utils3d.torch.project_cv
    return _call_based_on_args('project_cv', args, kwargs)

@suppress_traceback
def project_gl(*args, **kwargs):
    """Project gl."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_gl, utils3d.torch.project_gl
    return _call_based_on_args('project_gl', args, kwargs)

@suppress_traceback
def project(*args, **kwargs):
    """Project."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project, utils3d.torch.project
    return _call_based_on_args('project', args, kwargs)

@suppress_traceback
def unproject(*args, **kwargs):
    """Unproject."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject, utils3d.torch.unproject
    return _call_based_on_args('unproject', args, kwargs)

@suppress_traceback
def screen_coord_to_view_coord(*args, **kwargs):
    """Screen coord to view coord."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.screen_coord_to_view_coord, None
    return _call_based_on_args('screen_coord_to_view_coord', args, kwargs)

@suppress_traceback
def quaternion_to_matrix(*args, **kwargs):
    """Quaternion to matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_to_matrix, utils3d.torch.quaternion_to_matrix
    return _call_based_on_args('quaternion_to_matrix', args, kwargs)

@suppress_traceback
def axis_angle_to_matrix(*args, **kwargs):
    """Axis angle to matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.axis_angle_to_matrix, utils3d.torch.axis_angle_to_matrix
    return _call_based_on_args('axis_angle_to_matrix', args, kwargs)

@suppress_traceback
def matrix_to_quaternion(*args, **kwargs):
    """Matrix to quaternion."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_quaternion, utils3d.torch.matrix_to_quaternion
    return _call_based_on_args('matrix_to_quaternion', args, kwargs)

@suppress_traceback
def extrinsics_to_essential(*args, **kwargs):
    """Extrinsics to essential."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_essential, utils3d.torch.extrinsics_to_essential
    return _call_based_on_args('extrinsics_to_essential', args, kwargs)

@suppress_traceback
def axis_angle_to_quaternion(*args, **kwargs):
    """Axis angle to quaternion."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.axis_angle_to_quaternion, utils3d.torch.axis_angle_to_quaternion
    return _call_based_on_args('axis_angle_to_quaternion', args, kwargs)

@suppress_traceback
def euler_axis_angle_rotation(*args, **kwargs):
    """Euler axis angle rotation."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_axis_angle_rotation, utils3d.torch.euler_axis_angle_rotation
    return _call_based_on_args('euler_axis_angle_rotation', args, kwargs)

@suppress_traceback
def euler_angles_to_matrix(*args, **kwargs):
    """Euler angles to matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_angles_to_matrix, utils3d.torch.euler_angles_to_matrix
    return _call_based_on_args('euler_angles_to_matrix', args, kwargs)

@suppress_traceback
def matrix_to_axis_angle(*args, **kwargs):
    """Matrix to axis angle."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_axis_angle, utils3d.torch.matrix_to_axis_angle
    return _call_based_on_args('matrix_to_axis_angle', args, kwargs)

@suppress_traceback
def matrix_to_euler_angles(*args, **kwargs):
    """Matrix to euler angles."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_euler_angles, utils3d.torch.matrix_to_euler_angles
    return _call_based_on_args('matrix_to_euler_angles', args, kwargs)

@suppress_traceback
def quaternion_to_axis_angle(*args, **kwargs):
    """Quaternion to axis angle."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_to_axis_angle, utils3d.torch.quaternion_to_axis_angle
    return _call_based_on_args('quaternion_to_axis_angle', args, kwargs)

@suppress_traceback
def skew_symmetric(*args, **kwargs):
    """Skew symmetric."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.skew_symmetric, utils3d.torch.skew_symmetric
    return _call_based_on_args('skew_symmetric', args, kwargs)

@suppress_traceback
def rotation_matrix_from_vectors(*args, **kwargs):
    """Rotation matrix from vectors."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rotation_matrix_from_vectors, utils3d.torch.rotation_matrix_from_vectors
    return _call_based_on_args('rotation_matrix_from_vectors', args, kwargs)

@suppress_traceback
def ray_intersection(*args, **kwargs):
    """Ray intersection."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.ray_intersection, None
    return _call_based_on_args('ray_intersection', args, kwargs)

@suppress_traceback
def make_affine_matrix(*args, **kwargs):
    """Make affine matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.make_affine_matrix, utils3d.torch.make_affine_matrix
    return _call_based_on_args('make_affine_matrix', args, kwargs)

@suppress_traceback
def random_rotation_matrix(*args, **kwargs):
    """Random rotation matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.random_rotation_matrix, utils3d.torch.random_rotation_matrix
    return _call_based_on_args('random_rotation_matrix', args, kwargs)

@suppress_traceback
def lerp(*args, **kwargs):
    """Lerp."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lerp, utils3d.torch.lerp
    return _call_based_on_args('lerp', args, kwargs)

@suppress_traceback
def slerp(*args, **kwargs):
    """Slerp."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp, utils3d.torch.slerp
    return _call_based_on_args('slerp', args, kwargs)

@suppress_traceback
def slerp_rotation_matrix(*args, **kwargs):
    """Slerp rotation matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp_rotation_matrix, utils3d.torch.slerp_rotation_matrix
    return _call_based_on_args('slerp_rotation_matrix', args, kwargs)

@suppress_traceback
def interpolate_se3_matrix(*args, **kwargs):
    """Interpolate se3 matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.interpolate_se3_matrix, utils3d.torch.interpolate_se3_matrix
    return _call_based_on_args('interpolate_se3_matrix', args, kwargs)

@suppress_traceback
def piecewise_lerp(*args, **kwargs):
    """Piecewise lerp."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_lerp, None
    return _call_based_on_args('piecewise_lerp', args, kwargs)

@suppress_traceback
def piecewise_interpolate_se3_matrix(*args, **kwargs):
    """Piecewise interpolate se3 matrix."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_interpolate_se3_matrix, None
    return _call_based_on_args('piecewise_interpolate_se3_matrix', args, kwargs)

@suppress_traceback
def transform_points(*args, **kwargs):
    """Transform points."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.transform_points, utils3d.torch.transform_points
    return _call_based_on_args('transform_points', args, kwargs)

@suppress_traceback
def angle_between(*args, **kwargs):
    """Angle between."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.angle_between, utils3d.torch.angle_between
    return _call_based_on_args('angle_between', args, kwargs)

@suppress_traceback
def vector_outer(*args, **kwargs):
    """Vector outer."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.vector_outer, None
    return _call_based_on_args('vector_outer', args, kwargs)

@suppress_traceback
def procrustes(*args, **kwargs):
    """Procrustes."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.procrustes, None
    return _call_based_on_args('procrustes', args, kwargs)

@suppress_traceback
def solve_pose(*args, **kwargs):
    """Solve pose."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.solve_pose, None
    return _call_based_on_args('solve_pose', args, kwargs)

@suppress_traceback
def solve_poses_sequential(*args, **kwargs):
    """Solve poses sequential."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.solve_poses_sequential, None
    return _call_based_on_args('solve_poses_sequential', args, kwargs)

@suppress_traceback
def triangulate_mesh(*args, **kwargs):
    """Triangulate mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.triangulate_mesh, utils3d.torch.triangulate_mesh
    return _call_based_on_args('triangulate_mesh', args, kwargs)

@suppress_traceback
def compute_face_corner_angles(*args, **kwargs):
    """Compute face corner angles."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_angles, utils3d.torch.compute_face_corner_angles
    return _call_based_on_args('compute_face_corner_angles', args, kwargs)

@suppress_traceback
def compute_face_corner_normals(*args, **kwargs):
    """Compute face corner normals."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_normals, utils3d.torch.compute_face_corner_normals
    return _call_based_on_args('compute_face_corner_normals', args, kwargs)

@suppress_traceback
def compute_face_corner_tangents(*args, **kwargs):
    """Compute face corner tangents."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_tangents, utils3d.torch.compute_face_corner_tangents
    return _call_based_on_args('compute_face_corner_tangents', args, kwargs)

@suppress_traceback
def compute_face_normals(*args, **kwargs):
    """Compute face normals."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_normals, utils3d.torch.compute_face_normals
    return _call_based_on_args('compute_face_normals', args, kwargs)

@suppress_traceback
def compute_face_tangents(*args, **kwargs):
    """Compute face tangents."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_tangents, utils3d.torch.compute_face_tangents
    return _call_based_on_args('compute_face_tangents', args, kwargs)

@suppress_traceback
def compute_vertex_normals(*args, **kwargs):
    """Compute vertex normals."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_vertex_normals, None
    return _call_based_on_args('compute_vertex_normals', args, kwargs)

@suppress_traceback
def remove_corrupted_faces(*args, **kwargs):
    """Remove corrupted faces."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_corrupted_faces, utils3d.torch.remove_corrupted_faces
    return _call_based_on_args('remove_corrupted_faces', args, kwargs)

@suppress_traceback
def merge_duplicate_vertices(*args, **kwargs):
    """Merge duplicate vertices."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.merge_duplicate_vertices, utils3d.torch.merge_duplicate_vertices
    return _call_based_on_args('merge_duplicate_vertices', args, kwargs)

@suppress_traceback
def remove_unused_vertices(*args, **kwargs):
    """Remove unused vertices."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_unused_vertices, utils3d.torch.remove_unused_vertices
    return _call_based_on_args('remove_unused_vertices', args, kwargs)

@suppress_traceback
def subdivide_mesh(*args, **kwargs):
    """Subdivide mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.subdivide_mesh, utils3d.torch.subdivide_mesh
    return _call_based_on_args('subdivide_mesh', args, kwargs)

@suppress_traceback
def mesh_edges(*args, **kwargs):
    """Mesh edges."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_edges, utils3d.torch.mesh_edges
    return _call_based_on_args('mesh_edges', args, kwargs)

@suppress_traceback
def mesh_half_edges(*args, **kwargs):
    """Mesh half edges."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_half_edges, utils3d.torch.mesh_half_edges
    return _call_based_on_args('mesh_half_edges', args, kwargs)

@suppress_traceback
def mesh_connected_components(*args, **kwargs):
    """Mesh connected components."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_connected_components, utils3d.torch.mesh_connected_components
    return _call_based_on_args('mesh_connected_components', args, kwargs)

@suppress_traceback
def graph_connected_components(*args, **kwargs):
    """Graph connected components."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.graph_connected_components, utils3d.torch.graph_connected_components
    return _call_based_on_args('graph_connected_components', args, kwargs)

@suppress_traceback
def mesh_adjacency_graph(*args, **kwargs):
    """Mesh adjacency graph."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_adjacency_graph, None
    return _call_based_on_args('mesh_adjacency_graph', args, kwargs)

@suppress_traceback
def flatten_mesh_indices(*args, **kwargs):
    """Flatten mesh indices."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.flatten_mesh_indices, None
    return _call_based_on_args('flatten_mesh_indices', args, kwargs)

@suppress_traceback
def create_cube_mesh(*args, **kwargs):
    """Create cube mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_cube_mesh, utils3d.torch.create_cube_mesh
    return _call_based_on_args('create_cube_mesh', args, kwargs)

@suppress_traceback
def create_icosahedron_mesh(*args, **kwargs):
    """Create icosahedron mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_icosahedron_mesh, utils3d.torch.create_icosahedron_mesh
    return _call_based_on_args('create_icosahedron_mesh', args, kwargs)

@suppress_traceback
def create_square_mesh(*args, **kwargs):
    """Create square mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_square_mesh, None
    return _call_based_on_args('create_square_mesh', args, kwargs)

@suppress_traceback
def create_camera_frustum_mesh(*args, **kwargs):
    """Create camera frustum mesh."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_camera_frustum_mesh, utils3d.torch.create_camera_frustum_mesh
    return _call_based_on_args('create_camera_frustum_mesh', args, kwargs)

@suppress_traceback
def merge_meshes(*args, **kwargs):
    """Merge meshes."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.merge_meshes, None
    return _call_based_on_args('merge_meshes', args, kwargs)

@suppress_traceback
def uv_map(*args, **kwargs):
    """Uv map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.uv_map, utils3d.torch.uv_map
    return _call_based_on_args('uv_map', args, kwargs)

@suppress_traceback
def pixel_coord_map(*args, **kwargs):
    """Pixel coord map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_coord_map, utils3d.torch.pixel_coord_map
    return _call_based_on_args('pixel_coord_map', args, kwargs)

@suppress_traceback
def screen_coord_map(*args, **kwargs):
    """Screen coord map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.screen_coord_map, utils3d.torch.screen_coord_map
    return _call_based_on_args('screen_coord_map', args, kwargs)

@suppress_traceback
def build_mesh_from_map(*args, **kwargs):
    """Build mesh from map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.build_mesh_from_map, utils3d.torch.build_mesh_from_map
    return _call_based_on_args('build_mesh_from_map', args, kwargs)

@suppress_traceback
def build_mesh_from_depth_map(*args, **kwargs):
    """Build mesh from depth map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.build_mesh_from_depth_map, utils3d.torch.build_mesh_from_depth_map
    return _call_based_on_args('build_mesh_from_depth_map', args, kwargs)

@suppress_traceback
def depth_map_edge(*args, **kwargs):
    """Depth map edge."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_edge, utils3d.torch.depth_map_edge
    return _call_based_on_args('depth_map_edge', args, kwargs)

@suppress_traceback
def depth_map_aliasing(*args, **kwargs):
    """Depth map aliasing."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_aliasing, utils3d.torch.depth_map_aliasing
    return _call_based_on_args('depth_map_aliasing', args, kwargs)

@suppress_traceback
def normal_map_edge(*args, **kwargs):
    """Normal map edge."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normal_map_edge, None
    return _call_based_on_args('normal_map_edge', args, kwargs)

@suppress_traceback
def point_map_to_normal_map(*args, **kwargs):
    """Point map to normal map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.point_map_to_normal_map, utils3d.torch.point_map_to_normal_map
    return _call_based_on_args('point_map_to_normal_map', args, kwargs)

@suppress_traceback
def depth_map_to_point_map(*args, **kwargs):
    """Depth map to point map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_to_point_map, utils3d.torch.depth_map_to_point_map
    return _call_based_on_args('depth_map_to_point_map', args, kwargs)

@suppress_traceback
def depth_map_to_normal_map(*args, **kwargs):
    """Depth map to normal map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_to_normal_map, utils3d.torch.depth_map_to_normal_map
    return _call_based_on_args('depth_map_to_normal_map', args, kwargs)

@suppress_traceback
def chessboard(*args, **kwargs):
    """Chessboard."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.chessboard, utils3d.torch.chessboard
    return _call_based_on_args('chessboard', args, kwargs)

@suppress_traceback
def masked_nearest_resize(*args, **kwargs):
    """Masked nearest resize."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.masked_nearest_resize, utils3d.torch.masked_nearest_resize
    return _call_based_on_args('masked_nearest_resize', args, kwargs)

@suppress_traceback
def masked_area_resize(*args, **kwargs):
    """Masked area resize."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.masked_area_resize, utils3d.torch.masked_area_resize
    return _call_based_on_args('masked_area_resize', args, kwargs)

@suppress_traceback
def colorize_depth_map(*args, **kwargs):
    """Colorize depth map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_depth_map, None
    return _call_based_on_args('colorize_depth_map', args, kwargs)

@suppress_traceback
def colorize_normal_map(*args, **kwargs):
    """Colorize normal map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_normal_map, None
    return _call_based_on_args('colorize_normal_map', args, kwargs)

@suppress_traceback
def flood_fill(*args, **kwargs):
    """Flood fill."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.flood_fill, utils3d.torch.flood_fill
    return _call_based_on_args('flood_fill', args, kwargs)

@suppress_traceback
def perlin_noise(*args, **kwargs):
    """Perlin noise."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perlin_noise, utils3d.torch.perlin_noise
    return _call_based_on_args('perlin_noise', args, kwargs)

@suppress_traceback
def perlin_noise_map(*args, **kwargs):
    """Perlin noise map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perlin_noise_map, utils3d.torch.perlin_noise_map
    return _call_based_on_args('perlin_noise_map', args, kwargs)

@suppress_traceback
def fractal_perlin_noise_map(*args, **kwargs):
    """Fractal perlin noise map."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.fractal_perlin_noise_map, utils3d.torch.fractal_perlin_noise_map
    return _call_based_on_args('fractal_perlin_noise_map', args, kwargs)

@suppress_traceback
def RastContext(*args, **kwargs):
    """Rastcontext."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.RastContext, utils3d.torch.RastContext
    return _call_based_on_args('RastContext', args, kwargs)

@suppress_traceback
def rasterize_triangles(*args, **kwargs):
    """Rasterize triangles."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_triangles, utils3d.torch.rasterize_triangles
    return _call_based_on_args('rasterize_triangles', args, kwargs)

@suppress_traceback
def rasterize_triangles_peeling(*args, **kwargs):
    """Rasterize triangles peeling."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_triangles_peeling, utils3d.torch.rasterize_triangles_peeling
    return _call_based_on_args('rasterize_triangles_peeling', args, kwargs)

@suppress_traceback
def rasterize_lines(*args, **kwargs):
    """Rasterize lines."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_lines, None
    return _call_based_on_args('rasterize_lines', args, kwargs)

@suppress_traceback
def rasterize_point_cloud(*args, **kwargs):
    """Rasterize point cloud."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_point_cloud, None
    return _call_based_on_args('rasterize_point_cloud', args, kwargs)

@suppress_traceback
def sample_texture(*args, **kwargs):
    """Sample texture."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sample_texture, utils3d.torch.sample_texture
    return _call_based_on_args('sample_texture', args, kwargs)

@suppress_traceback
def test_rasterization(*args, **kwargs):
    """Test rasterization."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.test_rasterization, None
    return _call_based_on_args('test_rasterization', args, kwargs)

@suppress_traceback
def read_extrinsics_from_colmap(*args, **kwargs):
    """Read extrinsics from colmap."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_extrinsics_from_colmap, None
    return _call_based_on_args('read_extrinsics_from_colmap', args, kwargs)

@suppress_traceback
def read_intrinsics_from_colmap(*args, **kwargs):
    """Read intrinsics from colmap."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_intrinsics_from_colmap, None
    return _call_based_on_args('read_intrinsics_from_colmap', args, kwargs)

@suppress_traceback
def write_extrinsics_as_colmap(*args, **kwargs):
    """Write extrinsics as colmap."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_extrinsics_as_colmap, None
    return _call_based_on_args('write_extrinsics_as_colmap', args, kwargs)

@suppress_traceback
def write_intrinsics_as_colmap(*args, **kwargs):
    """Write intrinsics as colmap."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_intrinsics_as_colmap, None
    return _call_based_on_args('write_intrinsics_as_colmap', args, kwargs)

@suppress_traceback
def read_obj(*args, **kwargs):
    """Read obj."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_obj, None
    return _call_based_on_args('read_obj', args, kwargs)

@suppress_traceback
def write_obj(*args, **kwargs):
    """Write obj."""
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_obj, None
    return _call_based_on_args('write_obj', args, kwargs)

@suppress_traceback
def masked_min(*args, **kwargs):
    """Masked min."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_min
    return _call_based_on_args('masked_min', args, kwargs)

@suppress_traceback
def masked_max(*args, **kwargs):
    """Masked max."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_max
    return _call_based_on_args('masked_max', args, kwargs)

@suppress_traceback
def csr_eliminate_zeros(*args, **kwargs):
    """Csr eliminate zeros."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.csr_eliminate_zeros
    return _call_based_on_args('csr_eliminate_zeros', args, kwargs)

@suppress_traceback
def rotation_matrix_2d(*args, **kwargs):
    """Rotation matrix 2d."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotation_matrix_2d
    return _call_based_on_args('rotation_matrix_2d', args, kwargs)

@suppress_traceback
def rotate_2d(*args, **kwargs):
    """Rotate 2d."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotate_2d
    return _call_based_on_args('rotate_2d', args, kwargs)

@suppress_traceback
def translate_2d(*args, **kwargs):
    """Translate 2d."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.translate_2d
    return _call_based_on_args('translate_2d', args, kwargs)

@suppress_traceback
def scale_2d(*args, **kwargs):
    """Scale 2d."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.scale_2d
    return _call_based_on_args('scale_2d', args, kwargs)

@suppress_traceback
def mesh_dual_graph(*args, **kwargs):
    """Mesh dual graph."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.mesh_dual_graph
    return _call_based_on_args('mesh_dual_graph', args, kwargs)

@suppress_traceback
def compute_boundaries(*args, **kwargs):
    """Compute boundaries."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_boundaries
    return _call_based_on_args('compute_boundaries', args, kwargs)

@suppress_traceback
def remove_isolated_pieces(*args, **kwargs):
    """Remove isolated pieces."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.remove_isolated_pieces
    return _call_based_on_args('remove_isolated_pieces', args, kwargs)

@suppress_traceback
def compute_mesh_laplacian(*args, **kwargs):
    """Compute mesh laplacian."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_mesh_laplacian
    return _call_based_on_args('compute_mesh_laplacian', args, kwargs)

@suppress_traceback
def laplacian_smooth_mesh(*args, **kwargs):
    """Laplacian smooth mesh."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_smooth_mesh
    return _call_based_on_args('laplacian_smooth_mesh', args, kwargs)

@suppress_traceback
def taubin_smooth_mesh(*args, **kwargs):
    """Taubin smooth mesh."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.taubin_smooth_mesh
    return _call_based_on_args('taubin_smooth_mesh', args, kwargs)

@suppress_traceback
def laplacian_hc_smooth_mesh(*args, **kwargs):
    """Laplacian hc smooth mesh."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_hc_smooth_mesh
    return _call_based_on_args('laplacian_hc_smooth_mesh', args, kwargs)

@suppress_traceback
def bounding_rect_from_mask(*args, **kwargs):
    """Bounding rect from mask."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.bounding_rect_from_mask
    return _call_based_on_args('bounding_rect_from_mask', args, kwargs)

@suppress_traceback
def texture_composite(*args, **kwargs):
    """Texture composite."""
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.texture_composite
    return _call_based_on_args('texture_composite', args, kwargs)
