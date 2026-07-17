"""
camera_controller.py — Real-time incremental camera pose controller
===================================================================

Maps WASD (translation) + arrow keys (rotation) or web vector input
to incremental updates of the world-to-camera (w2c) 4x4 matrix.

Camera coordinate convention (consistent with GEN3C/camera_utils.py
look_at_matrix):
  - World Y axis points up
  - Camera Z axis = look direction (forward)
  - Camera X axis = right
  - Camera Y axis = up
  - w2c = [R | t], where t = -R @ C_world

Keyboard mapping:
  W           -> Forward  (Zoom In  / translate along camera +Z)
  S           -> Backward (Zoom Out / translate along camera -Z)
  A           -> Strafe Left  (translate along camera -X)
  D           -> Strafe Right (translate along camera +X)
  ArrowUp     -> Pitch Up   (rotate around camera X axis)
  ArrowDown   -> Pitch Down (rotate around camera X axis)
  ArrowLeft   -> Yaw Left   (rotate around world Y axis)
  ArrowRight  -> Yaw Right  (rotate around world Y axis)

Rotation mode: GLOBAL coordinate rotation.
  Yaw is always applied around the WORLD up axis (Y), regardless of
  the current pitch angle. Pitch is applied around the camera's local
  X axis (which lies in the horizontal plane when yaw is the only
  other rotation). This prevents the disorienting "roll" effect that
  occurs with local-axis rotation when looking up/down and then
  turning left/right.

  Implementation: cumulative Euler angles (yaw, pitch) are tracked
  internally. The rotation matrix is rebuilt as R = R_pitch @ R_yaw,
  ensuring yaw is always around the world Y axis.

Derivation:
  Let C be the camera world position, then t = -R @ C.
  Translation in camera local frame by d_cam:
    C_new = C + R^T @ d_cam
    t_new = -R @ C_new = t - d_cam      (simply subtract d_cam)
  Rotation (global Euler angles):
    R_new = R_pitch(total_pitch) @ R_yaw(total_yaw)
    t_new = -R_new @ C                  (camera position unchanged)
"""

import math
from typing import Set

import torch


class CameraController:
    """
    Incrementally updates the world-to-camera (w2c) matrix from input.

    Uses GLOBAL coordinate rotation: yaw is always around the world Y
    axis, preventing roll artifacts when pitching and then yawing.

    Attributes:
        w2c          : [4, 4] float32 world-to-camera transform (on device)
        intrinsics   : [3, 3] float32 camera intrinsics in pixels (on device)
        translation_step : translation step per update (world units)
        rotation_step_deg: rotation step per update (degrees)
    """

    # All valid key names
    VALID_KEYS = frozenset([
        'w', 's', 'a', 'd',
        'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    ])

    # Maximum pitch angle to prevent gimbal lock (degrees)
    MAX_PITCH = 89.0

    def __init__(
        self,
        initial_w2c: torch.Tensor,       # [4, 4] initial world-to-camera matrix
        intrinsics: torch.Tensor,         # [3, 3] camera intrinsics (pixels)
        translation_step: float = 0.05,   # translation step per tick (world units)
        rotation_step_deg: float = 3.0,   # rotation step per tick (degrees)
        device: str = "cuda",
    ):
        self.w2c = initial_w2c.float().clone().to(device)
        self.intrinsics = intrinsics.float().clone().to(device)
        self.translation_step = translation_step
        self.rotation_step_deg = rotation_step_deg
        self.device = device

        # Extract initial Euler angles from the w2c rotation matrix.
        # R = R_pitch @ R_yaw, so:
        #   pitch = asin(R[2,1])
        #   yaw   = atan2(R[2,0], R[2,2])
        R = self.w2c[:3, :3]
        self._total_pitch = math.degrees(math.asin(
            float(torch.clamp(R[2, 1], -1.0, 1.0))
        ))
        self._total_yaw = math.degrees(math.atan2(
            float(R[2, 0]), float(R[2, 2])
        ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, keys: Set[str]) -> torch.Tensor:
        """
        Update camera pose from the set of currently pressed keys.

        Args:
            keys: Set of currently pressed key names.
                  Valid values: 'w', 's', 'a', 'd',
                  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'

        Returns:
            Updated self.w2c, shape [4, 4].
        """
        s = self.translation_step

        # -- Translation (in camera local frame) --
        if 'w' in keys:
            self._translate_cam([0.0,  0.0,  s])   # Forward (Zoom In)
        if 's' in keys:
            self._translate_cam([0.0,  0.0, -s])   # Backward (Zoom Out)
        if 'a' in keys:
            self._translate_cam([-s,   0.0,  0.0]) # Strafe Left
        if 'd' in keys:
            self._translate_cam([ s,   0.0,  0.0]) # Strafe Right

        # -- Rotation (global Euler angles) --
        r = self.rotation_step_deg
        if 'ArrowUp'    in keys:
            self._rotate_pitch(-r)  # Pitch up: look toward +Y
        if 'ArrowDown'  in keys:
            self._rotate_pitch( r)  # Pitch down: look toward -Y
        if 'ArrowLeft'  in keys:
            self._rotate_yaw(-r)    # Yaw left
        if 'ArrowRight' in keys:
            self._rotate_yaw( r)    # Yaw right

        return self.w2c

    def update_from_vectors(
        self,
        move_local: list,
        look_delta: list,
    ) -> torch.Tensor:
        """
        Update camera pose from local translation and look delta vectors.
        Used by the web frontend vector control interface.

        Difference from update(keys):
          - update(keys) uses fixed step sizes (translation_step /
            rotation_step_deg), suitable for per-key-tick movement.
          - update_from_vectors uses the provided values directly,
            suitable for frontend control with arbitrary granularity.

        Args:
            move_local: [x, y, z] translation in camera local frame
                        (world units, same sign convention as update):
                        x>0 strafe right, z>0 forward, y>0 up.
            look_delta: [yaw_rad, pitch_rad] rotation deltas (radians):
                        yaw>0 turn right, pitch>0 look up.

        Returns:
            Updated self.w2c, shape [4, 4].
        """
        # Translation
        if move_local[0] != 0 or move_local[1] != 0 or move_local[2] != 0:
            self._translate_cam(move_local)

        # Rotation (global Euler angles)
        if len(look_delta) >= 2:
            if look_delta[0] != 0:
                self._rotate_yaw(math.degrees(look_delta[0]))
            if look_delta[1] != 0:
                # pitch>0 = look up -> _rotate_pitch takes negative value
                self._rotate_pitch(-math.degrees(look_delta[1]))

        return self.w2c

    def get_w2c_batch(self) -> torch.Tensor:
        """Return w2c as [1, 1, 4, 4] (for Cache3D.render_cache)."""
        return self.w2c.unsqueeze(0).unsqueeze(0)   # [1, 1, 4, 4]

    def get_intrinsics_batch(self) -> torch.Tensor:
        """Return intrinsics as [1, 1, 3, 3] (for Cache3D.render_cache)."""
        return self.intrinsics.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    def reset(self, initial_w2c: torch.Tensor) -> None:
        """Reset camera pose to the specified initial matrix."""
        self.w2c = initial_w2c.float().clone().to(self.device)
        # Re-extract Euler angles from the reset matrix
        R = self.w2c[:3, :3]
        self._total_pitch = math.degrees(math.asin(
            float(torch.clamp(R[2, 1], -1.0, 1.0))
        ))
        self._total_yaw = math.degrees(math.atan2(
            float(R[2, 0]), float(R[2, 2])
        ))

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _translate_cam(self, d_cam):
        """
        Translate camera in its local coordinate frame.

        Derivation:
          t_new = t - d_cam
          (equivalent to moving camera world position along R^T @ d_cam)
        """
        d = torch.tensor(d_cam, dtype=self.w2c.dtype, device=self.device)
        self.w2c[:3, 3] -= d

    def _rotate_pitch(self, degrees: float):
        """
        Rotate pitch (look up/down) using GLOBAL coordinate rotation.

        Updates the cumulative pitch angle and rebuilds the rotation
        matrix. Yaw remains around the world Y axis regardless of pitch.

        Args:
            degrees: positive -> pitch down (forward tilts toward -Y).
        """
        self._total_pitch += degrees
        # Clamp to prevent gimbal lock
        self._total_pitch = max(-self.MAX_PITCH, min(self.MAX_PITCH, self._total_pitch))
        self._rebuild_rotation()

    def _rotate_yaw(self, degrees: float):
        """
        Rotate yaw (turn left/right) using GLOBAL coordinate rotation.

        Updates the cumulative yaw angle and rebuilds the rotation
        matrix. Yaw is always around the world Y axis, preventing
        roll artifacts when pitched.

        Args:
            degrees: positive -> turn right (gaze shifts toward +X).
        """
        self._total_yaw += degrees
        self._rebuild_rotation()

    def _rebuild_rotation(self):
        """
        Rebuild the w2c rotation from cumulative Euler angles.

        Constructs R = R_pitch(total_pitch) @ R_yaw(total_yaw),
        ensuring yaw is applied around the WORLD Y axis first,
        then pitch around the local X axis.

        Camera world position is preserved; only the translation
        vector is recomputed as t = -R_new @ C.
        """
        theta_yaw = math.radians(self._total_yaw)
        theta_pitch = math.radians(self._total_pitch)
        cy, sy = math.cos(theta_yaw), math.sin(theta_yaw)
        cp, sp = math.cos(theta_pitch), math.sin(theta_pitch)

        # R = R_pitch @ R_yaw
        #   R_yaw = [[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]]
        #   R_pitch = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
        #   R = [[cy, 0, -sy], [-sp*sy, cp, -sp*cy], [cp*sy, sp, cp*cy]]
        R_new = torch.tensor([
            [ cy,    0.0,  -sy,   0.0],
            [-sp*sy,  cp,  -sp*cy, 0.0],
            [ cp*sy,  sp,   cp*cy, 0.0],
            [ 0.0,   0.0,   0.0,   1.0],
        ], dtype=self.w2c.dtype, device=self.device)

        # Preserve camera world position: C = -R_old^T @ t_old
        R_old = self.w2c[:3, :3]
        t_old = self.w2c[:3, 3]
        C = -R_old.T @ t_old  # camera position in world

        # Update rotation and recompute translation: t = -R_new @ C
        self.w2c[:3, :3] = R_new[:3, :3]
        self.w2c[:3, 3] = -R_new[:3, :3] @ C


# ---------------------------------------------------------------------------
# Convenience function: construct CameraController from MoGe output
# ---------------------------------------------------------------------------

def make_camera_controller_from_moge(
    moge_initial_w2c_b144: torch.Tensor,   # [1, 1, 4, 4]
    moge_intrinsics_b133: torch.Tensor,    # [1, 1, 3, 3]
    translation_step: float = 0.05,
    rotation_step_deg: float = 3.0,
    device: str = "cuda",
) -> CameraController:
    """
    Construct a CameraController from MoGe depth estimation output.

    Args:
        moge_initial_w2c_b144: initial w2c from MoGe, shape [1, 1, 4, 4].
        moge_intrinsics_b133 : intrinsics from MoGe, shape [1, 1, 3, 3].
        translation_step     : translation step per tick (world units, default 0.05).
        rotation_step_deg    : rotation step per tick (degrees, default 3.0).
        device               : target device (default "cuda").

    Returns:
        CameraController instance.
    """
    initial_w2c = moge_initial_w2c_b144[0, 0]   # [4, 4]
    intrinsics   = moge_intrinsics_b133[0, 0]    # [3, 3]
    return CameraController(
        initial_w2c=initial_w2c,
        intrinsics=intrinsics,
        translation_step=translation_step,
        rotation_step_deg=rotation_step_deg,
        device=device,
    )
