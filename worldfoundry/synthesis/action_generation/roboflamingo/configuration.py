# Inference-only RoboFlamingo source retained in-tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoboFlamingoArchitectureConfig:
    """RoboFlamingo model architecture options.

    Args:
        llm_name: Official MPT/LLaMA variant key.
        vision_encoder_path: OpenCLIP vision encoder identifier.
        vision_encoder_pretrained: OpenCLIP pretrained weights identifier.
        cross_attn_every_n_layers: Flamingo cross-attention interval.
        window_size: Evaluation visual-history window size.
        fusion_mode: RGB/gripper fusion mode.
        use_gripper: Whether the policy consumes gripper-camera images.
        use_state: Whether low-dimensional robot state is consumed.
        decoder_type: Action decoder type.
        head_type: Action head family.
        precision: Runtime precision string.
    """

    llm_name: str
    vision_encoder_path: str
    vision_encoder_pretrained: str
    cross_attn_every_n_layers: int
    window_size: int
    fusion_mode: str
    use_gripper: bool
    use_state: bool
    decoder_type: str
    head_type: str
    n_obs_steps: int
    precision: str

    @property
    def signature(self) -> str:
        return "|".join(
            (
                self.llm_name,
                self.vision_encoder_path,
                self.vision_encoder_pretrained,
                str(self.cross_attn_every_n_layers),
                str(self.window_size),
                self.fusion_mode,
                str(self.use_gripper).lower(),
                str(self.use_state).lower(),
                self.decoder_type,
                self.head_type,
                str(self.n_obs_steps),
                self.precision,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_name": self.llm_name,
            "vision_encoder_path": self.vision_encoder_path,
            "vision_encoder_pretrained": self.vision_encoder_pretrained,
            "cross_attn_every_n_layers": self.cross_attn_every_n_layers,
            "window_size": self.window_size,
            "fusion_mode": self.fusion_mode,
            "use_gripper": self.use_gripper,
            "use_state": self.use_state,
            "decoder_type": self.decoder_type,
            "head_type": self.head_type,
            "n_obs_steps": self.n_obs_steps,
            "precision": self.precision,
        }


def action_trace_contract(config: RoboFlamingoArchitectureConfig) -> dict[str, Any]:
    """Describe the in-tree RoboFlamingo action-head contract.

    Args:
        config: Resolved RoboFlamingo architecture options.
    """

    return {
        "policy_family": "flamingo_vlm_robot_policy",
        "vlm_backbone": {
            "language_model": config.llm_name,
            "vision_encoder": config.vision_encoder_path,
            "vision_weights": config.vision_encoder_pretrained,
            "cross_attention_interval": config.cross_attn_every_n_layers,
        },
        "observation_layout": {
            "visual_history_window": config.window_size,
            "fusion_mode": config.fusion_mode,
            "uses_gripper_camera": config.use_gripper,
            "uses_robot_state": config.use_state,
            "diffusion_history_steps": config.n_obs_steps,
        },
        "action_head": {
            "decoder_type": config.decoder_type,
            "head_type": config.head_type,
            "action_dimensions": 7,
            "pose_dimensions": 6,
            "gripper_dimensions": 1,
        },
    }
