from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OFFICIAL_SOURCE = {
    "repo_url": "https://github.com/RoboFlamingo/RoboFlamingo",
    "reference_checkout": "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/RoboFlamingo--RoboFlamingo during porting",
    "source_files": [
        "robot_flamingo/models/factory.py",
        "robot_flamingo/models/flamingo_bc.py",
        "robot_flamingo/models/flamingo_mpt.py",
        "robot_flamingo/models/action_head.py",
        "robot_flamingo/eval/eval_utils.py",
    ],
    "shared_base_models": [
        "worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo",
    ],
}


MPT_DOLLY_3B_DEFAULTS = {
    "llm_name": "mpt_dolly_3b",
    "cross_attn_every_n_layers": 1,
    "vision_encoder_path": "ViT-L-14",
    "vision_encoder_pretrained": "openai",
    "window_size": 12,
    "fusion_mode": "post",
    "use_gripper": True,
    "use_state": False,
    "decoder_type": "lstm",
    "head_type": "lstm",
    "precision": "fp32",
}


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

    llm_name: str = "mpt_dolly_3b"
    vision_encoder_path: str = "ViT-L-14"
    vision_encoder_pretrained: str = "openai"
    cross_attn_every_n_layers: int = 1
    window_size: int = 12
    fusion_mode: str = "post"
    use_gripper: bool = True
    use_state: bool = False
    decoder_type: str = "lstm"
    head_type: str = "lstm"
    precision: str = "fp32"

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
        },
        "action_head": {
            "decoder_type": config.decoder_type,
            "head_type": config.head_type,
            "action_dimensions": 7,
            "pose_dimensions": 6,
            "gripper_dimensions": 1,
        },
    }
