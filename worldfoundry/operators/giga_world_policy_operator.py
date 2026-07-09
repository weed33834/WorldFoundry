"""Module for the GigaWorldPolicy operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class GigaWorldPolicyOperator(EmbodiedActionOperator):
    """Operator for GigaWorld-Policy world-action model inference contracts."""

    MODEL_ID = "giga-world-policy"
    POLICY_FAMILY = "world_action_model"
    ACTION_REPRESENTATION = "continuous_action_chunk"
    OBSERVATION_LAYOUT = "three_view_rgb_state_instruction"
    DEFAULT_INPUT_SCHEMA = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": False,
        "state": True,
        "dataset_paths": True,
        "actions": ["world_actions", "robot_actions", "action_chunk", "delta_action"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "task_instruction")
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "model_track": "wam",
                "planning_target": kwargs.get("planning_target", "action_chunk_prediction"),
            },
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        image_mapping = images if isinstance(images, Mapping) else {}
        state = _first_present(
            kwargs,
            "observation.state",
            "state",
            "robot_state",
            "proprio",
            "state_vector",
        )
        observation = _compact_dict(
            {
                "images": images,
                "cam_high": _first_present(image_mapping, "observation.images.cam_high", "cam_high", "front", "front_image"),
                "cam_left_wrist": _first_present(
                    image_mapping,
                    "observation.images.cam_left_wrist",
                    "cam_left_wrist",
                    "left_wrist",
                    "left_image",
                ),
                "cam_right_wrist": _first_present(
                    image_mapping,
                    "observation.images.cam_right_wrist",
                    "cam_right_wrist",
                    "right_wrist",
                    "right_image",
                ),
                "state": state,
                "dataset_paths": _first_present(kwargs, "dataset_paths", "dataset_path"),
                "episode_idx": kwargs.get("episode_idx"),
                "action_chunk": kwargs.get("action_chunk"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "giga_world_policy_observation": observation,
                "giga_world_policy_input_contract": {
                    "modalities": ["three_view_rgb", "robot_state", "language_or_t5_embedding"],
                    "runtime": "official_server_client",
                    "output_contract": "continuous_action_chunk",
                    "requires_dataset_for_official_client": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(
            raw,
            "giga_world_actions",
            "world_actions",
            "robot_actions",
            "action_chunk",
            "delta_actions",
            "predicted_actions",
        )
        self.interaction_history.append(actions)
        action_space = raw.get("action_space") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="continuous_world_action_chunk",
                control_mode="open_loop_offline_dataset_evaluation",
                action_horizon=(raw.get("action_horizon") if isinstance(raw, Mapping) else None),
                supports_world_action_modeling=True,
            ),
            "action_space": action_space
            or self.input_schema.get("action_space")
            or {"kind": "continuous_robot_action", "dimensions": 14, "horizon": "runtime_configured"},
            "policy_controls": {
                "supports_world_action_model": True,
                "uses_three_camera_observation": True,
                "official_client_mode": "offline_open_loop_dataset",
                "output_semantics": "denormalized_action_chunk",
            },
        }
