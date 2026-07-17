"""Inference transforms for the RISE AgileX action policy."""

from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from worldfoundry.synthesis.action_generation.openpi import transforms
from worldfoundry.synthesis.action_generation.openpi.modeling import model as openpi_model
from worldfoundry.synthesis.action_generation.openpi.modeling import tokenizer as openpi_tokenizer


def discretize_advantage(value: float, bins: int = 10) -> int:
    """Discretize the released policy's scalar quality condition.

    The action-policy release uses ``np.digitize`` over the left edges of ten
    equal-width bins in ``[-1, 1]``.  Consequently its inference condition
    ``1.0`` is represented by the literal prompt value ``10``.
    """

    if bins <= 0:
        raise ValueError("advantage_bins must be positive")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("action advantage must be finite")
    edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float64)[:-1]
    return int(np.digitize(np.asarray(scalar), bins=edges))


def format_advantage_prompt(prompt: str, state: np.ndarray, *, advantage: float = 1.0, bins: int = 10) -> str:
    """Build the exact PaliGemma prefix used by the released RISE policy."""

    cleaned_text = str(prompt).strip().replace("_", " ").replace("\n", " ")
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if not np.isfinite(state_array).all():
        raise ValueError("RISE normalized state contains non-finite values")
    discretized_state = np.digitize(state_array, bins=np.linspace(-1.0, 1.0, 257)[:-1]) - 1
    state_text = " ".join(map(str, discretized_state))
    advantage_bin = discretize_advantage(advantage, bins)
    return f"Task: {cleaned_text}, State: {state_text}, Advantage: {advantage_bin};\nAction: "


class AdvantagePaligemmaTokenizer(openpi_tokenizer.PaligemmaTokenizer):
    """Local-only PaliGemma tokenizer with the release's quality prefix."""

    def __init__(
        self,
        max_len: int = 200,
        tokenizer_path: str | None = None,
        *,
        advantage: float = 1.0,
        advantage_bins: int = 10,
    ) -> None:
        super().__init__(max_len=max_len, tokenizer_path=tokenizer_path)
        self._advantage = float(advantage)
        self._advantage_bins = int(advantage_bins)

    @property
    def advantage_bin(self) -> int:
        return discretize_advantage(self._advantage, self._advantage_bins)

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if state is None:
            raise ValueError("RISE advantage-conditioned tokenization requires normalized robot state")
        full_prompt = format_advantage_prompt(
            prompt,
            state,
            advantage=self._advantage,
            bins=self._advantage_bins,
        )
        tokens = list(self._tokenizer.encode(full_prompt, add_bos=True))
        token_count = len(tokens)
        if token_count > self._max_len:
            logging.warning(
                "RISE prompt token length (%d) exceeds max length (%d); truncating",
                token_count,
                self._max_len,
            )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        else:
            pad_count = self._max_len - token_count
            tokens.extend([0] * pad_count)
            mask = [True] * token_count + [False] * pad_count
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Map three AgileX camera views and 14-D state into Pi0.5 inputs."""

    action_dim: int = 32
    model_type: openpi_model.ModelType = openpi_model.ModelType.PI05

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    RENAME_MAP: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"unexpected RISE camera keys: {sorted(unexpected)}")

        missing = [name for name in self.EXPECTED_CAMERAS if name not in in_images]
        if missing:
            raise ValueError(f"RISE requires all three camera views; missing {missing}")

        images: dict[str, np.ndarray] = {}
        masks: dict[str, np.bool_] = {}
        for name in self.EXPECTED_CAMERAS:
            image = np.asarray(in_images[name])
            if image.ndim != 3:
                raise ValueError(f"RISE camera {name!r} must be CHW or HWC RGB, got {image.shape}")
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] != 3:
                raise ValueError(f"RISE camera {name!r} is not RGB: {image.shape}")
            if np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all():
                    raise ValueError(f"RISE camera {name!r} contains non-finite values")
                low = float(image.min(initial=0.0))
                high = float(image.max(initial=0.0))
                if low < 0.0:
                    if low < -1.0 or high > 1.0:
                        raise ValueError(f"unsupported image range [{low}, {high}] for {name!r}")
                    image = (image + 1.0) * 127.5
                elif high <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            model_name = self.RENAME_MAP[name]
            images[model_name] = np.ascontiguousarray(image)
            masks[model_name] = np.True_

        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        if state.size != 14:
            raise ValueError(f"RISE requires a 14-D robot state, got {state.size}")
        if not np.isfinite(state).all():
            raise ValueError("RISE robot state contains non-finite values")
        state = transforms.pad_to_dim(state, self.action_dim)
        state = np.where(np.abs(state) > np.pi, 0.0, state).astype(np.float32, copy=False)

        result = {"image": images, "image_mask": masks, "state": state}
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Return the released policy's 14 deployed action dimensions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError(f"RISE policy returned an invalid action tensor: {actions.shape}")
        return {"actions": np.ascontiguousarray(actions[:, :14])}


__all__ = [
    "AdvantagePaligemmaTokenizer",
    "AgilexInputs",
    "AgilexOutputs",
    "discretize_advantage",
    "format_advantage_prompt",
]
