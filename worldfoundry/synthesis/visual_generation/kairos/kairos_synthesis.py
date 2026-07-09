"""Provides a synthesis facade for interacting with the Kairos Sensenova runtime.

This module defines the `KairosSynthesis` class, which acts as a high-level
interface for performing synthesis operations using the Kairos Sensenova model.
It wraps the underlying `KairosRuntime` to provide a consistent API for
model initialization and prediction, inheriting from `BaseSynthesis`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ...base_synthesis import BaseSynthesis
from .runtime import KairosRuntime


class KairosSynthesis(BaseSynthesis):
    """Thin synthesis facade over the official Kairos Sensenova runtime checkout.

    This class provides a high-level interface for initializing and interacting
    with the Kairos Sensenova model for synthesis tasks. It delegates all core
    logic to an internal `KairosRuntime` instance.
    """

    MODEL_ID = KairosRuntime.MODEL_ID
    DISPLAY_NAME = KairosRuntime.DISPLAY_NAME

    def __init__(self, runtime: KairosRuntime) -> None:
        """Initializes the KairosSynthesis facade with a KairosRuntime instance.

        Args:
            runtime: An initialized instance of `KairosRuntime` which handles
                     the actual model loading and inference.
        """
        super().__init__()
        self.runtime = runtime
        self.model_id = runtime.model_id
        self.model_name = runtime.model_name
        self.generation_type = runtime.generation_type
        self.device = runtime.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "KairosSynthesis":
        """Factory method to create a KairosSynthesis instance from a pretrained model.

        This method initializes the underlying `KairosRuntime` from a pretrained
        model and then wraps it in a `KairosSynthesis` instance.

        Args:
            pretrained_model_path: The path to the pretrained model or a model identifier.
                                   Type can vary based on the runtime's requirements.
            args: Placeholder for additional arguments (currently unused).
            device: The device to load the model on (e.g., "cuda", "cpu").
            model_id: An optional identifier for the model.
            **kwargs: Additional keyword arguments passed directly to `KairosRuntime.from_pretrained`.

        Returns:
            An instance of `KairosSynthesis` ready for prediction.
        """
        # The 'args' parameter is not used by the KairosRuntime, so it's explicitly deleted.
        del args
        runtime = KairosRuntime.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id,
            **kwargs,
        )
        return cls(runtime)

    def runtime_plan(self) -> dict[str, Any]:
        """Retrieves the runtime plan from the underlying KairosRuntime.

        This method delegates to the `runtime` instance to get details about its
        operational plan, such as supported features or configurations.

        Returns:
            A dictionary containing the runtime plan information.
        """
        return self.runtime.runtime_plan()

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Performs a prediction using the underlying KairosRuntime.

        This method acts as a wrapper for the `predict` method of the `KairosRuntime`
        instance, passing all arguments directly to it.

        Args:
            prompt: The text prompt for the generation.
            images: Optional input images. Type can vary based on the runtime's requirements.
            video: Optional input video. Type can vary based on the runtime's requirements.
            interactions: A sequence of interaction objects (e.g., for control, editing).
            output_path: Optional path to save the generated output.
            fps: Optional frames per second for video generation.
            **kwargs: Additional keyword arguments passed directly to `KairosRuntime.predict`.

        Returns:
            A dictionary containing the prediction results, typically including
            the path to the generated output or other relevant metadata.
        """
        return self.runtime.predict(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )


__all__ = ["KairosSynthesis"]