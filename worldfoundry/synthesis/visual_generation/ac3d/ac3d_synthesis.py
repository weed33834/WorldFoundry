"""
This module defines the AC3DSynthesis class, a facade that provides a high-level interface for interacting with the AC3D model's runtime.

It inherits from BaseSynthesis and wraps an AC3DRuntime instance to expose
its functionalities, such as loading pretrained models and making predictions,
in a consistent manner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ...base_synthesis import BaseSynthesis
from .runtime import AC3DRuntime


class AC3DSynthesis(BaseSynthesis):
    """
    A facade class for the AC3D model, providing a simplified interface over the
    official AC3D runtime.

    This class extends BaseSynthesis and encapsulates an AC3DRuntime instance,
    delegating core functionalities like model loading and inference to it.
    It standardizes the interaction with the AC3D model within the synthesis framework.
    """

    MODEL_ID = AC3DRuntime.MODEL_ID
    DISPLAY_NAME = AC3DRuntime.DISPLAY_NAME

    def __init__(self, runtime: AC3DRuntime) -> None:
        """
        Initializes the AC3DSynthesis instance with a provided AC3D runtime.

        Args:
            runtime: An initialized instance of AC3DRuntime which this synthesis
                     facade will wrap.
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
    ) -> "AC3DSynthesis":
        """
        Loads a pretrained AC3D model and initializes an AC3DSynthesis instance.

        This class method delegates the model loading to the AC3DRuntime's
        `from_pretrained` method and then wraps the resulting runtime.

        Args:
            pretrained_model_path: Path to the pretrained model or a model identifier.
            args: Placeholder for additional arguments (currently unused and deleted).
            device: The device to load the model on (e.g., "cuda", "cpu").
            model_id: Identifier for the specific model to load.
            **kwargs: Additional keyword arguments passed directly to the
                      AC3DRuntime's `from_pretrained` method.

        Returns:
            An instance of AC3DSynthesis initialized with the loaded runtime.
        """
        # The 'args' parameter is not used by AC3DRuntime and is explicitly removed.
        del args
        runtime = AC3DRuntime.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id,
            **kwargs,
        )
        return cls(runtime)

    def runtime_plan(self) -> dict[str, Any]:
        """
        Retrieves the execution plan from the underlying AC3D runtime.

        Returns:
            A dictionary representing the runtime's execution plan.
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
        """
        Generates predictions using the underlying AC3D runtime based on the provided inputs.

        This method acts as a direct passthrough to the AC3DRuntime's `predict` method.

        Args:
            prompt: Text prompt for generation.
            images: Input images for generation.
            video: Input video for generation.
            interactions: A sequence of interaction objects to guide generation.
            output_path: Optional path to save the generated output.
            fps: Frames per second for video generation (if applicable).
            **kwargs: Additional keyword arguments passed directly to the
                      AC3DRuntime's `predict` method.

        Returns:
            A dictionary containing the prediction results.
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


__all__ = ["AC3DSynthesis"]