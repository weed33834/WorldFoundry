"""
This module provides the MatrixGame2Synthesis class, which acts as a high-level wrapper
for the MatrixGame2Runtime. It facilitates model loading, configuration, and prediction
for MatrixGame2 tasks, delegating core functionalities to the underlying runtime.
It also exposes the `process_video` utility function for video processing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base_synthesis import BaseSynthesis
from worldfoundry.core.io.artifacts import process_game_control_video as process_video

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.worldfoundry_runtime import (
        MatrixGame2Runtime,
    )


def _runtime_module():
    """
    Dynamically imports and returns the `worldfoundry_runtime` module for MatrixGame2.

    This function helps to avoid circular imports and delays the import
    of the runtime module until it's actually needed.
    """
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime import (
        worldfoundry_runtime,
    )

    return worldfoundry_runtime


def _runtime_cls():
    """
    Dynamically retrieves the `MatrixGame2Runtime` class from its module.

    This function helps to avoid circular imports and delays the class lookup
    until it's actually needed.
    """
    return _runtime_module().MatrixGame2Runtime


class MatrixGame2Synthesis(BaseSynthesis):
    """
    A synthesis class for the MatrixGame2 task, serving as a high-level API
    that wraps and delegates operations to an underlying MatrixGame2Runtime instance.

    This class simplifies interaction with the MatrixGame2 generation pipeline,
    allowing initialization with various parameters or loading from pretrained models.
    It supports direct attribute access and method calls that are forwarded to
    the runtime instance.
    """

    def __init__(
        self,
        pipeline: Any = None,
        vae: Any = None,
        weight_dtype: Any = None,
        mode: str = "universal",
        device: str = "cuda",
        *,
        runtime: "MatrixGame2Runtime | None" = None,
    ):
        """
        Initializes the MatrixGame2Synthesis instance.

        Args:
            pipeline (Any, optional): The underlying pipeline object. Defaults to None.
            vae (Any, optional): The VAE component for the pipeline. Defaults to None.
            weight_dtype (Any, optional): The data type for model weights. Defaults to None.
            mode (str, optional): The operational mode (e.g., "universal"). Defaults to "universal".
            device (str, optional): The device to run the model on (e.g., "cuda"). Defaults to "cuda".
            runtime (MatrixGame2Runtime | None, optional): An existing MatrixGame2Runtime instance
                to wrap. If None, a new runtime instance will be created using the provided
                pipeline, vae, weight_dtype, mode, and device parameters. Defaults to None.
        """
        super().__init__()
        # If an existing runtime is provided, use it; otherwise, initialize a new one.
        self.runtime = runtime or _runtime_cls()(
            pipeline=pipeline,
            vae=vae,
            weight_dtype=weight_dtype,
            mode=mode,
            device=device,
        )

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying `self.runtime` instance.

        This allows methods and properties of `MatrixGame2Runtime` to be accessed
        directly through the `MatrixGame2Synthesis` instance.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The attribute value from `self.runtime`.

        Raises:
            AttributeError: If the attribute is 'runtime' (to prevent infinite recursion)
                            or if it does not exist on `self.runtime`.
        """
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        mode: str = "universal",
        device=None,
        weight_dtype: Any = None,
        **kwargs,
    ) -> "MatrixGame2Synthesis":
        """
        Creates a MatrixGame2Synthesis instance by loading a pretrained model.

        This class method delegates the model loading process to the
        `MatrixGame2Runtime.from_pretrained` method.

        Args:
            pretrained_model_path: The path to the pretrained model.
            mode (str, optional): The operational mode. Defaults to "universal".
            device (Any, optional): The device to load the model on. Defaults to None.
            weight_dtype (Any, optional): The data type for model weights. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the runtime's from_pretrained method.

        Returns:
            MatrixGame2Synthesis: A new instance of MatrixGame2Synthesis initialized
                                  with the loaded runtime.
        """
        runtime = _runtime_cls().from_pretrained(
            pretrained_model_path=pretrained_model_path,
            mode=mode,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(runtime=runtime)

    @staticmethod
    def _resolve_checkpoint_path(model_root: str, mode: str, checkpoint_path: str | None = None) -> str:
        """
        Resolves the full path to a model checkpoint.

        This static method delegates the path resolution to the `MatrixGame2Runtime` class.

        Args:
            model_root (str): The root directory for models.
            mode (str): The mode of the model (e.g., "universal").
            checkpoint_path (str | None, optional): An optional specific checkpoint path.
                                                    If None, a default path based on `model_root`
                                                    and `mode` will be resolved. Defaults to None.

        Returns:
            str: The resolved absolute path to the model checkpoint.
        """
        return _runtime_cls()._resolve_checkpoint_path(
            model_root=model_root,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )

    def predict(self, *args, **kwargs):
        """
        Executes the prediction task using the underlying `MatrixGame2Runtime` instance.

        All arguments and keyword arguments are directly forwarded to the runtime's
        predict method.

        Args:
            *args: Positional arguments for the runtime's predict method.
            **kwargs: Keyword arguments for the runtime's predict method.

        Returns:
            Any: The result of the prediction from the runtime.
        """
        # Dynamically set the process_video function on the runtime module
        # This ensures the runtime has access to the correct video processing utility.
        _runtime_module().process_video = process_video
        return self.runtime.predict(*args, **kwargs)


__all__ = ["MatrixGame2Synthesis", "process_video"]