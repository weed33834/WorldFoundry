"""
This module provides the MatrixGame1Synthesis class, a thin facade over the
MatrixGame1Runtime. It acts as a synthesis layer, simplifying interaction
with the underlying runtime model for Matrix Game 1.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_1_runtime import MatrixGame1Runtime

from ...base_synthesis import BaseSynthesis


class MatrixGame1Synthesis(BaseSynthesis):
    """
    A synthesis facade for the Matrix-Game-1 runtime.

    This class wraps the MatrixGame1Runtime, providing a higher-level interface
    and potentially adapting its outputs or inputs for broader compatibility
    within a synthesis framework. It delegates most operations to the underlying runtime.
    """

    MODEL_ID = MatrixGame1Runtime.MODEL_ID
    DISPLAY_NAME = MatrixGame1Runtime.DISPLAY_NAME

    def __init__(self, *, runtime: MatrixGame1Runtime | None = None, **kwargs: Any) -> None:
        """
        Initializes the MatrixGame1Synthesis instance.

        Args:
            runtime: An optional pre-instantiated MatrixGame1Runtime object.
                     If None, a new runtime instance will be created using `kwargs`.
            **kwargs: Additional keyword arguments to pass to the MatrixGame1Runtime
                      constructor if `runtime` is not provided.
        """
        super().__init__()
        # If a runtime instance is not provided, create a new one with the given keyword arguments.
        self.runtime = runtime or MatrixGame1Runtime(**kwargs)

    def __getattr__(self, name: str):
        """
        Proxies attribute access to the underlying MatrixGame1Runtime instance.

        This allows direct access to methods and properties of the runtime
        through the synthesis object, effectively acting as a pass-through.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying runtime.

        Raises:
            AttributeError: If the attribute 'runtime' itself is requested directly
                            via __getattr__ to prevent infinite recursion, or if
                            the attribute is not found on the runtime.
        """
        # Prevent infinite recursion if 'runtime' is requested through __getattr__
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "MatrixGame1Synthesis":
        """
        Constructs a MatrixGame1Synthesis instance from a pretrained model.

        This method delegates the loading of the pretrained model to the
        MatrixGame1Runtime's `from_pretrained` class method and then
        wraps the resulting runtime instance.

        Args:
            *args: Positional arguments to pass to MatrixGame1Runtime.from_pretrained.
            **kwargs: Keyword arguments to pass to MatrixGame1Runtime.from_pretrained.

        Returns:
            A new instance of MatrixGame1Synthesis initialized with the pretrained runtime.
        """
        return cls(runtime=MatrixGame1Runtime.from_pretrained(*args, **kwargs))

    def preflight(self) -> dict[str, Any]:
        """
        Performs pre-flight checks and returns a status dictionary.

        This method calls the underlying runtime's preflight check and
        then adapts the keys of the returned dictionary for consistency
        within the synthesis framework.

        Returns:
            A dictionary containing the results of the pre-flight checks,
            including statuses for missing assets and runtime files.
        """
        preflight = self.runtime.preflight()
        # Remap specific keys from the runtime's preflight check for broader synthesis compatibility.
        preflight["missing_assets"] = preflight["missing_checkpoint_files"]
        preflight["missing_runtime"] = preflight["missing_runtime_files"]
        return preflight

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Executes the prediction logic using the underlying MatrixGame1Runtime.

        This method directly delegates the prediction call and its arguments
        to the wrapped runtime instance.

        Args:
            *args: Positional arguments to pass to the runtime's predict method.
            **kwargs: Keyword arguments to pass to the runtime's predict method.

        Returns:
            A dictionary containing the prediction results from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["MatrixGame1Synthesis"]