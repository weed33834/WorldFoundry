"""Provides a high-level synthesis facade for interacting with the SCOPE runtime.

This module integrates the SCOPE runtime (for visual generation) within the
worldfoundry synthesis framework, offering a simplified interface for model
initialization, prediction, and attribute access delegation. It re-exports
essential constants and utilities from the underlying SCOPE runtime.
"""
from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.scope.worldfoundry_runtime import (
    DEFAULT_MODEL_DIR,
    SCOPERuntime,
    diffsynth_runtime_root,
    runtime_root,
)

from ...base_synthesis import BaseSynthesis


class SCOPESynthesis(BaseSynthesis):
    """Thin synthesis facade over the base-model SCOPE runtime."""

    MODEL_ID = SCOPERuntime.MODEL_ID
    DISPLAY_NAME = SCOPERuntime.DISPLAY_NAME

    def __init__(self, *, runtime: SCOPERuntime | None = None, **kwargs: Any) -> None:
        """Initializes the SCOPESynthesis facade.

        Args:
            runtime: An optional pre-existing SCOPERuntime instance. If not provided,
                     a new SCOPERuntime will be initialized with the given `kwargs`.
            **kwargs: Arbitrary keyword arguments to pass to the SCOPERuntime
                      constructor if a `runtime` instance is not provided.
        """
        super().__init__()
        # Initialize the internal runtime instance. If a runtime object is provided, use it;
        # otherwise, create a new SCOPERuntime using the passed keyword arguments.
        self.runtime = runtime or SCOPERuntime(**kwargs)

    def __getattr__(self, name: str):
        """Delegates attribute access to the underlying SCOPERuntime instance.

        This method is called when an attribute is not found in the SCOPESynthesis
        instance itself. It attempts to retrieve the attribute from the internal
        `self.runtime` object.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying `SCOPERuntime` instance.

        Raises:
            AttributeError: If the 'runtime' attribute itself is being accessed
                            via __getattr__ (to prevent infinite recursion), or
                            if the attribute is not found in either the synthesis
                            object or its runtime.
        """
        if name == "runtime":
            # Prevent infinite recursion if __getattr__ is called trying to get 'runtime' itself.
            # This can happen if self.runtime hasn't been fully initialized or is somehow null.
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "SCOPESynthesis":
        """Creates a SCOPESynthesis instance by loading a pretrained SCOPERuntime model.

        This class method acts as a factory, delegating the model loading
        process to the `SCOPERuntime.from_pretrained` method.

        Args:
            *args: Positional arguments to pass to `SCOPERuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `SCOPERuntime.from_pretrained`.

        Returns:
            A new instance of `SCOPESynthesis` with the pretrained SCOPERuntime.
        """
        return cls(runtime=SCOPERuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Performs a prediction using the underlying SCOPERuntime instance.

        This method directly delegates the prediction call and all its arguments
        to the `predict` method of the internal `self.runtime` object.

        Args:
            *args: Positional arguments to pass to the `SCOPERuntime.predict` method.
            **kwargs: Keyword arguments to pass to the `SCOPERuntime.predict` method.

        Returns:
            The prediction result from `SCOPERuntime.predict`, typically a dictionary.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = [
    "DEFAULT_MODEL_DIR",
    "SCOPESynthesis",
    "diffsynth_runtime_root",
    "runtime_root",
]