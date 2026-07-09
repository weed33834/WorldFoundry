"""Provides a synthesis facade for the Splatt3R 3D model runtime.

This module defines `Splatt3RSynthesis`, a wrapper class that provides
a simpler interface over the more complex `Splatt3RRuntime` base model.
It allows for easy integration into synthesis pipelines by proxying
calls to the underlying runtime.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.base_models.three_dimensions.general_3d.splatt3r import Splatt3RRuntime

from ...base_synthesis import BaseSynthesis


class Splatt3RSynthesis(BaseSynthesis):
    """Thin synthesis facade over the base-model Splatt3R runtime."""

    MODEL_ID = Splatt3RRuntime.MODEL_ID
    DISPLAY_NAME = Splatt3RRuntime.DISPLAY_NAME

    def __init__(self, *, runtime: Splatt3RRuntime | None = None, **kwargs: Any) -> None:
        """
        Initializes the Splatt3R synthesis facade.

        Args:
            runtime (Splatt3RRuntime | None): An optional pre-initialized Splatt3RRuntime instance.
                                              If None, a new runtime is created using kwargs.
            **kwargs (Any): Arbitrary keyword arguments passed to Splatt3RRuntime constructor
                            if `runtime` is not provided.
        """
        super().__init__()
        self.runtime = runtime or Splatt3RRuntime(**kwargs)

    def __getattr__(self, name: str):
        """
        Provides attribute access proxying to the underlying Splatt3R runtime instance.

        This method is called when an attribute is not found in the Splatt3RSynthesis instance itself.
        It attempts to retrieve the attribute from the `self.runtime` object.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the `self.runtime` object.

        Raises:
            AttributeError: If the attribute 'runtime' itself is requested directly
                            or if the attribute is not found in the underlying runtime.
        """
        # Prevent infinite recursion if 'runtime' is accessed directly via __getattr__
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Splatt3RSynthesis":
        """
        Creates a new Splatt3RSynthesis instance by loading a pretrained Splatt3RRuntime.

        This class method acts as a convenience constructor, passing all arguments
        to the `Splatt3RRuntime.from_pretrained` method.

        Args:
            *args (Any): Positional arguments to pass to `Splatt3RRuntime.from_pretrained`.
            **kwargs (Any): Keyword arguments to pass to `Splatt3RRuntime.from_pretrained`.

        Returns:
            Splatt3RSynthesis: A new instance of Splatt3RSynthesis with the loaded runtime.
        """
        return cls(runtime=Splatt3RRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Performs a prediction using the underlying Splatt3R runtime.

        This method acts as a direct proxy to the `predict` method of the
        internal `Splatt3RRuntime` instance.

        Args:
            *args (Any): Positional arguments to pass to `self.runtime.predict`.
            **kwargs (Any): Keyword arguments to pass to `self.runtime.predict`.

        Returns:
            dict[str, Any]: The prediction results from the `Splatt3RRuntime`.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["Splatt3RSynthesis"]