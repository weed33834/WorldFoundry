"""
This module provides the DreamDojoSynthesis class, which acts as a facade
over the DreamDojo runtime. It facilitates model interaction by delegating
calls to the underlying DreamDojoRuntime instance.
"""
from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis
from .worldfoundry_runtime import DreamDojoRuntime


class DreamDojoSynthesis(BaseSynthesis):
    """Thin synthesis facade over the DreamDojo runtime."""

    MODEL_ID = DreamDojoRuntime.MODEL_ID
    DISPLAY_NAME = DreamDojoRuntime.DISPLAY_NAME

    def __init__(self, *, runtime: DreamDojoRuntime | None = None, **kwargs: Any) -> None:
        """
        Initializes the DreamDojoSynthesis instance.

        Args:
            runtime: An optional pre-initialized DreamDojoRuntime instance.
                     If None, a new DreamDojoRuntime instance is created using kwargs.
            **kwargs: Arbitrary keyword arguments passed to DreamDojoRuntime if `runtime` is not provided.
        """
        super().__init__()
        # Initialize runtime; if not provided, create a new one with given keyword arguments.
        self.runtime = runtime or DreamDojoRuntime(**kwargs)

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying DreamDojoRuntime instance.

        This method is called when an attribute is not found in the DreamDojoSynthesis instance itself.
        It proxies the attribute lookup to the `runtime` object.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the runtime object.

        Raises:
            AttributeError: If the 'runtime' attribute itself is being accessed
                            via __getattr__, to prevent infinite recursion,
                            or if the attribute does not exist on the runtime.
        """
        # Prevent infinite recursion if 'runtime' attribute itself is accessed this way.
        if name == "runtime":
            raise AttributeError(name)
        # Delegate attribute access to the internal runtime object.
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "DreamDojoSynthesis":
        """
        Factory method to create a DreamDojoSynthesis instance from a pretrained model.

        This method delegates the `from_pretrained` call to the underlying DreamDojoRuntime,
        then wraps the resulting runtime instance in a new DreamDojoSynthesis object.

        Args:
            *args: Positional arguments passed to `DreamDojoRuntime.from_pretrained`.
            **kwargs: Keyword arguments passed to `DreamDojoRuntime.from_pretrained`.

        Returns:
            A new DreamDojoSynthesis instance initialized with the pretrained runtime.
        """
        return cls(runtime=DreamDojoRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Executes a prediction using the underlying DreamDojoRuntime.

        This method acts as a direct proxy to the `predict` method of the `runtime` object.

        Args:
            *args: Positional arguments passed to `self.runtime.predict`.
            **kwargs: Keyword arguments passed to `self.runtime.predict`.

        Returns:
            The prediction result from the runtime, typically a dictionary.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["DreamDojoSynthesis"]