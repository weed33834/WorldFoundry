"""
This module provides the Lyra1Synthesis class, a thin facade over the Lyra1Runtime.

It simplifies interaction with the underlying Lyra-1 model runtime by delegating
most operations and attributes to a Lyra1Runtime instance. This design promotes
code reusability and separation of concerns.
"""
from __future__ import annotations

from typing import Any

from .worldfoundry_runtime import Lyra1Runtime


class Lyra1Synthesis:
    """
    Thin synthesis facade over the Lyra-1 runtime.

    This class acts as a wrapper around the Lyra1Runtime, providing a simplified
    interface for interacting with the Lyra-1 model. It exposes model metadata
    and delegates core functionalities like prediction to the underlying runtime
    instance.
    """

    MODEL_ID = Lyra1Runtime.MODEL_ID
    DISPLAY_NAME = Lyra1Runtime.DISPLAY_NAME
    BLOCKED_REASONS = Lyra1Runtime.BLOCKED_REASONS
    MULTI_TRAJECTORY_INDEX = Lyra1Runtime.MULTI_TRAJECTORY_INDEX

    def __init__(self, *, runtime: Lyra1Runtime | None = None, **kwargs: Any) -> None:
        """
        Initializes a new Lyra1Synthesis instance.

        Args:
            runtime: An optional pre-initialized Lyra1Runtime instance. If not
                     provided, a new Lyra1Runtime instance will be created.
            **kwargs: Arbitrary keyword arguments to be passed to the Lyra1Runtime
                      constructor if a new runtime instance is created.
        """
        super().__init__()
        self.runtime = runtime or Lyra1Runtime(**kwargs)

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying Lyra1Runtime instance.

        This method is called when an attribute is not found in the Lyra1Synthesis
        instance itself. It attempts to retrieve the attribute from the
        'runtime' object, effectively making Lyra1Synthesis behave like
        Lyra1Runtime for most attribute lookups.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the Lyra1Runtime instance.

        Raises:
            AttributeError: If the 'runtime' attribute itself is being accessed
                            through this method (to prevent infinite recursion),
                            or if the attribute does not exist on the runtime.
        """
        # Prevent infinite recursion if 'runtime' is accessed via __getattr__
        # It should be directly accessible as self.runtime
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Lyra1Synthesis":
        """
        Creates a Lyra1Synthesis instance by loading a pretrained Lyra1Runtime.

        This class method acts as a convenience constructor, delegating the
        loading of a pretrained model to the Lyra1Runtime.from_pretrained
        method and then wrapping the resulting runtime.

        Args:
            *args: Positional arguments to pass to Lyra1Runtime.from_pretrained.
            **kwargs: Keyword arguments to pass to Lyra1Runtime.from_pretrained.

        Returns:
            A new Lyra1Synthesis instance initialized with the pretrained runtime.
        """
        return cls(runtime=Lyra1Runtime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Performs a prediction using the underlying Lyra1Runtime instance.

        This method directly delegates the prediction call to the 'predict'
        method of the encapsulated Lyra1Runtime.

        Args:
            *args: Positional arguments to pass to the runtime's predict method.
            **kwargs: Keyword arguments to pass to the runtime's predict method.

        Returns:
            A dictionary containing the prediction results from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["Lyra1Synthesis"]