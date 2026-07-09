"""
Provides a synthesis facade for the Warp-as-History runtime.

This module defines the `WarpAsHistorySynthesis` class, which acts as a lightweight wrapper
around the `WarpAsHistoryRuntime`. It simplifies interaction with the underlying runtime
by delegating most method calls and attribute accesses to the runtime instance, while
conforming to the `BaseSynthesis` interface.
"""
from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.warp_as_history.worldfoundry_runtime import (
    WarpAsHistoryRuntime,
)

from ...base_synthesis import BaseSynthesis


class WarpAsHistorySynthesis(BaseSynthesis):
    """
    A synthesis facade for the Warp-as-History runtime.

    This class serves as a high-level interface to the `WarpAsHistoryRuntime`,
    delegating core functionalities like `predict` and `from_pretrained`
    to the underlying runtime instance. It inherits from `BaseSynthesis`
    to provide a consistent interface within the synthesis framework.

    Attributes:
        MODEL_ID (str): The unique identifier for the Warp-as-History model,
                        delegated from `WarpAsHistoryRuntime`.
        DISPLAY_NAME (str): The human-readable name for the Warp-as-History model,
                            delegated from `WarpAsHistoryRuntime`.
        runtime (WarpAsHistoryRuntime): The underlying instance of the Warp-as-History runtime.
    """

    MODEL_ID = WarpAsHistoryRuntime.MODEL_ID
    DISPLAY_NAME = WarpAsHistoryRuntime.DISPLAY_NAME

    def __init__(self, *, runtime: WarpAsHistoryRuntime | None = None, **kwargs: Any) -> None:
        """
        Initializes the WarpAsHistorySynthesis instance.

        Args:
            runtime (WarpAsHistoryRuntime | None): An optional pre-existing
                                                   WarpAsHistoryRuntime instance.
                                                   If not provided, a new runtime
                                                   instance will be created using
                                                   `kwargs`.
            **kwargs (Any): Arbitrary keyword arguments passed directly to the
                            `WarpAsHistoryRuntime` constructor if a `runtime`
                            instance is not explicitly provided.
        """
        super().__init__()
        self.runtime = runtime or WarpAsHistoryRuntime(**kwargs)

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying `runtime` instance.

        This method is called when an attribute is not found in the
        `WarpAsHistorySynthesis` instance itself. It allows the synthesis
        object to behave like the runtime by transparently forwarding
        attribute lookups.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the `runtime` instance.

        Raises:
            AttributeError: If the 'runtime' attribute itself is being accessed
                            through `__getattr__` (which indicates an issue as
                            it should be directly available) or if the attribute
                            does not exist on the underlying `runtime` instance.
        """
        # Prevent infinite recursion if 'runtime' attribute itself is somehow looked up via __getattr__
        # (though `self.runtime` is set directly, this provides a safeguard for unexpected access patterns).
        if name == "runtime":
            raise AttributeError(name)
        # Delegate attribute access to the encapsulated WarpAsHistoryRuntime instance.
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "WarpAsHistorySynthesis":
        """
        Creates a new `WarpAsHistorySynthesis` instance by loading a pretrained runtime.

        This class method acts as a factory, delegating the loading of a
        pretrained model to the underlying `WarpAsHistoryRuntime.from_pretrained`
        method and then wrapping the resulting runtime instance.

        Args:
            *args (Any): Positional arguments passed to `WarpAsHistoryRuntime.from_pretrained`.
            **kwargs (Any): Keyword arguments passed to `WarpAsHistoryRuntime.from_pretrained`.

        Returns:
            WarpAsHistorySynthesis: A new instance of `WarpAsHistorySynthesis`
                                    with a loaded pretrained runtime.
        """
        return cls(runtime=WarpAsHistoryRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Performs a prediction using the underlying Warp-as-History runtime.

        This method delegates the prediction task directly to the `predict`
        method of the encapsulated `WarpAsHistoryRuntime` instance.

        Args:
            *args (Any): Positional arguments passed to `self.runtime.predict`.
            **kwargs (Any): Keyword arguments passed to `self.runtime.predict`.

        Returns:
            dict[str, Any]: The prediction results from the `WarpAsHistoryRuntime`.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["WarpAsHistorySynthesis"]