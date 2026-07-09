"""Provides a synthesis facade for the Open-MAGVIT2 model, enabling interaction with its runtime."""
from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis
from .worldfoundry_runtime import OpenMAGVIT2Runtime


class OpenMAGVIT2Synthesis(BaseSynthesis):
    """
    A synthesis facade for the Open-MAGVIT2 model.

    This class acts as a thin wrapper around the `OpenMAGVIT2Runtime`,
    delegating most operations like prediction and model loading to the underlying runtime
    instance. It provides a consistent interface within the synthesis framework while
    abstracting the specifics of the Open-MAGVIT2 implementation.
    """

    MODEL_ID = OpenMAGVIT2Runtime.MODEL_ID
    DISPLAY_NAME = OpenMAGVIT2Runtime.DISPLAY_NAME

    def __init__(self, *, runtime: OpenMAGVIT2Runtime | None = None, **kwargs: Any) -> None:
        """
        Initializes the OpenMAGVIT2Synthesis facade.

        Args:
            runtime: An optional pre-initialized `OpenMAGVIT2Runtime` instance.
                     If None, a new `OpenMAGVIT2Runtime` will be instantiated
                     using the provided `kwargs`.
            **kwargs: Keyword arguments to pass to the `OpenMAGVIT2Runtime`
                      constructor if `runtime` is not provided.
        """
        super().__init__()
        # Initialize the runtime, either by using the provided instance or creating a new one.
        self.runtime = runtime or OpenMAGVIT2Runtime(**kwargs)

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying `OpenMAGVIT2Runtime` instance.

        This method is called when an attribute is not found in the
        `OpenMAGVIT2Synthesis` instance itself. It allows direct access to
        methods and properties of the `self.runtime` object as if they were
        part of this facade.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying `OpenMAGVIT2Runtime` instance.

        Raises:
            AttributeError: If the attribute 'runtime' itself is accessed via
                            __getattr__, or if the attribute does not exist
                            on the underlying runtime.
        """
        # Prevent infinite recursion if 'runtime' itself is somehow accessed via __getattr__.
        # 'runtime' should be accessed directly via self.runtime.
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "OpenMAGVIT2Synthesis":
        """
        Creates an `OpenMAGVIT2Synthesis` instance by loading a pre-trained
        Open-MAGVIT2 model.

        This class method acts as a convenience constructor, delegating the
        loading of the pre-trained model to the `OpenMAGVIT2Runtime`.

        Args:
            *args: Positional arguments to pass to `OpenMAGVIT2Runtime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `OpenMAGVIT2Runtime.from_pretrained`.

        Returns:
            An instance of `OpenMAGVIT2Synthesis` with a pre-trained runtime.
        """
        return cls(runtime=OpenMAGVIT2Runtime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Performs a prediction using the underlying Open-MAGVIT2 runtime.

        This method delegates the actual prediction task to the `predict`
        method of the `OpenMAGVIT2Runtime` instance.

        Args:
            *args: Positional arguments to pass to `self.runtime.predict`.
            **kwargs: Keyword arguments to pass to `self.runtime.predict`.

        Returns:
            A dictionary containing the prediction results from the runtime.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["OpenMAGVIT2Synthesis"]