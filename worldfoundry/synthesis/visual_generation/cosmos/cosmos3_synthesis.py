"""
Provides a synthesis adapter for the Cosmos3 base runtime, enabling lazy loading
and delegation of operations to the underlying Cosmos3Runtime class.

This module acts as a wrapper, allowing `Cosmos3Synthesis` to conform to the
`BaseSynthesis` interface while leveraging the functionalities of `Cosmos3Runtime`
without direct import until needed.
"""
from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis


def _runtime_cls():
    """
    Lazily imports and returns the Cosmos3Runtime class.

    This function delays the import of `Cosmos3Runtime` until it is actually
    needed, preventing potential circular dependencies or unnecessary imports
    at module load time.

    Returns:
        The Cosmos3Runtime class object.
    """
    from ....base_models.diffusion_model.video.cosmos3.worldfoundry_runtime import Cosmos3Runtime

    return Cosmos3Runtime


class Cosmos3Synthesis(BaseSynthesis):
    """
    Lazy synthesis adapter for the Cosmos3 base runtime.

    This class wraps the `Cosmos3Runtime` and delegates method calls,
    providing a `BaseSynthesis` compatible interface with lazy initialization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the Cosmos3Synthesis adapter.

        Instantiates the underlying `Cosmos3Runtime` class, passing all
        arguments to its constructor.

        Args:
            *args: Positional arguments to pass to the Cosmos3Runtime constructor.
            **kwargs: Keyword arguments to pass to the Cosmos3Runtime constructor.
        """
        self._runtime = _runtime_cls()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying `Cosmos3Runtime` instance.

        If an attribute is not found on `Cosmos3Synthesis` itself, this method
        attempts to retrieve it from the `self._runtime` instance.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the `self._runtime` instance.

        Raises:
            AttributeError: If `_runtime` itself is accessed (to prevent
                            infinite recursion) or if the attribute
                            does not exist on the underlying runtime.
        """
        # Prevent infinite recursion if accessing '_runtime' itself
        if name == "_runtime":
            raise AttributeError(name)
        return getattr(self._runtime, name)

    @classmethod
    def plan(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Calls the `plan` class method of the underlying `Cosmos3Runtime` class.

        This method acts as a proxy for the `plan` method defined on the
        `Cosmos3Runtime` class.

        Args:
            *args: Positional arguments to pass to the runtime's `plan` method.
            **kwargs: Keyword arguments to pass to the runtime's `plan` method.

        Returns:
            The result of the runtime's `plan` method, typically a dictionary
            describing a plan or configuration.
        """
        return _runtime_cls().plan(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Cosmos3Synthesis":
        """
        Creates a new `Cosmos3Synthesis` instance from a pre-trained Cosmos3 runtime.

        This class method acts as an alternative constructor, allowing for
        instantiation of `Cosmos3Synthesis` by loading a pre-trained
        `Cosmos3Runtime` model.

        Args:
            *args: Positional arguments to pass to the runtime's `from_pretrained` method.
            **kwargs: Keyword arguments to pass to the runtime's `from_pretrained` method.

        Returns:
            A new `Cosmos3Synthesis` instance wrapping the loaded pre-trained runtime.
        """
        instance = cls.__new__(cls)
        # Directly assign the runtime created by from_pretrained, bypassing __init__
        instance._runtime = _runtime_cls().from_pretrained(*args, **kwargs)
        return instance

    def api_init(self, *args: Any, **kwargs: Any) -> Any:
        """
        Initializes the API for the underlying `Cosmos3Runtime` instance.

        Delegates the call to the `api_init` method of the wrapped runtime.

        Args:
            *args: Positional arguments to pass to the runtime's `api_init` method.
            **kwargs: Keyword arguments to pass to the runtime's `api_init` method.

        Returns:
            The result of the runtime's `api_init` method.
        """
        return self._runtime.api_init(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generates a prediction using the underlying `Cosmos3Runtime` instance.

        Delegates the prediction call to the `predict` method of the wrapped runtime.

        Args:
            *args: Positional arguments to pass to the runtime's `predict` method.
            **kwargs: Keyword arguments to pass to the runtime's `predict` method.

        Returns:
            The prediction result from the runtime.
        """
        return self._runtime.predict(*args, **kwargs)


__all__ = ["Cosmos3Synthesis"]