"""
Adapter for the Cosmos Predict2.5 base runtime, providing a lazy loading and proxy pattern.

This module defines `CosmosPredict2p5Synthesis`, which acts as a wrapper around
`CosmosPredict2p5Runtime`. It defers the actual import and instantiation of the
runtime class until it's first needed, optimizing import times and resource
usage when the runtime is not immediately required.
"""
from typing import Any, TYPE_CHECKING

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from ....base_models.diffusion_model.video.cosmos2p5.worldfoundry_predict_runtime import PipelineImageInput
else:
    PipelineImageInput = Any


def _runtime_cls():
    """
    Lazily imports and returns the CosmosPredict2p5Runtime class.

    This function defers the import of the runtime class to avoid circular
    dependencies or to optimize initial module load times if the runtime
    is not always immediately used.
    """
    from ....base_models.diffusion_model.video.cosmos2p5.worldfoundry_predict_runtime import (
        CosmosPredict2p5Runtime,
    )

    return CosmosPredict2p5Runtime


class CosmosPredict2p5Synthesis(BaseSynthesis):
    """
    Lazy synthesis adapter for the Cosmos Predict2.5 base runtime.

    This class acts as a proxy, delegating most of its functionality to an
    instance of `CosmosPredict2p5Runtime`. The actual runtime class is
    only imported and instantiated when `CosmosPredict2p5Synthesis`
    is initialized, or its class methods like `plan` or `from_pretrained`
    are called, improving module load performance.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initializes the CosmosPredict2p5Synthesis adapter.

        This creates an instance of the underlying CosmosPredict2p5Runtime
        and stores it, effectively initializing the runtime.

        Args:
            *args: Positional arguments to pass to the runtime's constructor.
            **kwargs: Keyword arguments to pass to the runtime's constructor.
        """
        # Lazily instantiate the actual runtime class using the deferred import.
        self._runtime = _runtime_cls()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying runtime instance.

        If an attribute is not found directly on this synthesis object,
        it attempts to retrieve it from the `_runtime` instance.

        Args:
            name: The name of the attribute to retrieve.

        Returns:
            The value of the attribute from the underlying runtime.

        Raises:
            AttributeError: If `_runtime` itself is requested to prevent
                            infinite recursion, or if the attribute
                            does not exist on the underlying runtime.
        """
        # Prevent infinite recursion if accessing the _runtime attribute itself.
        if name == "_runtime":
            raise AttributeError(name)
        return getattr(self._runtime, name)

    @classmethod
    def plan(cls, *args: Any, **kwargs: Any):
        """
        Delegates the 'plan' class method call to the underlying runtime class.

        Args:
            *args: Positional arguments to pass to the runtime's plan method.
            **kwargs: Keyword arguments to pass to the runtime's plan method.

        Returns:
            The result of the runtime's plan method.
        """
        return _runtime_cls().plan(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        """
        Delegates the 'from_pretrained' class method call to the underlying runtime class.

        This method instantiates the `CosmosPredict2p5Synthesis` class by
        calling the underlying runtime's `from_pretrained` and wrapping
        the resulting runtime instance.

        Args:
            *args: Positional arguments to pass to the runtime's from_pretrained method.
            **kwargs: Keyword arguments to pass to the runtime's from_pretrained method.

        Returns:
            An instance of CosmosPredict2p5Synthesis initialized with the
            pretrained runtime.
        """
        # Create an uninitialized instance of the synthesis class without calling __init__.
        instance = cls.__new__(cls)
        # Call from_pretrained on the actual runtime class and assign the result to the instance's _runtime.
        instance._runtime = _runtime_cls().from_pretrained(*args, **kwargs)
        return instance

    def api_init(self, *args: Any, **kwargs: Any):
        """
        Delegates the 'api_init' method call to the underlying runtime instance.

        Args:
            *args: Positional arguments to pass to the runtime's api_init method.
            **kwargs: Keyword arguments to pass to the runtime's api_init method.

        Returns:
            The result of the runtime's api_init method.
        """
        return self._runtime.api_init(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any):
        """
        Delegates the 'predict' method call to the underlying runtime instance.

        Args:
            *args: Positional arguments to pass to the runtime's predict method.
            **kwargs: Keyword arguments to pass to the runtime's predict method.

        Returns:
            The result of the runtime's predict method.
        """
        return self._runtime.predict(*args, **kwargs)


__all__ = ["CosmosPredict2p5Synthesis", "PipelineImageInput"]