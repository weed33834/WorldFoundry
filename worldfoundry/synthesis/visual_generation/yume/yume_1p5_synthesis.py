"""
This module provides a WorldFoundry synthesis wrapper for the Yume 1.5 visual generation runtime.

It allows integration of the Yume 1.5 model into the WorldFoundry evaluation framework by
conforming to the `BaseSynthesis` interface and delegating actual model operations
to the underlying Yume 1.5 runtime.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.yume.worldfoundry_runtime import Yume1p5Runtime


def _runtime_cls():
    """
    Dynamically imports and returns the Yume1p5Runtime class.

    This helper function is used to avoid potential circular imports or to defer
    the import until the class is actually needed.
    """
    from worldfoundry.synthesis.visual_generation.yume.worldfoundry_runtime import Yume1p5Runtime

    return Yume1p5Runtime


class Yume1p5Synthesis(BaseSynthesis):
    """
    A WorldFoundry synthesis wrapper around the Yume 1.5 runtime.

    This class adapts the Yume 1.5 visual generation model to the WorldFoundry
    `BaseSynthesis` interface, enabling it to be used within the WorldFoundry
    evaluation framework. It delegates all core model operations to an
    internal `Yume1p5Runtime` instance.
    """

    def __init__(self, model=None, device=None, weight_dtype=None, *, runtime: "Yume1p5Runtime | None" = None) -> None:
        """
        Initializes the Yume1p5Synthesis wrapper.

        An existing Yume1p5Runtime instance can be provided, or a new one will
        be instantiated using the provided model configuration parameters.

        Args:
            model: The pre-trained model object or path to load.
            device: The device to load the model onto (e.g., "cuda", "cpu").
            weight_dtype: The data type for model weights (e.g., torch.float16).
            runtime: An optional, pre-initialized Yume1p5Runtime instance.
        """
        super().__init__()
        # If a runtime instance is not provided, create a new one using the _runtime_cls helper.
        self.runtime = runtime or _runtime_cls()(model=model, device=device, weight_dtype=weight_dtype)

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying Yume1p5Runtime instance.

        If an attribute is not found directly on the Yume1p5Synthesis object,
        this method attempts to retrieve it from the encapsulated `self.runtime`
        object, effectively acting as a proxy.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the runtime object.

        Raises:
            AttributeError: If the attribute is not found on either the
                            Yume1p5Synthesis object or its runtime instance.
        """
        runtime = self.__dict__.get("runtime")
        if runtime is not None:
            return getattr(runtime, name)
        raise AttributeError(name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device,
        weight_dtype,
        fsdp,
    ) -> "Yume1p5Synthesis":
        """
        Creates a Yume1p5Synthesis instance by loading a pre-trained Yume 1.5 model.

        This class method simplifies the process of initializing the wrapper
        with a model loaded from a specified path, delegating the actual
        model loading to the `Yume1p5Runtime.from_pretrained` method.

        Args:
            pretrained_model_path: The path to the directory containing the
                                   pre-trained Yume 1.5 model.
            device: The device to load the model onto (e.g., "cuda", "cpu").
            weight_dtype: The data type for model weights (e.g., torch.float16).
            fsdp: Flag indicating whether FSDP is used for distributed training.

        Returns:
            An instance of Yume1p5Synthesis with the loaded model.
        """
        runtime_cls = _runtime_cls()
        return cls(
            runtime=runtime_cls.from_pretrained(
                pretrained_model_path=pretrained_model_path,
                device=device,
                weight_dtype=weight_dtype,
                fsdp=fsdp,
            )
        )

    def predict(self, *args: Any, **kwargs: Any):
        """
        Delegates the prediction call to the underlying Yume1p5Runtime instance.

        This method acts as a passthrough, forwarding all arguments to the
        `predict` method of the encapsulated runtime object.

        Args:
            *args: Positional arguments to pass to the runtime's predict method.
            **kwargs: Keyword arguments to pass to the runtime's predict method.

        Returns:
            The prediction result from the Yume1p5Runtime.
        """
        return self.runtime.predict(*args, **kwargs)

    def predict_per_interaction(self, *args: Any, **kwargs: Any):
        """
        Delegates the per-interaction prediction call to the Yume1p5Runtime instance.

        This method acts as a passthrough, forwarding all arguments to the
        `predict_per_interaction` method of the encapsulated runtime object.

        Args:
            *args: Positional arguments to pass to the runtime's predict_per_interaction method.
            **kwargs: Keyword arguments to pass to the runtime's predict_per_interaction method.

        Returns:
            The per-interaction prediction result from the Yume1p5Runtime.
        """
        return self.runtime.predict_per_interaction(*args, **kwargs)


__all__ = ["Yume1p5Synthesis"]