"""
Provides a `YumeSynthesis` class that acts as a thin wrapper around the `YumeRuntime` for synthesis tasks.

This module facilitates integration of the Yume visual generation runtime into the WorldFoundry framework,
allowing consistent interaction with the underlying model for prediction tasks.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.yume.worldfoundry_runtime import YumeRuntime


def _runtime_cls():
    """
    Dynamically imports and returns the YumeRuntime class.

    This function is used to lazily load the YumeRuntime class,
    avoiding potential circular import issues or unnecessary imports
    until the class is actually needed.
    """
    from worldfoundry.synthesis.visual_generation.yume.worldfoundry_runtime import YumeRuntime

    return YumeRuntime


class YumeSynthesis(BaseSynthesis):
    """
    Thin WorldFoundry synthesis wrapper around the Yume runtime.

    This class provides a high-level interface to the Yume visual generation model,
    delegating core synthesis operations to an underlying YumeRuntime instance.
    It allows for consistent integration within the WorldFoundry framework.
    """

    def __init__(self, model=None, device=None, weight_dtype=None, *, runtime: "YumeRuntime | None" = None) -> None:
        """
        Initializes the YumeSynthesis wrapper.

        Args:
            model: The model configuration or path to load, if a new runtime is created.
            device: The device to load the model onto (e.g., "cuda", "cpu").
            weight_dtype: The data type for model weights (e.g., torch.float16).
            runtime: An optional, pre-initialized YumeRuntime instance to use.
                     If None, a new YumeRuntime instance will be created using
                     the provided model, device, and weight_dtype.
        """
        super().__init__()
        # If a runtime is not provided, create a new one using the _runtime_cls factory.
        self.runtime = runtime or _runtime_cls()(model=model, device=device, weight_dtype=weight_dtype)

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying YumeRuntime instance.

        This method allows YumeSynthesis instances to behave like YumeRuntime
        instances for attributes not explicitly defined in YumeSynthesis itself.
        It enables transparent access to runtime methods and properties.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying runtime.

        Raises:
            AttributeError: If the attribute does not exist on either YumeSynthesis
                            or the wrapped YumeRuntime instance.
        """
        # Safely get the 'runtime' attribute from the instance's dictionary
        # to prevent infinite recursion if 'runtime' itself is not yet set.
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
    ) -> "YumeSynthesis":
        """
        Creates a YumeSynthesis instance by loading a pretrained Yume model.

        This class method acts as a factory, delegating the actual model loading
        to the YumeRuntime's `from_pretrained` method and then wrapping
        the resulting runtime instance.

        Args:
            pretrained_model_path: The path to the pretrained model.
            device: The device to load the model onto (e.g., "cuda", "cpu").
            weight_dtype: The data type for model weights (e.g., torch.float16).
            fsdp: Flag indicating whether to use FSDP for model loading.

        Returns:
            A new YumeSynthesis instance with the pretrained model loaded.
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
        Performs a prediction using the wrapped YumeRuntime.

        All arguments are passed directly to the underlying `runtime.predict` method.

        Args:
            *args: Positional arguments to pass to `runtime.predict`.
            **kwargs: Keyword arguments to pass to `runtime.predict`.

        Returns:
            The result of the `runtime.predict` call.
        """
        return self.runtime.predict(*args, **kwargs)

    def predict_per_interaction(self, *args: Any, **kwargs: Any):
        """
        Performs a prediction per interaction using the wrapped YumeRuntime.

        All arguments are passed directly to the underlying `runtime.predict_per_interaction` method.

        Args:
            *args: Positional arguments to pass to `runtime.predict_per_interaction`.
            **kwargs: Keyword arguments to pass to `runtime.predict_per_interaction`.

        Returns:
            The result of the `runtime.predict_per_interaction` call.
        """
        return self.runtime.predict_per_interaction(*args, **kwargs)


__all__ = ["YumeSynthesis"]