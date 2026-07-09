"""Adapter for the MotionCtrl synthesis runtime, providing a high-level interface for MotionCtrl model interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from .worldfoundry_runtime import MotionCtrlRuntime


def _runtime_module():
    """Lazily imports and returns the `worldfoundry_runtime` module.

    This avoids a circular import dependency and ensures the module is only loaded
    when explicitly needed.
    """
    from . import worldfoundry_runtime

    return worldfoundry_runtime


def _runtime_cls():
    """Lazily imports and returns the `MotionCtrlRuntime` class.

    This is an internal helper to retrieve the runtime class dynamically.
    """
    return _runtime_module().MotionCtrlRuntime


# Default configuration values for MotionCtrl, sourced from the runtime module.
DEFAULT_MOTIONCTRL_CONFIG = _runtime_module().DEFAULT_MOTIONCTRL_CONFIG
DEFAULT_MOTIONCTRL_COND_DIR = _runtime_module().DEFAULT_MOTIONCTRL_COND_DIR
DEFAULT_MOTIONCTRL_CKPT = _runtime_module().DEFAULT_MOTIONCTRL_CKPT


class MotionCtrlSynthesis(BaseSynthesis):
    """Provides a synthesis adapter for MotionCtrl, wrapping the core `MotionCtrlRuntime`.

    This class offers a high-level interface consistent with the BaseSynthesis API,
    delegating actual model operations to an internal MotionCtrlRuntime instance.
    """

    MODEL_ID = "motionctrl"
    DISPLAY_NAME = "MotionCtrl"

    def __init__(self, *, runtime: "MotionCtrlRuntime | None" = None, **runtime_kwargs: Any) -> None:
        """Initializes the MotionCtrlSynthesis adapter.

        Args:
            runtime: An optional pre-initialized `MotionCtrlRuntime` instance.
                     If `None`, a new runtime will be created using `runtime_kwargs`.
            runtime_kwargs: Keyword arguments passed to the `MotionCtrlRuntime`
                            constructor if a new runtime is initialized.
        """
        super().__init__()
        # If no runtime instance is provided, create one using the default runtime class
        # and any provided keyword arguments.
        self.runtime = runtime or _runtime_cls()(**runtime_kwargs)

    def __getattr__(self, name: str):
        """Delegates attribute access to the underlying `runtime` instance.

        This allows direct access to methods and properties of the `MotionCtrlRuntime`
        instance through the `MotionCtrlSynthesis` adapter.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the `runtime` instance.

        Raises:
            AttributeError: If the 'runtime' attribute itself is being accessed
                            through delegation, to prevent infinite recursion.
        """
        # Prevent infinite recursion if 'getattr' is called on 'self.runtime' itself.
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "MotionCtrlSynthesis":
        """Factory method to create a `MotionCtrlSynthesis` instance from a pretrained model.

        This method forwards the call to the underlying `MotionCtrlRuntime.from_pretrained`
        method and wraps the resulting runtime in a new `MotionCtrlSynthesis` instance.

        Args:
            pretrained_model_path: Path to the pretrained model or its identifier.
            args: Additional arguments passed to the runtime's `from_pretrained` method.
            device: The device to load the model on (e.g., 'cpu', 'cuda').
            model_id: Identifier for the model, if applicable.
            kwargs: Additional keyword arguments passed to the runtime's `from_pretrained` method.

        Returns:
            An initialized `MotionCtrlSynthesis` instance.
        """
        runtime = _runtime_cls().from_pretrained(
            pretrained_model_path=pretrained_model_path,
            args=args,
            device=device,
            model_id=model_id,
            **kwargs,
        )
        return cls(runtime=runtime)

    def predict(self, *args: Any, **kwargs: Any):
        """Delegates the prediction call to the underlying `runtime` instance.

        This method directly passes all arguments to the `predict` method of the
        internal `MotionCtrlRuntime` instance.

        Args:
            *args: Positional arguments to pass to the runtime's `predict` method.
            **kwargs: Keyword arguments to pass to the runtime's `predict` method.

        Returns:
            The result of the runtime's predict method.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = [
    "DEFAULT_MOTIONCTRL_CKPT",
    "DEFAULT_MOTIONCTRL_COND_DIR",
    "DEFAULT_MOTIONCTRL_CONFIG",
    "MotionCtrlSynthesis",
]