"""
Adapter module for integrating the Hunyuan World Voyager runtime into a broader synthesis framework.

This module provides the `HunyuanWorldVoyagerSynthesis` class, which acts as a lightweight wrapper
around the `HunyuanWorldVoyagerRuntime`. It facilitates the use of the Voyager runtime's
capabilities (like prediction and input generation) within a common synthesis interface
defined by `BaseSynthesis`. It also re-exports key components and utilities from the
`hunyuan_world_voyager.runtime` module for convenience.
"""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.runtime import (
    HunyuanWorldVoyagerRuntime,
    get_1d_rotary_pos_embed_riflex,
    load_models,
    load_runtime,
    parallelize_transformer,
)

from ...base_synthesis import BaseSynthesis


class HunyuanWorldVoyagerSynthesis(BaseSynthesis):
    """
    Thin synthesis adapter around the canonical Voyager runtime.

    This class extends `BaseSynthesis` and delegates most of its core functionality
    to an encapsulated `HunyuanWorldVoyagerRuntime` instance. It provides a consistent
    interface for integrating the Voyager model into synthesis workflows.
    """

    def __init__(self, runtime: HunyuanWorldVoyagerRuntime):
        """
        Initializes the HunyuanWorldVoyagerSynthesis adapter.

        Args:
            runtime: An initialized instance of `HunyuanWorldVoyagerRuntime`
                     that this adapter will wrap and delegate calls to.
        """
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "HunyuanWorldVoyagerSynthesis":
        """
        Creates an instance of HunyuanWorldVoyagerSynthesis by loading a pretrained Voyager runtime.

        This is a convenience class method that directly calls `HunyuanWorldVoyagerRuntime.from_pretrained`
        and then wraps the resulting runtime object.

        Args:
            *args: Positional arguments to pass to `HunyuanWorldVoyagerRuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `HunyuanWorldVoyagerRuntime.from_pretrained`.

        Returns:
            A new instance of `HunyuanWorldVoyagerSynthesis` with the loaded runtime.
        """
        return cls(HunyuanWorldVoyagerRuntime.from_pretrained(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the underlying `HunyuanWorldVoyagerRuntime` instance.

        If an attribute is not found directly on this adapter class, Python will
        attempt to retrieve it from the `self.runtime` object. This allows direct
        access to methods and properties of the wrapped runtime.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the `self.runtime` object.

        Raises:
            AttributeError: If the attribute is not found on `self.runtime`.
        """
        return getattr(self.runtime, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the prediction logic by delegating to the underlying runtime's predict method.

        This method serves as a pass-through to `self.runtime.predict`.

        Args:
            *args: Positional arguments to pass to `self.runtime.predict`.
            **kwargs: Keyword arguments to pass to `self.runtime.predict`.

        Returns:
            The result of the `self.runtime.predict` call.
        """
        return self.runtime.predict(*args, **kwargs)

    def create_hunyuan_video_input(self, *args: Any, **kwargs: Any) -> Any:
        """
        Creates video input specific to the Hunyuan format by delegating to the runtime.

        This method serves as a pass-through to `self.runtime.create_hunyuan_video_input`.

        Args:
            *args: Positional arguments to pass to `self.runtime.create_hunyuan_video_input`.
            **kwargs: Keyword arguments to pass to `self.runtime.create_hunyuan_video_input`.

        Returns:
            The result of the `self.runtime.create_hunyuan_video_input` call.
        """
        return self.runtime.create_hunyuan_video_input(*args, **kwargs)


__all__ = [
    "HunyuanWorldVoyagerRuntime",
    "HunyuanWorldVoyagerSynthesis",
    "get_1d_rotary_pos_embed_riflex",
    "load_models",
    "load_runtime",
    "parallelize_transformer",
]