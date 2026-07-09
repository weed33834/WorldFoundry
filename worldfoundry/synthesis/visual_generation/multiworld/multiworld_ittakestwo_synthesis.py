"""A synthesis facade for interacting with the MultiWorld ItTakesTwo runtime.

This module provides a high-level interface (`MultiWorldItTakesTwoSynthesis`)
to the `MultiWorldItTakesTwoRuntime` for visual generation tasks. It acts as
a thin wrapper, delegating core functionality like initialization and prediction
to the underlying runtime implementation.
"""

from __future__ import annotations

from typing import Any, Optional

from worldfoundry.synthesis.visual_generation.multiworld.ittakestwo_runtime import (
    MultiWorldItTakesTwoRuntime,
)

from ...base_synthesis import BaseSynthesis


class MultiWorldItTakesTwoSynthesis(BaseSynthesis):
    """Thin synthesis facade over the base-model MultiWorld ItTakesTwo runtime.

    This class simplifies interaction with the MultiWorld ItTakesTwo visual generation
    runtime by providing a unified interface for initialization and prediction,
    acting as a convenience layer.
    """

    def __init__(
        self,
        runtime_root: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *,
        runtime: Optional[MultiWorldItTakesTwoRuntime] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the MultiWorldItTakesTwoSynthesis layer.

        This constructor allows for initialization in two ways:
        1. By providing an existing `MultiWorldItTakesTwoRuntime` instance via the `runtime` argument.
        2. By providing `runtime_root`, `config_path`, and `checkpoint_path` to
           construct a new `MultiWorldItTakesTwoRuntime` internally.

        Args:
            runtime_root: The root directory for the runtime environment. Required if `runtime` is not provided.
            config_path: Path to the configuration file for the runtime. Required if `runtime` is not provided.
            checkpoint_path: Path to the model checkpoint. Required if `runtime` is not provided.
            runtime: An optional pre-initialized `MultiWorldItTakesTwoRuntime` instance.
            **kwargs: Additional keyword arguments passed directly to the `MultiWorldItTakesTwoRuntime`
                      constructor if a new runtime is being created.

        Raises:
            ValueError: If `runtime` is not provided and any of `runtime_root`,
                        `config_path`, or `checkpoint_path` are missing.
        """
        super().__init__()
        if runtime is None:
            # If no runtime instance is explicitly provided, all necessary path arguments
            # must be supplied to construct a new MultiWorldItTakesTwoRuntime internally.
            if runtime_root is None or config_path is None or checkpoint_path is None:
                raise ValueError(
                    "runtime_root, config_path, and checkpoint_path are required when runtime is not provided."
                )
            runtime = MultiWorldItTakesTwoRuntime(
                runtime_root=runtime_root,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                **kwargs,
            )
        self.runtime = runtime

    @property
    def runtime_root(self) -> str:
        """
        The root directory of the underlying MultiWorldItTakesTwo runtime environment.

        Returns:
            The runtime root path as a string.
        """
        return self.runtime.runtime_root

    @property
    def config_path(self) -> str:
        """
        The path to the configuration file used by the underlying runtime.

        Returns:
            The configuration path as a string.
        """
        return self.runtime.config_path

    @property
    def checkpoint_path(self) -> str:
        """
        The path to the model checkpoint used by the underlying runtime.

        Returns:
            The checkpoint path as a string.
        """
        return self.runtime.checkpoint_path

    @property
    def python_executable(self) -> str:
        """
        The path to the Python executable used by the underlying runtime.

        Returns:
            The Python executable path as a string.
        """
        return self.runtime.python_executable

    @property
    def device(self) -> str:
        """
        The compute device (e.g., 'cpu', 'cuda:0') on which the underlying runtime operates.

        Returns:
            The device identifier as a string.
        """
        return self.runtime.device

    @property
    def defaults(self) -> dict[str, Any]:
        """
        Default parameters or settings configured for the underlying runtime.

        Returns:
            A dictionary of default settings.
        """
        return self.runtime.defaults

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "MultiWorldItTakesTwoSynthesis":
        """
        Constructs a MultiWorldItTakesTwoSynthesis instance by loading a pretrained runtime.

        This class method acts as a convenience constructor, delegating the loading
        process to `MultiWorldItTakesTwoRuntime.from_pretrained`.

        Args:
            *args: Positional arguments to pass to `MultiWorldItTakesTwoRuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `MultiWorldItTakesTwoRuntime.from_pretrained`.

        Returns:
            An instance of `MultiWorldItTakesTwoSynthesis` with the loaded runtime.
        """
        return cls(runtime=MultiWorldItTakesTwoRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args, **kwargs):
        """
        Performs a prediction using the underlying MultiWorldItTakesTwo runtime.

        All arguments are passed directly to the `predict` method of the
        contained `MultiWorldItTakesTwoRuntime` instance.

        Args:
            *args: Positional arguments for the runtime's predict method.
            **kwargs: Keyword arguments for the runtime's predict method.

        Returns:
            The result of the runtime's predict method.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["MultiWorldItTakesTwoSynthesis"]