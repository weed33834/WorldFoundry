"""
WorldFoundry synthesis wrapper for NeoVerse navigation video generation.

This module provides a `NeoVerseSynthesis` class that acts as an adapter
for the `NeoVerseOfficialRuntime` to integrate with the WorldFoundry synthesis
framework. It allows generating navigation videos using the NeoVerse model
by delegating calls to an underlying runtime instance.
"""
from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.neoverse.worldfoundry_runtime import (
    DEFAULT_PROMPT,
    NeoVerseOfficialRuntime,
)

from ...base_synthesis import BaseSynthesis


class NeoVerseSynthesis(BaseSynthesis):
    """WorldFoundry synthesis wrapper for NeoVerse navigation video generation.

    This class adapts the `NeoVerseOfficialRuntime` for use within the WorldFoundry
    synthesis framework, specifically for generating navigation videos. It provides
    methods to initialize, load pretrained models, and perform predictions by
    delegating to an internal `NeoVerseOfficialRuntime` instance.
    """

    def __init__(
        self,
        runtime: NeoVerseOfficialRuntime | None = None,
        **runtime_kwargs: Any,
    ) -> None:
        """Initializes the NeoVerseSynthesis wrapper.

        Args:
            runtime: An optional pre-initialized `NeoVerseOfficialRuntime` instance.
                     If None, a new runtime will be created using `runtime_kwargs`.
            **runtime_kwargs: Keyword arguments passed to `NeoVerseOfficialRuntime`
                              if a new runtime instance needs to be created.
        """
        super().__init__()
        # If a runtime is not provided, initialize a new NeoVerseOfficialRuntime instance
        # using the provided keyword arguments.
        self.runtime = runtime or NeoVerseOfficialRuntime(**runtime_kwargs)

    @classmethod
    def bundled_runtime_root(cls) -> str:
        """Returns the root directory where the NeoVerse runtime is bundled.

        This delegates to the `bundled_runtime_root` method of the underlying
        `NeoVerseOfficialRuntime` class.

        Returns:
            The string path to the bundled runtime root directory.
        """
        return NeoVerseOfficialRuntime.bundled_runtime_root()

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "NeoVerseSynthesis":
        """Instantiates `NeoVerseSynthesis` from a pretrained NeoVerse runtime.

        This class method acts as a factory to create a `NeoVerseSynthesis` instance
        by first loading a `NeoVerseOfficialRuntime` from pretrained weights.

        Args:
            *args: Positional arguments passed to `NeoVerseOfficialRuntime.from_pretrained`.
            **kwargs: Keyword arguments passed to `NeoVerseOfficialRuntime.from_pretrained`.

        Returns:
            A new instance of `NeoVerseSynthesis` initialized with the pretrained runtime.
        """
        return cls(NeoVerseOfficialRuntime.from_pretrained(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the underlying runtime object.

        This method allows attributes (e.g., methods or properties) of the
        `NeoVerseOfficialRuntime` instance to be accessed directly through the
        `NeoVerseSynthesis` wrapper.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The value of the attribute from the underlying `self.runtime` object.

        Raises:
            AttributeError: If the attribute 'runtime' itself is requested
                            (to prevent infinite recursion), or if the attribute
                            does not exist on the underlying runtime object.
        """
        # Prevent infinite recursion if 'runtime' attribute is accessed directly on self.runtime.
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Performs a prediction using the underlying NeoVerse runtime.

        This method forwards all arguments to the `predict` method of the
        `NeoVerseOfficialRuntime` instance.

        Args:
            *args: Positional arguments passed to `self.runtime.predict`.
            **kwargs: Keyword arguments passed to `self.runtime.predict`.

        Returns:
            The result of the prediction from the `NeoVerseOfficialRuntime`.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["DEFAULT_PROMPT", "NeoVerseSynthesis"]