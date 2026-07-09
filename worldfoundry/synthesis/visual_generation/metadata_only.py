"""
This module defines a marker synthesis backend, `MetadataOnlySynthesis`, for models that are
listed in WorldFoundry but do not have an in-tree runnable implementation.
It also defines a custom error, `MetadataOnlyModelError`, raised when attempting to run such a model.
"""

from __future__ import annotations

from typing import Any

try:
    # Attempt to import the real BaseSynthesis from the synthesis framework.
    from ..base_synthesis import BaseSynthesis
except ModuleNotFoundError as exc:
    # If the ModuleNotFoundError is not for 'torch', re-raise it.
    # This check allows the module to be imported even if the full torch-dependent
    # synthesis framework is not installed, as long as the exception is specifically
    # about 'torch'.
    if exc.name != "torch":
        raise

    class BaseSynthesis:  # type: ignore[no-redef]
        """
        A placeholder base class for synthesis backends.

        This stub is used when the full `torch`-dependent `BaseSynthesis`
        from `..base_synthesis` is not available (e.g., if torch is not installed).
        It allows `MetadataOnlySynthesis` to be defined without full framework dependencies.
        """

        def __init__(self) -> None:
            """
            Initializes the BaseSynthesis object.

            In this stub implementation, it performs no actions.
            """
            pass


class MetadataOnlyModelError(RuntimeError):
    """Raised when a catalog-only model is requested as a runnable backend."""


class MetadataOnlySynthesis(BaseSynthesis):
    """
    A marker synthesis backend for models that are documented in WorldFoundry
    (e.g., with official source, paper, checkpoints, and environment notes)
    but do not have an in-tree runnable implementation within the `worldfoundry` framework.

    Attempts to use this backend for actual generation (e.g., via `from_pretrained`,
    `predict`, or `api_init`) will raise a `MetadataOnlyModelError`.

    Attributes:
        MODEL_ID: A default string identifier for the model.
        DISPLAY_NAME: A user-friendly name for the model, used for display purposes.
        IN_TREE_BACKEND: A boolean flag indicating whether this backend provides
                         runnable synthesis logic. Always `False` for `MetadataOnlySynthesis`.
    """

    MODEL_ID: str | None = None
    DISPLAY_NAME: str | None = None
    IN_TREE_BACKEND = False

    def __init__(self, model_id: str | None = None) -> None:
        """
        Initializes the MetadataOnlySynthesis marker.

        Args:
            model_id: An optional string identifier for the model.
                      If not provided, it falls back to `MODEL_ID` or the class name.
        """
        super().__init__()
        # Determine the effective model ID, prioritizing the passed `model_id`,
        # then the class-level `MODEL_ID`, and finally the class name.
        self.model_id = model_id or self.MODEL_ID or self.__class__.__name__
        # Set the display name, preferring the class-level `DISPLAY_NAME` over the resolved `model_id`.
        self.model_name = self.DISPLAY_NAME or self.model_id

    @classmethod
    def _error_message(cls, model_id: str | None = None) -> str:
        """
        Generates a standard error message for metadata-only models.

        This message explains that the model is cataloged for provenance but not runnable
        within the current framework, guiding the user on how to port it.

        Args:
            model_id: An optional string identifier for the model to include in the message.
                      If not provided, it falls back to `MODEL_ID` or the class name.
        Returns:
            A string containing the informative error message.
        """
        # Determine the model identifier to be used in the error message,
        # prioritizing the passed `model_id`, then the class-level `MODEL_ID`,
        # and finally the class name.
        resolved = model_id or cls.MODEL_ID or cls.__name__
        return (
            f"{resolved} is metadata-only in WorldFoundry: official source, paper, checkpoint, "
            "and environment notes may be recorded for provenance, but no runnable in-tree "
            "synthesis backend is claimed. Port the model architecture into worldfoundry "
            "before registering it as a pipeline adapter."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "MetadataOnlySynthesis":
        """
        Attempts to load a metadata-only model, but always raises an error.

        This method is overridden to explicitly prevent instantiation for runnable tasks,
        as this backend is a placeholder for non-ported models. It serves as a clear
        signal that models associated with this backend are for cataloging purposes only.

        Args:
            pretrained_model_path: Path to a pretrained model (ignored).
            args: Additional arguments (ignored).
            device: The device to load the model on (ignored).
            model_id: An optional identifier for the model to include in the error message.
            kwargs: Arbitrary keyword arguments (ignored).

        Raises:
            MetadataOnlyModelError: Always raised, indicating that the model cannot be run.
        """
        # Explicitly deletes unused arguments to ensure they are not accidentally used
        # and to signal intent that these parameters are ignored by this backend.
        del pretrained_model_path, args, device, kwargs
        raise MetadataOnlyModelError(cls._error_message(model_id))

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Attempts to perform prediction, but always raises an error.

        As this is a metadata-only backend, it does not support actual prediction.
        This method ensures that any attempt to use it for generation results in an error.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            dict[str, Any]: This method never returns; it always raises an error.

        Raises:
            MetadataOnlyModelError: Always raised, indicating that the model cannot be run.
        """
        # Explicitly deletes unused arguments to ensure they are not accidentally used
        # and to signal intent that these parameters are ignored by this backend.
        del args, kwargs
        raise MetadataOnlyModelError(self._error_message(self.model_id))

    def api_init(self, api_key: str, endpoint: str) -> None:
        """
        Attempts to initialize an API connection, but always raises an error.

        This method is a placeholder for API-based models and is not supported
        by a metadata-only backend. It ensures that any attempt to use this backend
        with API credentials results in an error.

        Args:
            api_key: The API key (ignored).
            endpoint: The API endpoint (ignored).

        Raises:
            MetadataOnlyModelError: Always raised, indicating that the model cannot be run.
        """
        # Explicitly deletes unused arguments to ensure they are not accidentally used
        # and to signal intent that these parameters are ignored by this backend.
        del api_key, endpoint
        raise MetadataOnlyModelError(self._error_message(self.model_id))