"""Module for defining RuntimeMemory, a state store for model-specific runtime artifacts.

This module provides the `RuntimeMemory` class, which extends `BaseMemory` to offer a
specialized memory solution for storing and retrieving runtime data specific to a
particular model. It allows associating records with a model ID and supports
common memory operations like recording, selecting, and managing entries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldfoundry.core.memory import BaseMemory


class RuntimeMemory(BaseMemory):
    """Small state store for model-specific runtime artifacts.

    This class provides a memory implementation specifically designed to store
    runtime data and results associated with a particular model. It extends
    `BaseMemory` and adds functionality to automatically tag records with a
    model ID.

    Attributes:
        MODEL_ID (str | None): A class-level default model identifier.
            If `model_id` is not provided during initialization, this value is used.
    """

    MODEL_ID: str | None = None

    def __init__(self, capacity: Optional[int] = None, model_id: str | None = None, **kwargs: Any):
        """Initializes a new instance of RuntimeMemory.

        Args:
            capacity: Optional maximum number of retained entries. If specified,
                the memory will evict older records when capacity is exceeded.
            model_id: An optional string identifier for the model associated
                with this memory instance. If None, it defaults to the class's
                `MODEL_ID`.
            **kwargs: Additional BaseMemory initialization options.
        """
        super().__init__(capacity=capacity, **kwargs)
        # Prioritize the instance-specific model_id; otherwise, fall back to the class-level MODEL_ID.
        self.model_id = model_id or self.MODEL_ID

    def record(self, data: Any, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Records new data into the memory.

        This method adds a new entry to the memory, automatically associating
        it with the `model_id` if one is set for this instance.

        Args:
            data: The actual data content to be stored. This can be any serializable object.
            metadata: Optional dictionary of additional metadata to store with the record.
                If 'type' is not present in metadata, it defaults to 'runtime_result'.
            **kwargs: Ignored keyword arguments.
        """
        del kwargs
        entry_metadata = dict(metadata or {})
        # Automatically prepend the model_id to the metadata if available.
        if self.model_id is not None:
            entry_metadata = {"model_id": self.model_id, **entry_metadata}
        self.append_record(
            data,
            kind=str(entry_metadata.get("type", "runtime_result")),
            metadata=entry_metadata,
        )

    def select(self, context_query: Any = None, prefer_type: str | None = None, **kwargs: Any):
        """Retrieves the latest record from memory.

        This method fetches the most recently added record. If `prefer_type`
        is specified, it attempts to find the latest record matching that type.

        Args:
            context_query: Ignored for this implementation.
            prefer_type: Optional string indicating a preferred type of record to retrieve.
                If provided, the method will try to find the latest record matching this type.
            **kwargs: Ignored keyword arguments.

        Returns:
            The content of the latest record (or latest matching `prefer_type`) if found,
            otherwise None.
        """
        del context_query, kwargs
        record = self.latest_record(prefer_type=prefer_type)
        return record["content"] if record is not None else None

    def manage(self, action: str = "reset", **kwargs: Any):
        """Performs management actions on the memory.

        Supported actions include 'reset' (clears all records) and 'evict'
        (removes older records to comply with capacity, if capacity is set).

        Args:
            action: The management action to perform.
                - "reset": Clears all records from memory.
                - "evict": Evicts older records if capacity is set and exceeded.
            **kwargs: Ignored keyword arguments.
        """
        del kwargs
        if action == "reset":
            self.reset_records()
            return
        # Only attempt to evict if a capacity limit has been set.
        if action == "evict" and self.capacity is not None:
            self._store.evict()