"""Multimodal memory ABC: ingestion, retrieval, compression, and lifecycle hooks."""

from abc import ABC
from typing import Any, Iterable, Mapping

from .store import MemoryStore


class BaseMemory(ABC):
    """Generic multimodal memory template for VLM and generative tasks.

    Subclasses implement the five-stage pipeline:

    Command methods (mutate state):
        - ``record(data, ...)`` — ingest raw interaction data.
        - ``manage()`` — evict, merge, or consolidate memories.

    Query methods (read state):
        - ``select(context_query, ...)`` — retrieve relevant snippets.
        - ``compress(memory_items, ...)`` — distill selected memories.
        - ``process(refined_data, ...)`` — adapt memories to model input formats.
    """

    TEMPLATE_KEYS = ("content", "type", "timestamp", "metadata")
    SUPPORTED_TYPES = ("image", "video", "text", "audio", "action", "other")

    def __init__(self, capacity=None, **kwargs):
        """Initialize bounded storage and optional capacity limit."""
        del kwargs
        self.capacity = capacity
        self._store = MemoryStore(capacity=capacity)

    @property
    def storage(self) -> list[dict[str, Any]]:
        return self._store.records

    @storage.setter
    def storage(self, records: Iterable[Mapping[str, Any]]) -> None:
        self._store = MemoryStore(capacity=self.capacity, records=records)

    def check_template(self, **kwargs):
        """Return required record keys and supported content types."""
        return {"required_keys": self.TEMPLATE_KEYS, "supported_types": self.SUPPORTED_TYPES}

    def append_record(
        self,
        content: Any,
        *,
        kind: str = "other",
        timestamp: int | float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._store.append(content, kind=kind, timestamp=timestamp, metadata=metadata)

    def latest_record(self, prefer_type: str | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        return self._store.latest(prefer_type=prefer_type, metadata=metadata)

    def reset_records(self) -> None:
        self._store.reset()

    def record(self, data, metadata=None, **kwargs):
        """Ingest raw interaction data and assign metadata tags."""
        raise NotImplementedError(f"{type(self).__name__}.record() must be implemented by subclasses.")

    def select(self, context_query, **kwargs):
        """Retrieve memory snippets relevant to the current task context."""
        raise NotImplementedError(f"{type(self).__name__}.select() must be implemented by subclasses.")

    def compress(self, _memory_items, **kwargs):
        """Distill selected memories to reduce dimensionality or token count."""
        raise NotImplementedError(f"{type(self).__name__}.compress() must be implemented by subclasses.")

    def process(self, _refined_data, _target_format="kv_cache", **kwargs):
        """Convert refined memories into a model-ready format such as KV cache."""
        raise NotImplementedError(f"{type(self).__name__}.process() must be implemented by subclasses.")

    def manage(self, **kwargs):
        """Maintain memory lifecycle: eviction, merging, and STM→LTM transfer."""
        raise NotImplementedError(f"{type(self).__name__}.manage() must be implemented by subclasses.")
