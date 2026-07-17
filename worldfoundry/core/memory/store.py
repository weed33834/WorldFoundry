"""Bounded in-process memory records, queries, and deterministic retrieval scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class MemoryRecord:
    """One bounded memory entry with normalized provenance fields."""

    content: Any
    kind: str = "other"
    timestamp: int | float = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        kind = value.get("kind", value.get("type", "other"))
        return cls(
            content=value.get("content"),
            kind=str(kind),
            timestamp=value.get("timestamp", 0),
            metadata=dict(value.get("metadata") or {}),
            score=None if value.get("score") is None else float(value["score"]),
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "content": self.content,
            "type": self.kind,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.score is not None:
            row["score"] = self.score
        return row


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Selection query shared by artifact, runtime, and action memories."""

    text: str | None = None
    prefer_type: str | None = None
    metadata: Mapping[str, Any] | None = None
    top_k: int = 1
    recency_weight: float = 1.0
    metadata_weight: float = 1.0


@dataclass(slots=True)
class MemorySelection:
    """Top-k retrieval result returned by :meth:`MemoryStore.select`."""

    records: list[MemoryRecord]

    @property
    def content(self) -> Any:
        return self.records[0].content if self.records else None

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]


class MemoryStore:
    """Bounded in-process memory store used by all concrete WorldFoundry memories."""

    def __init__(self, capacity: int | None = None, records: Iterable[Mapping[str, Any] | MemoryRecord] = ()) -> None:
        self.capacity = capacity
        self.records: list[dict[str, Any]] = []
        for record in records:
            self.append_record(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def append(
        self,
        content: Any,
        *,
        kind: str = "other",
        timestamp: int | float | None = None,
        metadata: Mapping[str, Any] | None = None,
        score: float | None = None,
    ) -> dict[str, Any]:
        record = MemoryRecord(
            content=content,
            kind=str(kind),
            timestamp=len(self.records) if timestamp is None else timestamp,
            metadata=dict(metadata or {}),
            score=score,
        ).to_dict()
        self.records.append(record)
        self.evict()
        return record

    def append_record(self, record: Mapping[str, Any] | MemoryRecord) -> dict[str, Any]:
        row = record.to_dict() if isinstance(record, MemoryRecord) else MemoryRecord.from_dict(record).to_dict()
        self.records.append(row)
        self.evict()
        return row

    def latest(
        self, prefer_type: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        for record in reversed(self.records):
            if prefer_type is not None and record.get("type") != prefer_type:
                continue
            if metadata is not None and not _metadata_matches(record.get("metadata") or {}, metadata):
                continue
            return record
        return None

    def select(self, query: MemoryQuery | None = None, **overrides: Any) -> MemorySelection:
        """Rank stored records and return the top-*query.top_k* matches."""
        from .retrieval import select_records

        if query is None:
            query = MemoryQuery(**overrides)
        elif overrides:
            payload = {
                "text": query.text,
                "prefer_type": query.prefer_type,
                "metadata": query.metadata,
                "top_k": query.top_k,
                "recency_weight": query.recency_weight,
                "metadata_weight": query.metadata_weight,
                **overrides,
            }
            query = MemoryQuery(**payload)
        return MemorySelection(select_records(self.records, query))

    def evict(self) -> None:
        if self.capacity is not None and self.capacity >= 0 and len(self.records) > self.capacity:
            del self.records[: len(self.records) - self.capacity]

    def reset(self) -> None:
        self.records.clear()

    def replace(self, records: Iterable[Mapping[str, Any] | MemoryRecord]) -> None:
        self.records = []
        for record in records:
            self.append_record(record)


def _metadata_matches(record_metadata: Mapping[str, Any], query_metadata: Mapping[str, Any]) -> bool:
    for key, value in query_metadata.items():
        if record_metadata.get(key) != value:
            return False
    return True


__all__ = [
    "MemoryQuery",
    "MemoryRecord",
    "MemorySelection",
    "MemoryStore",
]
