"""Provider contract for in-tree platform detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from .types import AcceleratorDescriptor, PlatformKind


@runtime_checkable
class PlatformProvider(Protocol):
    """Structural interface implemented by accelerator probes."""

    kind: PlatformKind

    def detect(self) -> list[AcceleratorDescriptor]:
        """Return devices visible to this process, or an empty list."""


class BasePlatformProvider(ABC):
    """Optional nominal base class for built-in and test providers."""

    kind: PlatformKind

    @abstractmethod
    def detect(self) -> list[AcceleratorDescriptor]:
        """Return devices visible to this process, or an empty list."""
