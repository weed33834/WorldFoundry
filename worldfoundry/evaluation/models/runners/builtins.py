"""Built-in runtime runner definitions for the WorldFoundry model runner registry.

Registers the default ``worldfoundry.pipeline`` runner (and its aliases) as the
only builtin runtime entry.  Provides lookup helpers used by the registry
during eager registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.evaluation.tasks.embodied.rollout_runner import EmbodiedClosedLoopRunner


@dataclass(frozen=True)
class BuiltinRuntimeRunnerEntry:
    """Configuration entry for a built-in runtime runner.

    Attributes:
        name: Primary identifier for the runner.
        runner_class: The Python class that implements the runner.
        aliases: Alternative lookup keys that resolve to this entry.
        description: Human-readable summary of the runner's purpose.
    """

    name: str
    runner_class: type
    aliases: tuple[str, ...] = ()
    description: str = ""

    def keys(self) -> tuple[str, ...]:
        """Compile all unique lookup keys (name and aliases) for this entry."""
        return tuple(dict.fromkeys((self.name, *self.aliases)))

    def to_dict(self) -> dict[str, Any]:
        """Convert the builtin runner entry to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "runner_class": f"{self.runner_class.__module__}:{self.runner_class.__qualname__}",
            "description": self.description,
        }


# ── Builtin runner entries ─────────────────────────────────────────────
BUILTIN_RUNTIME_RUNNERS: tuple[BuiltinRuntimeRunnerEntry, ...] = (
    BuiltinRuntimeRunnerEntry(
        name="worldfoundry.pipeline",
        aliases=(
            "worldfoundry:pipeline",
            "worldfoundry-pipeline",
        ),
        runner_class=WorldFoundryPipelineRunner,
        description="WorldFoundry pipeline runner for data-backed pipeline bindings and runtime profiles.",
    ),
    BuiltinRuntimeRunnerEntry(
        name="worldfoundry.embodied-closed-loop",
        aliases=(
            "worldfoundry:embodied-closed-loop",
            "embodied-closed-loop",
            "embodied.rollout",
        ),
        runner_class=EmbodiedClosedLoopRunner,
        description="Native embodied simulator closed-loop rollout runner.",
    ),
)


def _runner_key(value: str) -> str:
    """Normalize a runner target key for matching."""
    return value.strip().lower().replace("_", "-")


def get_builtin_runtime_runner_class(name: str) -> type | None:
    """Retrieve a built-in runtime runner class by name or alias.

    Performs case-insensitive, dash-normalised lookup across all
    :data:`BUILTIN_RUNTIME_RUNNERS` entries.

    Args:
        name: Runner name or alias to search for.

    Returns:
        The matching ``runner_class``, or ``None`` if no entry matches.
    """
    key = _runner_key(name)
    for entry in BUILTIN_RUNTIME_RUNNERS:
        if key in {_runner_key(item) for item in entry.keys()}:
            return entry.runner_class
    return None


def list_builtin_runtime_runners() -> tuple[BuiltinRuntimeRunnerEntry, ...]:
    """Return the list of all registered built-in runtime runners."""
    return BUILTIN_RUNTIME_RUNNERS


__all__ = [
    "BUILTIN_RUNTIME_RUNNERS",
    "BuiltinRuntimeRunnerEntry",
    "EmbodiedClosedLoopRunner",
    "WorldFoundryPipelineRunner",
    "get_builtin_runtime_runner_class",
    "list_builtin_runtime_runners",
]
