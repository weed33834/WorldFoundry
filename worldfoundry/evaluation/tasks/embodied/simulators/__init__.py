"""Native embodied simulator interfaces and registry."""

from worldfoundry.evaluation.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evaluation.tasks.embodied.simulators.registry import (
    SIMULATOR_ENTRIES,
    SimulatorEntry,
    get_simulator_entry,
    list_simulator_ids,
    resolve_simulator_class,
)

__all__ = [
    "BaseSimulator",
    "SIMULATOR_ENTRIES",
    "SimulatorEntry",
    "StepResult",
    "get_simulator_entry",
    "list_simulator_ids",
    "resolve_simulator_class",
]
