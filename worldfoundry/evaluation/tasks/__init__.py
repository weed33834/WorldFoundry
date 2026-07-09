"""Lazy public facade for task, benchmark, YAML, and runner discovery."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES: dict[str, str] = {
    "BENCHMARK_YAML_SCHEMA_VERSION": "worldfoundry.evaluation.tasks.catalog.yaml",
    "TASK_YAML_SCHEMA_VERSION": "worldfoundry.evaluation.tasks.catalog.yaml",
    "BenchmarkSpec": "worldfoundry.evaluation.tasks.catalog",
    "CatalogRegistry": "worldfoundry.evaluation.tasks.catalog",
    "DuplicateTaskRegistryKeyError": "worldfoundry.evaluation.tasks.catalog.registry",
    "TaskRegistry": "worldfoundry.evaluation.tasks.catalog.registry",
    "TaskRegistryEntry": "worldfoundry.evaluation.tasks.catalog.registry",
    "TaskRegistryError": "worldfoundry.evaluation.tasks.catalog.registry",
    "UnknownTaskRegistryKeyError": "worldfoundry.evaluation.tasks.catalog.registry",
    "WorldTaskConfig": "worldfoundry.evaluation.tasks.catalog",
    "benchmark_spec_from_yaml_mapping": "worldfoundry.evaluation.tasks.catalog.yaml",
    "iter_task_yaml_paths": "worldfoundry.evaluation.tasks.catalog.registry",
    "load_benchmark_zoo_registry": "worldfoundry.evaluation.tasks.catalog.zoo_registry",
    "load_benchmark_yaml": "worldfoundry.evaluation.tasks.catalog.yaml",
    "load_catalog_yaml": "worldfoundry.evaluation.tasks.catalog.yaml",
    "load_task_registry_from_paths": "worldfoundry.evaluation.tasks.catalog.registry",
    "load_task_yaml": "worldfoundry.evaluation.tasks.catalog.yaml",
    "load_world_task_yaml": "worldfoundry.evaluation.tasks.catalog.yaml",
    "load_yaml_mapping_with_extends": "worldfoundry.evaluation.tasks.catalog.yaml",
    "validate_task_yaml_file": "worldfoundry.evaluation.tasks.catalog.registry",
    "world_task_config_from_yaml_mapping": "worldfoundry.evaluation.tasks.catalog.yaml",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Dynamically import and resolve exported modules on demand (lazy loading).

    Args:
        name: The name of the attribute to import.

    Returns:
        The imported attribute value.

    Raises:
        AttributeError: If the name is not in the export mapping.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Provide a list of all exported attributes and defined globals for autocompletion.

    Returns:
        A sorted list of attribute names.
    """
    return sorted({*globals(), *__all__})
