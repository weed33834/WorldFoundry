"""Runs single model inference generations combined with subsequent benchmark score evaluations.

This module provides the orchestrator that takes a model (HuggingFace, custom, etc.), resolves
its configuration, generates outputs (e.g. videos), materializes output files, runs those files
through official benchmark validation tools on the host system, and writes an integrated scorecard.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import write_json, write_jsonl
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.api.artifacts import local_path_for_uri
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import resolve_benchmark_manifest_path
from worldfoundry.evaluation.tasks.execution.orchestration.run_mode import normalize_benchmark_run_mode
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution
from worldfoundry.evaluation.utils import BENCHMARK_TASK_ROOT
from worldfoundry.evaluation.reporting import RUN_SUMMARY_SCHEMA_VERSION, write_run_manifest_artifacts

from .evaluate import EvaluateRunRequest, EvaluateRunResult, execute_evaluate_run
from .plan import build_run_plan_from_task_registry, evaluate_request_from_run_plan, write_run_plan


MODEL_BENCHMARK_RUN_SCHEMA_VERSION = "worldfoundry-model-benchmark-run"
MODEL_BENCHMARK_RESULT_SCHEMA_VERSION = "worldfoundry-model-benchmark-result"
DEFAULT_BENCHMARK_TASK_ROOT = BENCHMARK_TASK_ROOT
CONTRACT_VALIDATION_ID = "contract-validation"

# Static MP4 video binary encoded in base64 utilized to fill placeholders during contract runs
_CONTRACT_FIXTURE_MP4_B64 = (
    "AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAAAhmcmVlAAAMWW1kYXQAAAGzABAHAAABthGBthUViwYKxYWViwsrFgKqYWga6QeCpsYKxYMFYsLqxYXViwLEwtCcuHgqbGKsWDBWLCysWFlYsCxMLQnLh4KmxgrFgwViwurFhdWLAsTC0KC4eCpsurFhdWLA5g8B++g8B/GpQYSh+PAgCEPFSsdq00EtWrHc/pd4u1tOmaaa8qVqmdT6yq/v/7/PMezJMV6rtBZDwGAMB4D+DEkFGENUCEAemVaynVWFw7H3qxuqy5M2rV6O2q3WgYraY3Zpd5VjSuFkSDhnoLIfYHbEbjFtU28RLdqJwLJcCiSgiAp1GDjikbxSoiPeGlik05ssrFhdWLAkKxYXViwZTC0RLh8Kmy6sWF1YsLqxYXViwaTC0Jy4fCpsurFhdWLCysWF1YsGkwtCcuHwqbNA8B+yg8BA+gHgowhpAUINEiZK39N4eCSPsqQeJ91is60JOp07bDCXrF8PPzditjExdPB+xdB8GATOA8BA5g8BA+goUgB4+BDBQjrB0Ph56F6ZIXh8ynL1atV+JgYqn90dfHW7hftH+NJVSXyr3lRaOWwWQKlWLBgrFg0wqa1uN3veqeID4RFzacDUSFShQWd4VqDSPiJZEiFbZdWLCysWDBWLC6sWDSYWiJcPkRxssrFhdWLC6sWFlYsGUwtCcuHwqbLqxYWViwurFhdWLBlMLQoSD5EcbLqxYXViwurFhdWLHTC0LUgMYbLqxYXViwurFhZWLBpMLQnLh8RNl1YsLqxYXViwurFg0mFoTlwMYbLqxYXViwsrFhZWLBpMLQnSD4ifAAABtlPAyF0MHwICcMwvMHwICsS8hMFls4cB8CAnCHkJgs5yHgfAgKQg5CYKYHSAMBgzB8CApBABgzBoJYNAh8VQvgkKRIUqR5B/8Hgv+8IXi8SC8SgUcVKviVC/xeDAdVD/ysvBRfEgDxcpB8D/xAMVCMTg3y6+xYjCxYPAQQYPAQM4PAfzolg8B/eiWEEGgNQgD4fAHj4fUA8uLwDorgkfEifHw++qVqh+Px+rg/isf+vvz//F/rZYPqPgfC/9QYAxUrhf6f8aBqXl3x98efNhfB8CApB8D/vC+D4EBSD4H/eFMDqBoDBmAcJQl+xR+9i4eEwIAMGYPgQFoQC7jgD56UMgpgdCAQBLH8+OvJRsUkwPgQFoPgQFoPgQFolHwa+8oh4Ltg+BASg+B/1hdsHwICcHwP+sWBZA+BAThchg+BAThmFzDg+BAUg+B/xhcw4PgQFIPgf8bwAAAbZVgMhcTB8CAnALC8gfAgJXBZMJOHAYIR8SchMFnOHAYIR8SchMFwsEwfAgKQfA/4wpgbCgfAgLwfAgLQDPBCiiayOAmBh8XlwHkjx9jwb499p4LmYPgQE4Pgf9YXbB8CAnB8D/rCmCMAcSQYMwyCGP35KrsVRv9nolpwGAO8aB4D+lLjQU2DBCVCSqURTVH8zu3cgj6nQ7zvD4MqEkvitQoaiUd1ImXAqQg+BAWg+BAXlyl4NYXelU9PhdQfAgKQfA/7wvg+BATg+B/2hcCxA+BAThkFwxA+BAUhkF2MHwICkHwP+ELpA+BAVg+B/4vwAAAbZXwMhYDCcD4EBSAUF7B8CArAKC7YPgQE7gqsZMTA+BAU/MhXCwRvPg+BAVg+B/whTAywfAgLQfAgKwYfD4u+JZeXD5Vikfl6ofetEZSX/UTW7nViAfUIQkApPl+b4d3FNz35/mX3vK/0eqB39UPFXoP7APgwBI+krwa/V+828LCYPAQGoPAQH4MAaDwP+eDF0APpcCEXqi+iQJPoBoENV5V+/H3bQPfhf658dqrqrVagD4BYMEJuR4BYXUHwICcAsLwcHwICcHwP+sKYH0D4EBaPoJJdRFUY3sqRidRdwhB8CAtB8CAtH0oBANVaqK29RCE8LeDwECSDwECmDAGF4MCADD4eBBA8DAhhDLwUIljyTS6KPeLghfH8BhHBCHf+AovF4/L4XKYDdBngwB05jgCwuMLB8CApB8D/vC4UQPgQFao8F2YPgQFIZhdsHwICsAsLzB8CArB8D/tQAAAbZZgMhcMQPgQFIZBfB8CApDIL4PgQE4BQVTDx58GEjTpeQBfB8CApcFYFQLiUEESfAHBCEkIKtTVQQhKLxJVzwj3w+VVX9Rtk3JzbOmQfAgJQfA/zQqUDwEESDF4PAf2IMAaDeBhLCGAd8fA0gNn/yKlBern1RePC8Slf1YIfv1X4fYIwIXgOq1Jdo8BjwNAYuBr4GLwQBIBB8qpdBLlA5eqP+/8uVzWYp9VF5vst968cDAHKe0MqyeCremtNA+BATgFhcMQPgQE4ZBQwUAwAYDfAM8DKwaAHggK1GBDEovLlP1QHx5+wFEPx5c34+VqB/7sqn0s/6nXg+BAUg+BAWhTGMHgIF0HgP4sHgID8GCCDwP+yJANB8AYXCSXhALhLHokiUJJdfRQX+HQ9+Jfi7ylTC/xcP1SoA+9HUHhf4MgYIAMCADKgeBgJQYuEoA61V8vin4lq1XsisvV/2p1aiYPor/36ou8OvCM5zgYfXLQCRLHkl06F4wfAgKQfA/7QsZwHwICnx0L4PgQFYZBdMHwICsHwP/MLzB8CApB8D/vQAAAbZbwMhcKED4EBaDBkF2wfAgKwCgvg+BAVgFBSMZyWHJTQPgQFKs8F0gfAgLQYMguCoFwfAgJwfA/0QqAqBcEAGLwa+Bi4EASQDf+quD6eA57in6tX4uVTWIp/FN7nts98Rng8BBCiUDwH9SDfpfqsuCBVQ60u95X5q1So+B/9n1cub8e+ij2KhHAJB8CArB4D/N+r8qNhYBwKlwlD+CWPi4S/aoH5cqL/a3ivyma306D4EBWD4H+OFwoQPgQFYMGQWjDuB8CApB8D/HCmDgSBhIVfUKwsHwKFUBkrbbDghB8CAtB8CAvH2nADfl0Vt7CILYMEiQAcJECAEISghTAbhd5Vs8oU2Qdq1C6v+q729rbwfAgKwh5mYYC4YgYIB8MguCIgfAgKQCgrBhHAfAgKwCgvCwfAgKwfA/7X8AAAG2XYDIXCAgHwICkHwIBULmHB8CAnDIKyTwfAgLQYMgqt6ajwYSHTxsLtg+BASuC+D4EBOGQVgTAmDD4GCGDfB4GAjCAChUD1X9XB8XQRB7+f/qtlRJ68nbzw61SGQPgQE4PAf49s9FNacFhBYPAQLoQgeA/wQhgfBgOqi6qr8dxV6z/pmAc1WuO8V7ttin0wAgHwICtWPhLLh/o8ERsXBciB8CAtBgyC6QPgQFoM4LQJAQBgQAeA/yQYu+DF4QICgHgNolKwPl2zquKLPiVFYKQe8oGL9Uqn8gHQyBhInDQPAQGqvw99/1aeFlhYPAQNJeDAgA1EsGH4QQYdA2CVQDy/R/Var/sEce+z+l6nd8PB0poHr/yhWAR7JJkmSckjUkYPAwlWn6JABxcJA9HinVbP0zUMBYbp4HwICf2PC7YPgQFcPBcOC4PgQE4Pgf94XUHwICsHwP/N8AAAG2X8DIWAgGcD4EBSD4EAmF7B8CApB8CATC+D4EBWD4EAqFUw/qanzQMEKG1RkKkweAgZweA/mweA/q1YPAQJoQgDIEIIXhLBBBlfy8FBRL0GHhcqCACgBt+XxUrVF6v9BQqvD5V6qp35f8vBRQfAwBYPAQX4PAQIoPAfvoMXgggwlgHA0BhLEgIRePgQRJBsVF4QFXy4Sx8PwgiSDAohHH9LlYQh8PYJGZBLBCwf/H4kj9V+A+B/5lxcDBCANH4QAYENUqgIQ7EUd/irVNkSyxcmUBAH/y4EPihL24bC+D4EBSD4EAmFwPgHg+BAUg+B/ihcHAmD4EBWD4H+OFmQPAQRoPAfwIPAQK4BoPAf4ZcDQEEGCEqEpWDfBoPR8rEqgeHw/ANCEJfy5RgQS/6pUDwUAuritX4Sh/+Kr4SQh/VK4PwUQPgf+YPAQe4PAQJYPAQOoPAf5asGEsA4fgwkggeAMBvBCCGB8SQYe0SBLBlIMCEJOfAOgQPF/6r8EH6qyD9XIEBWEDRJLviX6j5UD4H/nRKB4D/PANEgAwGBQeA8PB3PK76gfVRXu4O8bu+Uy4kw94GUl8VAeYMBfB8CApB8CAVC4EADQfAgKwfAgQQqgiA0PgYMxKBgzBhI8bBghKgY2FQw5w4D4EBSGQXGFg+BATw6FwRgPB8CApB8D/tC8LB8CApB8D/vcAAANlbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAo90cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAEAAAABAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAIHbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABsm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAXJzdGJsAAAA2nN0c2QAAAAAAAAAAQAAAMptcDR2AAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAAYGVzZHMAAAAAA4CAgE8AAQAEgICAQSARAAAAAAEAAAAAYogFgICALwAAAbABAAABtYkTAAABAAAAASAAxI2IAEUCBAgUYwAAAbJMYXZjNTkuMzcuMTAwBoCAgAECAAAAFGJ0cnQAAAAAAAEAAAAAYogAAAAYc3R0cwAAAAAAAAABAAAACAAACAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAABxzdHNjAAAAAAAAAAEAAAABAAAACAAAAAEAAAA0c3RzegAAAAAAAAAAAAAACAAAAlkAAAFaAAAA8wAAAV8AAAGkAAABPwAAAVwAAAINAAAAFHN0Y28AAAAAAAAAAQAAACwAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjU5LjI3LjEwMA=="
)
_CONTRACT_FIXTURE_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov"}


@dataclass(frozen=True)
class ModelBenchmarkRunRequest:
    """Configures execution parameters to run models and official benchmarks together.

    Attributes:
        output_dir: Main output workspace path.
        benchmark_id: Canonical target benchmark identifier (from the zoo).
        benchmark_manifest_path: Path on disk pointing to the benchmark zoo catalog shard.
        benchmark_mode: Target execution mode ('official-validation', 'official-run').
        model_id: Target model identifier under evaluation.
        model_runner: Overriding model runner implementation.
        model_zoo_manifest_dir: Path to Model Zoo manifest directory.
        model_variant_id: selected weights variant ID.
        model_parameters: Hyperparameters passed to the model runner.
        model_runtime: System configurations passed to model runner environment.
        model_config: Raw model parameter mappings.
        requests: Custom pre-materialized request inputs.
        requests_path: Preflight requests path on disk.
        task_name: Task key evaluated within the registry.
        task_roots: Local paths scanned to discover task yaml catalogs.
        task_benchmark: Scope constraint for registry lookups.
        task_recursive: If True, recursively scans task_roots.
        task_root_dir: Anchor directory for relative catalog paths.
        dataset_root: Physical path to dataset files on disk.
        dataset_id: Selected dataset identifier.
        split: Selected dataset split (e.g. "default", "validation").
        num_samples: Maximum count of samples evaluated.
        generated_artifact_dir: Custom output directory for generated artifacts.
        output_artifact: Override representing the expected generated asset type.
        required_artifacts: List of artifact kinds expected to be verified.
        metrics: Sequence of scorecard metrics computed.
        generation_cache_dir: Custom directory path for SQLite generation caching.
        generation_cache_mode: Caching mode.
        generation_cache_namespace: Caching namespace partition.
        run_id: Unique trace ID.
        benchmark_timeout_seconds: Bounded timeout for running official benchmarks.
        benchmark_workdir: Target working directory for benchmark runners.
        benchmark_env: Custom environment overrides for benchmark runners.
        materialize_placeholders: If True, copies placeholder media during contract runs.
        contract_fixture: If True, allows running mock contract validators without actual models.
        fail_on_generation_error: If True, fails immediately if any generation fails.
    """

    output_dir: str | Path
    benchmark_id: str
    benchmark_manifest_path: str | Path
    benchmark_mode: str = "official-run"
    model_id: str = ""
    model_runner: str | None = None
    model_zoo_manifest_dir: str | Path | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    requests: Sequence[Any] | None = None
    requests_path: str | Path | None = None
    task_name: str | None = None
    task_roots: Sequence[str | Path] | None = None
    task_benchmark: str | None = None
    task_recursive: bool = False
    task_root_dir: str | Path | None = None
    dataset_root: str | Path | None = None
    dataset_id: str | None = None
    split: str = "default"
    num_samples: int | None = None
    generated_artifact_dir: str | Path | None = None
    output_artifact: str = "generated_video"
    required_artifacts: Sequence[str] = ("generated_video",)
    metrics: Sequence[str] = ("artifact_count", "required_artifacts_present")
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "model_benchmark"
    run_id: str | None = None
    benchmark_timeout_seconds: float | None = None
    benchmark_workdir: str | Path | None = None
    benchmark_env: Mapping[str, Any] | None = None
    materialize_placeholders: bool | None = None
    contract_fixture: bool = False
    fail_on_generation_error: bool = False


@dataclass(frozen=True)
class ModelBenchmarkRunResult:
    """Encapsulates the aggregated summary, report paths, and exit codes of single run.

    Attributes:
        schema_version: Standard schema identification string.
        status: Run status ("succeeded" or "failed").
        exit_code: Process exit code.
        output_dir: Output workspace directory.
        run_manifest_path: Path to serialised run manifest JSON.
        generation_result: Detailed result metrics returned from generation.
        benchmark_result: Scorecard details returned from official evaluation.
        generated_artifact_dir: Directory containing copied generated artifact files.
        artifact_manifest_path: Path to serialized manifest indexing generated artifacts.
        artifacts: Map of produced summary and report paths.
    """
    schema_version: str
    status: str
    exit_code: int
    output_dir: Path
    run_manifest_path: Path
    generation_result: EvaluateRunResult | None
    benchmark_result: Mapping[str, Any]
    generated_artifact_dir: Path
    artifact_manifest_path: Path
    artifacts: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        """Determines if both generation and evaluation completed with success."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Converts the run outcome into a plain, serializable dictionary."""
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["run_manifest_path"] = str(self.run_manifest_path)
        payload["generated_artifact_dir"] = str(self.generated_artifact_dir)
        payload["artifact_manifest_path"] = str(self.artifact_manifest_path)
        payload["generation_result"] = None if self.generation_result is None else self.generation_result.to_dict()
        payload["benchmark_result"] = dict(self.benchmark_result)
        payload["artifacts"] = dict(self.artifacts)
        payload["ok"] = self.ok
        return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Safely decodes and parses list records from a JSONL file."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _default_task_roots(benchmark_manifest_path: str | Path) -> tuple[Path, ...]:
    """Resolves relative fallback directories scanned to search for task YAML definitions."""
    manifest_path = resolve_benchmark_manifest_path(benchmark_manifest_path)
    sibling_tasks = manifest_path.parent / "tasks"
    if sibling_tasks.exists():
        return (sibling_tasks,)
    return (DEFAULT_BENCHMARK_TASK_ROOT,)


def _default_requests(benchmark_id: str, output_artifact: str) -> tuple[GenerationRequest, ...]:
    """Generates a list containing one generic fallback GenerationRequest during contract runs."""
    return (
        GenerationRequest(
            sample_id="sample-0000",
            task_name=benchmark_id,
            inputs={"prompt": "WorldFoundry contract fixture"},
            output_schema={output_artifact: {"kind": output_artifact}},
        ),
    )


def _safe_name(value: str) -> str:
    """Sanitizes strings for safe usage in file paths and manifest IDs."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._") or "artifact"


def _artifact_suffix(name: str, uri: str) -> str:
    """Determines the standard file extension to apply when copying intermediate generated files."""
    suffix = Path(uri).suffix
    if suffix:
        return suffix
    lowered = name.lower()
    if "video" in lowered:
        return ".mp4"
    if "image" in lowered:
        return ".png"
    return ".bin"


def _write_placeholder_artifact(destination: Path, metadata: Mapping[str, Any]) -> None:
    """Writes a mock placeholder file to act as model generation output during contract runs."""
    if destination.suffix.lower() in _CONTRACT_FIXTURE_VIDEO_SUFFIXES:
        destination.write_bytes(base64.b64decode(_CONTRACT_FIXTURE_MP4_B64))
        return
    destination.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _placeholder_allowed(mode: str, explicit: bool | None, *, contract_fixture: bool = False) -> bool:
    """Dictates whether writing fake media placeholders is permitted under the active configuration."""
    if explicit is not None:
        return bool(explicit)
    return mode == "contract" and contract_fixture


def _task_plan_path(root: Path) -> Path:
    """Returns the filepath representing the intermediate task execution plan."""
    return root / "task_run_plan.json"


def _run_generation_from_task_registry(request: ModelBenchmarkRunRequest, root: Path) -> EvaluateRunResult:
    """Materializes requests and runs model generations utilising the local task catalog registry."""
    if not request.task_name:
        raise ValueError("task_name is required for task-registry materialization")
    if request.dataset_root is None:
        raise ValueError("task-registry model-benchmark runs require dataset_root/data_path to materialize requests")

    task_roots = tuple(Path(path) for path in request.task_roots) if request.task_roots else _default_task_roots(request.benchmark_manifest_path)
    plan = build_run_plan_from_task_registry(
        task_name=request.task_name,
        task_roots=task_roots,
        output_dir=root / "generation",
        benchmark=request.task_benchmark or request.benchmark_id,
        recursive=request.task_recursive,
        root_dir=request.task_root_dir,
        mode="model",
        dataset_root=request.dataset_root,
        dataset_id=request.dataset_id or f"{request.benchmark_id}:generated",
        split=request.split,
        model_id=request.model_id,
        model_runner=request.model_runner,
        model_zoo_manifest_dir=request.model_zoo_manifest_dir,
        model_variant_id=request.model_variant_id,
        model_parameters=request.model_parameters,
        model_runtime=request.model_runtime,
        model_config=request.model_config,
        metrics=tuple(request.metrics),
        required_artifacts=tuple(request.required_artifacts),
        generation_cache_dir=request.generation_cache_dir,
        generation_cache_mode=request.generation_cache_mode,
        generation_cache_namespace=request.generation_cache_namespace,
        limit=request.num_samples,
        materialize_requests=True,
        run_id=None if request.run_id is None else f"{request.run_id}:generation",
        fail_on_sample_error=request.fail_on_generation_error,
    )
    if not plan.requests:
        raise ValueError(
            "task-registry materialization produced zero requests; check task metadata_path, dataset_root/data_path, "
            "split, and num_samples"
        )
    write_run_plan(plan, _task_plan_path(root))
    return execute_evaluate_run(evaluate_request_from_run_plan(plan))


def _materialize_generated_artifacts(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str,
    allow_placeholders: bool,
) -> tuple[int, int]:
    """Copies physical output files from generation outputs into a unified benchmark input folder."""
    rows: list[dict[str, Any]] = []
    results = [
        GenerationResult.from_dict(row)
        for row in _read_jsonl(generation_output_dir / "results.jsonl")
    ]
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        for name, artifact in result.artifacts.items():
            if output_artifact and name != output_artifact:
                continue
            suffix = _artifact_suffix(name, artifact.uri)
            destination = generated_artifact_dir / f"{_safe_name(result.sample_id)}__{_safe_name(name)}{suffix}"
            source_path = local_path_for_uri(artifact.uri)
            row = {
                "sample_id": result.sample_id,
                "artifact_name": name,
                "source_uri": artifact.uri,
                "destination": str(destination),
                "status": "missing",
                "placeholder": False,
            }
            if source_path is not None and source_path.is_file():
                if source_path.resolve() != destination.resolve():
                    shutil.copy2(source_path, destination)
                row["status"] = "copied"
            elif allow_placeholders:
                _write_placeholder_artifact(
                    destination,
                    {
                        "placeholder": True,
                        "sample_id": result.sample_id,
                        "artifact_name": name,
                        "source_uri": artifact.uri,
                    },
                )
                row["status"] = "placeholder"
                row["placeholder"] = True
            rows.append(row)

    write_jsonl(artifact_manifest_path, rows, atomic=False)
    materialized_count = sum(1 for row in rows if row["status"] in {"copied", "placeholder"})
    placeholder_count = sum(1 for row in rows if row["status"] == "placeholder")
    return materialized_count, placeholder_count


def _materialize_contract_validation_artifacts(
    *,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str,
) -> tuple[int, int]:
    """Generates dummy placeholder artifacts directly when contract_fixture is executed in mock mode."""
    artifact_name = output_artifact or "generated_artifact"
    destination = generated_artifact_dir / f"sample-0000__{_safe_name(artifact_name)}{_artifact_suffix(artifact_name, '')}"
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_placeholder_artifact(
        destination,
        {
            "placeholder": True,
            "sample_id": "sample-0000",
            "artifact_name": artifact_name,
            "source": "benchmark_contract_validation",
        },
    )
    write_jsonl(
        artifact_manifest_path,
        [
            {
                "sample_id": "sample-0000",
                "artifact_name": artifact_name,
                "source_uri": "",
                "destination": str(destination),
                "status": "placeholder",
                "placeholder": True,
            }
        ],
        atomic=False,
    )
    return 1, 1


def _run_generation(request: ModelBenchmarkRunRequest, root: Path) -> EvaluateRunResult | None:
    """Dispatches generation workloads depending on selected custom request structures or task registries."""
    if request.generated_artifact_dir is not None:
        return None
    requests = request.requests
    if requests is None and request.requests_path is None:
        if request.benchmark_id == "videoverse":
            from worldfoundry.evaluation.tasks.execution.runners.videoverse.videoverse_prompts import (
                materialize_videoverse_generation_requests,
            )

            requests = materialize_videoverse_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "VideoVerse prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_VIDEOVERSE_ROOT or WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "phyfps-bench-gen":
            from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_prompts import (
                materialize_phyfps_generation_requests,
            )

            requests = materialize_phyfps_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "PhyFPS-Bench-Gen prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYFPS_BENCH_GEN_ROOT or WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "physvidbench":
            from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_prompts import (
                materialize_physvidbench_generation_requests,
            )

            requests = materialize_physvidbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "PhysVidBench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYSVIDBENCH_ROOT or WORLDFOUNDRY_PHYSVIDBENCH_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "physics-iq":
            from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
                materialize_physics_iq_generation_requests,
            )

            requests = materialize_physics_iq_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "Physics-IQ prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYSICS_IQ_ROOT or WORLDFOUNDRY_PHYSICS_IQ_DESCRIPTIONS."
                )
        elif request.benchmark_id == "phygenbench":
            from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import (
                materialize_phygenbench_generation_requests,
            )

            requests = materialize_phygenbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "PhyGenBench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYGENBENCH_ROOT or WORLDFOUNDRY_PHYGENBENCH_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "videophy":
            from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_prompts import (
                materialize_videophy_generation_requests,
            )

            requests = materialize_videophy_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "VideoPhy prompt materialization produced zero requests; bundled assets are missing."
                )
        elif request.benchmark_id == "videophy2":
            from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_prompts import (
                materialize_videophy2_generation_requests,
            )

            requests = materialize_videophy2_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "VideoPhy2 prompt materialization produced zero requests; bundled assets are missing."
                )
        elif request.benchmark_id == "mirabench":
            from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_prompts import (
                materialize_mirabench_generation_requests,
            )

            requests = materialize_mirabench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "MiraBench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_MIRABENCH_ROOT or WORLDFOUNDRY_MIRABENCH_META_CSV."
                )
        elif request.benchmark_id == "phyground":
            from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_prompts import (
                materialize_phyground_generation_requests,
            )

            requests = materialize_phyground_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "PhyGround prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYGROUND_DATA_ROOT or WORLDFOUNDRY_PHYGROUND_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "phyeduvideo":
            from worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.phyeduvideo_prompts import (
                materialize_phyeduvideo_generation_requests,
            )

            requests = materialize_phyeduvideo_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "PhyEduVideo prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_PHYEDUVIDEO_ROOT or WORLDFOUNDRY_PHYEDUVIDEO_PROMPTS_FILE."
                )
        elif request.benchmark_id == "aigcbench":
            from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_prompts import (
                materialize_aigcbench_generation_requests,
            )

            requests = materialize_aigcbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "AIGCBench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_AIGCBENCH_DATASET_ROOT or WORLDFOUNDRY_AIGCBENCH_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "vmbench":
            from worldfoundry.evaluation.tasks.execution.runners.vmbench.vmbench_prompts import (
                materialize_vmbench_generation_requests,
            )

            requests = materialize_vmbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "VMBench prompt materialization produced zero requests; bundled prompts should be available "
                    "under worldfoundry/data/benchmarks/assets/vmbench/prompts/."
                )
        elif request.benchmark_id == "world-in-world":
            from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
                materialize_world_in_world_generation_requests,
            )

            requests = materialize_world_in_world_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "World-in-World prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_WORLD_IN_WORLD_ASSETS_ROOT or episode manifest env vars."
                )
        elif request.benchmark_id == "iworld-bench":
            from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
                materialize_iworldbench_generation_requests,
            )

            requests = materialize_iworldbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "iWorld-Bench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT or WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "ipv-bench":
            from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_prompts import (
                materialize_ipv_bench_generation_requests,
            )

            requests = materialize_ipv_bench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "IPV-Bench prompt materialization produced zero requests; set "
                    "WORLDFOUNDRY_IPV_BENCH_ROOT or WORLDFOUNDRY_IPV_BENCH_PROMPT_MANIFEST."
                )
        elif request.benchmark_id == "ewmbench":
            from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_prompts import (
                materialize_ewmbench_generation_requests,
            )

            requests = materialize_ewmbench_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "EWMBench prompt materialization produced zero requests; bundled task_manifest.json "
                    "should be available under worldfoundry/data/benchmarks/assets/ewmbench/."
                )
        elif request.benchmark_id == "evalcrafter":
            from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
                materialize_evalcrafter_generation_requests,
            )

            requests = materialize_evalcrafter_generation_requests(limit=request.num_samples)
            if not requests:
                raise ValueError(
                    "EvalCrafter prompt materialization produced zero requests; bundled prompt700.txt "
                    "should be available under worldfoundry/data/benchmarks/assets/evalcrafter/."
                )
        if request.task_name and requests is None:
            return _run_generation_from_task_registry(request, root)
        if request.contract_fixture:
            return None
        if requests is None:
            raise ValueError(
                "model-benchmark runs require generated inputs. Provide task_name+dataset_root, "
                "requests_path, generated_artifact_dir, or set contract_fixture=True for benchmark "
                "contract validation placeholders."
            )
    return execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=root / "generation",
            mode="model",
            requests=requests,
            requests_path=request.requests_path,
            metrics=tuple(request.metrics),
            required_artifacts=tuple(request.required_artifacts),
            benchmark={
                "suite": "benchmark_zoo",
                "benchmark_name": request.benchmark_id,
                "task_type": request.task_name or request.benchmark_id,
                "evaluation_protocol": "model_generation",
            },
            model_id=request.model_id,
            model_runner=request.model_runner,
            model_zoo_manifest_dir=request.model_zoo_manifest_dir,
            model_variant_id=request.model_variant_id,
            model_parameters=request.model_parameters,
            model_runtime=request.model_runtime,
            model_config=request.model_config,
            generation_cache_dir=request.generation_cache_dir,
            generation_cache_mode=request.generation_cache_mode,
            generation_cache_namespace=request.generation_cache_namespace,
            dataset_id=f"{request.benchmark_id}:generated",
            run_id=None if request.run_id is None else f"{request.run_id}:generation",
            fail_on_sample_error=request.fail_on_generation_error,
        )
    )


def _model_benchmark_run_summary(
    *,
    request: ModelBenchmarkRunRequest,
    status: str,
    mode: str,
    root: Path,
    materialized_count: int,
    placeholder_count: int,
    generation_result: EvaluateRunResult | None,
    benchmark_payload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compiles the primary run summary record, detailing generation and benchmark execution metrics."""
    sample_count = generation_result.sample_count if generation_result is not None else materialized_count
    successful_samples = (
        generation_result.successful_sample_count if generation_result is not None else materialized_count
    )
    failed_samples = generation_result.failed_sample_count if generation_result is not None else 0
    generation_success_rate = successful_samples / sample_count if sample_count else 0.0
    benchmark_ok = 1.0 if benchmark_payload.get("ok") is True else 0.0
    leaderboard_valid = (
        status == "succeeded"
        and benchmark_payload.get("ok") is True
        and not request.contract_fixture
        and placeholder_count == 0
    )
    score_valid = leaderboard_valid
    leaderboard_blockers = _model_benchmark_leaderboard_blockers(status, benchmark_payload)
    if request.contract_fixture and "model-benchmark run used contract fixture" not in leaderboard_blockers:
        leaderboard_blockers.append("model-benchmark run used contract fixture")
    if placeholder_count and "generated artifacts include placeholders" not in leaderboard_blockers:
        leaderboard_blockers.append("generated artifacts include placeholders")
    leaderboard = {
        "materialized_artifact_count": float(materialized_count),
        "placeholder_artifact_count": float(placeholder_count),
        "real_artifact_count": float(max(materialized_count - placeholder_count, 0)),
        "generation_success_rate": float(generation_success_rate),
        "benchmark_ok": benchmark_ok,
    }
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "source_schema_version": MODEL_BENCHMARK_RUN_SCHEMA_VERSION,
        "run": {
            "run_id": request.run_id,
            "status": status,
            "started_at": None,
            "finished_at": None,
            "worldfoundry_version": None,
            "run_fingerprint": None,
        },
        "benchmark": {
            "benchmark_name": request.benchmark_id,
            "task_type": request.task_name or f"{request.benchmark_id}:model_benchmark",
            "suite": "benchmark_zoo",
            "evaluation_protocol": f"model_benchmark:{mode}",
        },
        "model": {
            "model_id": request.model_id,
            "model_name": request.model_id,
            "model_type": "model_benchmark",
        },
        "dataset": {
            "dataset_id": request.dataset_id or f"{request.benchmark_id}:generated",
            "name": request.dataset_id or f"{request.benchmark_id}:generated",
            "split": request.split,
            "sample_count": sample_count,
        },
        "counts": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": [],
        },
        "generation": {
            "successful": successful_samples,
            "failed": failed_samples,
            "materialized_artifact_count": materialized_count,
            "placeholder_artifact_count": placeholder_count,
            "real_artifact_count": max(materialized_count - placeholder_count, 0),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": {
                metric_id: {"mean": value, "higher_is_better": True}
                for metric_id, value in leaderboard.items()
            },
            "summary": {
                "sample_count": sample_count,
                "successful_samples": successful_samples,
                "failed_samples": failed_samples,
                "failed_sample_ids": [],
            },
        },
        "leaderboard": leaderboard,
        "eligibility": {
            "score_valid": score_valid,
            "leaderboard_valid": leaderboard_valid,
            "leaderboard_eligible": leaderboard_valid,
            "reasons": leaderboard_blockers,
            "blocking_reasons": leaderboard_blockers,
        },
        "artifacts": {str(key): value for key, value in artifacts.items() if value not in (None, "")},
        "wrapper": {
            "output_dir": str(root),
            "benchmark_mode": mode,
            "materialized_artifact_count": materialized_count,
            "placeholder_artifact_count": placeholder_count,
            "contract_fixture": request.contract_fixture,
        },
    }


def _model_benchmark_leaderboard_blockers(status: str, benchmark_payload: Mapping[str, Any]) -> list[str]:
    """Compiles blocking eligibility reasons if a run cannot be published to official leaderboards."""
    if status != "succeeded":
        return [status]
    if benchmark_payload.get("ok") is True:
        return []

    metadata = benchmark_payload.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if metadata.get("contract_only") is True:
        return ["benchmark runner ran in contract-only mode"]
    if metadata.get("normalizer_only") is True:
        return ["benchmark runner produced normalizer-only evidence"]
    if benchmark_payload.get("official_benchmark_verified") is not True:
        return ["official benchmark verification missing"]
    if benchmark_payload.get("integration_evidence") is not True:
        return ["benchmark integration evidence missing"]
    return ["benchmark runner did not produce leaderboard-valid evidence"]


def run_model_benchmark(request: ModelBenchmarkRunRequest | Mapping[str, Any] | None = None, **kwargs: Any) -> ModelBenchmarkRunResult:
    """Executes a complete 1:1 model generation and benchmark scoring sequence.

    First triggers target model generation (unless generated_artifact_dir is supplied directly),
    materializes the files, dispatches host execution commands to score outputs, and compiles scorecard summaries.

    Args:
        request: Configured ModelBenchmarkRunRequest payload.
        **kwargs: Inline overrides merged directly into request properties.

    Returns:
        The generated ModelBenchmarkRunResult summary.
    """
    if isinstance(request, ModelBenchmarkRunRequest):
        if kwargs:
            payload = asdict(request)
            payload.update(kwargs)
            request = ModelBenchmarkRunRequest(**payload)
    else:
        payload = dict(kwargs)
        if isinstance(request, Mapping):
            payload = {**dict(request), **payload}
        request = ModelBenchmarkRunRequest(**payload)

    mode = normalize_benchmark_run_mode(request.benchmark_mode)
    if not request.model_id and not request.contract_fixture:
        raise ValueError("model-benchmark runs require model_id unless contract_fixture=True is set.")
    root = Path(request.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "model_benchmark_run.json"
    run_manifest_path = root / "run_manifest.json"
    environment_path = root / "environment.json"
    env_requirements_path = root / "env_requirements.json"
    summary_path = root / "summary.json"
    artifact_manifest_path = root / "generated_artifacts.jsonl"

    generation_result = _run_generation(request, root)
    placeholder_count = 0
    if request.generated_artifact_dir is not None:
        generated_artifact_dir = Path(request.generated_artifact_dir).expanduser().resolve()
        materialized_count = len([path for path in generated_artifact_dir.rglob("*") if path.is_file()]) if generated_artifact_dir.exists() else 0
        write_jsonl(artifact_manifest_path, [], atomic=False)
    else:
        generated_artifact_dir = root / "generated_artifacts"
        if request.contract_fixture and generation_result is None:
            materialized_count, placeholder_count = _materialize_contract_validation_artifacts(
                generated_artifact_dir=generated_artifact_dir,
                artifact_manifest_path=artifact_manifest_path,
                output_artifact=request.output_artifact,
            )
        else:
            if request.benchmark_id == "videoverse":
                from worldfoundry.evaluation.tasks.execution.runners.videoverse.videoverse_prompts import (
                    copy_videoverse_generated_videos,
                )

                materialized_count, placeholder_count = copy_videoverse_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "phyfps-bench-gen":
                from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_prompts import (
                    copy_phyfps_generated_videos,
                )

                materialized_count, placeholder_count = copy_phyfps_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "physvidbench":
                from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_prompts import (
                    copy_physvidbench_generated_videos,
                )

                materialized_count, placeholder_count = copy_physvidbench_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "physics-iq":
                from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_prompts import (
                    copy_physics_iq_generated_videos,
                )

                materialized_count, placeholder_count = copy_physics_iq_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "phygenbench":
                from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import (
                    copy_phygenbench_generated_videos,
                )

                materialized_count, placeholder_count = copy_phygenbench_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "videophy":
                from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_prompts import (
                    copy_videophy_generated_videos,
                )

                materialized_count, placeholder_count = copy_videophy_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "videophy2":
                from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_prompts import (
                    copy_videophy2_generated_videos,
                )

                materialized_count, placeholder_count = copy_videophy2_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "mirabench":
                from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_prompts import (
                    copy_mirabench_generated_videos,
                )

                materialized_count, placeholder_count = copy_mirabench_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "ewmbench":
                from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_prompts import (
                    copy_ewmbench_generated_videos,
                )

                materialized_count, placeholder_count = copy_ewmbench_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "evalcrafter":
                from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
                    copy_evalcrafter_generated_videos,
                )

                materialized_count, placeholder_count = copy_evalcrafter_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "phyground":
                from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_prompts import (
                    copy_phyground_generated_videos,
                )

                materialized_count, placeholder_count = copy_phyground_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "phyeduvideo":
                from worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.phyeduvideo_prompts import (
                    copy_phyeduvideo_generated_videos,
                )

                materialized_count, placeholder_count = copy_phyeduvideo_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            elif request.benchmark_id == "aigcbench":
                from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_prompts import (
                    copy_aigcbench_generated_videos,
                )

                materialized_count, placeholder_count = copy_aigcbench_generated_videos(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            else:
                materialized_count, placeholder_count = _materialize_generated_artifacts(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                    allow_placeholders=_placeholder_allowed(
                        mode,
                        request.materialize_placeholders,
                        contract_fixture=request.contract_fixture,
                    ),
                )

    benchmark_result = run_benchmark_execution(
        request.benchmark_id,
        output_dir=root / "benchmark",
        manifest_path=resolve_benchmark_manifest_path(request.benchmark_manifest_path, request.benchmark_id),
        mode=mode,
        generated_artifact_dir=generated_artifact_dir,
        timeout_seconds=request.benchmark_timeout_seconds,
        workdir=request.benchmark_workdir,
        env_overrides=dict(request.benchmark_env or {}),
    )
    benchmark_payload = benchmark_result.to_dict()
    generation_exit_code = 0 if generation_result is None else generation_result.exit_code
    artifact_exit_code = (
        1
        if generation_result is not None
        and generation_result.successful_sample_count > 0
        and materialized_count == 0
        else 0
    )
    official_mode_requires_evidence = mode in {"official-validation", "official-run"}
    benchmark_exit_code = 0 if not official_mode_requires_evidence or benchmark_result.ok else 1
    exit_code = generation_exit_code or artifact_exit_code or benchmark_exit_code
    status = "succeeded" if exit_code == 0 else "failed"
    sample_count = generation_result.sample_count if generation_result is not None else materialized_count
    successful_samples = (
        generation_result.successful_sample_count if generation_result is not None else materialized_count
    )
    failed_samples = generation_result.failed_sample_count if generation_result is not None else 0
    artifacts = {
        "run_manifest": str(manifest_path),
        "standard_run_manifest": str(run_manifest_path),
        "environment": str(environment_path),
        "env_requirements": str(env_requirements_path),
        "run_summary": str(summary_path),
        "generated_artifact_dir": str(generated_artifact_dir),
        "generated_artifact_manifest": str(artifact_manifest_path),
        "benchmark_scorecard": benchmark_payload.get("scorecard_path"),
    }
    if generation_result is not None:
        artifacts["generation_scorecard"] = str(generation_result.scorecard_path)
    task_run_plan_path = _task_plan_path(root)
    if task_run_plan_path.is_file():
        artifacts["task_run_plan"] = str(task_run_plan_path)

    manifest = {
        "schema_version": MODEL_BENCHMARK_RUN_SCHEMA_VERSION,
        "run_id": request.run_id,
        "status": status,
        "benchmark_id": request.benchmark_id,
        "benchmark_mode": mode,
        "model_id": request.model_id or CONTRACT_VALIDATION_ID,
        "task": {
            "task_name": request.task_name,
            "task_benchmark": request.task_benchmark or (request.benchmark_id if request.task_name else None),
            "task_roots": [str(path) for path in request.task_roots] if request.task_roots else [],
            "dataset_root": None if request.dataset_root is None else str(request.dataset_root),
            "dataset_id": request.dataset_id,
            "split": request.split,
            "num_samples": request.num_samples,
        },
        "output_dir": str(root),
        "generated_artifact_dir": str(generated_artifact_dir),
        "materialized_artifact_count": materialized_count,
        "placeholder_artifact_count": placeholder_count,
        "generation": None if generation_result is None else generation_result.to_dict(),
        "benchmark": benchmark_payload,
        "artifacts": dict(artifacts),
    }
    write_json(manifest_path, manifest, atomic=False)
    write_run_manifest_artifacts(
        output_dir=root,
        base_manifest={
            "schema_version": "worldfoundry-run-manifest",
            "run_id": request.run_id,
            "runner": "model_benchmark_runner",
            "status": status,
            "exit_code": exit_code,
            "output_dir": str(root),
            "benchmark": {
                "benchmark_id": request.benchmark_id,
                "benchmark_mode": mode,
                "manifest_path": str(resolve_benchmark_manifest_path(request.benchmark_manifest_path, request.benchmark_id)),
            },
            "model": {
                "model_id": request.model_id or CONTRACT_VALIDATION_ID,
                "model_runner": request.model_runner,
                "variant_id": request.model_variant_id,
            },
            "dataset": {
                "dataset_id": request.dataset_id or f"{request.benchmark_id}:generated",
                "split": request.split,
                "sample_count": sample_count,
            },
            "sample_count": sample_count,
            "successful_sample_count": successful_samples,
            "failed_sample_count": failed_samples,
            "artifacts": dict(artifacts),
        },
        config={
            "benchmark_mode": mode,
            "output_artifact": request.output_artifact,
            "required_artifacts": tuple(request.required_artifacts),
            "metrics": tuple(request.metrics),
            "materialize_placeholders": request.materialize_placeholders,
            "contract_fixture": request.contract_fixture,
            "placeholder_artifact_count": placeholder_count,
            "benchmark_timeout_seconds": request.benchmark_timeout_seconds,
            "model_parameters": dict(request.model_parameters or {}),
            "model_runtime": dict(request.model_runtime or {}),
            "benchmark_env": dict(request.benchmark_env or {}),
        },
        required_paths=(
            request.benchmark_manifest_path,
            *((request.generated_artifact_dir,) if request.generated_artifact_dir is not None else ()),
        ),
        cache_paths={
            "generated_artifact_dir": generated_artifact_dir,
            "benchmark_output_dir": root / "benchmark",
            "generation_output_dir": root / "generation",
        },
        package_names=("worldfoundry", "numpy", "pandas"),
        manifest_path=run_manifest_path,
        environment_path=environment_path,
        env_requirements_path=env_requirements_path,
    )
    write_json(
        summary_path,
        _model_benchmark_run_summary(
            request=request,
            status=status,
            mode=mode,
            root=root,
            materialized_count=materialized_count,
            placeholder_count=placeholder_count,
            generation_result=generation_result,
            benchmark_payload=benchmark_payload,
            artifacts=artifacts,
        ),
        atomic=False,
    )

    return ModelBenchmarkRunResult(
        schema_version=MODEL_BENCHMARK_RESULT_SCHEMA_VERSION,
        status=status,
        exit_code=exit_code,
        output_dir=root,
        run_manifest_path=manifest_path,
        generation_result=generation_result,
        benchmark_result=benchmark_payload,
        generated_artifact_dir=generated_artifact_dir,
        artifact_manifest_path=artifact_manifest_path,
        artifacts=artifacts,
    )


__all__ = [
    "CONTRACT_VALIDATION_ID",
    "DEFAULT_BENCHMARK_TASK_ROOT",
    "MODEL_BENCHMARK_RESULT_SCHEMA_VERSION",
    "MODEL_BENCHMARK_RUN_SCHEMA_VERSION",
    "ModelBenchmarkRunRequest",
    "ModelBenchmarkRunResult",
    "run_model_benchmark",
]
