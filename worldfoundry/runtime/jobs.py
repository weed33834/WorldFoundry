"""Local process/job helpers for CLI surfaces and MCP/UI execution."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.core.time import utc_now_iso as _utc_now_iso


TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})


def python_module_command(command: Sequence[str], *, python_executable: str | None = None) -> tuple[str, ...]:
    """Run a ``worldfoundry-eval`` command through the current Python interpreter."""

    items = tuple(str(item) for item in command)
    if not items:
        raise ValueError("command cannot be empty")
    if items[0] in {"worldfoundry", "worldfoundry-eval"}:
        return (python_executable or sys.executable, "-m", "worldfoundry.evaluation", *items[1:])
    return items


def _decode_process_text(value: str | bytes | None) -> str:
    """Decode subprocess output to a string, replacing invalid bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Force-kill the entire process group for a subprocess on POSIX systems."""
    if sys.platform == "win32":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int,
    kill_timeout: int = 5,
) -> dict[str, Any]:
    """Run a command with a hard timeout and always return captured output.

    This helper is intended for official benchmark subprocesses. Some simulator
    or CUDA-backed scripts can ignore ordinary timeout handling while stuck in
    native code, so timeout failures are converted into structured results that
    callers can write into scorecards instead of surfacing a traceback.
    """

    command_tuple = tuple(str(item) for item in command)
    if not command_tuple:
        raise ValueError("command cannot be empty")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    start = time.monotonic()
    process = subprocess.Popen(
        command_tuple,
        cwd=None if cwd is None else str(cwd),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    timed_out = False
    kill_stuck = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Native CUDA simulators can ignore regular interrupt signals, so we
        # terminate the whole process group first and then force-kill if needed.
        timed_out = True
        _kill_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=kill_timeout)
        except subprocess.TimeoutExpired as kill_exc:
            kill_stuck = True
            stdout = _decode_process_text(kill_exc.stdout or exc.stdout)
            stderr = _decode_process_text(kill_exc.stderr or exc.stderr)
        stderr = (
            f"{_decode_process_text(stderr)}\n"
            f"TimeoutExpired: command exceeded {timeout}s"
        ).strip()
    return {
        "command": list(command_tuple),
        "stdout": _decode_process_text(stdout),
        "stderr": _decode_process_text(stderr),
        "returncode": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "kill_stuck": kill_stuck,
        "duration_seconds": time.monotonic() - start,
    }


@dataclass
class CommandJob:
    """Track the lifecycle and output of an asynchronous subprocess command.

    Attributes:
        job_id: Unique identifier for the job.
        command: Full command tuple executed by the subprocess.
        display_command: Human-readable command tuple for UI surfaces.
        cwd: Working directory for the subprocess, if set.
        output_dir: Directory for persistent output artifacts, if set.
        metadata: Arbitrary metadata attached by the submitter.
        status: Current lifecycle status (queued, running, completed, failed, cancelled).
        created_at: ISO timestamp when the job was created.
        started_at: ISO timestamp when the job began running.
        completed_at: ISO timestamp when the job finished.
        returncode: Process exit code, or ``None`` if not finished.
        error: Error message if the job failed.
        result: Parsed JSON result extracted from stdout.
        logs: Chronological list of stdout/stderr log entries.
    """

    job_id: str
    command: tuple[str, ...]
    display_command: tuple[str, ...]
    cwd: str | None = None
    output_dir: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    result: Any | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        """Return whether the job has reached a terminal status."""
        return self.status in TERMINAL_JOB_STATUSES

    def append_log(self, stream: str, text: str) -> None:
        """Append a timestamped log entry for *stream* (stdout or stderr)."""
        if text:
            self.logs.append({"time": _utc_now_iso(), "stream": stream, "text": text})

    def log_text(self, *, stream: str | None = None, limit: int | None = None) -> str:
        """Return concatenated text from log entries, optionally filtered by stream."""
        rows = [row for row in self.logs if stream is None or row.get("stream") == stream]
        if limit is not None:
            rows = rows[-limit:]
        return "".join(str(row.get("text") or "") for row in rows)

    def to_summary(self, *, log_tail: int = 40) -> dict[str, Any]:
        """Return a summary dict suitable for UI polling endpoints."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "output_dir": self.output_dir,
            "command": list(self.display_command),
            "cwd": self.cwd,
            "metadata": dict(self.metadata),
            "error": self.error,
            "logs": self.logs[-log_tail:] if log_tail else [],
        }

    def to_result(self, *, include_logs: bool = False, log_tail: int = 200) -> dict[str, Any]:
        """Return a full result dict with optional log text."""
        payload = self.to_summary(log_tail=log_tail if include_logs else 0)
        payload["result"] = self.result
        if include_logs:
            payload["stdout"] = self.log_text(stream="stdout", limit=log_tail)
            payload["stderr"] = self.log_text(stream="stderr", limit=log_tail)
        return payload


class AsyncCommandJobStore:
    """Small in-process command runner used by local UI and MCP surfaces."""

    def __init__(self, *, max_log_lines: int = 4000) -> None:
        self.max_log_lines = max_log_lines
        self._jobs: dict[str, CommandJob] = {}

    def submit(
        self,
        command: Sequence[str],
        *,
        display_command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        output_dir: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> CommandJob:
        """Submit a command as an async job and start it immediately.

        Args:
            command: Command and arguments to execute.
            display_command: Optional human-readable override for UI surfaces.
            cwd: Working directory for the subprocess.
            env: Additional environment variables merged into ``os.environ``.
            output_dir: Directory for persistent output artifacts.
            metadata: Arbitrary metadata attached to the job.
            job_id: Optional explicit job identifier; auto-generated if ``None``.

        Returns:
            The newly created :class:`CommandJob`.

        Raises:
            ValueError: If *job_id* already exists in the store.
        """
        resolved_job_id = job_id or uuid.uuid4().hex[:12]
        if resolved_job_id in self._jobs:
            raise ValueError(f"job already exists: {resolved_job_id}")
        # Jobs are tracked in-process so UI/MCP calls can poll by id and inspect
        # both state transitions and captured stdout/stderr.
        job = CommandJob(
            job_id=resolved_job_id,
            command=tuple(str(item) for item in command),
            display_command=tuple(str(item) for item in (display_command or command)),
            cwd=str(cwd) if cwd is not None else None,
            output_dir=str(output_dir) if output_dir is not None else None,
            metadata=dict(metadata or {}),
        )
        self._jobs[job.job_id] = job
        job._task = asyncio.create_task(self._run(job, dict(env or {})))
        return job

    def get(self, job_id: str) -> CommandJob | None:
        """Retrieve a job by its identifier, or ``None`` if not found."""
        return self._jobs.get(job_id)

    def list(self) -> list[CommandJob]:
        """Return all jobs sorted by creation time (most recent first)."""
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    async def cancel(self, job_id: str) -> tuple[bool, str]:
        """Cancel a running or queued job by identifier.

        Args:
            job_id: The job to cancel.

        Returns:
            ``(True, "cancelled")`` on success, or ``(False, reason)`` on failure.
        """
        job = self.get(job_id)
        if job is None:
            return False, f"unknown job: {job_id}"
        if job.terminal:
            return False, f"job is already {job.status}"
        job.status = "cancelled"
        job.error = "cancelled by request"
        await self._terminate_process(job)
        if job._task is not None and not job._task.done():
            job._task.cancel()
        job.completed_at = job.completed_at or _utc_now_iso()
        return True, "cancelled"

    async def _run(self, job: CommandJob, env: Mapping[str, str]) -> None:
        """Execute the job subprocess, stream logs, and set the final status."""
        # Async runner lifecycle: spawn process -> stream logs -> parse result -> set final status.
        job.status = "running"
        job.started_at = _utc_now_iso()
        process_env = os.environ.copy()
        process_env.update(env)
        try:
            if job.output_dir:
                Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            job._process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=job.cwd,
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            await asyncio.gather(
                self._read_stream(job, "stdout", job._process.stdout),
                self._read_stream(job, "stderr", job._process.stderr),
            )
            job.returncode = await job._process.wait()
            if job.status == "cancelled":
                return
            job.result = _extract_json_from_logs(job.logs)
            job.status = "completed" if job.returncode == 0 else "failed"
            if job.status == "failed":
                job.error = f"command exited with code {job.returncode}"
        except asyncio.CancelledError:
            await self._terminate_process(job)
            job.status = "cancelled"
            job.error = job.error or "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through UI/MCP status.
            job.status = "failed"
            job.error = str(exc)
            job.append_log("stderr", f"{type(exc).__name__}: {exc}\n")
        finally:
            if job.completed_at is None:
                job.completed_at = _utc_now_iso()

    async def _read_stream(
        self,
        job: CommandJob,
        stream_name: str,
        stream: asyncio.StreamReader | None,
    ) -> None:
        """Read lines from a subprocess stream and append them to the job log."""
        if stream is None:
            return
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            job.append_log(stream_name, text)
            if len(job.logs) > self.max_log_lines:
                del job.logs[: len(job.logs) - self.max_log_lines]

    async def _terminate_process(self, job: CommandJob) -> None:
        """Gracefully terminate the job's subprocess, escalating to SIGKILL if needed."""
        # Graceful shutdown first, then hard stop, so benchmark runners can flush
        # partial state before the process exits.
        process = job._process
        if process is None or process.returncode is not None:
            return
        try:
            if sys.platform == "win32":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            if sys.platform == "win32":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        except ProcessLookupError:
            pass


def _extract_json_from_logs(logs: Sequence[Mapping[str, Any]]) -> Any | None:
    """Attempt to parse a JSON result object from stdout log entries."""
    stdout = "".join(str(row.get("text") or "") for row in logs if row.get("stream") == "stdout")
    for candidate in _json_candidates(stdout):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str) -> list[str]:
    """Generate candidate JSON strings from raw stdout for result extraction."""
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates.extend(reversed(lines))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    return candidates


__all__ = [
    "AsyncCommandJobStore",
    "CommandJob",
    "TERMINAL_JOB_STATUSES",
    "python_module_command",
    "run_bounded_command",
]
