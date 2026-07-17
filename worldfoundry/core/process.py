"""Import-light subprocess launch helpers shared by model runtimes."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence


def torchrun_module_command(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    python_executable: str = sys.executable,
) -> list[str]:
    """Build a single-node torchrun command for a Python module."""

    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive")
    return [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "-m",
        module,
        *map(str, args),
    ]


def run_torchrun_module(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Run a single-node Python module under torchrun and capture its logs."""

    return subprocess.run(
        torchrun_module_command(
            module,
            nproc_per_node=nproc_per_node,
            args=args,
            python_executable=python_executable,
        ),
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        check=False,
    )


__all__ = ["run_torchrun_module", "torchrun_module_command"]
