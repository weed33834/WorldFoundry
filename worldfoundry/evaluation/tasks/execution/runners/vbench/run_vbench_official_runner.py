#!/usr/bin/env python3
"""Official runner for VBench."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import dispatch_main as main


if __name__ == "__main__":
    raise SystemExit(main())
