#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common_module():
    path = Path(__file__).resolve().with_name("matrix_game2_demo_common.py")
    spec = importlib.util.spec_from_file_location("_worldfoundry_matrix_game2_demo_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    return load_common_module().run(None, variant="official")


if __name__ == "__main__":
    raise SystemExit(main())
