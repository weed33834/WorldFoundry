"""Path helpers for the in-tree YOLO-World runtime."""

from __future__ import annotations

import sys
from pathlib import Path


def root_path() -> Path:
    return Path(__file__).resolve().parent


def config_path() -> Path:
    return root_path() / "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"


def ensure_import_path() -> None:
    for path in (root_path(), root_path() / "mmyolo"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
