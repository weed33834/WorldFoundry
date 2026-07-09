# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for MASt3R
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> mast3r -> utils -> misc.py functionality."""

import os
import hashlib


def mkdir_for(f):
    """Mkdir for.

    Args:
        f: The f.
    """
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f


def hash_md5(s):
    """Hash md5.

    Args:
        s: The s.
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()
