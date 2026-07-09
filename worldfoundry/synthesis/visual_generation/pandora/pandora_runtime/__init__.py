"""Vendored Pandora runtime package."""

from __future__ import annotations

import sys

from . import ChatUniVi as _ChatUniVi
from worldfoundry.synthesis.visual_generation.dynamicrafter_pandora import DynamiCrafter as _DynamiCrafter

from . import configuration as _configuration

sys.modules.setdefault("ChatUniVi", _ChatUniVi)
sys.modules.setdefault("configuration", _configuration)
sys.modules.setdefault("DynamiCrafter", _DynamiCrafter)

from . import model as _model

sys.modules.setdefault("model", _model)
