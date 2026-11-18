#!/usr/bin/env python3
"""兼容入口：唯一真源已收敛至 pipeline/train_template/tools/filter_by_cutoff.py。"""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

_SSOT = Path(__file__).resolve().parents[2] / "pipeline" / "train_template" / "tools" / "filter_by_cutoff.py"

if __name__ == "__main__":
    runpy.run_path(str(_SSOT), run_name="__main__")
else:
    _SPEC = importlib.util.spec_from_file_location("_gt_ssot_03_filter_by_cutoff", _SSOT)
    assert _SPEC is not None and _SPEC.loader is not None
    _MOD = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MOD
    _SPEC.loader.exec_module(_MOD)
    globals().update({k: v for k, v in vars(_MOD).items() if not k.startswith("__")})
