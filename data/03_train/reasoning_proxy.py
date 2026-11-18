#!/usr/bin/env python3
"""兼容入口：推理代理的唯一真源已收敛至 server/reasoning_proxy.py。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.reasoning_proxy import _StreamSplitter, _rebuild_nonstream, _split_reasoning, main, make_app  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
