"""管线入口：python -m pipeline.run <stage1|stage2|stage3|review|all>"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List

from .common.config import load_config, stage_dirs
from .common.store import now_iso, read_json, write_json_atomic


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _record(stage: str, status: str, started: str, result=None, error: str = None) -> None:
    path = stage_dirs(load_config())["state"] / "runs.json"
    hist = read_json(path, default=[]) or []
    hist.insert(0, {
        "id": f"cli-{int(time.time())}",
        "stage": stage,
        "force": False,
        "status": status,
        "created_at": started,
        "finished_at": now_iso(),
        "error": error,
        "result": result,
    })
    write_json_atomic(path, hist[:100])


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GoldThread · Agent 训练数据维护工具")
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3", "review", "all"])
    parser.add_argument("--force", action="store_true", help="忽略增量状态，强制重建")
    parser.add_argument("--session", action="append", help="review 指定会话 id（可多次）")
    parser.add_argument("--config", help="配置文件路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages: List[str] = ["stage1", "stage2", "stage3"] if args.stage == "all" else [args.stage]

    for name in stages:
        started = now_iso()
        try:
            if name == "stage1":
                from .stage1_extract import run
                result = run(force=args.force, log=_log, cfg=cfg)
            elif name == "stage2":
                from .stage2_gold import run
                result = run(force=args.force, log=_log, cfg=cfg)
            elif name == "stage3":
                from .stage3_synth import run
                result = run(force=args.force, log=_log, cfg=cfg)
            elif name == "review":
                from .review import run_review
                result = run_review(session_ids=args.session, force=args.force, log=_log, cfg=cfg)
            _record(name, "done", started, result)
        except Exception as exc:  # noqa: BLE001
            _record(name, "failed", started, error=str(exc))
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())