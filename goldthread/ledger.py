"""统一操作账本（ledger）：所有变更与发布动作的线程安全只追加审计日志。"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import load_index, now_iso, read_json

_LEDGER_LOCK = threading.RLock()


def ledger_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["state"] / "ledger.jsonl"


def corpus_snapshot_summary(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取当前 01_raw / 02_gold / LATEST_BUILD 的轻量摘要。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_idx = load_index(dirs["raw"] / "index.json")
    gold_idx = load_index(dirs["gold"] / "index.json")
    raw_sessions = raw_idx.get("sessions") or {}
    gold_sessions = gold_idx.get("sessions") or {}

    projects = sorted(
        {
            str(e.get("project_group") or "")
            for e in (*raw_sessions.values(), *gold_sessions.values())
            if isinstance(e, dict) and e.get("project_group")
        }
    )
    latest_build = read_json(
        dirs["root"] / "04_release" / "builds" / "LATEST_BUILD.json", default={}
    ) or {}
    return {
        "raw_sessions": len(raw_sessions),
        "gold_sessions": len(gold_sessions),
        "projects": projects,
        "latest_build_id": latest_build.get("build_id") or None,
    }


def _normalize_affected(affected: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = dict(affected or {})
    out: Dict[str, Any] = {
        "sids": [str(x) for x in (src.get("sids") or []) if x],
        "projects": [str(x) for x in (src.get("projects") or []) if x],
        "samples": [str(x) for x in (src.get("samples") or []) if x],
        "build_id": src.get("build_id") or None,
        "pack": src.get("pack") or None,
    }
    for k, v in src.items():
        if k not in out:
            out[k] = v
    return out


def _new_op_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"op_{stamp}_{uuid.uuid4().hex[:4]}"


def record_operation(
    tool: str,
    *,
    reason: str = "",
    actor: str = "agent",
    params: Optional[Dict[str, Any]] = None,
    affected: Optional[Dict[str, Any]] = None,
    before_summary: Optional[Dict[str, Any]] = None,
    after_summary: Optional[Dict[str, Any]] = None,
    status: str = "ok",
    duration_ms: int = 0,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """线程安全地追加一条操作事件到 data/00_state/ledger.jsonl。"""
    event: Dict[str, Any] = {
        "op_id": _new_op_id(),
        "timestamp": now_iso(),
        "actor": str(actor or "agent"),
        "tool": str(tool),
        "reason": str(reason or ""),
        "params": dict(params or {}),
        "affected": _normalize_affected(affected),
        "before_summary": dict(before_summary or {}),
        "after_summary": dict(after_summary or {}),
        "status": str(status or "ok"),
        "duration_ms": int(duration_ms or 0),
    }
    path = ledger_path(cfg)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _LEDGER_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
    return event


def _matches_project(proj: str, query: str) -> bool:
    pl = proj.lower()
    ql = query.lower()
    return pl == ql or pl == f"game::{ql}" or pl.split("::")[-1] == ql


def read_ledger(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 100,
    sid: Optional[str] = None,
    project: Optional[str] = None,
    tool: Optional[str] = None,
    build_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按时间倒序（最新在前）读取并过滤操作账本。"""
    path = ledger_path(cfg)
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    with _LEDGER_LOCK:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw in reversed(lines):
        s = raw.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        aff = ev.get("affected") if isinstance(ev.get("affected"), dict) else {}
        if sid:
            sids = [str(x) for x in (aff.get("sids") or [])]
            if not any(x == sid or x.startswith(sid) for x in sids):
                continue
        if project:
            projs = [str(x) for x in (aff.get("projects") or [])]
            if not any(_matches_project(p, project) for p in projs):
                continue
        if tool and ev.get("tool") != tool:
            continue
        if build_id:
            ev_bid = str(aff.get("build_id") or "")
            if not ev_bid or (ev_bid != build_id and not ev_bid.startswith(build_id)):
                continue
        if actor and ev.get("actor") != actor:
            continue
        events.append(ev)
        if limit and limit > 0 and len(events) >= limit:
            break
    return events
