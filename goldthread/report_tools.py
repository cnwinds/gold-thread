"""评估报告注册与版本演进关联（04_release/reports）。"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import now_iso, read_json, replace_atomic, write_json_atomic

from . import eval_tools, ledger

_REPORT_LOCK = threading.RLock()
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def reports_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["root"] / "04_release" / "reports"


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        replace_atomic(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _auto_summary(md_content: str, max_chars: int = 240) -> str:
    for line in md_content.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">") or s.startswith("---") or s.startswith("|"):
            continue
        cleaned = re.sub(r"[`*_]+", "", s).strip()
        if cleaned:
            return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 1] + "…"
    return ""


def _validate_report_id(report_id: Optional[str]) -> str:
    if report_id:
        rid = str(report_id).strip()
        if not _VALID_ID_RE.fullmatch(rid):
            raise ValueError(f"非法 report_id: {rid!r}")
        return rid
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def publish_report(
    *,
    title: str,
    md_content: str,
    report_id: Optional[str] = None,
    before_pack: Optional[str] = None,
    after_pack: Optional[str] = None,
    build_id: Optional[str] = None,
    summary: str = "",
    actor: str = "agent",
    reason: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将评估报告 Markdown 与结构化元数据原子归档到 04_release/reports/ 并记入 ledger。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    title_s = str(title or "").strip()
    if not title_s:
        raise ValueError("报告标题不能为空")
    md_s = str(md_content or "")
    if not md_s.strip():
        raise ValueError("报告正文不能为空")

    rid = _validate_report_id(report_id)
    before_snapshot = ledger.corpus_snapshot_summary(cfg)

    diff_summary: Optional[Dict[str, Any]] = None
    diff_error: Optional[str] = None
    if before_pack and after_pack:
        try:
            diff_summary = eval_tools.eval_diff(before_pack, after_pack, cfg=cfg)
        except (FileNotFoundError, ValueError, OSError) as exc:
            diff_error = str(exc)

    after_summary_eval: Optional[Dict[str, Any]] = None
    if after_pack and diff_summary is None:
        try:
            after_summary_eval = eval_tools._compact_eval_summary(  # noqa: SLF001
                eval_tools.eval_pack(after_pack, cfg=cfg)
            )
        except (FileNotFoundError, ValueError, OSError):
            after_summary_eval = None

    recent_ops = [op.get("op_id") for op in ledger.read_ledger(cfg=cfg, limit=10) if op.get("op_id")]

    root = reports_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    md_path = root / f"{rid}.md"
    meta_path = root / f"{rid}.json"
    created_at = now_iso()
    summary_text = str(summary or "").strip() or _auto_summary(md_s)

    record: Dict[str, Any] = {
        "report_id": rid,
        "title": title_s,
        "summary": summary_text,
        "created_at": created_at,
        "actor": str(actor or "agent"),
        "reason": str(reason or ""),
        "before_pack": str(before_pack) if before_pack else None,
        "after_pack": str(after_pack) if after_pack else None,
        "build_id": str(build_id) if build_id else (
            (diff_summary or {}).get("after", {}).get("build_id")
            or (after_summary_eval or {}).get("build_id")
            or None
        ),
        "diff_summary": diff_summary,
        "after_eval_summary": after_summary_eval,
        "diff_error": diff_error,
        "related_op_ids": recent_ops,
        "md_chars": len(md_s),
        "md_file": str(md_path),
        "meta_file": str(meta_path),
    }

    index_entry = {
        "report_id": rid,
        "title": title_s,
        "summary": summary_text,
        "created_at": created_at,
        "actor": record["actor"],
        "before_pack": record["before_pack"],
        "after_pack": record["after_pack"],
        "build_id": record["build_id"],
        "has_diff": diff_summary is not None,
        "md_chars": len(md_s),
        "md_file": str(md_path),
        "meta_file": str(meta_path),
    }

    with _REPORT_LOCK:
        _write_text_atomic(md_path, md_s if md_s.endswith("\n") else md_s + "\n")
        write_json_atomic(meta_path, record)

        idx_path = root / "index.json"
        idx_data = read_json(idx_path, default={"reports": []}) or {"reports": []}
        existing = [
            r for r in (idx_data.get("reports") or [])
            if isinstance(r, dict) and r.get("report_id") != rid
        ]
        existing.insert(0, index_entry)
        existing.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        write_json_atomic(idx_path, {"updated_at": created_at, "reports": existing})

    duration_ms = int((time.monotonic() - t0) * 1000)
    op = ledger.record_operation(
        "report.publish",
        reason=reason or f"发布评估报告: {title_s}",
        actor=actor,
        params={
            "report_id": rid,
            "title": title_s,
            "before_pack": before_pack,
            "after_pack": after_pack,
            "build_id": record["build_id"],
        },
        affected={
            "build_id": record["build_id"],
            "pack": str(after_pack) if after_pack else None,
            "report_id": rid,
        },
        before_summary=before_snapshot,
        after_summary=ledger.corpus_snapshot_summary(cfg),
        status="ok",
        duration_ms=duration_ms,
        cfg=cfg,
    )
    record["op_id"] = op["op_id"]
    return record


def list_reports(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    idx_path = reports_root(cfg) / "index.json"
    data = read_json(idx_path, default={"reports": []}) or {"reports": []}
    rows = [r for r in (data.get("reports") or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return rows


def get_report(report_id: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    rid = _validate_report_id(report_id)
    root = reports_root(cfg)
    meta_path = root / f"{rid}.json"
    md_path = root / f"{rid}.md"
    if not meta_path.is_file():
        raise KeyError(f"评估报告不存在: {rid}")
    meta = read_json(meta_path, default={}) or {}
    md_text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    return {**meta, "markdown": md_text}
