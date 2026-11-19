"""内容寻址快照池（CAS）：按 SHA256 去重归档构建时刻的原始与黄金会话产物，支持 100% 精确时间旅行回放。"""
from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common.config import load_config, stage_dirs
from .common.store import (
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    replace_atomic,
    sha256_file,
    write_json_atomic,
)

_CAS_LOCK = threading.RLock()
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SESSION_FILE_KEYS = (
    "raw_file",
    "meta_file",
    "gold_file",
    "annotation_file",
    "reject_file",
    "dpo_file",
    "diag_file",
)


def cas_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["root"] / "04_release" / "cas"


def _object_path(sha256: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    digest = str(sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"非法 CAS sha256: {sha256!r}")
    return cas_root(cfg) / digest[:2] / digest


def put_file(path: Optional[str | Path], cfg: Optional[Dict[str, Any]] = None) -> str:
    """将本地文件按 sha256 原子归档至 CAS 对象池；若已存在则直接去重复用。"""
    if not path:
        return ""
    src = Path(path)
    if not src.is_file():
        return ""
    digest = sha256_file(src)
    dest = _object_path(digest, cfg)
    if dest.is_file():
        return digest
    with _CAS_LOCK:
        if dest.is_file():
            return digest
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        os.close(fd)
        try:
            shutil.copy2(src, tmp)
            replace_atomic(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return digest


def get_cas_path(sha256: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    digest = str(sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        return None
    target = cas_root(cfg) / digest[:2] / digest
    return target if target.is_file() else None


def read_cas_json(sha256: Optional[str], cfg: Optional[Dict[str, Any]] = None, default: Any = None) -> Any:
    path = get_cas_path(sha256, cfg)
    if path is None:
        return default
    return read_json(path, default=default)


def read_cas_jsonl(sha256: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    path = get_cas_path(sha256, cfg)
    if path is None:
        return []
    return read_jsonl(path)


def count_cas_objects(cfg: Optional[Dict[str, Any]] = None) -> int:
    root = cas_root(cfg)
    if not root.exists():
        return 0
    total = 0
    for sub in root.iterdir():
        if sub.is_dir() and len(sub.name) == 2:
            total += sum(1 for f in sub.iterdir() if f.is_file() and _SHA256_RE.fullmatch(f.name))
    return total


def _dominant_model_from_entries(gold_e: Dict[str, Any], raw_e: Dict[str, Any]) -> str:
    if gold_e.get("dominant_model"):
        return str(gold_e["dominant_model"])
    models = raw_e.get("models")
    if isinstance(models, dict) and models:
        return max(models.items(), key=lambda kv: int(kv[1] or 0))[0]
    return "unknown"


def capture_build_cas(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """捕获当前工作区全部黄金会话及其关联原始/标注/剪枝文件的 CAS 快照清单。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    step_decisions = read_json(dirs["state"] / "step_decisions.json", default={}) or {}

    sessions_out: Dict[str, Dict[str, Any]] = {}
    unique_digests: set[str] = set()

    for sid in sorted(gold_sessions.keys()):
        gold_e = gold_sessions.get(sid) or {}
        raw_e = raw_sessions.get(sid) or {}
        src_info = gold_e.get("source") if isinstance(gold_e.get("source"), dict) else {}

        candidate_paths = {
            "raw_file": raw_e.get("raw_file") or src_info.get("raw_file") or "",
            "meta_file": raw_e.get("meta_file") or src_info.get("meta_file") or "",
            "gold_file": gold_e.get("gold_file") or "",
            "annotation_file": gold_e.get("annotation_file") or "",
            "reject_file": gold_e.get("reject_file") or "",
            "dpo_file": gold_e.get("dpo_file") or "",
            "diag_file": gold_e.get("diag_file") or "",
        }
        files_sha: Dict[str, str] = {}
        for key in _SESSION_FILE_KEYS:
            digest = put_file(candidate_paths.get(key), cfg)
            files_sha[key] = digest
            if digest:
                unique_digests.add(digest)

        sd_entry = step_decisions.get(sid) if isinstance(step_decisions.get(sid), dict) else {}
        sessions_out[sid] = {
            "session_id": sid,
            "project_group": str(gold_e.get("project_group") or raw_e.get("project_group") or "unknown"),
            "dominant_model": _dominant_model_from_entries(gold_e, raw_e),
            "cwd": str(gold_e.get("cwd") or raw_e.get("cwd") or ""),
            "origin": str(raw_e.get("origin") or src_info.get("origin") or "pi"),
            "timestamp": str(raw_e.get("timestamp") or gold_e.get("last_run") or ""),
            "models": copy.deepcopy(raw_e.get("models") or {}),
            "raw_stats": copy.deepcopy(raw_e.get("stats") or {}),
            "gold_stats": copy.deepcopy(gold_e.get("stats") or {}),
            "dpo_count": int(gold_e.get("dpo_count") or 0),
            "model_report": copy.deepcopy(gold_e.get("model_report") or None),
            "step_decisions": copy.deepcopy(sd_entry),
            "files": files_sha,
        }

    return {
        "version": 1,
        "fidelity": "full",
        "captured_at": now_iso(),
        "object_count": len(unique_digests),
        "sessions": sessions_out,
    }


def backfill_build_cas(
    build_dir: Path,
    build_manifest: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    readonly: bool = False,
) -> Dict[str, Any]:
    """为尚未携带 cas_manifest 的历史 Build 生成侧车回填清单（readonly=True 时纯只读零磁盘写入）。"""
    cfg = cfg or load_config()
    backfill_path = Path(build_dir) / "cas_backfill.json"
    existing = read_json(backfill_path, default=None)
    if isinstance(existing, dict) and isinstance(existing.get("sessions"), dict):
        out = dict(existing)
        out.setdefault("fidelity", "backfilled")
        return out

    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    step_decisions = read_json(dirs["state"] / "step_decisions.json", default={}) or {}

    sources = ((build_manifest.get("source_revisions") or {}).get("sources") or [])
    sessions_out: Dict[str, Dict[str, Any]] = {}
    unique_digests: set[str] = set()

    for src in sources:
        if not isinstance(src, dict) or not src.get("session_id"):
            continue
        sid = str(src["session_id"])
        gold_e = gold_sessions.get(sid) or {}
        raw_e = raw_sessions.get(sid) or {}
        src_info = gold_e.get("source") if isinstance(gold_e.get("source"), dict) else {}

        files_sha: Dict[str, str] = {k: "" for k in _SESSION_FILE_KEYS}
        # 严格校验历史记录过的哈希；仅当已在 CAS 中或工作区当前文件哈希与历史完全一致时记录
        for verified_key in ("gold_file", "annotation_file", "diag_file"):
            expected_sha = str(src.get(verified_key) or "").strip().lower()
            cand = gold_e.get(verified_key) or ""
            if expected_sha and get_cas_path(expected_sha, cfg):
                files_sha[verified_key] = expected_sha
                unique_digests.add(expected_sha)
            elif not readonly and expected_sha and cand and Path(cand).is_file() and sha256_file(Path(cand)) == expected_sha:
                files_sha[verified_key] = put_file(cand, cfg)
                unique_digests.add(expected_sha)

        raw_cand = raw_e.get("raw_file") or src_info.get("raw_file") or ""
        if raw_cand and Path(raw_cand).is_file():
            r_sha = sha256_file(Path(raw_cand)) if readonly else put_file(raw_cand, cfg)
            if not readonly or get_cas_path(r_sha, cfg):
                files_sha["raw_file"] = r_sha
                if r_sha:
                    unique_digests.add(r_sha)

        gold_exact = bool(files_sha["gold_file"] and files_sha["gold_file"] == str(src.get("gold_file") or ""))
        if gold_exact:
            for aux_key, cand_path in (
                ("meta_file", raw_e.get("meta_file") or src_info.get("meta_file") or ""),
                ("reject_file", gold_e.get("reject_file") or ""),
                ("dpo_file", gold_e.get("dpo_file") or ""),
            ):
                if cand_path and Path(cand_path).is_file():
                    d_sha = sha256_file(Path(cand_path)) if readonly else put_file(cand_path, cfg)
                    if not readonly or get_cas_path(d_sha, cfg):
                        files_sha[aux_key] = d_sha
                        if d_sha:
                            unique_digests.add(d_sha)

        sd_entry = step_decisions.get(sid) if (gold_exact and isinstance(step_decisions.get(sid), dict)) else {}
        sessions_out[sid] = {
            "session_id": sid,
            "project_group": str(gold_e.get("project_group") or raw_e.get("project_group") or "unknown"),
            "dominant_model": _dominant_model_from_entries(gold_e, raw_e),
            "cwd": str(gold_e.get("cwd") or raw_e.get("cwd") or ""),
            "origin": str(raw_e.get("origin") or src_info.get("origin") or "pi"),
            "timestamp": str(raw_e.get("timestamp") or gold_e.get("last_run") or ""),
            "models": copy.deepcopy(raw_e.get("models") or {}),
            "raw_stats": copy.deepcopy(raw_e.get("stats") or {}),
            "gold_stats": copy.deepcopy(gold_e.get("stats") or {}) if gold_exact else {},
            "dpo_count": int(gold_e.get("dpo_count") or 0) if gold_exact else 0,
            "model_report": copy.deepcopy(gold_e.get("model_report") or None) if gold_exact else None,
            "step_decisions": copy.deepcopy(sd_entry),
            "files": files_sha,
            "expected_revisions": {
                "gold_file": str(src.get("gold_file") or ""),
                "annotation_file": str(src.get("annotation_file") or ""),
                "diag_file": str(src.get("diag_file") or ""),
            },
        }

    result = {
        "version": 1,
        "fidelity": "backfilled",
        "backfilled": True,
        "captured_at": now_iso(),
        "object_count": len(unique_digests),
        "sessions": sessions_out,
    }
    if not readonly and Path(build_dir).is_dir():
        try:
            write_json_atomic(backfill_path, result)
        except OSError:
            pass
    return result


def load_build_cas_manifest(
    build_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    readonly: bool = False,
) -> Dict[str, Any]:
    """读取指定 Build 的 CAS 快照清单；readonly=True 时纯只读不落盘侧车文件。"""
    from . import dataset_build

    cfg = cfg or load_config()
    build_dir, manifest = dataset_build.resolve_build(cfg, build_id)
    cas_man = manifest.get("cas_manifest")
    if isinstance(cas_man, dict) and isinstance(cas_man.get("sessions"), dict):
        out = dict(cas_man)
        out.setdefault("fidelity", "full")
        return out
    return backfill_build_cas(build_dir, manifest, cfg, readonly=readonly)
