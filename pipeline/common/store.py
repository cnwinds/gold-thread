"""文件与状态存储工具：JSONL 原子读写、指纹、索引。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return n


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path) -> Dict[str, Any]:
    """增量指纹：size + mtime_ns + 尾部 4KB sha256（大文件秒级跳过）。"""
    st = Path(path).stat()
    tail = b""
    with Path(path).open("rb") as f:
        if st.st_size > 4096:
            f.seek(st.st_size - 4096)
        tail = f.read(4096)
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "tail_sha256": sha256_bytes(tail),
    }


def fingerprint_equal(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return False
    return a.get("size") == b.get("size") and a.get("tail_sha256") == b.get("tail_sha256")


_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def safe_name(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "unnamed"


def load_index(path: Path) -> Dict[str, Any]:
    return read_json(path, default={"generated_at": None, "sessions": {}}) or {
        "generated_at": None,
        "sessions": {},
    }


def dump_index(path: Path, sessions: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(extra or {})
    payload["generated_at"] = now_iso()
    payload["sessions"] = sessions
    write_json_atomic(path, payload)