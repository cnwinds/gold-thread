"""发布、激活与回滚。构建和打包不改当前训练版本。"""
from __future__ import annotations

import hashlib
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import dataset_build
from .common.config import config_hash, load_config, stage_dirs
from .common.store import now_iso, read_json, sha256_file, write_json_atomic

_VERSION = re.compile(r"^v[0-9]{8}-[0-9]{4}-[a-f0-9]{8,64}$")
_LOG_NAME = "activation_log.json"
_RELEASE_LOCK = threading.Lock()


def release_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["root"] / "04_release"


def _valid_version(version: str) -> str:
    value = str(version or "")
    if not _VERSION.fullmatch(value):
        raise ValueError("非法发布版本")
    return value


def _active_path(cfg: Dict[str, Any]) -> Path:
    return release_root(cfg) / "ACTIVE_RELEASE.txt"


def _log_path(cfg: Dict[str, Any]) -> Path:
    return release_root(cfg) / _LOG_NAME


def _releases_dir(cfg: Dict[str, Any]) -> Path:
    return release_root(cfg) / "releases"


def read_active_version(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_config()
    path = _active_path(cfg)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else ""


def _write_active(cfg: Dict[str, Any], version: str) -> None:
    data_dir = str((cfg.get("paths") or {}).get("data_dir") or "data")
    text = (
        f"{version}\n"
        f"synced_at={now_iso()}\n"
        f"source={data_dir}/04_release/releases/{version}\n"
    )
    path = _active_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_events(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = read_json(_log_path(cfg), default={}) or {}
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return []
    return [row for row in events if isinstance(row, dict)]


def _save_events(cfg: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    write_json_atomic(_log_path(cfg), {"events": events})


def _release_record(path: Path) -> Optional[Dict[str, Any]]:
    record = read_json(path / "RELEASE.json", default=None)
    if isinstance(record, dict) and record.get("version") == path.name:
        return record
    manifest = read_json(path / "manifest.json", default=None)
    if not isinstance(manifest, dict) or manifest.get("version") != path.name:
        return None
    if not any(path.glob("*.jsonl")):
        return None
    return {
        "version": path.name,
        "created_at": manifest.get("created_at") or "",
        "origin": "legacy",
        "build_id": "",
        "counts": manifest.get("counts") or {},
    }


def _imported_ids(cfg: Dict[str, Any]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for manifest in dataset_build.list_builds(cfg):
        version = str(manifest.get("legacy_version") or "")
        build_id = str(manifest.get("build_id") or "")
        if version and build_id:
            found[version] = build_id
    return found


def _summary(record: Dict[str, Any], active: str, imported: Dict[str, str]) -> Dict[str, Any]:
    version = str(record.get("version") or "")
    origin = str(record.get("origin") or "build")
    build_id = str(record.get("build_id") or "")
    if origin == "legacy":
        build_id = imported.get(version, "")
    return {
        "version": version,
        "build_id": build_id,
        "created_at": record.get("created_at") or "",
        "origin": origin,
        "counts": record.get("counts") or {},
        "active": version == active,
        "imported_build_id": imported.get(version, "") if origin == "legacy" else build_id,
    }


def _can_rollback(cfg: Dict[str, Any], active: str) -> bool:
    if not active:
        return False
    for event in reversed(_load_events(cfg)):
        if event.get("version") != active:
            continue
        previous = str(event.get("previous") or "")
        if previous and previous != active and (_releases_dir(cfg) / previous).is_dir():
            return True
        return False
    return False


def list_releases(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    active = read_active_version(cfg)
    imported = _imported_ids(cfg)
    rows: List[Dict[str, Any]] = []
    root = _releases_dir(cfg)
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            record = _release_record(path)
            if record:
                rows.append(_summary(record, active, imported))
    rows.sort(key=lambda item: (item.get("created_at") or "", item["version"]), reverse=True)
    return {
        "active_version": active,
        "can_rollback": _can_rollback(cfg, active),
        "releases": rows,
    }


def _version_for(build_id: str, when: str) -> str:
    head = when[:16].replace("-", "").replace("T", "-").replace(":", "")
    return _valid_version(f"v{head}-{build_id[:8]}")


def _find_published(cfg: Dict[str, Any], build_id: str) -> Optional[Dict[str, Any]]:
    root = _releases_dir(cfg)
    if not root.is_dir():
        return None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        record = read_json(path / "RELEASE.json", default=None)
        if isinstance(record, dict) and record.get("build_id") == build_id:
            return record
    return None


def publish(cfg: Optional[Dict[str, Any]] = None, build_id: Optional[str] = None) -> Dict[str, Any]:
    """把一个 Build 冻结成发布版本，不修改 ACTIVE_RELEASE。"""
    cfg = cfg or load_config()
    with _RELEASE_LOCK:
        build_dir, manifest = dataset_build.resolve_build(cfg, build_id)
        selected = str(manifest["build_id"])
        existing = _find_published(cfg, selected)
        if existing:
            summary = _summary(existing, read_active_version(cfg), _imported_ids(cfg))
            return {**summary, "activated": False, "reused": True}

        version = _version_for(selected, now_iso())
        target = _releases_dir(cfg) / version
        if target.exists():
            raise RuntimeError(f"发布版本已存在: {version}")
        names = [
            str(item.get("path") or "")
            for item in (manifest.get("files") or [])
            if item.get("path") and item.get("path") != "BUILD.json"
        ]
        tmp = _releases_dir(cfg) / f".{version}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            for name in names:
                if Path(name).name != name or name.startswith("."):
                    raise ValueError(f"Build 含非法文件名: {name}")
                source = (build_dir / name).resolve()
                source.relative_to(build_dir.resolve())
                if not source.is_file():
                    raise FileNotFoundError(f"Build 缺少文件: {name}")
                shutil.copy2(source, tmp / name)
            record = {
                "version": version,
                "build_id": selected,
                "created_at": now_iso(),
                "origin": "build",
                "counts": dict(manifest.get("counts") or {}),
            }
            write_json_atomic(tmp / "RELEASE.json", record)
            tmp.replace(target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        summary = _summary(record, read_active_version(cfg), {})
        return {**summary, "activated": False, "reused": False}


def activate(cfg: Optional[Dict[str, Any]] = None, version: str = "") -> Dict[str, Any]:
    """把已发布版本设为当前训练版本。"""
    cfg = cfg or load_config()
    selected = _valid_version(version)
    with _RELEASE_LOCK:
        path = _releases_dir(cfg) / selected
        record = _release_record(path) if path.is_dir() else None
        if not record:
            raise FileNotFoundError(f"发布版本不存在: {selected}")
        previous = read_active_version(cfg)
        if previous != selected:
            events = _load_events(cfg)
            events.append({
                "action": "activate",
                "version": selected,
                "previous": previous,
                "at": now_iso(),
            })
            _save_events(cfg, events)
            _write_active(cfg, selected)
        return {"version": selected, "previous": previous, "active_version": selected}


def rollback(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """回到当前版本被激活之前的那个版本。再点一次不会弹回去。"""
    cfg = cfg or load_config()
    with _RELEASE_LOCK:
        active = read_active_version(cfg)
        events = _load_events(cfg)
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            if event.get("version") != active:
                continue
            previous = str(event.get("previous") or "")
            if not previous or previous == active:
                break
            target = _releases_dir(cfg) / previous
            if not target.is_dir() or not _release_record(target):
                raise FileNotFoundError(f"上一发布版本已不存在: {previous}")
            del events[index]
            _save_events(cfg, events)
            _write_active(cfg, previous)
            return {"version": previous, "previous": active, "active_version": previous}
        raise RuntimeError("没有可回滚的上一版本")


def _legacy_files(path: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    declared = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    names = sorted(p.name for p in path.iterdir() if p.is_file() and (
        p.suffix == ".jsonl" or p.name in {"dataset_info.json", "manifest.json"}
    ))
    if not any(name.endswith(".jsonl") for name in names):
        raise RuntimeError(f"历史发布里没有数据集文件: {path.name}")
    items: List[Dict[str, Any]] = []
    for name in names:
        source = path / name
        digest = sha256_file(source)
        meta = declared.get(name) if isinstance(declared.get(name), dict) else {}
        expected = str(meta.get("sha256") or "")
        if expected and expected != digest:
            raise RuntimeError(f"历史发布文件与 manifest 不一致: {name}")
        item: Dict[str, Any] = {
            "path": name,
            "sha256": digest,
            "size": source.stat().st_size,
        }
        if meta.get("count") is not None:
            item["rows"] = int(meta["count"])
        items.append(item)
    return items


def _legacy_build_id(items: List[Dict[str, Any]]) -> str:
    blob = "\n".join(f"{item['path']} {item['sha256']}" for item in items)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def import_legacy(cfg: Optional[Dict[str, Any]] = None, version: str = "") -> Dict[str, Any]:
    """把历史发布目录固化成 Build。不改 ACTIVE_RELEASE，也不改最新 Build 指针。"""
    cfg = cfg or load_config()
    selected = _valid_version(version)
    with _RELEASE_LOCK:
        source = _releases_dir(cfg) / selected
        manifest_path = source / "manifest.json"
        legacy = read_json(manifest_path, default=None)
        if not source.is_dir() or not isinstance(legacy, dict) or legacy.get("version") != selected:
            raise FileNotFoundError(f"历史发布不存在: {selected}")
        if (source / "RELEASE.json").is_file():
            raise RuntimeError("该版本来自 Build，无需迁入")
        items = _legacy_files(source, legacy)
        build_id = _legacy_build_id(items)
        root = dataset_build.builds_root(cfg)
        target = root / build_id
        latest_path = root / "LATEST_BUILD.json"
        latest_before = latest_path.read_bytes() if latest_path.is_file() else None
        existing = read_json(target / "BUILD.json", default=None)
        if isinstance(existing, dict) and existing.get("build_id") == build_id:
            current = {item["path"]: item["sha256"] for item in items}
            previous = {
                item.get("path"): item.get("sha256")
                for item in (existing.get("files") or [])
                if item.get("path") in current
            }
            if previous != current:
                raise RuntimeError(f"Build 已存在但内容不同: {build_id}")
            return {
                "build_id": build_id,
                "version": selected,
                "reused": True,
                "activated": False,
            }
        if target.exists():
            raise RuntimeError(f"Build 目录存在但缺少有效 BUILD.json: {target.name}")

        snapshot = {
            "legacy_import": {
                "version": selected,
                "manifest_config_hash": str(legacy.get("config_hash") or ""),
            }
        }
        tmp = root / f".{build_id}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            files: List[Dict[str, Any]] = []
            for item in items:
                shutil.copy2(source / item["path"], tmp / item["path"])
                files.append(item)
            write_json_atomic(tmp / "build_config.json", snapshot)
            files.append({
                "path": "build_config.json",
                "sha256": sha256_file(tmp / "build_config.json"),
                "size": (tmp / "build_config.json").stat().st_size,
            })
            record = {
                "version": 1,
                "build_id": build_id,
                "plan_hash": build_id,
                "created_at": legacy.get("created_at") or now_iso(),
                "config_hash": config_hash(snapshot),
                "origin": "legacy_release",
                "legacy_version": selected,
                "source_revisions": {},
                "split_plan": {},
                "decisions": {},
                "counts": dict(legacy.get("counts") or {}),
                "files": files,
                "samples": {},
            }
            write_json_atomic(tmp / "BUILD.json", record)
            tmp.replace(target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if latest_before is None:
            if latest_path.exists():
                latest_path.unlink()
        elif latest_path.read_bytes() != latest_before:
            latest_path.write_bytes(latest_before)
        return {
            "build_id": build_id,
            "version": selected,
            "reused": False,
            "activated": False,
        }
