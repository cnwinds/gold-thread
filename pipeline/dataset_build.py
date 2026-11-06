"""不可变数据集 Build：固定 Stage 3 输入、产物与交付来源。"""
from __future__ import annotations

import copy
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import dataset_draft
from .common.config import config_hash, load_config, stage_dirs
from .common.store import now_iso, read_json, read_jsonl, sha256_file, write_json_atomic
from .dataset_draft import DRAFT_LOCK

_BUILD_ID = re.compile(r"^[a-f0-9]{16,64}$")
_LATEST = "LATEST_BUILD.json"


def builds_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["root"] / "04_release" / "builds"


def _valid_id(build_id: str) -> str:
    value = str(build_id or "")
    if not _BUILD_ID.fullmatch(value):
        raise ValueError("非法 build_id")
    return value


def _config_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}


def _output_names(train_root: Path, counts: Dict[str, int]) -> List[str]:
    names = sorted({name for name in counts if (train_root / name).is_file()})
    names.extend(sorted(p.name for p in train_root.glob("*.yaml") if p.is_file()))
    for name in (
        "handoff.md", "RESULT.md", "dataset_info.json", "stats.json", "index.json",
        "ds_z2_config.json", "ds_z3_config.json",
    ):
        if (train_root / name).is_file():
            names.append(name)
    return sorted(set(names))


def _verify_reuse(train_root: Path, existing: Dict[str, Any], names: List[str]) -> None:
    """同指纹复用前比较确定性产物；统计时间戳不参与比较。"""
    deterministic = {
        name for name in names
        if name.endswith(".jsonl") or name.endswith(".yaml") or name == "handoff.md"
    }
    previous = {
        item.get("path"): item.get("sha256")
        for item in (existing.get("files") or [])
        if item.get("path") in deterministic
    }
    if set(previous) != deterministic:
        raise RuntimeError("Build 指纹相同但本次产物文件集合不同，拒绝复用")
    for name in sorted(deterministic):
        if previous.get(name) != sha256_file(train_root / name):
            raise RuntimeError(f"Build 指纹相同但产物内容不同: {name}，请升级生成器版本")


def create_build(
    train_root: Path,
    cfg: Dict[str, Any],
    plan_hash: str,
    draft: Dict[str, Any],
    counts: Dict[str, int],
    validate_input: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """原子固化一次 Stage 3；相同 plan_hash 复用而不覆盖历史内容。"""
    train_root = Path(train_root).resolve()
    build_id = _valid_id(plan_hash)
    root = builds_root(cfg)
    target = root / build_id
    with DRAFT_LOCK:
        names = _output_names(train_root, counts)
        existing = read_json(target / "BUILD.json", default=None)
        if isinstance(existing, dict) and existing.get("build_id") == build_id:
            if validate_input:
                revisions = dataset_draft.source_revisions(cfg, include_aux_outputs=True)
                actual_hash = dataset_draft.plan_fingerprint(
                    cfg, draft, include_aux_outputs=True, revisions=revisions
                )
                if actual_hash != plan_hash:
                    raise RuntimeError("Build 输入在复用前发生变化，请重新运行 Stage 3")
            _verify_reuse(train_root, existing, names)
            write_json_atomic(root / _LATEST, {"build_id": build_id})
            return target, existing
        if target.exists():
            raise RuntimeError(f"Build 目录存在但缺少有效 BUILD.json: {target}")

        tmp = root / f".{build_id}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            files: List[Dict[str, Any]] = []
            samples: Dict[str, List[str]] = {}
            for name in names:
                source = (train_root / name).resolve()
                source.relative_to(train_root)
                destination = tmp / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                item: Dict[str, Any] = {
                    "path": name,
                    "sha256": sha256_file(destination),
                    "size": destination.stat().st_size,
                }
                if name in counts:
                    item["rows"] = int(counts[name])
                if name.endswith(".jsonl"):
                    ids = [
                        str(row.get("sample_id") or row.get("diagnostic_id") or "")
                        for row in read_jsonl(destination)
                    ]
                    samples[name] = [sample_id for sample_id in ids if sample_id]
                files.append(item)

            write_json_atomic(tmp / "build_config.json", _config_snapshot(cfg))
            files.append({
                "path": "build_config.json",
                "sha256": sha256_file(tmp / "build_config.json"),
                "size": (tmp / "build_config.json").stat().st_size,
            })
            revisions = dataset_draft.source_revisions(cfg, include_aux_outputs=True)
            actual_hash = dataset_draft.plan_fingerprint(
                cfg, draft, include_aux_outputs=True, revisions=revisions
            )
            if validate_input and actual_hash != plan_hash:
                raise RuntimeError("Build 输入在固化前发生变化，请重新运行 Stage 3")
            manifest = {
                "version": 1,
                "build_id": build_id,
                "plan_hash": plan_hash,
                "created_at": now_iso(),
                "config_hash": config_hash(cfg),
                "source_revisions": revisions,
                "split_plan": copy.deepcopy(draft.get("split_plan") or {}),
                "decisions": copy.deepcopy(draft.get("decisions") or {}),
                "counts": dict(counts),
                "files": files,
                "samples": samples,
            }
            write_json_atomic(tmp / "BUILD.json", manifest)
            tmp.replace(target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        write_json_atomic(root / _LATEST, {"build_id": build_id})
        return target, manifest


def resolve_build(
    cfg: Optional[Dict[str, Any]] = None, build_id: Optional[str] = None
) -> Tuple[Path, Dict[str, Any]]:
    cfg = cfg or load_config()
    root = builds_root(cfg)
    if build_id:
        selected = _valid_id(build_id)
    else:
        latest = read_json(root / _LATEST, default={}) or {}
        selected = str(latest.get("build_id") or "")
        if not selected:
            rows = list_builds(cfg)
            selected = rows[0]["build_id"] if rows else ""
        if not selected:
            raise FileNotFoundError("尚无数据集 Build，请先构建数据集")
    path = root / _valid_id(selected)
    manifest = read_json(path / "BUILD.json", default=None)
    if not isinstance(manifest, dict) or manifest.get("build_id") != selected:
        raise FileNotFoundError(f"数据集 Build 不存在或清单无效: {selected}")
    return path, manifest


def list_builds(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    root = builds_root(cfg)
    rows = []
    if not root.exists():
        return rows
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        manifest = read_json(path / "BUILD.json", default=None)
        if isinstance(manifest, dict) and manifest.get("build_id") == path.name:
            rows.append(manifest)
    rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return rows


def load_build_config(build_dir: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """读取并校验 Build 配置快照；缺失或漂移时禁止用当前配置替代。"""
    snapshot = read_json(Path(build_dir) / "build_config.json", default=None)
    if not isinstance(snapshot, dict) or not snapshot:
        raise RuntimeError("Build 配置快照缺失或无效，禁止打包")
    manifest = read_json(Path(build_dir) / "BUILD.json", default={}) or {}
    expected = str(manifest.get("config_hash") or "")
    if expected and config_hash(snapshot) != expected:
        raise RuntimeError("Build 配置快照与 BUILD.json 中的 config_hash 不一致")
    return snapshot
