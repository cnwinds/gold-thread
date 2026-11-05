"""从当前数据集删除一条会话及其跨阶段派生产物。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import now_iso


_SID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")
_FROZEN_TRAIN_DIRS = {"releases", "delivery"}


def validate_sid(sid: str) -> str:
    if not _SID_RE.fullmatch(str(sid or "")):
        raise ValueError("非法会话 ID")
    return sid


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON：{path}: {exc}") from exc


def _inside(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"会话文件越出数据目录：{path}") from exc
    return resolved


def _id_matches(value: Any, sid: str) -> bool:
    if value is None:
        return False
    text = str(value)
    return text == sid or sid in text.split("::")


def _row_belongs(row: Dict[str, Any], sid: str) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    record = row.get("record") if isinstance(row.get("record"), dict) else {}
    for value in (
        row.get("session_id"), metadata.get("session_id"), record.get("session_id"),
        row.get("sample_id"), metadata.get("sample_id"), row.get("diagnostic_id"),
        row.get("gold_id"),
    ):
        if _id_matches(value, sid):
            return True
    return False


class _Transaction:
    """先暂存全部变更；提交失败时恢复已经应用的目标文件。"""

    def __init__(self, workspace: Path):
        workspace.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=".session-delete-", dir=workspace))
        self.actions: list[tuple[Path, Optional[Path]]] = []

    def write_text(self, target: Path, text: str) -> None:
        target = Path(target)
        if not target.is_file():
            return
        staged = self.root / f"{len(self.actions)}.new"
        staged.write_text(text, encoding="utf-8", newline="")
        self.actions.append((target, staged))

    def write_json(self, target: Path, value: Any) -> None:
        self.write_text(target, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def delete(self, target: Path) -> None:
        target = Path(target)
        if target.is_file() and all(target != existing for existing, _ in self.actions):
            self.actions.append((target, None))

    def commit(self) -> None:
        backups: Dict[Path, Path] = {}
        applied: list[Path] = []
        try:
            for index, (target, _) in enumerate(self.actions):
                if target.exists():
                    backup = self.root / f"{index}.bak"
                    shutil.copy2(target, backup)
                    backups[target] = backup
            for target, staged in self.actions:
                if staged is None:
                    target.unlink()
                else:
                    os.replace(staged, target)
                applied.append(target)
        except Exception:
            for target in reversed(applied):
                backup = backups.get(target)
                if backup and backup.exists():
                    shutil.copy2(backup, target)
                elif target.exists():
                    target.unlink()
            raise
        finally:
            shutil.rmtree(self.root, ignore_errors=True)


def _filter_jsonl(tx: _Transaction, path: Path, sid: str) -> tuple[int, int]:
    """按血缘字段过滤，保留其它行的原始文本；返回 (删除数, 剩余数)。"""
    kept: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                kept.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析 {path} 第 {line_number} 行：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"训练数据必须是 JSON 对象：{path} 第 {line_number} 行")
            if _row_belongs(row, sid):
                removed += 1
            else:
                kept.append(line)
    if removed:
        tx.write_text(path, "".join(kept))
    return removed, sum(1 for line in kept if line.strip())


def _train_jsonl_files(train_root: Path) -> Iterable[Path]:
    if not train_root.exists():
        return []
    return (
        path for path in train_root.rglob("*.jsonl")
        if path.is_file() and not (_FROZEN_TRAIN_DIRS & set(path.relative_to(train_root).parts))
    )


def _remove_map_entry(tx: _Transaction, path: Path, sid: str, key: str = "") -> None:
    data = _read_json(path)
    if not isinstance(data, dict):
        return
    container = data.get(key) if key else data
    if not isinstance(container, dict) or sid not in container:
        return
    container.pop(sid)
    tx.write_json(path, data)


def _update_state(tx: _Transaction, state: Path, sid: str) -> None:
    _remove_map_entry(tx, state / "labels.json", sid)
    _remove_map_entry(tx, state / "turn_overrides.json", sid)
    _remove_map_entry(tx, state / "stage1.json", sid, "files")
    _remove_map_entry(tx, state / "stage2.json", sid, "files")

    collected_path = state / "collected.json"
    collected = _read_json(collected_path)
    if isinstance(collected, dict):
        changed = False
        if isinstance(collected.get("ids"), list):
            ids = [x for x in collected["ids"] if x != sid]
            changed = changed or ids != collected["ids"]
            collected["ids"] = ids
        if isinstance(collected.get("items"), dict) and sid in collected["items"]:
            collected["items"].pop(sid)
            changed = True
        if changed:
            tx.write_json(collected_path, collected)

    for name in ("dpo_exclusions.json",):
        path = state / name
        data = _read_json(path)
        if isinstance(data, dict):
            kept = {k: v for k, v in data.items() if not _id_matches(k, sid)}
            if len(kept) != len(data):
                tx.write_json(path, kept)

    for path in (state / "inbox").glob(f"**/{sid}.jsonl") if (state / "inbox").exists() else []:
        tx.delete(_inside(path, state))


def _update_sample_reviews(tx: _Transaction, gold: Path, sid: str) -> None:
    path = gold / "sample_reviews.json"
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        return
    items = data["items"]
    kept = {k: v for k, v in items.items() if not _id_matches(k, sid)}
    if len(kept) == len(items):
        return
    data["items"] = kept
    snapshot = data.get("review_snapshot")
    if isinstance(snapshot, dict):
        snapshot["store_items"] = len(kept)
    tx.write_json(path, data)


def _session_files(raw: Path, gold: Path, raw_entry: Dict[str, Any], gold_entry: Dict[str, Any], sid: str) -> set[Path]:
    files: set[Path] = set()
    for key in ("raw_file", "meta_file"):
        if raw_entry.get(key):
            path = _inside(Path(raw_entry[key]), raw)
            expected = f"{sid}.jsonl" if key == "raw_file" else f"{sid}.meta.json"
            if path.name != expected:
                raise ValueError(f"原始索引路径与会话不符：{path}")
            relative = path.relative_to(raw.resolve())
            if len(relative.parts) != 3 or relative.parts[0] not in {"positive", "negative", "pending"}:
                raise ValueError(f"原始索引路径不符合会话布局：{path}")
            files.add(path)
    for key, suffix in (
        ("gold_file", ".gold.jsonl"), ("reject_file", ".pruned.jsonl"),
        ("dpo_file", ".dpo.jsonl"), ("diag_file", ".diagnostics.jsonl"),
        ("annotation_file", ".json"),
    ):
        if gold_entry.get(key):
            path = _inside(Path(gold_entry[key]), gold)
            if path.name != sid + suffix:
                raise ValueError(f"黄金索引路径与会话不符：{path}")
            relative = path.relative_to(gold.resolve())
            valid_layout = (
                key == "gold_file" and len(relative.parts) == 3 and relative.parts[0] == "sessions"
            ) or (
                key != "gold_file" and len(relative.parts) == 2
                and relative.parts[0] == ("annotations" if key == "annotation_file" else "rejects")
            )
            if not valid_layout:
                raise ValueError(f"黄金索引路径不符合会话布局：{path}")
            files.add(path)
    for root, pattern in (
        (raw, f"**/{sid}.jsonl"), (raw, f"**/{sid}.meta.json"),
        (gold / "sessions", f"**/{sid}.gold.jsonl"),
        (gold / "rejects", f"{sid}.*.jsonl"),
        (gold / "annotations", f"{sid}.json"),
    ):
        if root.exists():
            for path in root.glob(pattern):
                if path.is_file():
                    files.add(_inside(path, gold if root != raw else raw))
    return files


def _counter_rows(rows: list[Dict[str, Any]], key: Callable[[Dict[str, Any]], Any]) -> Dict[str, int]:
    return dict(Counter(str(key(row) or "?") for row in rows))


def _remaining_rows(path: Path, sid: str) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if not _row_belongs(row, sid)]


def _update_train_metadata(
    tx: _Transaction, train: Path, sid: str, remaining: Dict[Path, int], staged_bytes: Dict[Path, bytes]
) -> None:
    stats_path = train / "stats.json"
    stats = _read_json(stats_path)
    if isinstance(stats, dict):
        funnels = stats.get("session_funnels")
        if isinstance(funnels, dict):
            funnels.pop(sid, None)
            stats["label_counts"] = _counter_rows(list(funnels.values()), lambda x: x.get("label"))
            stats["review_states"] = _counter_rows(list(funnels.values()), lambda x: x.get("effective_state"))
            stats["split"] = {
                str(v.get("project_group") or "unknown"): str(v.get("split") or "train")
                for v in funnels.values()
            }
        else:
            # 老版本统计缺少血缘时，不保留可能仍引用已删项目的聚合字段。
            for key in ("label_counts", "review_states", "split"):
                stats.pop(key, None)

        canonical = {
            "sft_step_windows_sharegpt.jsonl": _remaining_rows(train / "sft_step_windows_sharegpt.jsonl", sid),
            "sft_gold_trajectories_sharegpt.jsonl": _remaining_rows(train / "sft_gold_trajectories_sharegpt.jsonl", sid),
            "dpo_preference_pairs.jsonl": _remaining_rows(train / "dpo_preference_pairs.jsonl", sid),
            "negative_diagnostics.jsonl": _remaining_rows(train / "negative_diagnostics.jsonl", sid),
        }
        counts = stats.get("counts")
        if isinstance(counts, dict):
            for path, count in remaining.items():
                if train in path.parents:
                    counts[path.relative_to(train).as_posix()] = count
        windows = canonical["sft_step_windows_sharegpt.jsonl"]
        trajectories = canonical["sft_gold_trajectories_sharegpt.jsonl"]
        dpo = canonical["dpo_preference_pairs.jsonl"]
        diagnostics = canonical["negative_diagnostics.jsonl"]
        stats["by_project"] = _counter_rows(windows, lambda x: (x.get("metadata") or {}).get("project_group"))
        stats["by_stage"] = _counter_rows(windows, lambda x: (x.get("metadata") or {}).get("stage"))
        stats["by_model"] = _counter_rows(trajectories, lambda x: (x.get("metadata") or {}).get("model"))
        stats["weighted_steps"] = sum(int((x.get("metadata") or {}).get("sample_weight", 1) or 1) for x in windows)
        stats["dpo_by_source"] = _counter_rows(dpo, lambda x: (x.get("metadata") or {}).get("source"))
        stats["diagnostics_by_pathology"] = dict(Counter(
            str(p).split(":")[0] for row in diagnostics for p in row.get("pathologies", [])
        ))
        stats["generated_at"] = now_iso()
        tx.write_json(stats_path, stats)

        index_path = train / "index.json"
        index = _read_json(index_path)
        if isinstance(index, dict):
            index["stats"] = stats
            index["generated_at"] = stats["generated_at"]
            if isinstance(index.get("_files"), dict):
                for path, count in remaining.items():
                    if train in path.parents:
                        index["_files"][path.relative_to(train).as_posix()] = count
            tx.write_json(index_path, index)

    manifest_path = train / "manifest.json"
    manifest = _read_json(manifest_path)
    if isinstance(manifest, dict):
        if isinstance(manifest.get("sessions"), list):
            manifest["sessions"] = [x for x in manifest["sessions"] if x != sid]
        if isinstance(manifest.get("excluded_samples"), list):
            manifest["excluded_samples"] = [x for x in manifest["excluded_samples"] if not _id_matches(x, sid)]
        files = manifest.get("files")
        if isinstance(files, dict):
            for path, content in staged_bytes.items():
                if train not in path.parents:
                    continue
                name = path.relative_to(train).as_posix()
                if name in files and isinstance(files[name], dict):
                    files[name]["count"] = remaining[path]
                    files[name]["sha256"] = hashlib.sha256(content).hexdigest()
        counts = manifest.get("counts")
        if isinstance(counts, dict):
            names = {"sft_train": "sft.train.jsonl", "sft_eval": "sft.eval.jsonl", "dpo_train": "dpo.train.jsonl", "dpo_eval": "dpo.eval.jsonl"}
            for key, name in names.items():
                path = train / name
                if key in counts and path in remaining:
                    counts[key] = remaining[path]
        tx.write_json(manifest_path, manifest)


def delete_session(
    sid: str,
    cfg: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = lambda _: None,
) -> Dict[str, Any]:
    """删除当前数据集中的 sid；源会话与冻结交付包不在操作范围内。"""
    sid = validate_sid(sid)
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw, gold, train, state = dirs["raw"], dirs["gold"], dirs["train"], dirs["state"]
    raw_index_path, gold_index_path = raw / "index.json", gold / "index.json"
    raw_index = _read_json(raw_index_path, {"sessions": {}})
    gold_index = _read_json(gold_index_path, {"sessions": {}})
    if not isinstance(raw_index.get("sessions"), dict) or not isinstance(gold_index.get("sessions"), dict):
        raise ValueError("原始或黄金索引格式不正确")
    raw_entry = raw_index["sessions"].get(sid) or {}
    gold_entry = gold_index["sessions"].get(sid) or {}
    if not raw_entry and not gold_entry:
        raise KeyError(f"会话不存在: {sid}")

    tx = _Transaction(state)
    try:
        files = _session_files(raw, gold, raw_entry, gold_entry, sid)
        for path in sorted(files):
            tx.delete(path)
        log(f"[delete] 已校验 {len(files)} 个会话专属文件")

        raw_index["sessions"].pop(sid, None)
        raw_index["counts"] = _counter_rows(list(raw_index["sessions"].values()), lambda x: x.get("label") or "pending")
        raw_index["generated_at"] = now_iso()
        tx.write_json(raw_index_path, raw_index)

        gold_index["sessions"].pop(sid, None)
        gold_index["counts"] = _counter_rows(list(gold_index["sessions"].values()), lambda x: x.get("label") or "pending")
        gold_index["generated_at"] = now_iso()

        removed_rows = 0
        gold_counts: Dict[str, int] = {}
        for name in ("dpo_pairs.jsonl", "diagnostics.jsonl"):
            path = gold / name
            if path.exists():
                removed, left = _filter_jsonl(tx, path, sid)
                removed_rows += removed
                gold_counts[name] = left
        if gold_counts:
            gold_index["dpo_pairs"] = gold_counts.get("dpo_pairs.jsonl", int(gold_index.get("dpo_pairs") or 0))
            gold_index["diagnostics"] = gold_counts.get("diagnostics.jsonl", int(gold_index.get("diagnostics") or 0))
        tx.write_json(gold_index_path, gold_index)

        _update_state(tx, state, sid)
        _update_sample_reviews(tx, gold, sid)

        remaining: Dict[Path, int] = {}
        staged_bytes: Dict[Path, bytes] = {}
        for path in _train_jsonl_files(train):
            path = _inside(path, train)
            removed, left = _filter_jsonl(tx, path, sid)
            if removed:
                remaining[path] = left
                # tx 中的暂存文本才是提交后的内容，用于 manifest 哈希。
                staged = next(staged for target, staged in tx.actions if target == path and staged is not None)
                staged_bytes[path] = staged.read_bytes()
                removed_rows += removed
        _update_train_metadata(tx, train, sid, remaining, staged_bytes)
        log(f"[delete] 从聚合与训练文件移除 {removed_rows} 行，涉及 {len(remaining)} 份训练文件")

        tx.commit()
    except Exception:
        shutil.rmtree(tx.root, ignore_errors=True)
        raise

    result = {
        "ok": True,
        "session_id": sid,
        "deleted_files": sum(1 for _, staged in tx.actions if staged is None),
        "updated_files": sum(1 for _, staged in tx.actions if staged is not None),
        "removed_rows": removed_rows,
        "source_preserved": True,
        "frozen_releases_preserved": True,
    }
    log(f"[delete] 会话 {sid} 已从当前数据集移除")
    return result
