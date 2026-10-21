"""Stage 1：从 pi 会话日志中提取原始数据，正/负例分目录落盘（不可变副本 + 元数据）。"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import pilog
from .common.config import config_hash, data_root, load_config, resolve_path, section_hash, stage_dirs
from .turnclassify import classify_session
from .common.quality import analyze_step_quality
from .common.store import (
    dump_index,
    fingerprint,
    fingerprint_equal,
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    safe_name,
    write_json_atomic,
    write_jsonl_atomic,
)

STAGE = "stage1"


def infer_project_group(cwd: str, rel_path: str) -> str:
    if cwd:
        norm = cwd.replace("\\", "/").rstrip("/")
        leaf = norm.split("/")[-1].lower()
        if "/crack/" in norm.lower() + "/":
            return "game::" + leaf.split("-")[0]
        return f"workspace::{leaf}"
    folder = rel_path.replace("\\", "/").split("/")[0].lower()
    if folder.startswith("--d--crack-"):
        sub = folder[len("--d--crack-") :].rstrip("-")
        return "game::" + sub.split("-")[0]
    return f"session_group::{folder}"


def _admitted(rel: str, cfg: Dict[str, Any]) -> bool:
    folder = rel.replace("\\", "/").split("/")[0].lower()
    acfg = cfg.get("admission", {})
    excluded = {f.lower() for f in acfg.get("exclude_folders", [])}
    if folder in excluded:
        return False
    return any(folder.startswith(p.lower()) for p in acfg.get("workspace_prefixes", []))


def label_for(models: Dict[str, int], cfg: Dict[str, Any]) -> tuple[str, str, float]:
    lcfg = cfg.get("labeling", {})
    positive = [re.compile(p, re.IGNORECASE) for p in lcfg.get("positive_model_patterns", [])]
    negative = [re.compile(p, re.IGNORECASE) for p in lcfg.get("negative_model_patterns", [])]
    if not models:
        return lcfg.get("unknown_label", "pending"), "no_model_signal", 0.0
    total = sum(models.values())
    dominant, count = max(models.items(), key=lambda kv: kv[1])
    ratio = count / total if total else 0.0
    unknown = lcfg.get("unknown_label", "pending")
    if ratio < float(lcfg.get("dominant_ratio_min", 0.6)):
        return unknown, f"mixed_models(dominant={dominant},{ratio:.0%})", ratio
    if any(p.search(dominant) for p in positive):
        return "positive", f"teacher_model:{dominant}", ratio
    if any(p.search(dominant) for p in negative):
        return "negative", f"student_model:{dominant}", ratio
    return unknown, f"unmatched_model:{dominant}", ratio


def _write_raw_copy(src: Path, dest_dir: Path, session_id: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{session_id}.jsonl"
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return dest
    shutil.copyfile(src, dest)
    return dest


def _prev_state_path(cfg: Dict[str, Any]) -> Path:
    return stage_dirs(cfg)["state"] / "stage1.json"


def run(force: bool = False, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_root = dirs["raw"]
    raw_root.mkdir(parents=True, exist_ok=True)
    dirs["state"].mkdir(parents=True, exist_ok=True)

    sessions_dir = resolve_path(cfg["paths"]["sessions_dir"])
    if not sessions_dir.exists():
        raise FileNotFoundError(f"会话目录不存在: {sessions_dir}")

    max_tool_chars = int(cfg["synth"].get("max_tool_output_chars", 12000))
    overrides = read_json(dirs["state"] / "labels.json", default={}) or {}
    hash_cfg = {**cfg, "paths": {**cfg["paths"], "sessions_dir": str(sessions_dir)}}
    cfgh = section_hash(
        hash_cfg, ["admission", "labeling", "turns", "llm.model", "paths.sessions_dir", "synth.max_tool_output_chars"]
    )
    state = read_json(_prev_state_path(cfg), default={}) or {}
    prev_files: Dict[str, Any] = state.get("files", {}) if state.get("config_hash") == cfgh else {}
    prev_index = load_index(raw_root / "index.json")

    candidates: List[Dict[str, Any]] = []
    skipped = {"non_crack_folder": 0, "excluded_folder": 0, "empty": 0}
    folder_stats: Dict[str, int] = defaultdict(int)

    for path in sorted(sessions_dir.rglob("*.jsonl")):
        rel = path.relative_to(sessions_dir).as_posix()
        folder = rel.split("/")[0]
        folder_stats[folder] += 1
        if not _admitted(rel, cfg):
            folder_l = folder.lower()
            if folder_l in {f.lower() for f in cfg["admission"].get("exclude_folders", [])}:
                skipped["excluded_folder"] += 1
            else:
                skipped["non_crack_folder"] += 1
            continue
        candidates.append({"path": path, "rel": rel})

    log(f"[stage1] 扫描 {sum(folder_stats.values())} 个会话文件，准入 {len(candidates)} 个（忽略 {skipped['non_crack_folder']} 个非项目目录）")

    parsed: List[Dict[str, Any]] = []
    for item in candidates:
        try:
            session = pilog.parse_session(item["path"], max_tool_output_chars=max_tool_chars)
        except Exception as exc:  # noqa: BLE001
            log(f"[stage1] 解析失败 {item['rel']}: {exc}")
            continue
        if not session["messages"]:
            skipped["empty"] += 1
            continue
        session["rel"] = item["rel"]
        session["source_path"] = str(item["path"])
        parsed.append(session)

    # 去重：迁移/复制产生的同 cwd + 同首条用户消息副本，保留最长的一份
    duplicate_of: Dict[str, str] = {}
    if cfg["admission"].get("dedupe_duplicate_prefix", True):
        buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for s in parsed:
            first_user = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")
            buckets[(s["cwd"].lower(), first_user.strip()[:200])].append(s)
        for _, group in buckets.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda s: s["line_count"], reverse=True)
            keeper = group[0]
            for dup in group[1:]:
                # 仅当两者差异巨大（前 20 轮重复副本）才判定为重复
                if dup["line_count"] < keeper["line_count"] * 0.5:
                    duplicate_of[dup["session_id"]] = keeper["session_id"]

    sessions_index: Dict[str, Any] = {}
    state_files: Dict[str, Any] = {}
    written = 0
    reused = 0
    removed_stale = 0
    label_counts = Counter()

    for s in parsed:
        sid = s["session_id"]
        src = Path(s["source_path"])
        fp = fingerprint(src)

        if sid in duplicate_of:
            log(f"[stage1] 跳过重复副本 {sid}（与 {duplicate_of[sid]} 同源）")
            continue

        label, reason, ratio = label_for(s["models"], cfg)
        if sid in overrides and overrides[sid] in ("positive", "negative", "pending"):
            label, reason = overrides[sid], "manual_override"
        label_counts[label] += 1
        prev = prev_files.get(sid)
        prev_entry = (prev_index.get("sessions") or {}).get(sid)

        unchanged = (
            not force
            and prev
            and fingerprint_equal(prev.get("fingerprint"), fp)
            and prev.get("label") == label
            and prev.get("flow_version") == pilog.FLOW_VERSION
            and prev_entry
            and prev_entry.get("label") == label
            and all(Path(p).exists() for p in prev.get("outputs", []))
        )
        if unchanged:
            sessions_index[sid] = prev_entry
            state_files[sid] = prev
            reused += 1
            continue

        annotations = None
        if cfg.get("turns", {}).get("use_model", False):
            try:
                annotations = classify_session(s, cfg=cfg, log=log)
            except Exception as exc:  # noqa: BLE001
                log(f"[stage1] 模型回合分类失败 {sid}: {exc}（回退正则）")
        flow, marks, flow_stats = pilog.build_flow(s, cfg, annotations)
        asst = [m for m in flow if m["role"] == "assistant"]
        qualities = [
            analyze_step_quality(
                m.get("thinking", ""),
                m.get("text", ""),
                m.get("tool_calls", []),
                m.get("tool_results", []),
                s["cwd"],
                max_think_chars=4500,
            )
            for m in asst
        ]
        pathologies = Counter()
        for q in qualities:
            for p in q["pathologies"]:
                pathologies[p.split(":")[0]] += 1
        score_avg = round(sum(q["score"] for q in qualities) / len(qualities), 2) if qualities else 0.0
        clean_steps = sum(1 for q in qualities if q["is_clean"])

        project = infer_project_group(s["cwd"], s["rel"])
        proj_dir = safe_name(project)
        dest_dir = raw_root / label / proj_dir
        raw_file = _write_raw_copy(src, dest_dir, sid)

        meta = {
            "session_id": sid,
            "label": label,
            "label_reason": reason,
            "label_ratio": round(ratio, 3),
            "project_group": project,
            "cwd": s["cwd"],
            "timestamp": s["timestamp"],
            "source": {"rel_path": s["rel"], "abs_path": s["source_path"]},
            "fingerprint": fp,
            "models": s["models"],
            "providers": s["providers"],
            "platform": s["platform_counts"],
            "line_count": s["line_count"],
            "bad_lines": s["bad_lines"],
            "stats": {
                "assistant_steps": len(asst),
                "clean_steps": clean_steps,
                "quality_avg": score_avg,
                "turns_total": flow_stats["turns_total"],
                "turns_kept": flow_stats["turns_kept"],
                "dropped": flow_stats["dropped"],
                "has_root": flow_stats["has_root"],
                "root_prompt": flow_stats["root_prompt"],
                "constraints": flow_stats["constraints"],
                "root_index": flow_stats["root_index"],
                "model_classified": flow_stats["model_classified"],
            },
            "turn_annotations": annotations,
            "pathologies": dict(pathologies),
            "turns": marks,
            "raw_file": str(raw_file),
            "meta_file": "",
            "extracted_at": now_iso(),
        }
        meta_path = dest_dir / f"{sid}.meta.json"
        meta["meta_file"] = str(meta_path)
        write_json_atomic(meta_path, meta)

        # 标签变化时清掉旧目录下的同名文件
        if prev and prev.get("label") and prev["label"] != label:
            for old in prev.get("outputs", []):
                try:
                    Path(old).unlink(missing_ok=True)
                except OSError:
                    pass

        sessions_index[sid] = meta
        state_files[sid] = {
            "label": label,
            "fingerprint": fp,
            "annotations_root": (annotations or {}).get("root_turn"),
            "flow_version": pilog.FLOW_VERSION,
            "outputs": [str(raw_file), str(meta_path)],
        }
        written += 1
        log(f"[stage1] {label:<8} {project:<22} {sid[:8]}  steps={len(asst):<4} turns={flow_stats['turns_total']:<4} kept={flow_stats['turns_kept']}")

    # 清理已删除的源会话
    for sid, entry in list((prev_index.get("sessions") or {}).items()):
        if sid in sessions_index:
            continue
        for p in (entry.get("raw_file"), entry.get("meta_file")):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                    removed_stale += 1
                except OSError:
                    pass
        state_files.pop(sid, None)

    extra = {
        "stage": STAGE,
        "config_hash": cfgh,
        "sessions_dir": str(sessions_dir),
        "counts": dict(label_counts),
        "skipped": skipped,
        "folder_stats": dict(folder_stats),
        "duplicates": duplicate_of,
    }
    dump_index(raw_root / "index.json", sessions_index, extra)
    write_json_atomic(
        _prev_state_path(cfg),
        {
            "stage": STAGE,
            "version": 1,
            "config_hash": cfgh,
            "last_run": now_iso(),
            "files": state_files,
        },
    )

    summary = {
        "stage": STAGE,
        "admitted": len(candidates),
        "written": written,
        "reused": reused,
        "removed_stale": removed_stale,
        "counts": dict(label_counts),
        "skipped": skipped,
    }
    log(
        f"[stage1] 完成：写入 {written} / 复用 {reused} / 清理 {removed_stale}；"
        f"正例 {label_counts.get('positive', 0)}，反例 {label_counts.get('negative', 0)}，待定 {label_counts.get('pending', 0)}"
    )
    return summary