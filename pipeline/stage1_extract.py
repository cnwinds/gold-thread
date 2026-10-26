"""Stage 1：从 pi 会话日志中提取原始数据，正/负例分目录落盘（不可变副本 + 元数据）。

支持双源模式：
1. 标准增量采集：扫描 `paths.sessions_dir` 下的全部 `.jsonl` 会话，执行领域准入与同源去重；
2. 离线重放（Replay Fallback）：若 `sessions_dir` 不存在但 `01_raw` 已有不可变副本，直接基于副本重算元数据与回合清洗。
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import pilog
from .common.config import load_config, resolve_path, section_hash, stage_dirs
from .common.quality import analyze_step_quality
from .common.store import (
    dump_index,
    fingerprint,
    fingerprint_equal,
    load_index,
    now_iso,
    read_json,
    safe_name,
    sha256_text,
    write_json_atomic,
)
from . import steps as stepcat
from .turnclassify import classify_session

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
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return dest
    shutil.copyfile(src, dest)
    return dest


def _prev_state_path(cfg: Dict[str, Any]) -> Path:
    return stage_dirs(cfg)["state"] / "stage1.json"


def _turn_overrides_path(cfg: Dict[str, Any]) -> Path:
    return stage_dirs(cfg)["state"] / "turn_overrides.json"


def _compute_meta_from_flow(
    s: Dict[str, Any],
    sid: str,
    label: str,
    reason: str,
    ratio: float,
    project: str,
    fp: Dict[str, Any],
    raw_file: Path,
    meta_path: Path,
    cfg: Dict[str, Any],
    annotations: Optional[Dict[str, Any]],
    manual_turn_ov: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    flow, marks, flow_stats = pilog.build_flow(s, cfg, annotations, manual_turn_ov)
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
    strengths = Counter()
    for q in qualities:
        for p in q["pathologies"]:
            pathologies[p.split(":")[0]] += 1
        for st in q["strengths"]:
            strengths[st.split(":")[0]] += 1
    score_avg = round(sum(q["score"] for q in qualities) / len(qualities), 2) if qualities else 0.0
    clean_steps = sum(1 for q in qualities if q["is_clean"])

    return {
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
            "turns_direct": flow_stats["turns_direct"],
            "turns_folded": flow_stats["turns_folded"],
            "turns_dropped": flow_stats["turns_dropped"],
            "dropped_steps": flow_stats["dropped_steps"],
            "dropped": flow_stats["dropped"],
            "has_root": flow_stats["has_root"],
            "root_prompt": flow_stats["root_prompt"],
            "raw_root_prompt": flow_stats["raw_root_prompt"],
            "root_normalized": flow_stats["root_normalized"],
            "constraints": flow_stats["constraints"],
            "root_index": flow_stats["root_index"],
            "model_classified": flow_stats["model_classified"],
            "manual_overridden": flow_stats["manual_overridden"],
        },
        "turn_annotations": annotations,
        "manual_turn_overrides": manual_turn_ov or {},
        "pathologies": dict(pathologies),
        "strengths": dict(strengths),
        "turns": marks,
        "raw_file": str(raw_file),
        "meta_file": str(meta_path),
        "extracted_at": now_iso(),
    }


def run(force: bool = False, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_root = dirs["raw"]
    raw_root.mkdir(parents=True, exist_ok=True)
    dirs["state"].mkdir(parents=True, exist_ok=True)

    sessions_dir = resolve_path(cfg["paths"]["sessions_dir"])
    max_tool_chars = int(cfg["synth"].get("max_tool_output_chars", 12000))
    overrides = read_json(dirs["state"] / "labels.json", default={}) or {}
    turn_overrides_all = read_json(_turn_overrides_path(cfg), default={}) or {}
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
    rejected_samples: List[Dict[str, str]] = []
    replay_mode = False

    dedupe_n = 0
    if sessions_dir.exists():
        for path in sorted(sessions_dir.rglob("*.jsonl")):
            rel = path.relative_to(sessions_dir).as_posix()
            folder = rel.split("/")[0]
            folder_stats[folder] += 1
            if not _admitted(rel, cfg):
                folder_l = folder.lower()
                if folder_l in {f.lower() for f in cfg["admission"].get("exclude_folders", [])}:
                    skipped["excluded_folder"] += 1
                    reason_tag = "excluded_folder"
                else:
                    skipped["non_crack_folder"] += 1
                    reason_tag = "non_crack_folder"
                if len(rejected_samples) < 40:
                    rejected_samples.append({"rel_path": rel, "folder": folder, "reason": reason_tag})
                continue
            candidates.append({"path": path, "rel": rel})
        stepcat.emit(log, "s1.scan", sum(folder_stats.values()), sum(folder_stats.values()),
                     f"扫描 {sum(folder_stats.values())} 个文件")
        stepcat.emit(log, "s1.admit", len(candidates), sum(folder_stats.values()),
                     f"准入 {len(candidates)} 个")
        log(
            f"[stage1] 扫描 {sum(folder_stats.values())} 个会话文件，准入 {len(candidates)} 个"
            f"（忽略 {skipped['non_crack_folder']} 个非项目目录，排除 {skipped['excluded_folder']} 个根目录）"
        )
    else:
        # 离线重放回退：若源目录不存在，尝试直接从 01_raw 现有不可变副本恢复
        existing_raws = sorted(raw_root.rglob("*.jsonl"))
        if not existing_raws:
            raise FileNotFoundError(f"会话目录不存在且 01_raw 无可用副本: {sessions_dir}")
        replay_mode = True
        prev_sessions = prev_index.get("sessions") or {}
        for path in existing_raws:
            sid = path.stem
            prev_e = prev_sessions.get(sid) or {}
            rel = (prev_e.get("source") or {}).get("rel_path") or path.relative_to(raw_root).as_posix()
            candidates.append({"path": path, "rel": rel, "replay_prev": prev_e})
        skipped = prev_index.get("skipped") or skipped
        folder_stats = defaultdict(int, prev_index.get("folder_stats") or {})
        log(f"[stage1] 源目录 {sessions_dir} 不可达，启用 01_raw 不可变副本重放模式（共 {len(candidates)} 个会话）")

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
        session["replay_prev"] = item.get("replay_prev")
        parsed.append(session)

    # 去重：迁移/复制产生的同 cwd + 同首条用户消息副本，保留最长的一份
    duplicate_of: Dict[str, str] = {}
    if not replay_mode and cfg["admission"].get("dedupe_duplicate_prefix", True):
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
                if dup["line_count"] < keeper["line_count"] * 0.5:
                    duplicate_of[dup["session_id"]] = keeper["session_id"]

    if duplicate_of:
        dedupe_n = len(duplicate_of)
    stepcat.emit(log, "s1.dedupe", dedupe_n, dedupe_n, f"去重 {dedupe_n} 个副本")

    sessions_index: Dict[str, Any] = {}
    state_files: Dict[str, Any] = {}
    written = 0
    reused = 0
    removed_stale = 0
    label_counts = Counter()

    total_parsed = len(parsed)
    for done_i, s in enumerate(parsed, 1):
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
        manual_turn_ov = turn_overrides_all.get(sid) or {}
        turn_ov_hash = sha256_text(json.dumps(manual_turn_ov, sort_keys=True, ensure_ascii=False))[:16]

        unchanged = (
            not force
            and prev
            and fingerprint_equal(prev.get("fingerprint"), fp)
            and prev.get("label") == label
            and prev.get("flow_version") == pilog.FLOW_VERSION
            and prev.get("turn_ov_hash", "") == turn_ov_hash
            and prev_entry
            and prev_entry.get("label") == label
            and all(Path(p).exists() for p in prev.get("outputs", []))
        )
        if unchanged:
            sessions_index[sid] = prev_entry
            state_files[sid] = prev
            reused += 1
            continue

        # 复用已有模型回合分类（若源文件未变且已有 turn_annotations）
        existing_meta_path = Path(prev_entry["meta_file"]) if prev_entry and prev_entry.get("meta_file") else None
        existing_meta = read_json(existing_meta_path, default={}) if existing_meta_path and existing_meta_path.exists() else {}
        annotations = (existing_meta or {}).get("turn_annotations")
        if cfg.get("turns", {}).get("use_model", False) and (force or not annotations):
            stepcat.emit(log, "s1.classify", done_i, total_parsed, f"分类 {sid[:8]}")
            try:
                annotations = classify_session(s, cfg=cfg, log=log)
            except Exception as exc:  # noqa: BLE001
                log(f"[stage1] 模型回合分类失败 {sid}: {exc}（保留已有或回退正则）")

        project = infer_project_group(s["cwd"], s["rel"])
        proj_dir = safe_name(project)
        dest_dir = raw_root / label / proj_dir
        raw_file = _write_raw_copy(src, dest_dir, sid)
        meta_path = dest_dir / f"{sid}.meta.json"

        meta = _compute_meta_from_flow(
            s=s,
            sid=sid,
            label=label,
            reason=reason,
            ratio=ratio,
            project=project,
            fp=fp,
            raw_file=raw_file,
            meta_path=meta_path,
            cfg=cfg,
            annotations=annotations,
            manual_turn_ov=manual_turn_ov,
        )
        write_json_atomic(meta_path, meta)

        # 标签变化时清掉旧目录下的同名文件
        if prev and prev.get("label") and prev["label"] != label:
            for old in prev.get("outputs", []):
                if Path(old).resolve() not in (raw_file.resolve(), meta_path.resolve()):
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
            "turn_ov_hash": turn_ov_hash,
            "outputs": [str(raw_file), str(meta_path)],
        }
        written += 1
        stepcat.emit(log, "s1.flow", done_i, total_parsed, f"成流 {sid[:8]}")
        st = meta["stats"]
        log(
            f"[stage1] {label:<8} {project:<22} {sid[:8]}  "
            f"steps={st['assistant_steps']:<4} turns={st['turns_total']:<3} "
            f"(直留={st['turns_direct']} 折叠={st['turns_folded']} 剥离={st['turns_dropped']})"
        )

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
        "replay_mode": replay_mode,
        "counts": dict(label_counts),
        "skipped": skipped,
        "folder_stats": dict(folder_stats),
        "duplicates": duplicate_of,
        "rejected_samples": rejected_samples,
    }
    dump_index(raw_root / "index.json", sessions_index, extra)
    write_json_atomic(
        _prev_state_path(cfg),
        {
            "stage": STAGE,
            "version": 2,
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
        "duplicates": duplicate_of,
    }
    stepcat.emit(log, "s1.flow", total_parsed, total_parsed, f"写入 {written} / 复用 {reused}")
    log(
        f"[stage1] 完成：写入 {written} / 复用 {reused} / 清理 {removed_stale}；"
        f"正例 {label_counts.get('positive', 0)}，反例 {label_counts.get('negative', 0)}，待定 {label_counts.get('pending', 0)}"
    )
    return summary


def relabel_session(sid: str, new_label: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """即时修改会话标签：写入 labels.json，移动 01_raw 文件并同步更新 stage1 / stage2 索引。"""
    if new_label not in ("positive", "negative", "pending"):
        raise ValueError(f"非法标签: {new_label}")
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    labels_path = dirs["state"] / "labels.json"
    overrides = read_json(labels_path, default={}) or {}
    overrides[sid] = new_label
    write_json_atomic(labels_path, overrides)

    raw_index_path = dirs["raw"] / "index.json"
    raw_index = load_index(raw_index_path)
    entry = (raw_index.get("sessions") or {}).get(sid)
    if not entry:
        return {"ok": True, "session_id": sid, "label": new_label, "synced": False}

    old_raw = Path(entry["raw_file"])
    old_meta = Path(entry["meta_file"])
    proj_dir = safe_name(entry.get("project_group", "unknown"))
    dest_dir = dirs["raw"] / new_label / proj_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    new_raw = dest_dir / f"{sid}.jsonl"
    new_meta = dest_dir / f"{sid}.meta.json"

    if old_raw.exists() and old_raw.resolve() != new_raw.resolve():
        shutil.move(str(old_raw), str(new_raw))
    if old_meta.exists() and old_meta.resolve() != new_meta.resolve():
        shutil.move(str(old_meta), str(new_meta))

    meta = read_json(new_meta, default=entry) or entry
    meta["label"] = new_label
    meta["label_reason"] = "manual_override"
    meta["raw_file"] = str(new_raw)
    meta["meta_file"] = str(new_meta)
    write_json_atomic(new_meta, meta)

    raw_index["sessions"][sid] = meta
    counts = Counter(v.get("label", "pending") for v in raw_index["sessions"].values())
    raw_index["counts"] = dict(counts)
    write_json_atomic(raw_index_path, raw_index)

    # 更新 stage1.json 状态
    s1_path = _prev_state_path(cfg)
    s1 = read_json(s1_path, default={}) or {}
    if sid in (s1.get("files") or {}):
        s1["files"][sid]["label"] = new_label
        s1["files"][sid]["outputs"] = [str(new_raw), str(new_meta)]
        write_json_atomic(s1_path, s1)

    # 若 stage2 已有该会话，同步更新其标签
    gold_index_path = dirs["gold"] / "index.json"
    gold_index = load_index(gold_index_path)
    gold_entry = (gold_index.get("sessions") or {}).get(sid)
    if gold_entry:
        min_steps = int(cfg.get("gold", {}).get("min_gold_steps", 10))
        gold_steps = int(gold_entry.get("stats", {}).get("gold_steps", 0))
        new_status = "gold" if (new_label == "positive" and gold_steps >= min_steps) else (
            "raw_only" if new_label == "pending" else "negative_source"
        )
        gold_entry["label"] = new_label
        gold_entry["status"] = new_status
        if gold_entry.get("source"):
            gold_entry["source"]["raw_file"] = str(new_raw)
        gold_file = Path(gold_entry.get("gold_file", ""))
        if gold_file.exists():
            from .common.store import read_jsonl, write_jsonl_atomic
            rows = read_jsonl(gold_file)
            if rows:
                rows[0]["label"] = new_label
                if rows[0].get("source"):
                    rows[0]["source"]["raw_file"] = str(new_raw)
                write_jsonl_atomic(gold_file, rows)
        ann_file = Path(gold_entry.get("annotation_file", ""))
        if ann_file.exists():
            ann = read_json(ann_file, default={}) or {}
            ann["label"] = new_label
            ann["status"] = new_status
            write_json_atomic(ann_file, ann)
        gold_index["counts"] = dict(Counter(v.get("label", "pending") for v in gold_index["sessions"].values()))
        write_json_atomic(gold_index_path, gold_index)

    return {"ok": True, "session_id": sid, "label": new_label, "synced": True}


def update_turn_overrides(
    sid: str,
    turns_patch: Optional[Dict[str, Any]] = None,
    root_prompt: Optional[str] = None,
    reset: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """人工更新某条会话的回合分类或根任务指令，并立即重算该会话的清洗流与元数据。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    ov_path = _turn_overrides_path(cfg)
    all_ov = read_json(ov_path, default={}) or {}

    if reset:
        all_ov.pop(sid, None)
    else:
        cur = dict(all_ov.get(sid) or {})
        cur_turns = dict(cur.get("turns") or {})
        for k, v in (turns_patch or {}).items():
            if v is None or v == "" or (isinstance(v, dict) and not v.get("kind")):
                cur_turns.pop(str(k), None)
            else:
                cur_turns[str(k)] = v if isinstance(v, dict) else {"kind": str(v)}
        cur["turns"] = cur_turns
        if root_prompt is not None:
            cur["root_prompt"] = root_prompt.strip()
        if not cur.get("turns") and not cur.get("root_prompt"):
            all_ov.pop(sid, None)
        else:
            all_ov[sid] = cur

    write_json_atomic(ov_path, all_ov)

    raw_index_path = dirs["raw"] / "index.json"
    raw_index = load_index(raw_index_path)
    entry = (raw_index.get("sessions") or {}).get(sid)
    if not entry:
        raise KeyError(f"会话不存在: {sid}")

    raw_file = Path(entry["raw_file"])
    meta_path = Path(entry["meta_file"])
    max_tool_chars = int(cfg["synth"].get("max_tool_output_chars", 12000))
    s = pilog.parse_session(raw_file, max_tool_output_chars=max_tool_chars)
    s["rel"] = (entry.get("source") or {}).get("rel_path", "")
    s["source_path"] = (entry.get("source") or {}).get("abs_path", str(raw_file))
    old_meta = read_json(meta_path, default=entry) or entry
    annotations = old_meta.get("turn_annotations")
    manual_turn_ov = all_ov.get(sid) or {}

    new_meta = _compute_meta_from_flow(
        s=s,
        sid=sid,
        label=entry["label"],
        reason=entry.get("label_reason", ""),
        ratio=entry.get("label_ratio", 1.0),
        project=entry.get("project_group", "unknown"),
        fp=entry.get("fingerprint") or fingerprint(raw_file),
        raw_file=raw_file,
        meta_path=meta_path,
        cfg=cfg,
        annotations=annotations,
        manual_turn_ov=manual_turn_ov,
    )
    write_json_atomic(meta_path, new_meta)
    raw_index["sessions"][sid] = new_meta
    write_json_atomic(raw_index_path, raw_index)

    return {
        "ok": True,
        "session_id": sid,
        "stats": new_meta["stats"],
        "turns": new_meta["turns"],
        "manual_turn_overrides": manual_turn_ov,
    }
