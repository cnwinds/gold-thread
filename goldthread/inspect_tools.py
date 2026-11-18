"""Token 友好的分级摘要检查器（source / corpus / session / step）。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline import curation, dataset_draft, stage1_extract
from pipeline.common import pilog
from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import load_index, read_json, read_jsonl


def _truncate_text(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        omitted = len(value) - max_chars
        return f"{value[:max_chars]}... [truncated {omitted} chars]"
    if isinstance(value, (dict, list)):
        raw = json.dumps(value, ensure_ascii=False)
        if len(raw) <= max_chars:
            return value
        omitted = len(raw) - max_chars
        return f"{raw[:max_chars]}... [truncated {omitted} chars]"
    return value


def _truncate_args(args: Any, max_chars: int = 600) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {"_raw": _truncate_text(args, max_chars)}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        out[str(k)] = _truncate_text(v, max_chars)
    return out


def _match_project_filter(project_group: str, query: Optional[str]) -> bool:
    if not query:
        return True
    p = (project_group or "").lower()
    q = query.strip().lower()
    return p == q or p == f"game::{q}" or p.split("::")[-1] == q


def _resolve_session_id(sid_or_prefix: str, candidates: Iterable[str]) -> str:
    q = str(sid_or_prefix or "").strip()
    if not q:
        raise ValueError("session_id 不能为空")
    pool = sorted({str(c) for c in candidates if c})
    if q in pool:
        return q
    matches = [c for c in pool if c.startswith(q)]
    if not matches:
        raise KeyError(f"未找到匹配的会话: {q}")
    if len(matches) > 1:
        preview = ", ".join(matches[:5])
        raise ValueError(f"会话前缀 {q!r} 匹配多条会话 ({len(matches)} 条): {preview}")
    return matches[0]


def _match_adopted_sid(
    parsed_sid: str,
    file_path: Path,
    raw_sessions: Dict[str, Any],
    abs_to_sid: Dict[str, str],
) -> Optional[str]:
    if parsed_sid in raw_sessions:
        return parsed_sid
    stem = file_path.stem
    tail = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    candidates_tokens = [tok for tok in (parsed_sid, stem, tail) if tok and len(tok) >= 6]
    for tok in candidates_tokens:
        if tok in raw_sessions:
            return tok
        prefix_hits = [
            rsid
            for rsid in raw_sessions
            if rsid.startswith(tok) or tok.startswith(rsid)
        ]
        if len(prefix_hits) == 1:
            return prefix_hits[0]
        if len(prefix_hits) > 1:
            return sorted(prefix_hits)[0]
    try:
        resolved = str(file_path.resolve())
    except OSError:
        resolved = str(file_path)
    return abs_to_sid.get(resolved)


def inspect_source(
    source_dir: Path | str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    unadopted_only: bool = False,
) -> Dict[str, Any]:
    """扫描源目录中的 .jsonl 会话，对比 01_raw/index.json 判断入库状态与项目归组。"""
    cfg = cfg or load_config()
    src_root = Path(source_dir)
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"源会话目录不存在: {src_root}")

    dirs = stage_dirs(cfg)
    raw_idx = load_index(dirs["raw"] / "index.json")
    raw_sessions = raw_idx.get("sessions") or {}
    corpus_projects = {
        str(e.get("project_group") or "")
        for e in raw_sessions.values()
        if isinstance(e, dict) and e.get("project_group")
    }
    abs_to_sid: Dict[str, str] = {}
    for rsid, entry in raw_sessions.items():
        if not isinstance(entry, dict):
            continue
        abs_p = (entry.get("source") or {}).get("abs_path")
        if abs_p:
            try:
                abs_to_sid[str(Path(abs_p).resolve())] = rsid
            except OSError:
                abs_to_sid[str(abs_p)] = rsid

    max_tool_chars = int((cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
    sessions_out: List[Dict[str, Any]] = []
    by_project: Dict[str, Dict[str, Any]] = {}
    parse_errors: List[Dict[str, str]] = []
    total_files = 0
    adopted_count = 0
    unadopted_count = 0

    for path in sorted(src_root.rglob("*.jsonl")):
        total_files += 1
        rel_path = path.relative_to(src_root).as_posix()
        try:
            parsed = pilog.parse_session(path, max_tool_output_chars=max_tool_chars)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append({"rel_path": rel_path, "error": str(exc)})
            continue
        if not parsed.get("messages"):
            continue

        sid = str(parsed["session_id"])
        cwd = str(parsed.get("cwd") or "")
        project_group = stage1_extract.infer_project_group(cwd, rel_path)
        models = parsed.get("models") or {}
        dominant_model = (
            max(models.items(), key=lambda kv: kv[1])[0] if models else "unknown"
        )
        label, _reason, _ratio = stage1_extract.label_for(models, cfg)
        msgs = parsed["messages"]
        user_turns = sum(1 for m in msgs if m.get("role") == "user")
        asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assistant_steps = len(asst_msgs)
        tool_calls = sum(len(m.get("tool_calls") or []) for m in asst_msgs)

        adopted_sid = _match_adopted_sid(sid, path, raw_sessions, abs_to_sid)
        adopted = adopted_sid is not None
        if adopted:
            adopted_count += 1
        else:
            unadopted_count += 1

        proj_bucket = by_project.setdefault(
            project_group,
            {
                "project_group": project_group,
                "total": 0,
                "adopted": 0,
                "unadopted": 0,
                "in_corpus": project_group in corpus_projects,
            },
        )
        proj_bucket["total"] += 1
        if adopted:
            proj_bucket["adopted"] += 1
        else:
            proj_bucket["unadopted"] += 1

        item = {
            "session_id": sid,
            "rel_path": rel_path,
            "project_group": project_group,
            "dominant_model": dominant_model,
            "label": label,
            "user_turns": user_turns,
            "assistant_steps": assistant_steps,
            "tool_calls": tool_calls,
            "adopted": adopted,
            "adopted_sid": adopted_sid,
        }
        if not unadopted_only or not adopted:
            sessions_out.append(item)

    projects = sorted(by_project)
    new_projects = sorted(p for p in projects if p not in corpus_projects)
    return {
        "source_dir": str(src_root),
        "total_files": total_files,
        "parsed_sessions": adopted_count + unadopted_count,
        "adopted_count": adopted_count,
        "unadopted_count": unadopted_count,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "projects": projects,
        "new_projects": new_projects,
        "by_project": {k: by_project[k] for k in sorted(by_project)},
        "sessions": sessions_out,
    }


def inspect_corpus(
    *,
    cfg: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """汇总 01_raw / 02_gold / 03_train / dataset_draft / LATEST_BUILD 全景摘要。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_idx = load_index(dirs["raw"] / "index.json")
    gold_idx = load_index(dirs["gold"] / "index.json")
    raw_sessions = raw_idx.get("sessions") or {}
    gold_sessions = gold_idx.get("sessions") or {}
    stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    draft = dataset_draft.load_draft(cfg)
    split_plan = draft.get("split_plan") or {}
    latest_build = read_json(
        dirs["root"] / "04_release" / "builds" / "LATEST_BUILD.json", default={}
    ) or {}
    funnels = stats.get("session_funnels") or {}

    all_sids = sorted(set(raw_sessions) | set(gold_sessions))
    sessions_out: List[Dict[str, Any]] = []
    by_project: Dict[str, Dict[str, Any]] = {}
    split_projects: Dict[str, set[str]] = {"train": set(), "eval": set()}

    for sid in all_sids:
        raw_e = raw_sessions.get(sid) if isinstance(raw_sessions.get(sid), dict) else {}
        gold_e = gold_sessions.get(sid) if isinstance(gold_sessions.get(sid), dict) else {}
        proj = str(
            gold_e.get("project_group") or raw_e.get("project_group") or "unknown"
        )
        if not _match_project_filter(proj, project):
            continue

        funnel_e = funnels.get(sid) if isinstance(funnels.get(sid), dict) else {}
        split = funnel_e.get("split")
        if not split:
            if split_plan.get("assignments") or split_plan.get("session_assignments"):
                split = dataset_draft.effective_split(proj, sid, split_plan)
            else:
                split = (stats.get("split") or {}).get(proj) or "train"
        if split in split_projects:
            split_projects[split].add(proj)

        models = raw_e.get("models") or {}
        dominant_model = str(
            gold_e.get("dominant_model")
            or (max(models.items(), key=lambda kv: kv[1])[0] if models else "unknown")
        )
        g_stats = gold_e.get("stats") or {}
        r_stats = raw_e.get("stats") or {}
        raw_steps = int(g_stats.get("raw_steps") or r_stats.get("assistant_steps") or 0)
        gold_steps = int(g_stats.get("gold_steps") or 0)
        turns_folded = int(g_stats.get("turns_folded") or r_stats.get("turns_folded") or 0)
        sft_windows = int((funnel_e.get("funnel") or {}).get("admitted_steps") or 0)
        dpo_written = int(
            funnel_e.get("dpo_written")
            if funnel_e.get("dpo_written") is not None
            else (gold_e.get("dpo_count") or 0)
        )

        mrep = gold_e.get("model_report")
        ann: Optional[Dict[str, Any]] = None
        if not mrep or not funnel_e.get("effective_state"):
            ann_file = gold_e.get("annotation_file")
            if ann_file and Path(ann_file).is_file():
                ann = read_json(Path(ann_file), default={}) or {}
                if not mrep:
                    mrep = ann.get("model_report")

        eff_state = funnel_e.get("effective_state")
        if not eff_state:
            if ann is not None:
                eff_state = curation.resolve(ann, cfg).get("state")
            else:
                eff_state = gold_e.get("status") or ("raw_only" if not gold_e else "unreviewed")

        verdict = mrep.get("verdict") if isinstance(mrep, dict) else None
        pathologies = dict(gold_e.get("pathologies") or raw_e.get("pathologies") or {})

        pb = by_project.setdefault(
            proj,
            {
                "project_group": proj,
                "split": split,
                "sessions": 0,
                "gold_sessions": 0,
                "raw_steps": 0,
                "gold_steps": 0,
                "sft_windows": 0,
                "dpo_written": 0,
                "labels": Counter(),
            },
        )
        pb["sessions"] += 1
        if gold_e:
            pb["gold_sessions"] += 1
        pb["raw_steps"] += raw_steps
        pb["gold_steps"] += gold_steps
        pb["sft_windows"] += sft_windows
        pb["dpo_written"] += dpo_written
        pb["labels"][str(gold_e.get("label") or raw_e.get("label") or "pending")] += 1

        sessions_out.append(
            {
                "session_id": sid,
                "project_group": proj,
                "label": gold_e.get("label") or raw_e.get("label") or "pending",
                "split": split,
                "dominant_model": dominant_model,
                "raw_steps": raw_steps,
                "gold_steps": gold_steps,
                "turns_folded": turns_folded,
                "sft_windows": sft_windows,
                "dpo_written": dpo_written,
                "effective_state": eff_state,
                "verdict": verdict,
                "pathologies": pathologies,
            }
        )

    by_project_clean: Dict[str, Any] = {}
    for proj in sorted(by_project):
        item = dict(by_project[proj])
        item["labels"] = dict(item["labels"])
        by_project_clean[proj] = item

    counts = stats.get("counts") or {}
    canonical = [p for p in by_project_clean if p.startswith("game::")]
    non_canonical = [p for p in by_project_clean if not p.startswith("game::")]

    return {
        "filter_project": project,
        "raw_sessions": sum(1 for s in sessions_out if s["session_id"] in raw_sessions),
        "gold_sessions": sum(1 for s in sessions_out if s["session_id"] in gold_sessions),
        "project_count": len(by_project_clean),
        "canonical_projects": len(canonical),
        "non_canonical_projects": non_canonical,
        "split_projects": {
            "train": sorted(split_projects["train"]),
            "eval": sorted(split_projects["eval"]),
        },
        "train_counts": {
            "sft_trajectories": int(counts.get("sft_gold_trajectories_sharegpt.jsonl") or 0),
            "sft_windows": int(counts.get("sft_step_windows_sharegpt.jsonl") or 0),
            "sft_windows_train": int(counts.get("sft_step_windows_sharegpt.train.jsonl") or 0),
            "sft_windows_eval": int(counts.get("sft_step_windows_sharegpt.eval.jsonl") or 0),
            "dpo_total": int(counts.get("dpo_preference_pairs.jsonl") or 0),
            "dpo_train": int(counts.get("dpo_preference_pairs.train.jsonl") or 0),
            "dpo_eval": int(counts.get("dpo_preference_pairs.eval.jsonl") or 0),
            "weighted_steps": int(stats.get("weighted_steps") or 0),
        },
        "latest_build_id": latest_build.get("build_id") or None,
        "by_project": by_project_clean,
        "sessions": sessions_out,
    }


def _load_session_bundle(
    sid_or_prefix: str, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    dirs = stage_dirs(cfg)
    raw_idx = load_index(dirs["raw"] / "index.json")
    gold_idx = load_index(dirs["gold"] / "index.json")
    raw_sessions = raw_idx.get("sessions") or {}
    gold_sessions = gold_idx.get("sessions") or {}
    sid = _resolve_session_id(sid_or_prefix, set(raw_sessions) | set(gold_sessions))

    raw_e = raw_sessions.get(sid) or {}
    gold_e = gold_sessions.get(sid) or {}
    raw_meta = (
        read_json(Path(raw_e["meta_file"]), default={})
        if raw_e.get("meta_file") and Path(raw_e["meta_file"]).is_file()
        else raw_e
    ) or {}
    gold_rec: Dict[str, Any] = {}
    if gold_e.get("gold_file") and Path(gold_e["gold_file"]).is_file():
        rows = read_jsonl(Path(gold_e["gold_file"]))
        if rows:
            gold_rec = rows[0]
    ann = (
        read_json(Path(gold_e["annotation_file"]), default={})
        if gold_e.get("annotation_file") and Path(gold_e["annotation_file"]).is_file()
        else {}
    ) or {}
    stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    funnel = (stats.get("session_funnels") or {}).get(sid)

    return {
        "dirs": dirs,
        "sid": sid,
        "raw_e": raw_e,
        "gold_e": gold_e,
        "raw_meta": raw_meta,
        "gold_rec": gold_rec,
        "ann": ann,
        "funnel": funnel,
    }


def _session_messages_for_inspect(
    bundle: Dict[str, Any], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    gold_rec = bundle["gold_rec"]
    if gold_rec.get("messages"):
        return list(gold_rec["messages"])
    raw_file = (
        (bundle["gold_e"].get("source") or {}).get("raw_file")
        or bundle["raw_e"].get("raw_file")
        or ""
    )
    if raw_file and Path(raw_file).is_file():
        max_tool_chars = int((cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
        parsed = pilog.parse_session(Path(raw_file), max_tool_output_chars=max_tool_chars)
        flow, _marks, _fstats = pilog.build_flow(
            parsed,
            cfg,
            bundle["raw_meta"].get("turn_annotations"),
            bundle["raw_meta"].get("manual_turn_overrides"),
        )
        return flow
    return []


def inspect_session(
    sid_or_prefix: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    include_steps: bool = False,
) -> Dict[str, Any]:
    """检查单条会话的元数据、根任务、回合清洗标记、质量报告与步骤摘要。"""
    cfg = cfg or load_config()
    bundle = _load_session_bundle(sid_or_prefix, cfg)
    sid = bundle["sid"]
    raw_e = bundle["raw_e"]
    gold_e = bundle["gold_e"]
    raw_meta = bundle["raw_meta"]
    gold_rec = bundle["gold_rec"]
    ann = bundle["ann"]

    effective = curation.resolve(ann, cfg) if ann else {
        "state": "unreviewed",
        "rewrites": {},
        "excluded": [],
        "kept": [],
    }
    mrep = ann.get("model_report") or gold_rec.get("model_report") or gold_e.get("model_report")
    g_stats = gold_rec.get("stats") or gold_e.get("stats") or {}
    r_stats = raw_meta.get("stats") or raw_e.get("stats") or {}

    turn_marks = [
        {
            "turn_index": m.get("turn_index"),
            "kind": m.get("kind"),
            "action": m.get("action"),
            "dropped": bool(m.get("dropped")),
            "folded": bool(m.get("folded")),
            "reason": m.get("reason") or "",
            "assistant_steps": int(m.get("assistant_steps") or 0),
            "kept_steps": int(m.get("kept_steps") or 0),
            "text_preview": m.get("text_preview") or "",
        }
        for m in (raw_meta.get("turns") or [])
        if isinstance(m, dict)
    ]

    models = raw_meta.get("models") or raw_e.get("models") or {}
    dominant_model = str(
        gold_rec.get("dominant_model")
        or gold_e.get("dominant_model")
        or (max(models.items(), key=lambda kv: kv[1])[0] if models else "unknown")
    )

    out: Dict[str, Any] = {
        "session_id": sid,
        "label": gold_e.get("label") or raw_e.get("label") or "pending",
        "status": gold_e.get("status") or ("raw_only" if not gold_e else "unknown"),
        "project_group": gold_e.get("project_group") or raw_e.get("project_group") or "unknown",
        "dominant_model": dominant_model,
        "cwd": gold_rec.get("cwd") or gold_e.get("cwd") or raw_e.get("cwd") or "",
        "timestamp": raw_meta.get("timestamp") or raw_e.get("timestamp") or "",
        "root_prompt": g_stats.get("root_prompt") or r_stats.get("root_prompt") or "",
        "raw_root_prompt": g_stats.get("raw_root_prompt") or r_stats.get("raw_root_prompt") or "",
        "constraints": g_stats.get("constraints") or r_stats.get("constraints") or [],
        "turn_marks": turn_marks,
        "model_report": mrep,
        "effective": effective,
        "funnel": bundle["funnel"],
        "stats": g_stats or r_stats,
    }

    if include_steps:
        msgs = _session_messages_for_inspect(bundle, cfg)
        excluded_set = set(effective.get("excluded") or [])
        rewrites = effective.get("rewrites") or {}
        asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
        compact_steps: List[Dict[str, Any]] = []
        for idx, m in enumerate(asst_msgs):
            mid = str(m.get("id") or "")
            eff_think = rewrites.get(mid, m.get("thinking") or "")
            raw_think = m.get("raw_thinking") or m.get("thinking") or ""
            q = m.get("quality") or m.get("raw_quality") or {}
            is_excl = bool(m.get("excluded")) or (mid in excluded_set)
            excl_reason = m.get("exclude_reason") or ("excluded_by_curation" if mid in excluded_set else "")
            compact_steps.append(
                {
                    "step_index": idx,
                    "msg_id": mid,
                    "turn_index": m.get("turn_index"),
                    "think_chars": len(eff_think),
                    "raw_think_chars": len(raw_think),
                    "text_chars": len(m.get("text") or ""),
                    "tool_calls": [str(tc.get("name") or "") for tc in (m.get("tool_calls") or [])],
                    "flags": list(m.get("flags") or []),
                    "judge_reason": m.get("judge_reason") or "",
                    "excluded": is_excl,
                    "exclude_reason": excl_reason,
                    "pathologies": list(q.get("pathologies") or []),
                }
            )
        out["steps"] = compact_steps

    return out


def _collect_tool_results_for_step(
    msgs: List[Dict[str, Any]],
    asst_msg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_trs = asst_msg.get("tool_results") or []
    if not raw_trs:
        call_ids = {tc.get("id") for tc in (asst_msg.get("tool_calls") or []) if tc.get("id")}
        if call_ids:
            raw_trs = [
                m for m in msgs
                if m.get("role") == "tool" and m.get("tool_call_id") in call_ids
            ]
    out: List[Dict[str, Any]] = []
    for tr in raw_trs:
        if not isinstance(tr, dict):
            continue
        out.append(
            {
                "tool_call_id": tr.get("tool_call_id") or "",
                "name": tr.get("name") or "tool",
                "isError": bool(tr.get("isError")),
                "raw_len": int(tr.get("raw_len") or len(tr.get("content") or "")),
                "content_preview": _truncate_text(tr.get("content") or "", 800),
            }
        )
    return out


def _find_related_dpo_pairs(
    bundle: Dict[str, Any], msg_id: str
) -> List[Dict[str, Any]]:
    dirs = bundle["dirs"]
    sid = bundle["sid"]
    candidates: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    dpo_files = []
    if bundle["gold_e"].get("dpo_file"):
        dpo_files.append(Path(bundle["gold_e"]["dpo_file"]))
    dpo_files.append(dirs["train"] / "dpo_preference_pairs.jsonl")
    for p in dpo_files:
        if not p.is_file():
            continue
        for row in read_jsonl(p):
            meta = row.get("metadata") or {}
            if meta.get("session_id") and meta.get("session_id") != sid:
                continue
            rel_ids = {
                str(meta.get("chosen_msg_id") or ""),
                str(meta.get("rejected_msg_id") or ""),
                str(meta.get("msg_id") or ""),
            }
            if msg_id not in rel_ids:
                continue
            sample_id = str(row.get("sample_id") or "")
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            c_val = chosen.get("value", "") if isinstance(chosen, dict) else str(chosen or "")
            r_val = rejected.get("value", "") if isinstance(rejected, dict) else str(rejected or "")
            candidates.append(
                {
                    "sample_id": sample_id,
                    "source": meta.get("source") or "",
                    "reason": meta.get("reason") or "",
                    "chosen_msg_id": meta.get("chosen_msg_id"),
                    "rejected_msg_id": meta.get("rejected_msg_id"),
                    "msg_id": meta.get("msg_id"),
                    "chosen_len": len(c_val),
                    "rejected_len": len(r_val),
                }
            )
    return candidates


def inspect_step(
    sid_or_prefix: str,
    step_ref: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """查看单步完整三态血缘（raw / gold / curation）及关联 DPO 偏好对。"""
    cfg = cfg or load_config()
    bundle = _load_session_bundle(sid_or_prefix, cfg)
    sid = bundle["sid"]
    msgs = _session_messages_for_inspect(bundle, cfg)
    asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
    if not asst_msgs:
        raise KeyError(f"会话 {sid} 没有 assistant 步骤")

    ref = str(step_ref or "").strip()
    if not ref:
        raise ValueError("step_ref 不能为空")

    target_idx: Optional[int] = None
    exact_matches = [i for i, m in enumerate(asst_msgs) if str(m.get("id") or "") == ref]
    if exact_matches:
        target_idx = exact_matches[0]
    elif ref.isdigit() or (ref.startswith("#") and ref[1:].isdigit()):
        idx = int(ref[1:] if ref.startswith("#") else ref)
        if 0 <= idx < len(asst_msgs):
            target_idx = idx
        else:
            raise IndexError(f"步骤序号越界: {idx} (有效范围 0..{len(asst_msgs) - 1})")
    else:
        prefix_matches = [
            i for i, m in enumerate(asst_msgs) if str(m.get("id") or "").startswith(ref)
        ]
        if len(prefix_matches) == 1:
            target_idx = prefix_matches[0]
        elif len(prefix_matches) > 1:
            ids = ", ".join(str(asst_msgs[i].get("id")) for i in prefix_matches[:5])
            raise ValueError(f"step_ref {ref!r} 匹配多个 msg_id: {ids}")

    if target_idx is None:
        raise KeyError(f"未找到步骤 {ref!r} (会话 {sid})")

    m = asst_msgs[target_idx]
    mid = str(m.get("id") or "")
    ann = bundle["ann"]
    effective = curation.resolve(ann, cfg) if ann else {
        "state": "unreviewed",
        "rewrites": {},
        "excluded": [],
        "kept": [],
    }
    excluded_set = set(effective.get("excluded") or [])
    rewrites = effective.get("rewrites") or {}

    tool_calls = [
        {
            "id": tc.get("id") or "",
            "name": tc.get("name") or "",
            "arguments": _truncate_args(tc.get("arguments") or {}, 600),
        }
        for tc in (m.get("tool_calls") or [])
        if isinstance(tc, dict)
    ]

    return {
        "session_id": sid,
        "step_index": target_idx,
        "total_steps": len(asst_msgs),
        "msg_id": mid,
        "turn_index": m.get("turn_index"),
        "raw_thinking": m.get("raw_thinking") if "raw_thinking" in m else (m.get("thinking") or ""),
        "thinking": m.get("thinking") or "",
        "effective_rewrite": rewrites.get(mid),
        "raw_text": m.get("raw_text") if "raw_text" in m else (m.get("text") or ""),
        "text": m.get("text") or "",
        "flags": list(m.get("flags") or []),
        "judge_reason": m.get("judge_reason") or "",
        "folded_from": m.get("folded_from"),
        "fold_details": m.get("fold_details"),
        "pruned_detour": m.get("pruned_detour"),
        "pruned_details": m.get("pruned_details"),
        "excluded": bool(m.get("excluded")) or (mid in excluded_set),
        "exclude_reason": m.get("exclude_reason") or ("excluded_by_curation" if mid in excluded_set else ""),
        "raw_quality": m.get("raw_quality"),
        "quality": m.get("quality"),
        "tool_calls": tool_calls,
        "tool_results": _collect_tool_results_for_step(msgs, m),
        "dpo_pairs": _find_related_dpo_pairs(bundle, mid),
    }
