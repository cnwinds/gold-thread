"""纯确定性策展工具（adopt / drop-session / turns / steps / dpo / split / exclude）。

所有写操作强制关闭隐式 LLM，执行前严格校验安全阀不变量，并自动记入 ledger.jsonl。
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline import dataset_draft, session_delete, stage1_extract, stage2_gold
from pipeline.common import pilog
from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import load_index, now_iso, read_json, read_jsonl, write_json_atomic

from . import inspect_tools, ledger

VALID_TURN_KINDS = frozenset({"root", "task", "setup", "noise", "continuation", "correction"})
VALID_STEP_ACTIONS = frozenset({"keep", "prune", "fold", "rewrite", "exclude", "clear_text"})
VALID_REPORT_VERDICTS = frozenset({"trainable", "discard", "needs_human"})
VALID_DPO_VERDICTS = frozenset({"good", "swap", "drop"})


def _offline_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg or load_config())
    out.setdefault("turns", {})["use_model"] = False
    out.setdefault("judge", {})["enabled"] = False
    return out


def _load_json_arg(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return copy.deepcopy(raw)
    if isinstance(raw, Path):
        return json.loads(raw.read_text(encoding="utf-8"))
    if not isinstance(raw, str):
        raise ValueError("JSON 参数格式无效")
    s = raw.strip()
    if not s:
        raise ValueError("JSON 参数不能为空")
    if s.startswith("@"):
        p = Path(s[1:].strip())
        if not p.is_file():
            raise FileNotFoundError(f"JSON 文件不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    if not (s.startswith("{") or s.startswith("[")):
        p = Path(s)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _flatten_tokens(values: Optional[Sequence[str] | str]) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)
    out: List[str] = []
    for item in items:
        for part in str(item or "").split(","):
            tok = part.strip()
            if tok:
                out.append(tok)
    return out


def _match_source_session(query: str, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    q = query.strip()
    exact = [
        s for s in sessions
        if s["session_id"] == q or Path(s["rel_path"]).stem == q
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"选择条件 {q!r} 匹配多条源会话: {[s['session_id'] for s in exact[:5]]}")

    hits: List[Dict[str, Any]] = []
    for s in sessions:
        sid = s["session_id"]
        stem = Path(s["rel_path"]).stem
        tail = stem.rsplit("_", 1)[-1] if "_" in stem else stem
        if sid.startswith(q) or stem.startswith(q) or tail.startswith(q):
            hits.append(s)
    if not hits:
        raise KeyError(f"源目录未找到匹配的会话: {q}")
    if len(hits) > 1:
        raise ValueError(f"选择前缀 {q!r} 匹配多条源会话: {[s['session_id'] for s in hits[:5]]}")
    return hits[0]


def adopt_sessions(
    from_dir: Path | str,
    *,
    select: Optional[Sequence[str] | str] = None,
    all_unadopted: bool = False,
    origin: str = "pi",
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """批量采纳源会话进 01_raw（强制零 LLM、自动剥离 <file> 标签与规范归组）。"""
    t0 = time.monotonic()
    if origin not in ("pi", "cursor"):
        raise ValueError(f"非法 origin: {origin} (仅支持 pi|cursor)")
    select_tokens = _flatten_tokens(select)
    if not select_tokens and not all_unadopted:
        raise ValueError("必须指定 select 或 all_unadopted=True")

    offline_cfg = _offline_cfg(cfg)
    src_root = Path(from_dir)
    scanned = inspect_tools.inspect_source(src_root, cfg=offline_cfg, unadopted_only=False)
    sessions = scanned["sessions"]

    if all_unadopted:
        chosen = [s for s in sessions if not s["adopted"]]
    else:
        chosen_map: Dict[str, Dict[str, Any]] = {}
        for tok in select_tokens:
            hit = _match_source_session(tok, sessions)
            chosen_map[hit["session_id"]] = hit
        chosen = list(chosen_map.values())

    before_summary = ledger.corpus_snapshot_summary(offline_cfg)
    adopted_sids: List[str] = []
    projects: set[str] = set()
    catalog_id = f"curate:{src_root.name}"

    for item in chosen:
        rel_path = item["rel_path"]
        src_path = src_root / rel_path
        sid = stage1_extract.adopt_file(
            src_path,
            rel_path,
            origin=origin,
            catalog_id=catalog_id,
            cfg=offline_cfg,
            log=lambda *_: None,
        )
        adopted_sids.append(sid)
        projects.add(item["project_group"])

    raw_idx = load_index(stage_dirs(offline_cfg)["raw"] / "index.json")
    for sid in adopted_sids:
        pg = ((raw_idx.get("sessions") or {}).get(sid) or {}).get("project_group")
        if pg:
            projects.add(str(pg))

    after_summary = ledger.corpus_snapshot_summary(offline_cfg)
    ev = ledger.record_operation(
        "curate.adopt",
        reason=reason,
        actor=actor,
        params={
            "from_dir": str(src_root),
            "origin": origin,
            "select": select_tokens,
            "all_unadopted": bool(all_unadopted),
        },
        affected={"sids": adopted_sids, "projects": sorted(projects)},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=offline_cfg,
    )
    return {
        "ok": True,
        "adopted_count": len(adopted_sids),
        "sids": adopted_sids,
        "projects": sorted(projects),
        "op_id": ev["op_id"],
    }


def drop_session(
    sid_or_prefix: str,
    *,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """安全级联删除指定会话及其派生产物，并记入 ledger。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    sid = inspect_tools._resolve_session_id(sid_or_prefix, set(raw_sessions) | set(gold_sessions))  # noqa: SLF001

    proj = (
        (gold_sessions.get(sid) or {}).get("project_group")
        or (raw_sessions.get(sid) or {}).get("project_group")
        or ""
    )
    before_summary = ledger.corpus_snapshot_summary(cfg)
    res = session_delete.delete_session(sid, cfg=cfg)
    after_summary = ledger.corpus_snapshot_summary(cfg)

    ev = ledger.record_operation(
        "curate.drop_session",
        reason=reason,
        actor=actor,
        params={"sid": sid},
        affected={"sids": [sid], "projects": [proj] if proj else []},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=cfg,
    )
    return {**res, "op_id": ev["op_id"]}


def _normalize_turns_patch(raw_patch: Any) -> Dict[str, Any]:
    data = _load_json_arg(raw_patch)
    out: Dict[str, Any] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or "index" not in item:
                raise ValueError("列表格式的回合补丁每项必须包含 index 字段")
            idx = str(int(item["index"]))
            spec = {k: v for k, v in item.items() if k != "index"}
            if not spec or spec.get("kind") is None:
                out[idx] = None
            else:
                kind = str(spec.get("kind"))
                if kind not in VALID_TURN_KINDS:
                    raise ValueError(f"非法回合类型: {kind}")
                out[idx] = spec
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            idx = str(int(k))
            if v is None or v == "":
                out[idx] = None
            elif isinstance(v, str):
                if v not in VALID_TURN_KINDS:
                    raise ValueError(f"非法回合类型: {v}")
                out[idx] = {"kind": v}
            elif isinstance(v, dict):
                if not v or v.get("kind") is None:
                    out[idx] = None
                else:
                    kind = str(v.get("kind"))
                    if kind not in VALID_TURN_KINDS:
                        raise ValueError(f"非法回合类型: {kind}")
                    out[idx] = dict(v)
            else:
                raise ValueError(f"回合 {k} 的补丁值格式无效")
        return out

    raise ValueError("回合补丁必须是 JSON 对象或列表")


def curate_turns(
    sid_or_prefix: str,
    patch: Any = None,
    *,
    reset: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """写入回合覆盖并立即重算该会话 01_raw 流与元数据（不触发 LLM）。"""
    t0 = time.monotonic()
    offline_cfg = _offline_cfg(cfg)
    dirs = stage_dirs(offline_cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    sid = inspect_tools._resolve_session_id(sid_or_prefix, raw_sessions)  # noqa: SLF001

    if not reset and patch is None:
        raise ValueError("必须提供 turns patch 或指定 reset=True")
    normalized = {} if reset else _normalize_turns_patch(patch)

    proj = str((raw_sessions.get(sid) or {}).get("project_group") or "")
    before_summary = ledger.corpus_snapshot_summary(offline_cfg)
    res = stage1_extract.update_turn_overrides(
        sid, turns_patch=normalized, reset=reset, cfg=offline_cfg
    )
    after_summary = ledger.corpus_snapshot_summary(offline_cfg)

    ev = ledger.record_operation(
        "curate.turns",
        reason=reason,
        actor=actor,
        params={"sid": sid, "reset": bool(reset), "patch": normalized},
        affected={"sids": [sid], "projects": [proj] if proj else []},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=offline_cfg,
    )
    return {**res, "op_id": ev["op_id"]}


def _normalize_step_decision_entries(raw: Any) -> List[Dict[str, Any]]:
    data = _load_json_arg(raw)
    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        items = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("sessions"), dict):
            for sid_key, body in data["sessions"].items():
                if not isinstance(body, dict):
                    raise ValueError(f"会话 {sid_key} 的决策必须是对象")
                items.append({"session_id": sid_key, **body})
        elif isinstance(data.get("sessions"), list):
            items = [x for x in data["sessions"] if isinstance(x, dict)]
        elif "session_id" in data:
            items = [data]
        else:
            raise ValueError("步级决策 JSON 必须包含 session_id 或 sessions")
    else:
        raise ValueError("步级决策必须是 JSON 对象或列表")

    if not items:
        raise ValueError("步级决策列表为空")
    return items


def curate_steps(
    decisions_input: Any,
    *,
    apply: bool = True,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验并持久化结构化步级决策到 00_state/step_decisions.json，默认重算受影响黄金会话。"""
    t0 = time.monotonic()
    offline_cfg = _offline_cfg(cfg)
    dirs = stage_dirs(offline_cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    pool = set(raw_sessions) | set(gold_sessions)
    if not pool:
        raise RuntimeError("当前语料库没有任何会话")

    entries = _normalize_step_decision_entries(decisions_input)
    existing_store = stage2_gold.load_step_decisions(offline_cfg)
    staged_store = copy.deepcopy(existing_store)
    max_tool_chars = int((offline_cfg.get("synth") or {}).get("max_tool_output_chars", 12000))

    affected_sids: List[str] = []
    projects: set[str] = set()

    for item in entries:
        sid_ref = str(item.get("session_id") or "").strip()
        if not sid_ref:
            raise ValueError("缺少 session_id")
        sid = inspect_tools._resolve_session_id(sid_ref, pool)  # noqa: SLF001
        raw_e = raw_sessions.get(sid) or {}
        raw_file = raw_e.get("raw_file") or ((gold_sessions.get(sid) or {}).get("source") or {}).get("raw_file")
        if not raw_file or not Path(raw_file).is_file():
            raise FileNotFoundError(f"会话 {sid} 的原始文件不存在")

        meta_file = Path(raw_e["meta_file"]) if raw_e.get("meta_file") and Path(raw_e["meta_file"]).is_file() else None
        raw_meta = read_json(meta_file, default={}) if meta_file else {}
        parsed = pilog.parse_session(Path(raw_file), max_tool_output_chars=max_tool_chars)
        flow, _marks, _fstats = pilog.build_flow(
            parsed,
            offline_cfg,
            (raw_meta or {}).get("turn_annotations"),
            (raw_meta or {}).get("manual_turn_overrides"),
        )
        asst = [m for m in flow if m.get("role") == "assistant"]
        asst_by_id: Dict[str, Tuple[int, Dict[str, Any]]] = {
            str(m.get("id") or ""): (idx, m) for idx, m in enumerate(asst) if m.get("id")
        }

        cur_entry = copy.deepcopy(staged_store.get(sid) or {})
        merged_steps: Dict[str, Any] = dict(cur_entry.get("steps") or {})
        reset_steps: set[str] = {str(x) for x in (cur_entry.get("reset_steps") or []) if x}
        new_steps = item.get("steps")
        if new_steps is not None:
            if not isinstance(new_steps, dict):
                raise ValueError(f"会话 {sid} 的 steps 必须是对象")
            for mid, spec in new_steps.items():
                mid_s = str(mid)
                if spec is None:
                    merged_steps.pop(mid_s, None)
                    reset_steps.add(mid_s)
                    continue
                if not isinstance(spec, dict):
                    raise ValueError(f"步骤 {mid_s} 决策必须是对象")
                act = str(spec.get("action") or "").strip()
                if act not in VALID_STEP_ACTIONS:
                    raise ValueError(f"非法步骤动作: {act!r} (msg_id={mid_s})")
                if mid_s not in asst_by_id:
                    raise ValueError(f"步骤 {mid_s} 不在会话 {sid} 流中")
                reset_steps.discard(mid_s)
                rank, target_msg = asst_by_id[mid_s]
                if act == "prune" and rank == 0:
                    raise ValueError(f"首个助手步 {mid_s} 不允许 prune")
                if act == "rewrite":
                    rewritten = str(spec.get("rewritten") or "").strip()
                    if not rewritten:
                        raise ValueError(f"步骤 {mid_s} action=rewrite 时 rewritten 不能为空")
                    if not stage2_gold._rewrite_ok(rewritten, target_msg):  # noqa: SLF001
                        raise ValueError(
                            f"步骤 {mid_s} rewrite 未通过安全阀校验（空壳、超长或跨语种翻译）"
                        )
                    merged_steps[mid_s] = {**spec, "action": act, "rewritten": rewritten}
                else:
                    merged_steps[mid_s] = {**spec, "action": act}

        prune_count = sum(1 for v in merged_steps.values() if isinstance(v, dict) and v.get("action") == "prune")
        max_prune = max(1, int(len(asst) * 0.35)) if asst else 0
        if prune_count > max_prune:
            raise ValueError(
                f"会话 {sid} 累计 prune 步数 ({prune_count}) 超过 35% 安全上限 ({max_prune}, 总助手步 {len(asst)})"
            )

        merged_seams: Dict[str, Any] = dict(cur_entry.get("seam_patches") or {})
        new_seams = item.get("seam_patches")
        if new_seams is not None:
            if not isinstance(new_seams, dict):
                raise ValueError(f"会话 {sid} 的 seam_patches 必须是对象")
            for mid, pinfo in new_seams.items():
                mid_s = str(mid)
                if pinfo is None:
                    merged_seams.pop(mid_s, None)
                    continue
                if not isinstance(pinfo, dict):
                    raise ValueError(f"接缝补丁 {mid_s} 必须是对象")
                if mid_s not in asst_by_id:
                    raise ValueError(f"接缝补丁目标步骤 {mid_s} 不在会话 {sid} 流中")
                _rank, target_msg = asst_by_id[mid_s]
                patched = str(pinfo.get("patched_thinking") or "").strip()
                if not patched or not stage2_gold._rewrite_ok(patched, target_msg):  # noqa: SLF001
                    raise ValueError(f"接缝补丁 {mid_s} patched_thinking 未通过安全阀校验")
                merged_seams[mid_s] = {**pinfo, "patched_thinking": patched}

        model_report = item.get("model_report")
        if model_report is not None:
            if not isinstance(model_report, dict):
                raise ValueError(f"会话 {sid} 的 model_report 必须是对象")
            verdict = model_report.get("verdict")
            if verdict not in VALID_REPORT_VERDICTS:
                raise ValueError(f"非法 model_report.verdict: {verdict!r}")
        else:
            model_report = cur_entry.get("model_report")

        staged_store[sid] = {
            "updated_at": now_iso(),
            "by": actor,
            "reason": str(reason or item.get("reason") or cur_entry.get("reason") or ""),
            "model_report": model_report,
            "steps": merged_steps,
            "reset_steps": sorted(reset_steps),
            "seam_patches": merged_seams,
        }
        if sid not in affected_sids:
            affected_sids.append(sid)
        pg = raw_e.get("project_group") or (gold_sessions.get(sid) or {}).get("project_group")
        if pg:
            projects.add(str(pg))

    before_summary = ledger.corpus_snapshot_summary(offline_cfg)
    write_json_atomic(stage2_gold.step_decisions_path(offline_cfg), staged_store)

    stage2_res: Optional[Dict[str, Any]] = None
    if apply:
        stage2_res = stage2_gold.run(
            only=affected_sids,
            force=True,
            cfg=offline_cfg,
            log=lambda *_: None,
        )
    after_summary = ledger.corpus_snapshot_summary(offline_cfg)

    ev = ledger.record_operation(
        "curate.steps",
        reason=reason,
        actor=actor,
        params={"apply": bool(apply), "sessions": affected_sids},
        affected={"sids": affected_sids, "projects": sorted(projects)},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=offline_cfg,
    )
    return {
        "ok": True,
        "sessions": affected_sids,
        "applied": bool(apply),
        "stage2": stage2_res,
        "op_id": ev["op_id"],
    }


def _normalize_dpo_verdicts(raw: Any) -> Dict[str, Dict[str, str]]:
    data = _load_json_arg(raw)
    if isinstance(data, dict) and "pairs" in data and isinstance(data["pairs"], list):
        data = data["pairs"]
    elif isinstance(data, dict) and "verdicts" in data and isinstance(data["verdicts"], dict):
        data = data["verdicts"]

    out: Dict[str, Dict[str, str]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("DPO 裁决列表元素必须是对象")
            sample_id = str(item.get("sample_id") or "").strip()
            verdict = str(item.get("verdict") or "").strip()
            if not sample_id or verdict not in VALID_DPO_VERDICTS:
                raise ValueError(f"非法 DPO 裁决项: {item}")
            out[sample_id] = {"verdict": verdict, "reason": str(item.get("reason") or "").strip()}
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            sample_id = str(k or "").strip()
            if not sample_id:
                raise ValueError("DPO sample_id 不能为空")
            if isinstance(v, str):
                verdict = v.strip()
                item_reason = ""
            elif isinstance(v, dict):
                verdict = str(v.get("verdict") or "").strip()
                item_reason = str(v.get("reason") or "").strip()
            else:
                raise ValueError(f"DPO {sample_id} 裁决格式无效")
            if verdict not in VALID_DPO_VERDICTS:
                raise ValueError(f"非法 DPO verdict {verdict!r} (sample_id={sample_id})")
            out[sample_id] = {"verdict": verdict, "reason": item_reason}
        return out

    raise ValueError("DPO 裁决必须是 JSON 对象或列表")


def curate_dpo(
    verdicts_input: Any,
    *,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将 DPO 偏好对裁决 (good|swap|drop) 合并写入 00_state/dpo_verdicts.json。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    from pipeline import judge, stage3_synth  # noqa: PLC0415

    vpath = judge.dpo_verdicts_path(cfg)
    store = read_json(vpath, default={}) or {}
    if not isinstance(store, dict):
        store = {}

    dirs = stage_dirs(cfg)
    records = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    draft = dataset_draft.load_draft(cfg)
    assignments = (draft.get("split_plan") or {}).get("assignments") or {}
    cands, _ = stage3_synth.synthesize_dpo_candidates(records, cfg, assignments)
    known_cand_ids: set[str] = {
        str(c.get("sample_id") or "").strip()
        for c in cands
        if isinstance(c, dict) and str(c.get("sample_id") or "").strip()
    }
    for row in read_jsonl(dirs["gold"] / "dpo_pairs.jsonl"):
        if isinstance(row, dict) and str(row.get("sample_id") or "").strip():
            known_cand_ids.add(str(row["sample_id"]).strip())

    if isinstance(verdicts_input, str) and re.fullmatch(
        r"(all|pending)=(good|swap|drop)", verdicts_input.strip()
    ):
        scope_tok, target_verdict = verdicts_input.strip().split("=", 1)
        rcache = read_json(dirs["state"] / "dpo_review_cache.json", default={}) or {}
        normalized = {}
        for c in cands:
            sid_key = str(c.get("sample_id") or "").strip()
            if not sid_key:
                continue
            if scope_tok == "pending":
                existing_v = (store.get(sid_key) or {}).get("verdict")
                if existing_v in VALID_DPO_VERDICTS:
                    continue
                cached_hit = any(
                    isinstance(v, dict)
                    and v.get("sample_id") == sid_key
                    and v.get("verdict") in VALID_DPO_VERDICTS
                    for v in rcache.values()
                )
                if cached_hit:
                    continue
            normalized[sid_key] = {"verdict": target_verdict, "reason": reason}
    else:
        normalized = _normalize_dpo_verdicts(verdicts_input)
        if not normalized:
            raise ValueError("DPO 裁决不能为空")
        unknown_ids = sorted(set(normalized.keys()) - known_cand_ids)
        if unknown_ids:
            raise ValueError(
                f"未知 DPO sample_id（不在当前候选池中，共 {len(unknown_ids)} 条）: "
                f"{', '.join(unknown_ids[:5])}"
            )
    now = now_iso()
    sids: set[str] = set()
    for sample_id, item in normalized.items():
        store[sample_id] = {
            "verdict": item["verdict"],
            "reason": item["reason"] or reason,
            "by": actor,
            "at": now,
        }
        parts = sample_id.split("::")
        if len(parts) >= 3 and parts[1]:
            sids.add(parts[1])

    before_summary = ledger.corpus_snapshot_summary(cfg)
    write_json_atomic(vpath, store)
    after_summary = ledger.corpus_snapshot_summary(cfg)

    ev = ledger.record_operation(
        "curate.dpo",
        reason=reason,
        actor=actor,
        params={"count": len(normalized)},
        affected={"samples": sorted(normalized), "sids": sorted(sids)},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=cfg,
    )
    return {
        "ok": True,
        "updated": len(normalized),
        "total": len(store),
        "sample_ids": sorted(normalized),
        "op_id": ev["op_id"],
    }


def _parse_kv_assignments(items: Optional[Sequence[str] | Dict[str, str]]) -> Dict[str, str]:
    if not items:
        return {}
    if isinstance(items, dict):
        return {str(k).strip(): str(v).strip() for k, v in items.items() if str(k).strip()}
    out: Dict[str, str] = {}
    for raw in items:
        for part in str(raw or "").split(","):
            tok = part.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise ValueError(f"切分参数必须为 key=value 格式: {tok!r}")
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def curate_split(
    *,
    pin: Optional[Sequence[str] | Dict[str, str]] = None,
    session_pin: Optional[Sequence[str] | Dict[str, str]] = None,
    eval_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    replan: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """锁定项目切分 / 会话切分或重切分数据集 Split Plan。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    pool = set(raw_sessions) | set(gold_sessions)

    projects = sorted(
        {
            str(e.get("project_group") or "unknown")
            for e in list(gold_sessions.values()) + list(raw_sessions.values())
            if isinstance(e, dict)
        }
    )

    before_summary = ledger.corpus_snapshot_summary(cfg)
    plan = dataset_draft.resolve_split_plan(projects, cfg, persist=True)

    pin_map_raw = _parse_kv_assignments(pin)
    pin_updates: Dict[str, str] = {}
    known_projects = set((plan.get("assignments") or {}).keys())
    for proj, sp in pin_map_raw.items():
        if proj not in known_projects and f"game::{proj}" in known_projects:
            proj = f"game::{proj}"
        pin_updates[proj] = sp
    if pin_updates:
        plan = dataset_draft.set_project_splits(pin_updates, cfg)

    sess_map_raw = _parse_kv_assignments(session_pin)
    sess_updates: Dict[str, str] = {}
    for sref, sp in sess_map_raw.items():
        sid = inspect_tools._resolve_session_id(sref, pool) if pool else sref  # noqa: SLF001
        sess_updates[sid] = sp
    if sess_updates:
        plan = dataset_draft.set_session_splits(sess_updates, cfg)

    changes: Dict[str, Any] = {}
    if replan or eval_ratio is not None or seed is not None:
        scfg = cfg.get("synth") or {}
        eff_ratio = float(
            eval_ratio if eval_ratio is not None else (
                plan.get("eval_ratio") if plan.get("eval_ratio") is not None else scfg.get("eval_ratio", 0.25)
            )
        )
        eff_seed = int(
            seed if seed is not None else (
                plan.get("seed") if plan.get("seed") is not None else scfg.get("seed", 42)
            )
        )
        replan_res = dataset_draft.replan_split(
            cfg, eval_ratio=eff_ratio, seed=eff_seed, dry_run=False
        )
        plan = replan_res["split_plan"]
        changes = replan_res["changes"]

    after_summary = ledger.corpus_snapshot_summary(cfg)
    ev = ledger.record_operation(
        "curate.split",
        reason=reason,
        actor=actor,
        params={
            "pin": pin_updates,
            "session_pin": sess_updates,
            "eval_ratio": eval_ratio,
            "seed": seed,
            "replan": bool(replan),
        },
        affected={
            "projects": sorted(set(pin_updates) | set(changes)),
            "sids": sorted(sess_updates),
        },
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=cfg,
    )
    return {
        "ok": True,
        "split_plan": plan,
        "changes": changes,
        "op_id": ev["op_id"],
    }


def curate_exclude(
    *,
    samples: Optional[Sequence[str]] = None,
    scopes: Optional[Sequence[str]] = None,
    restore: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """停用或恢复样本（sample_id）或作用域（project:/session:/session_sft:/session_dpo:）。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    sample_list = _flatten_tokens(samples)
    scope_list = [str(s).strip() for s in (scopes or []) if str(s).strip()]
    if not sample_list and not scope_list:
        raise ValueError("必须指定至少一个 --sample 或 --scope")

    action = "auto" if restore else "exclude"
    if action == "exclude" and not str(reason or "").strip():
        raise ValueError("排除/停用必须填写 --reason")

    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})
    pool = set(raw_sessions) | set(gold_sessions)

    scope_updates: List[Dict[str, Any]] = []
    affected_sids: set[str] = set()
    affected_projects: set[str] = set()

    for raw_scope in scope_list:
        if ":" not in raw_scope:
            raise ValueError(f"scope 必须为 <kind>:<id> 格式: {raw_scope!r}")
        kind, ident = raw_scope.split(":", 1)
        kind = kind.strip()
        ident = ident.strip()
        if kind not in dataset_draft.SCOPE_KINDS:
            raise ValueError(f"非法 scope 类型: {kind!r}")
        if kind in ("session", "session_sft", "session_dpo"):
            if pool and ident not in pool:
                ident = inspect_tools._resolve_session_id(ident, pool)  # noqa: SLF001
            affected_sids.add(ident)
        elif kind == "project":
            affected_projects.add(ident)
        scope_updates.append({
            "scope": kind,
            "id": ident,
            "action": action,
            "reason": reason,
            "by": actor,
        })

    sample_updates: List[Dict[str, Any]] = []
    for sid_item in sample_list:
        sample_updates.append({
            "sample_id": sid_item,
            "action": action,
            "reason": reason,
            "by": actor,
        })
        parts = sid_item.split("::")
        if len(parts) >= 2 and parts[1] in pool:
            affected_sids.add(parts[1])

    before_summary = ledger.corpus_snapshot_summary(cfg)
    samples_res = dataset_draft.update_decisions(sample_updates, cfg) if sample_updates else {"changed": 0}
    scopes_res = dataset_draft.update_scope_decisions(scope_updates, cfg) if scope_updates else {"changed": 0}
    after_summary = ledger.corpus_snapshot_summary(cfg)

    ev = ledger.record_operation(
        "curate.exclude",
        reason=reason,
        actor=actor,
        params={"action": action, "samples": sample_list, "scopes": scope_list},
        affected={
            "samples": sample_list,
            "sids": sorted(affected_sids),
            "projects": sorted(affected_projects),
        },
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=cfg,
    )
    return {
        "ok": True,
        "action": action,
        "samples_changed": samples_res["changed"],
        "scopes_changed": scopes_res["changed"],
        "op_id": ev["op_id"],
    }
