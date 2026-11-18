"""显式可选的小模型初筛子命令（gt assist turns/judge/dpo）。

绝不在主构建链路暗箱触发，仅在显式调用时临时开启 LLM 执行初筛建议，可选 --apply 落盘并记入账本。
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline import judge, stage1_extract, stage2_gold, stage3_synth, turnclassify
from pipeline.common import pilog
from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import load_index, now_iso, read_json, write_json_atomic

from . import inspect_tools, ledger


def _offline_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg or load_config())
    out.setdefault("turns", {})["use_model"] = False
    out.setdefault("judge", {})["enabled"] = False
    return out


def assist_turns(
    sid_or_prefix: str,
    *,
    apply: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """显式调用 turnclassify.classify_session 输出回合分类建议；--apply 时写入 meta.json 并记账。"""
    t0 = time.monotonic()
    base_cfg = copy.deepcopy(cfg or load_config())
    online_cfg = copy.deepcopy(base_cfg)
    online_cfg.setdefault("turns", {})["use_model"] = True

    dirs = stage_dirs(online_cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    sid = inspect_tools._resolve_session_id(sid_or_prefix, raw_sessions)  # noqa: SLF001
    entry = raw_sessions[sid]
    raw_file = Path(entry["raw_file"])
    meta_file = Path(entry["meta_file"])
    max_tool_chars = int((online_cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
    session = pilog.parse_session(raw_file, max_tool_output_chars=max_tool_chars)

    annotations = turnclassify.classify_session(session, cfg=online_cfg, log=lambda *_: None)
    op_id: Optional[str] = None
    applied = False
    if apply and annotations is not None:
        before_summary = ledger.corpus_snapshot_summary(base_cfg)
        meta = read_json(meta_file, default={}) or {}
        meta["turn_annotations"] = annotations
        write_json_atomic(meta_file, meta)
        stage1_extract.update_turn_overrides(sid, cfg=_offline_cfg(base_cfg))
        after_summary = ledger.corpus_snapshot_summary(base_cfg)
        ev = ledger.record_operation(
            "assist.turns",
            reason=reason or "应用模型回合分类初筛",
            actor=actor,
            params={"sid": sid, "apply": True},
            affected={"sids": [sid], "projects": [entry.get("project_group") or ""]},
            before_summary=before_summary,
            after_summary=after_summary,
            duration_ms=int((time.monotonic() - t0) * 1000),
            cfg=base_cfg,
        )
        op_id = ev["op_id"]
        applied = True

    return {
        "ok": True,
        "session_id": sid,
        "applied": applied,
        "turn_annotations": annotations,
        "op_id": op_id,
    }


def assist_judge(
    sid_or_prefix: str,
    *,
    apply: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """显式调用 judge.decide_steps + judge.audit_quality 输出步级与质量建议；--apply 时落入 step_decisions.json。"""
    t0 = time.monotonic()
    base_cfg = copy.deepcopy(cfg or load_config())
    online_cfg = copy.deepcopy(base_cfg)
    online_cfg.setdefault("judge", {})["enabled"] = True
    offline_cfg = _offline_cfg(base_cfg)

    dirs = stage_dirs(online_cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    sid = inspect_tools._resolve_session_id(sid_or_prefix, raw_sessions)  # noqa: SLF001
    entry = raw_sessions[sid]
    raw_file = Path(entry["raw_file"])
    meta_file = Path(entry["meta_file"])
    raw_meta = read_json(meta_file, default={}) or {}
    max_tool_chars = int((online_cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
    session = pilog.parse_session(raw_file, max_tool_output_chars=max_tool_chars)
    flow, _marks, _fstats = pilog.build_flow(
        session,
        offline_cfg,
        raw_meta.get("turn_annotations"),
        raw_meta.get("manual_turn_overrides"),
    )
    asst_total = sum(1 for m in flow if m.get("role") == "assistant")

    jmeta: Dict[str, Any] = {}
    decisions = judge.decide_steps(flow, cfg=online_cfg, log=lambda *_: None, out_meta=jmeta)
    quality_report = judge.audit_quality(
        flow,
        {"raw_steps": asst_total, "gold_steps": asst_total},
        cfg=online_cfg,
        log=lambda *_: None,
    )

    op_id: Optional[str] = None
    applied = False
    stage2_res: Optional[Dict[str, Any]] = None
    if apply:
        before_summary = ledger.corpus_snapshot_summary(offline_cfg)
        sd_path = stage2_gold.step_decisions_path(offline_cfg)
        store = stage2_gold.load_step_decisions(offline_cfg)
        cur_entry = copy.deepcopy(store.get(sid) or {})
        merged_steps = dict(cur_entry.get("steps") or {})
        for mid, dec in (decisions or {}).items():
            if isinstance(dec, dict):
                merged_steps[str(mid)] = dict(dec)
        store[sid] = {
            "updated_at": now_iso(),
            "by": actor,
            "reason": reason or "应用模型步级初筛",
            "model_report": quality_report or cur_entry.get("model_report"),
            "steps": merged_steps,
            "seam_patches": dict(cur_entry.get("seam_patches") or {}),
        }
        write_json_atomic(sd_path, store)
        stage2_res = stage2_gold.run(
            only=[sid],
            force=True,
            cfg=offline_cfg,
            log=lambda *_: None,
        )
        after_summary = ledger.corpus_snapshot_summary(offline_cfg)
        ev = ledger.record_operation(
            "assist.judge",
            reason=reason or "应用模型步级初筛",
            actor=actor,
            params={"sid": sid, "apply": True, "decisions": len(decisions)},
            affected={"sids": [sid], "projects": [entry.get("project_group") or ""]},
            before_summary=before_summary,
            after_summary=after_summary,
            duration_ms=int((time.monotonic() - t0) * 1000),
            cfg=offline_cfg,
        )
        op_id = ev["op_id"]
        applied = True

    return {
        "ok": True,
        "session_id": sid,
        "applied": applied,
        "decisions": decisions,
        "model_report": quality_report,
        "meta": jmeta,
        "stage2": stage2_res,
        "op_id": op_id,
    }


def assist_dpo(
    *,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """显式开启 judge.enabled=True 对未审 DPO 偏好对跑小模型初筛缓存并记入 ledger。"""
    t0 = time.monotonic()
    online_cfg = copy.deepcopy(cfg or load_config())
    online_cfg.setdefault("judge", {})["enabled"] = True

    before_summary = ledger.corpus_snapshot_summary(online_cfg)
    res = stage3_synth.evaluate_dpo_pending(cfg=online_cfg, log=lambda *_: None)
    after_summary = ledger.corpus_snapshot_summary(online_cfg)

    ev = ledger.record_operation(
        "assist.dpo",
        reason=reason or "模型 DPO 初筛评估",
        actor=actor,
        params={
            "candidates": res.get("candidates", 0),
            "evaluated": res.get("evaluated", 0),
        },
        affected={},
        before_summary=before_summary,
        after_summary=after_summary,
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=online_cfg,
    )
    return {
        "ok": True,
        **res,
        "op_id": ev["op_id"],
    }
