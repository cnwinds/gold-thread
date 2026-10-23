"""Stage 3：从黄金会话合成训练数据（单步窗口 / 完整轨迹 / DPO / 诊断 / 切分 / 合成漏斗审计）。"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import curation, judge
from .common.config import config_hash, load_config, stage_dirs
from .common.format import conversation, to_openai, to_sharegpt
from .common.quality import compact_thinking, evaluate_effective_step
from .common.store import (
    dump_index,
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

STAGE = "stage3"


def _tools_json(flow_messages: List[Dict[str, Any]]) -> str:
    from .common.pilog import DEFAULT_TOOLS

    return json.dumps(DEFAULT_TOOLS, ensure_ascii=False)


def _pick_split(projects: List[str], cfg: Dict[str, Any]) -> Dict[str, str]:
    scfg = cfg["synth"]
    explicit = list(scfg.get("eval_projects") or [])
    ratio = float(scfg.get("eval_ratio", 0.0))
    unique = sorted(set(projects))
    if explicit:
        return {p: ("eval" if p in explicit else "train") for p in unique}
    if ratio <= 0 or len(unique) < 2:
        return {p: "train" for p in unique}
    n_eval = max(1, round(len(unique) * ratio))
    eval_set = set(unique[-n_eval:])
    return {p: ("eval" if p in eval_set else "train") for p in unique}


def _subsegment_ranges(sid: str, cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """检查会话是否命中 synth.sft_allow_subsegments 白名单子段配置。"""
    rules = cfg.get("synth", {}).get("sft_allow_subsegments") or []
    ranges: List[Tuple[int, int]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        pat = str(r.get("session_id") or "").strip()
        if not pat:
            continue
        if pat == sid or sid.startswith(pat) or pat in sid:
            s_step = int(r.get("start_step", 1))
            e_step = int(r.get("end_step", 999999))
            ranges.append((s_step, e_step))
    return ranges


def _build_anchored_history(
    msgs: List[Dict[str, Any]],
    idx: int,
    window_turns: int,
    sys_msg: Optional[Dict[str, Any]],
    root_msg: Optional[Dict[str, Any]],
    kept_fn,
) -> List[Dict[str, Any]]:
    history = [h for h in msgs[max(0, idx - window_turns) : idx] if kept_fn(h)]
    while history and history[0]["role"] == "tool":
        history = history[1:]

    anchored: List[Dict[str, Any]] = []
    if sys_msg and (not history or history[0]["role"] != "system"):
        anchored.append(sys_msg)
    if root_msg and not any(h.get("id") == root_msg.get("id") for h in history):
        anchored.append(root_msg)
    full_history = anchored + [h for h in history if h["role"] != "system"]
    if sys_msg and (not full_history or full_history[0].get("id") != sys_msg.get("id")):
        full_history.insert(0, sys_msg)

    cleaned: List[Dict[str, Any]] = []
    for h in full_history:
        if h["role"] == "tool" and (not cleaned or cleaned[-1]["role"] not in ("assistant", "tool")):
            continue
        cleaned.append(h)
    return cleaned


def _negative_dpo(
    msgs: List[Dict[str, Any]],
    sys_msg: Optional[Dict[str, Any]],
    root_msg: Optional[Dict[str, Any]],
    dcfg: Dict[str, Any],
    window_turns: int,
    sid: str,
    project: str,
    split: str,
    model: str,
) -> List[Dict[str, Any]]:
    """反例会话的病态步 -> DPO 对（chosen = 压缩修正后的 thinking）。"""
    pairs: List[Dict[str, Any]] = []
    min_chars = int(dcfg.get("min_thinking_chars", 80))
    for idx, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        raw_q = m.get("raw_quality") or m.get("quality") or {}
        bad = [
            p for p in raw_q.get("pathologies", [])
            if p in ("over_rumination_length", "over_rumination_doubt_loop", "mental_hex_checkmark")
        ]
        if not bad:
            continue
        original = m.get("raw_thinking") or m.get("thinking", "")
        if len(original) < min_chars:
            continue
        compacted = compact_thinking(original, 1500)
        if len(compacted) >= len(original) * 0.75:
            continue
        cleaned = _build_anchored_history(msgs, idx, window_turns, sys_msg, root_msg, lambda _: True)
        convs: List[Dict[str, str]] = []
        for h in cleaned:
            convs.extend(to_sharegpt(h, strip_thinking=True))
        if not convs:
            continue
        pairs.append(
            {
                "sample_id": f"dpo_neg::{sid}::{m['id']}",
                "conversations": convs,
                "chosen": to_sharegpt(m, override_thinking=compacted)[0],
                "rejected": to_sharegpt(m, override_thinking=original)[0],
                "metadata": {
                    "source": "negative_step",
                    "reason": ",".join(bad),
                    "session_id": sid,
                    "project_group": project,
                    "split": split,
                    "model": model,
                    "msg_id": m["id"],
                    "pathologies": bad,
                    "rejected_think_len": len(original),
                    "chosen_think_len": len(compacted),
                },
            }
        )
    return pairs


def _review_rewrite_dpo(
    msgs: List[Dict[str, Any]],
    rewrites: Dict[str, str],
    excluded: set[str],
    sys_msg: Optional[Dict[str, Any]],
    root_msg: Optional[Dict[str, Any]],
    window_turns: int,
    sid: str,
    project: str,
    split: str,
    model: str,
) -> List[Dict[str, Any]]:
    """黄金复核中采纳的模型重写/人工精修 -> 高质量思维链 DPO 偏好对。"""
    pairs: List[Dict[str, Any]] = []
    if not rewrites:
        return pairs
    for idx, m in enumerate(msgs):
        if m["role"] != "assistant" or m["id"] not in rewrites or m["id"] in excluded:
            continue
        raw_think = (m.get("raw_thinking") or m.get("thinking") or "").strip()
        new_think = (rewrites[m["id"]] or "").strip()
        if not raw_think or not new_think or raw_think == new_think:
            continue
        raw_q = m.get("raw_quality") or m.get("quality") or {}
        raw_pathos = raw_q.get("pathologies", [])
        # 当原文存在思维链病态，或重写显著压缩了冗余长度（至少短 20%）时构造偏好对
        if not raw_pathos and len(new_think) > len(raw_think) * 0.8:
            continue
        cleaned = _build_anchored_history(
            msgs, idx, window_turns, sys_msg, root_msg, lambda x: x.get("id") not in excluded
        )
        convs: List[Dict[str, str]] = []
        for h in cleaned:
            convs.extend(to_sharegpt(h, strip_thinking=True))
        if not convs:
            continue
        pairs.append(
            {
                "sample_id": f"dpo_review::{sid}::{m['id']}",
                "conversations": convs,
                "chosen": to_sharegpt(m, override_thinking=new_think)[0],
                "rejected": to_sharegpt(m, override_thinking=raw_think)[0],
                "metadata": {
                    "source": "review_rewrite",
                    "reason": ",".join(raw_pathos) if raw_pathos else "cot_conciseness_rewrite",
                    "session_id": sid,
                    "project_group": project,
                    "split": split,
                    "model": model,
                    "msg_id": m["id"],
                    "pathologies": raw_pathos,
                    "rejected_think_len": len(raw_think),
                    "chosen_think_len": len(new_think),
                },
            }
        )
    return pairs


def evaluate_session_synthesis(
    rec: Dict[str, Any],
    annotation: Dict[str, Any],
    cfg: Dict[str, Any],
    split: str = "train",
) -> Dict[str, Any]:
    """计算单条黄金会话在 Stage 3 的合成门控、每步去向审计与生成的窗口样本。"""
    scfg = cfg["synth"]
    window_turns = int(scfg.get("window_turns", 8))
    sft_labels = set(scfg.get("sft_labels", ["positive"]))
    weights = scfg.get("weights", {})
    max_think_chars = int(cfg["gold"].get("max_thinking_chars", 2400))
    max_think_target = int(cfg["gold"].get("target_thinking_chars", 1600))

    sid = rec["session_id"]
    label = rec.get("label", "pending")
    project = rec.get("project_group", "unknown")
    cwd = rec.get("cwd", "")
    model = rec.get("dominant_model") or "unknown"
    msgs: List[Dict[str, Any]] = rec.get("messages", [])
    sys_msg = next((m for m in msgs if m["role"] == "system"), None)
    root_msg = next((m for m in msgs if m["role"] == "user"), None)

    effective = curation.resolve(annotation, cfg)
    eff_state = effective["state"]
    rewrites: Dict[str, str] = effective["rewrites"]
    excluded = set(effective["excluded"])
    human_kept = set(effective.get("kept") or [])

    sub_ranges = _subsegment_ranges(sid, cfg)
    in_sft_label = label in sft_labels
    blocked_by_review = eff_state not in curation.IN_SFT
    # 模型质量报告否决：需要整条重写 / 判定丢弃 -> 不值得训练，直接报告不产样本
    mrep = rec.get("model_report") or {}
    model_block = bool(mrep.get("needs_wholesale_rewrite")) or (mrep.get("verdict") == "discard")

    # 完整轨迹仅在属于 sft_labels 且未被复核阻断时产出
    full_sft_allowed = in_sft_label and not blocked_by_review and not model_block
    # 单步窗口：允许完整 SFT 会话，或命中 sft_allow_subsegments 白名单子段且未被人工显式驳回的会话
    subsegment_only = (not full_sft_allowed) and bool(sub_ranges) and (eff_state != "rejected") and not model_block

    drop_calls = {
        tc.get("id")
        for m in msgs
        if m.get("role") == "assistant" and m.get("id") in excluded
        for tc in (m.get("tool_calls") or [])
        if tc.get("id")
    }

    def _kept_msg(m: Dict[str, Any]) -> bool:
        if m.get("role") == "assistant" and m.get("id") in excluded:
            return False
        if m.get("role") == "tool" and m.get("tool_call_id") in drop_calls:
            return False
        return True

    asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    total = len(asst_idx)

    step_audits: List[Dict[str, Any]] = []
    windows_sgpt: List[Dict[str, Any]] = []
    windows_oai: List[Dict[str, Any]] = []

    funnel = {
        "raw_steps": int(rec.get("stats", {}).get("raw_steps", total)),
        "gold_steps": total,
        "admitted_steps": 0,
        "rescued_steps": 0,
        "human_kept_steps": 0,
        "excluded_steps": 0,
        "pathology_dropped": 0,
        "synthesized_dropped": 0,
        "empty_dropped": 0,
        "blocked_or_subsegment_dropped": 0,
        "weighted_steps": 0,
    }

    tools_str = _tools_json(msgs)
    tools_obj = json.loads(tools_str)

    for rank, idx in enumerate(asst_idx):
        m = msgs[idx]
        mid = m["id"]
        step_no = rank + 1

        raw_think = m.get("raw_thinking") or m.get("thinking") or ""
        thinking = m.get("thinking") or ""
        is_rewritten = mid in rewrites
        if is_rewritten:
            thinking = rewrites[mid]
        elif len(thinking) > max_think_chars:
            thinking = compact_thinking(thinking, max_think_target)

        raw_q = m.get("raw_quality") or m.get("quality") or {}
        eff_q = evaluate_effective_step(m, cwd, override_thinking=thinking, max_think_chars=4500)

        hl_map = {h.get("msg_id"): h for h in (mrep.get("highlights") or [])}
        if mid in hl_map:
            stage_name, weight = "highlight", int(hl_map[mid].get("weight", 2))
        elif mid and mid == mrep.get("closing_msg_id"):
            stage_name, weight = "closing", int(weights.get("closing", 3))
        elif mid and mid == mrep.get("opening_msg_id"):
            stage_name, weight = "opening", int(weights.get("opening", 3))
        elif rank < 3:
            stage_name, weight = "opening", int(weights.get("opening", 3))
        elif rank >= max(3, total - 3):
            stage_name, weight = "closing", int(weights.get("closing", 3))
        elif len(m.get("tool_calls", [])) >= 2 or "script_verification" in eff_q.get("strengths", []) or any(
            tc["name"] in ("write", "edit") for tc in m.get("tool_calls", [])
        ):
            stage_name, weight = "key_decision", int(weights.get("key_decision", 2))
        else:
            stage_name, weight = "routine", int(weights.get("routine", 1))

        status = "dropped"
        reason = ""

        if not full_sft_allowed and not subsegment_only:
            reason = ("model_discard:" + str(mrep.get("verdict") or "needs_wholesale_rewrite")) if model_block else (
                f"session_blocked:{eff_state}" if in_sft_label else f"label_excluded:{label}")
            funnel["blocked_or_subsegment_dropped"] += 1
        elif subsegment_only and not any(s <= step_no <= e for s, e in sub_ranges):
            reason = "outside_subsegment_whitelist"
            funnel["blocked_or_subsegment_dropped"] += 1
        elif mid in excluded:
            reason = "excluded_by_review"
            funnel["excluded_steps"] += 1
        elif not (thinking or "").strip():
            reason = "empty_thinking"
            funnel["empty_dropped"] += 1
        elif "thinking_synthesized" in (m.get("flags") or []) and not is_rewritten and mid not in human_kept:
            reason = "synthesized_thinking_unverified"
            funnel["synthesized_dropped"] += 1
        elif mid in human_kept:
            status = "admitted"
            reason = "rescued_by_human_edit" if (is_rewritten and not raw_q.get("is_clean", True)) else "human_kept"
            funnel["human_kept_steps"] += 1
            if not raw_q.get("is_clean", True):
                funnel["rescued_steps"] += 1
        elif eff_q.get("is_clean"):
            if subsegment_only and len(thinking.strip()) < 80:
                reason = "subsegment_thinking_too_short"
                funnel["empty_dropped"] += 1
            else:
                status = "admitted"
                if eff_q.get("rescued") or (is_rewritten and not raw_q.get("is_clean", True)):
                    reason = "rescued_by_rewrite"
                    funnel["rescued_steps"] += 1
                elif subsegment_only:
                    reason = "whitelisted_subsegment"
                elif is_rewritten:
                    reason = "clean_rewritten"
                else:
                    reason = "clean_gold"
        else:
            reason = "pathology:" + ",".join(eff_q.get("pathologies", ["unknown"]))
            funnel["pathology_dropped"] += 1

        if status == "admitted":
            cleaned = _build_anchored_history(msgs, idx, window_turns, sys_msg, root_msg, _kept_msg)
            if not cleaned:
                status = "dropped"
                reason = "empty_context_history"
                funnel["empty_dropped"] += 1
            else:
                funnel["admitted_steps"] += 1
                funnel["weighted_steps"] += weight
                step_id = f"step::{sid}::{mid}"
                prompt_sgpt: List[Dict[str, str]] = []
                for h in cleaned:
                    prompt_sgpt.extend(to_sharegpt(h, strip_thinking=True))
                prompt_oai = [to_openai(h, strip_thinking=True) for h in cleaned]

                step_meta = {
                    "sample_id": step_id,
                    "session_id": sid,
                    "project_group": project,
                    "split": split,
                    "model": model,
                    "root_task": (root_msg or {}).get("content", "")[:160],
                    "msg_id": mid,
                    "step_rank": step_no,
                    "total_steps": total,
                    "stage": stage_name,
                    "sample_weight": weight,
                    "admit_reason": reason,
                    "rewritten": is_rewritten,
                    "quality_score": eff_q.get("score"),
                    "raw_quality_score": raw_q.get("score"),
                    "resolved_pathologies": eff_q.get("resolved_pathologies", []),
                    "strengths": eff_q.get("strengths", []),
                }
                target_sgpt = to_sharegpt(m, override_thinking=thinking)[0]
                windows_sgpt.append(
                    {
                        "sample_id": step_id,
                        "conversations": prompt_sgpt + [target_sgpt],
                        "tools": tools_str,
                        "metadata": step_meta,
                    }
                )
                windows_oai.append(
                    {
                        "sample_id": step_id,
                        "messages": prompt_oai + [to_openai(m, override_thinking=thinking)],
                        "tools": tools_obj,
                        "metadata": step_meta,
                    }
                )

        step_audits.append(
            {
                "step_rank": step_no,
                "msg_id": mid,
                "stage": stage_name,
                "weight": weight,
                "status": status,
                "reason": reason,
                "rewritten": is_rewritten,
                "flags": m.get("flags") or [],
                "raw_score": raw_q.get("score", 100.0),
                "eff_score": eff_q.get("score", 100.0),
                "raw_pathologies": raw_q.get("pathologies", []),
                "eff_pathologies": eff_q.get("pathologies", []),
                "resolved_pathologies": eff_q.get("resolved_pathologies", []),
                "think_len_raw": len(raw_think),
                "think_len_eff": len(thinking),
                "tools": [tc.get("name", "") for tc in m.get("tool_calls", [])],
                "gist": (thinking or m.get("text") or "").strip().split("\n")[0][:120],
            }
        )

    gold_traj: Optional[Dict[str, Any]] = None
    if full_sft_allowed:
        kept_msgs = [m for m in msgs if _kept_msg(m)]
        kept_asst_count = sum(1 for m in kept_msgs if m.get("role") == "assistant")
        if kept_asst_count > 0:
            gold_traj = {
                "sample_id": f"traj::{sid}",
                "conversations": conversation(kept_msgs, "sharegpt", strip_thinking=False, override=rewrites),
                "tools": tools_str,
                "metadata": {
                    "sample_id": f"traj::{sid}",
                    "session_id": sid,
                    "project_group": project,
                    "split": split,
                    "label": label,
                    "model": model,
                    "assistant_steps": kept_asst_count,
                    "rewritten_steps": len(rewrites),
                    "excluded_steps": len(excluded),
                },
            }

    return {
        "session_id": sid,
        "label": label,
        "project_group": project,
        "split": split,
        "model": model,
        "effective_state": eff_state,
        "model_report": mrep or None,
        "model_block": model_block,
        "full_sft_allowed": full_sft_allowed,
        "subsegment_only": subsegment_only,
        "subsegment_ranges": sub_ranges,
        "funnel": funnel,
        "step_audits": step_audits,
        "gold_traj": gold_traj,
        "windows_sgpt": windows_sgpt,
        "windows_oai": windows_oai,
    }


def preview_session_synthesis(sid: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """供前端实时预览某条黄金会话在 Stage 3 中的入库漏斗与每步命运。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    gold_index = load_index(dirs["gold"] / "index.json")
    entry = (gold_index.get("sessions") or {}).get(sid)
    if not entry:
        raise KeyError(f"黄金会话不存在: {sid}")
    rows = read_jsonl(Path(entry["gold_file"]))
    if not rows:
        raise RuntimeError(f"黄金文件为空: {entry['gold_file']}")
    rec = rows[0]
    ann = read_json(Path(entry.get("annotation_file", "")), default={}) or {}
    split_map = _pick_split([r.get("project_group", "unknown") for r in gold_index["sessions"].values()], cfg)
    split = split_map.get(entry.get("project_group", "unknown"), "train")
    res = evaluate_session_synthesis(rec, ann, cfg, split=split)
    return {
        "session_id": sid,
        "effective_state": res["effective_state"],
        "full_sft_allowed": res["full_sft_allowed"],
        "subsegment_only": res["subsegment_only"],
        "subsegment_ranges": res["subsegment_ranges"],
        "funnel": res["funnel"],
        "step_audits": res["step_audits"],
    }


def run(force: bool = False, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    gold_index = load_index(dirs["gold"] / "index.json")
    if not gold_index["sessions"]:
        raise RuntimeError("stage2 未产出数据，请先运行 stage2")

    train_root = dirs["train"]
    train_root.mkdir(parents=True, exist_ok=True)
    scfg = cfg["synth"]
    dcfg = scfg.get("dpo", {})
    fcfg = scfg.get("formats", {"sharegpt": True, "openai": True})
    window_turns = int(scfg.get("window_turns", 8))

    records = gold_index["sessions"]
    split_map = _pick_split([r.get("project_group", "unknown") for r in records.values()], cfg)

    gold_trajs: List[Dict[str, Any]] = []
    step_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    dpo_pairs: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    session_funnels: Dict[str, Any] = {}

    # 1) stage2 产出的天然 DPO 对
    if dcfg.get("from_pruned_steps", True):
        for pair in read_jsonl(dirs["gold"] / "dpo_pairs.jsonl"):
            pair["metadata"]["split"] = split_map.get(pair["metadata"].get("project_group"), "train")
            dpo_pairs.append(pair)

    stage_counts = Counter()
    review_counts = Counter()
    project_counts = Counter()
    model_counts = Counter()
    admit_reason_counts = Counter()
    drop_reason_counts = Counter()

    for sid, entry in sorted(records.items()):
        label = entry.get("label", "pending")
        project = entry.get("project_group", "unknown")
        split = split_map.get(project, "train")
        gold_file = Path(entry["gold_file"])
        rows = read_jsonl(gold_file)
        if not rows:
            continue
        rec = rows[0]
        msgs: List[Dict[str, Any]] = rec["messages"]
        model = rec.get("dominant_model") or entry.get("dominant_model") or "unknown"
        sys_msg = next((m for m in msgs if m["role"] == "system"), None)
        root_msg = next((m for m in msgs if m["role"] == "user"), None)

        annotation = read_json(Path(entry.get("annotation_file", "")), default={}) or {}
        eval_res = evaluate_session_synthesis(rec, annotation, cfg, split=split)
        eff_state = eval_res["effective_state"]
        review_counts[eff_state] += 1

        # 诊断库（全部标签）
        sess_diags = read_jsonl(Path(entry.get("diag_file", ""))) if entry.get("diag_file") else []
        for diag in sess_diags:
            diag["split"] = split
            diagnostics.append(diag)

        # DPO 来源 2：黄金复核改写对（正例/待定会话中模型重写或人工精修）
        effective_obj = curation.resolve(annotation, cfg)
        sess_review_dpos: List[Dict[str, Any]] = []
        if dcfg.get("from_review_rewrites", True) and effective_obj["rewrites"]:
            sess_review_dpos = _review_rewrite_dpo(
                msgs,
                effective_obj["rewrites"],
                set(effective_obj["excluded"]),
                sys_msg,
                root_msg,
                window_turns,
                sid,
                project,
                split,
                model,
            )
            dpo_pairs.extend(sess_review_dpos)

        # DPO 来源 3：反例会话病态步启发式压缩对
        sess_neg_dpos: List[Dict[str, Any]] = []
        if label == "negative" and dcfg.get("from_negative_steps", True):
            sess_neg_dpos = _negative_dpo(
                msgs, sys_msg, root_msg, dcfg, window_turns, sid, project, split, model
            )
            dpo_pairs.extend(sess_neg_dpos)

        for sa in eval_res["step_audits"]:
            if sa["status"] == "admitted":
                admit_reason_counts[sa["reason"]] += 1
            else:
                drop_reason_counts[sa["reason"].split(":")[0]] += 1

        session_funnels[sid] = {
            "session_id": sid,
            "label": label,
            "project_group": project,
            "model": model,
            "split": split,
            "effective_state": eff_state,
            "full_sft_allowed": eval_res["full_sft_allowed"],
            "subsegment_only": eval_res["subsegment_only"],
            "funnel": eval_res["funnel"],
            "dpo_pruned": int(entry.get("dpo_count", 0)),
            "dpo_review": len(sess_review_dpos),
            "dpo_negative": len(sess_neg_dpos),
            "diagnostics": len(sess_diags),
        }

        if not eval_res["full_sft_allowed"] and not eval_res["subsegment_only"]:
            if label in set(scfg.get("sft_labels", ["positive"])):
                log(f"[stage3] {curation.EFFECTIVE_ZH.get(eff_state, eff_state)}，不进 SFT: {sid}")
            continue

        if eval_res["gold_traj"]:
            gold_trajs.append(eval_res["gold_traj"])
            model_counts[model] += 1

        for w_sgpt, w_oai in zip(eval_res["windows_sgpt"], eval_res["windows_oai"]):
            st_name = w_sgpt["metadata"]["stage"]
            stage_counts[st_name] += 1
            project_counts[project] += 1
            step_pairs.append((w_sgpt, w_oai))

        log(
            f"[stage3] {sid[:8]} ({label:<8} {project:<18}) -> "
            f"黄金 {eval_res['funnel']['gold_steps']} 步, 入选窗口 {eval_res['funnel']['admitted_steps']} "
            f"(改写挽救 {eval_res['funnel']['rescued_steps']}, 排除 {eval_res['funnel']['excluded_steps']})"
        )

    # ---- DPO 偏好对：模型复核（多模型生成的对里，风格差异/长短偏好不算正误）----
    dpo_review_stats = {"checked": 0, "swapped": 0, "dropped": 0}
    if dpo_pairs and judge.enabled(cfg):
        verdicts = judge.review_dpo(dpo_pairs, cfg=cfg, log=log)
        dpo_review_stats["checked"] = len(dpo_pairs)
        kept_pairs = []
        for p in dpo_pairs:
            v = verdicts.get(p.get("sample_id")) or {}
            kind = v.get("verdict")
            if kind == "drop":
                dpo_review_stats["dropped"] += 1
                continue
            if kind == "swap":
                p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
                p.setdefault("metadata", {})["model_review"] = "swap"
                dpo_review_stats["swapped"] += 1
            if kind:
                p.setdefault("metadata", {})["model_review"] = kind
                p["metadata"]["model_review_reason"] = v.get("reason", "")
            kept_pairs.append(p)
        dpo_pairs = kept_pairs
        log(
            f"[judge] DPO 复核 {dpo_review_stats['checked']} 对：调换 {dpo_review_stats['swapped']}，"
            f"丢弃 {dpo_review_stats['dropped']}，保留 {len(dpo_pairs)}"
        )

    # 全局打乱（跨项目混训，且保证 ShareGPT 与 OpenAI 格式严格同序）
    if scfg.get("shuffle", True):
        rng = random.Random(int(scfg.get("seed", 42)))
        rng.shuffle(gold_trajs)
        rng.shuffle(step_pairs)

    sgpt = [p[0] for p in step_pairs] if fcfg.get("sharegpt", True) else []
    oai = [p[1] for p in step_pairs] if fcfg.get("openai", True) else []

    out = {
        "sft_gold_trajectories_sharegpt.jsonl": gold_trajs if fcfg.get("sharegpt", True) else [],
        "sft_step_windows_sharegpt.jsonl": sgpt,
        "sft_step_windows_openai.jsonl": oai,
        "dpo_preference_pairs.jsonl": dpo_pairs,
        "negative_diagnostics.jsonl": diagnostics,
    }
    counts = {}
    for name, rows in out.items():
        counts[name] = write_jsonl_atomic(train_root / name, rows)

    # 物理切分：train / eval 各写一份，训练框架可直接读取，无需自行过滤
    split_out: Dict[str, int] = {}
    for name, rows in out.items():
        if not rows:
            continue
        by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "eval": []}
        for r in rows:
            sp = (r.get("metadata") or {}).get("split", "train")
            by_split.setdefault(sp, []).append(r)
        for sp, part in by_split.items():
            if not part:
                continue
            fname = f"{name[:-6]}.{sp}.jsonl"
            split_out[fname] = write_jsonl_atomic(train_root / fname, part)
    counts.update(split_out)

    dataset_info = {
        "pi_sft_gold_trajectories": {
            "file_name": "sft_gold_trajectories_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "pi_sft_step_windows": {
            "file_name": "sft_step_windows_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
                "observation_tag": "observation",
            },
        },
        "pi_sft_step_windows_openai": {
            "file_name": "sft_step_windows_openai.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "observation_tag": "tool",
            },
        },
        "pi_dpo_preference_pairs": {
            "file_name": "dpo_preference_pairs.jsonl",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
                "observation_tag": "observation",
            },
        },
    }
    # 物理切分版本（train/eval 已分开，建议训练时直接用这两个）
    _SGPT_TAGS = {
        "role_tag": "from", "content_tag": "value", "user_tag": "human",
        "assistant_tag": "gpt", "system_tag": "system", "observation_tag": "observation",
    }
    for base in ("sft_step_windows_sharegpt", "sft_gold_trajectories_sharegpt"):
        for sp in ("train", "eval"):
            fname = f"{base}.{sp}.jsonl"
            if (train_root / fname).exists():
                key = f"pi_{base.replace('_sharegpt', '')}_{sp}"
                dataset_info[key] = {
                    "file_name": fname,
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "tools": "tools"},
                    "tags": dict(_SGPT_TAGS),
                }
    for sp in ("train", "eval"):
        fname = f"dpo_preference_pairs.{sp}.jsonl"
        if (train_root / fname).exists():
            dataset_info[f"pi_dpo_preference_pairs_{sp}"] = {
                "file_name": fname,
                "ranking": True,
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "chosen": "chosen",
                            "rejected": "rejected", "tools": "tools"},
                "tags": dict(_SGPT_TAGS),
            }
    write_json_atomic(train_root / "dataset_info.json", dataset_info)

    stats = {
        "generated_at": now_iso(),
        "config_hash": config_hash(cfg),
        "counts": counts,
        "split": split_map,
        "by_project": dict(project_counts),
        "by_stage": dict(stage_counts),
        "by_model": dict(model_counts),
        "admit_reasons": dict(admit_reason_counts),
        "drop_reasons": dict(drop_reason_counts),
        "session_funnels": session_funnels,
        "weighted_steps": sum(int(p[0]["metadata"].get("sample_weight", 1)) for p in step_pairs),
        "dpo_by_source": dict(Counter(p["metadata"].get("source", "?") for p in dpo_pairs)),
        "diagnostics_by_pathology": dict(
            Counter(p.split(":")[0] for d in diagnostics for p in d.get("pathologies", []))
        ),
        "label_counts": dict(Counter(r.get("label", "?") for r in records.values())),
        "review_states": dict(review_counts),
        "dpo_review": dpo_review_stats,
    }
    write_json_atomic(train_root / "stats.json", stats)

    dump_index(
        train_root / "index.json",
        {
            "_files": counts,
        },
        {"stage": STAGE, "stats": stats, "last_run": now_iso()},
    )

    log(
        f"[stage3] 完成：黄金轨迹 {counts.get('sft_gold_trajectories_sharegpt.jsonl', 0)}，"
        f"单步窗口 {counts.get('sft_step_windows_sharegpt.jsonl', 0)}，"
        f"DPO {counts.get('dpo_preference_pairs.jsonl', 0)}，诊断 {counts.get('negative_diagnostics.jsonl', 0)}"
    )
    return {"stage": STAGE, **counts, "split": split_map, "stats": stats}
