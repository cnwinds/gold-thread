"""模型参与层：黄金会话裁决 + 长 CoT 重写（输出标注，不直接改数据）。

拆成两类调用，避免单次输出过长：
1) judge  ：整条会话给一个存废结论 + 病态步 id（小输出）
2) rewrite：分批重写超长 thinking（每批 N 步，直接基于未经截断的 raw_thinking 重写）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import curation
from .common.config import load_config, stage_dirs
from .common.llm import chat_json
from .common.store import load_index, read_json, read_jsonl, write_json_atomic

PROMPT_VERSION = "review.v3"

JUDGE_SYSTEM = """你是工程分析 Agent 训练数据的审核专家。
给你一条已被脚本提纯的黄金轨迹（含工具调用摘要），判断它是否值得作为 SFT 正样本。
重点找：心算十六进制并打勾、脑补源码、反复自我推翻、跨目录偷看、盲调工具无思考、假阴性死胡同、明显未完成就收尾。
严格只输出 JSON。"""

JUDGE_SCHEMA = """输出 JSON：
{"verdict": "gold|needs_review|reject", "confidence": 0.0-1.0,
 "reasons": ["简短中文理由"], "bad_step_ids": ["消息 id"]}"""

REWRITE_SYSTEM = """你是 Agent 思维链提纯器。把冗长、反复自我推翻的分析 thinking 重写为紧凑版本：
- 保留：观测到的事实/字节/工具输出 -> 结构假设 -> 下一步验证动作；
- 删除：自我否定循环（Wait/Actually/Let me reconsider）、默写的大段代码、重复复述任务；
- 语言不变：输出语言必须与原文一致（英文原文就输出英文，中文原文就输出中文），严禁翻译；
- 保留段落结构：沿用原文的空行分段与列表格式（如 "- " 短横线项），一个主题一段；禁止把多个主题压成单一密块；
- 目标 300~1200 字，保持第一人称推理语气，不要输出任何解释或标题。
严格只输出 JSON。"""

REWRITE_SCHEMA = """输出 JSON：
{"rewrites": [{"msg_id": "原消息 id", "rewritten": "重写后的 thinking"}]}"""

REFINE_REWRITE_SYSTEM = REWRITE_SYSTEM.replace("严格只输出 JSON。", "") + """这是复核后的修订轮：人工审阅了你上一版改写（previous_rewrite），并给出意见。
- 以 original_thinking 为唯一事实来源，不得引入原文没有的偏移、数值、路径、结论；
- 必须落实 reviewer_comment 与 reviewer_note；被拒绝（reviewer_rejected_previous=true）说明上一版有实质问题，不要小修小补；
- reviewer_edited_previous=true 时 previous_rewrite 是人工改过的版本，在它的基础上吸收意见，不要退回更早的表述；
- 若意见与原文事实冲突，以原文为准，并在 reply 里说明。
严格只输出 JSON。"""

REFINE_REWRITE_SCHEMA = """输出 JSON：
{"rewrites": [{"msg_id": "原消息 id", "rewritten": "修订后的 thinking", "reply": "一句话说明按意见改了什么"}]}"""

OPTIMIZE_SYSTEM = """你是工程分析 Agent 训练数据的全面优化器。对给出的每一步 thinking 做生产级优化，目标是改完即可直接进入训练集：
- 保留：观测到的事实/字节/工具输出 -> 结构假设 -> 下一步验证动作，三段式结构完整；
- 修复：心算十六进制并打勾（改写为"用脚本实测验证"的表述）、脑补源码、反复自我推翻（Wait/Actually/Hold on）、与任务无关的复述；
- 语言不变：与原文语言一致，严禁翻译；
- 段落不变：沿用原文的空行分段与列表格式，一个主题一段；
- 事实红线：偏移量、密钥、字节数、路径、版本号、结论必须与原文一致，严禁引入原文没有的信息，严禁丢失关键数据；
- 若该步已足够好，输出轻修版本即可；但输入的每一条都必须返回。
严格只输出 JSON。"""

OPTIMIZE_SCHEMA = """输出 JSON：
{"rewrites": [{"msg_id": "原消息 id", "rewritten": "优化后的完整 thinking"}]}
要求：输入的每一条 msg_id 都必须在 rewrites 中出现。"""

AUDIT_SYSTEM = """你是独立的工程分析训练数据审计员，与优化器不是同一个模型。逐条审计优化后的 thinking 是否达到"可直接进入训练集"的标准：
- 事实保真：偏移/字节/密钥/路径/版本/结论是否与原版一致；是否引入原文没有的信息；是否丢失关键数据；
- 语言与结构：语言是否与原版一致；段落/列表结构是否保留；
- 质量达标：自我推翻循环是否清除；是否仍有心算打勾；是否保持第一人称；是否短到丢失实质内容。
逐条给出问题；全部达标时 passed=true 且 issues 为空。
严格只输出 JSON。"""

AUDIT_SCHEMA = """输出 JSON：
{"passed": true 或 false,
 "overall": "一句话总评",
 "issues": [{"msg_id": "原消息 id", "problem": "问题简述", "instruction": "给优化器的修改指令"}]}"""

OPTIMIZE_REVISE_SYSTEM = """你是工程分析 Agent 训练数据的全面优化器（修订轮）。独立审计员否决了部分条目并给出逐条修改指令：
- 以 original_thinking 为唯一事实来源，逐条落实 audit_instruction；
- 语言与段落结构与原文一致；严禁引入原文没有的信息；
- 每一条都必须返回修订后的完整 thinking。
严格只输出 JSON。"""

REFINE_JUDGE_SYSTEM = JUDGE_SYSTEM.replace("严格只输出 JSON。", "") + """这是复核轮：附上了你上一轮的结论（previous_review）和人工复核意见（reviewer_feedback）。
认真对待人工意见：意见成立就修正结论与 bad_step_ids；不成立要在 reply 里给出依据，不要无原则附和。
严格只输出 JSON。"""

REFINE_JUDGE_SCHEMA = """输出 JSON：
{"verdict": "gold|needs_review|reject", "confidence": 0.0-1.0,
 "reasons": ["简短中文理由"], "bad_step_ids": ["消息 id"], "reply": "对人工意见的简短回应"}"""


def _orig_thinking(m: Dict[str, Any]) -> str:
    return (m.get("raw_thinking") or m.get("thinking") or "").strip()


def _step(m: Dict[str, Any], max_think: int, max_result: int = 800) -> Dict[str, Any]:
    orig = _orig_thinking(m)
    raw_q = m.get("raw_quality") or m.get("quality") or {}
    return {
        "msg_id": m.get("id"),
        "thinking": orig[:max_think],
        "thinking_len": len(orig),
        "text": (m.get("text") or "")[:400],
        "tools": [
            {"name": tc.get("name"), "args": json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:400]}
            for tc in m.get("tool_calls", [])
        ],
        "results": [
            {"name": tr.get("name"), "isError": tr.get("isError"), "out": (tr.get("content") or "")[:max_result]}
            for tr in m.get("tool_results", [])
        ],
        "pathologies": raw_q.get("pathologies", []),
    }


def review_session(
    entry: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    log=print,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    rcfg = cfg.get("review", {})
    rows = read_jsonl(Path(entry["gold_file"]))
    if not rows:
        raise RuntimeError(f"黄金文件为空: {entry.get('gold_file')}")
    rec = rows[0]
    msgs = rec["messages"]
    asst = [m for m in msgs if m["role"] == "assistant"]

    judge_max = int(rcfg.get("judge_max_think_chars", 3000))
    sample = _sample_steps(asst, int(rcfg.get("max_steps_per_session", 40)))

    payload = {
        "session": {
            "session_id": rec.get("session_id"),
            "project_group": rec.get("project_group"),
            "label": rec.get("label"),
            "model": rec.get("dominant_model"),
            "root_prompt": rec.get("stats", {}).get("root_prompt", ""),
            "total_steps": len(asst),
            "gold_stats": rec.get("stats", {}),
        },
        "steps": [_step(m, judge_max) for m in sample],
    }
    log(f"[review] {entry.get('session_id')} 评审 {len(sample)}/{len(asst)} 步")
    verdict = chat_json(
        JUDGE_SYSTEM,
        JUDGE_SCHEMA + "\n\n待评审数据：\n" + json.dumps(payload, ensure_ascii=False),
        temperature=float(rcfg.get("temperature", 0.0)),
        prompt_version=PROMPT_VERSION + ".judge",
    )
    if not isinstance(verdict, dict):
        raise RuntimeError("评审返回结构异常")

    valid_ids = [m["id"] for m in asst]
    bad = [b for b in (verdict.get("bad_step_ids") or []) if b in valid_ids]

    # 分批重写长 CoT（基于 raw_thinking，避免脚本预截断造成信息丢失）
    threshold = int(rcfg.get("compress_thinking_over", 1200))
    candidates = [
        m for m in asst
        if len(_orig_thinking(m)) > threshold
        or "thinking_synthesized" in (m.get("flags") or [])
    ]
    batch_size = int(rcfg.get("rewrite_batch_size", 4))
    max_batches = int(rcfg.get("max_rewrite_batches", 6))
    rewrites: List[Dict[str, Any]] = []
    skipped_batches = 0
    for bi in range(0, min(len(candidates), batch_size * max_batches), batch_size):
        batch = candidates[bi : bi + batch_size]
        body = {
            "root_prompt": rec.get("stats", {}).get("root_prompt", ""),
            "steps": [{"msg_id": m["id"], "thinking": _orig_thinking(m)[:8000]} for m in batch],
        }
        try:
            out = chat_json(
                REWRITE_SYSTEM,
                REWRITE_SCHEMA + "\n\n待重写：\n" + json.dumps(body, ensure_ascii=False),
                temperature=float(rcfg.get("temperature", 0.0)),
                prompt_version=PROMPT_VERSION + ".rewrite",
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[review] 改写批次失败: {exc}")
            skipped_batches += 1
            continue
        for rw in (out or {}).get("rewrites", []):
            if not isinstance(rw, dict):
                continue
            mid = rw.get("msg_id")
            text = (rw.get("rewritten") or "").strip()
            if mid in valid_ids and text:
                rewrites.append({"msg_id": mid, "rewritten": text})

    confidence = float(verdict.get("confidence") or 0.0)
    return {
        "kind": "initial",
        "verdict": _normalize_verdict(verdict.get("verdict"), confidence, rcfg),
        "confidence": confidence,
        "reasons": verdict.get("reasons", []),
        "bad_step_ids": bad,
        "thinking_rewrites": rewrites,
        "reviewed_steps": len(sample),
        "total_steps": len(asst),
        "skipped_batches": skipped_batches,
        "model": cfg["llm"].get("model"),
        "prompt_version": PROMPT_VERSION,
    }


def _normalize_verdict(status: Any, confidence: float, rcfg: Dict[str, Any]) -> str:
    if status not in ("gold", "needs_review", "reject"):
        return "needs_review"
    if status == "gold" and confidence < float(rcfg.get("judge_confidence_min", 0.7)):
        return "needs_review"
    return status


def _sample_steps(asst: List[Dict[str, Any]], max_steps: int) -> List[Dict[str, Any]]:
    if len(asst) <= max_steps:
        return asst
    step_ids = {round(i * (len(asst) - 1) / (max_steps - 1)) for i in range(max_steps)}
    return [asst[i] for i in sorted(step_ids)]


def refine_session(
    entry: Dict[str, Any],
    ann: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    log=print,
) -> Dict[str, Any]:
    """按人工意见做定向修订：只重写被批注/被拒的步骤，并结合意见重新给会话结论。"""
    cfg = cfg or load_config()
    rcfg = cfg.get("review", {})
    rows = read_jsonl(Path(entry["gold_file"]))
    if not rows:
        raise RuntimeError(f"黄金文件为空: {entry.get('gold_file')}")
    rec = rows[0]
    asst = [m for m in rec["messages"] if m["role"] == "assistant"]
    by_id = {m["id"]: m for m in asst}

    prev = curation.latest_round(ann)
    if not prev:
        raise RuntimeError("该会话还没有模型评审，无法按意见修订")
    hr = ann.get("human_review") or {}
    note = (hr.get("note") or "").strip()
    targets = {mid: it for mid, it in curation.feedback_targets(ann).items() if mid in by_id}
    if not targets and not note:
        raise RuntimeError("没有可供模型修订的意见：请先批注步骤、拒绝改写或填写整体意见")

    prev_rw = {rw["msg_id"]: rw["rewritten"] for rw in (prev.get("thinking_rewrites") or [])}
    rw_targets = [mid for mid in targets if _orig_thinking(by_id[mid])]
    log(f"[refine] {entry.get('session_id')} 第 {prev.get('round')} 轮 -> 修订 {len(rw_targets)} 步，整体意见 {'有' if note else '无'}")

    new_rw: Dict[str, str] = {}
    replies: Dict[str, str] = {}
    batch_size = int(rcfg.get("rewrite_batch_size", 4))
    for bi in range(0, len(rw_targets), batch_size):
        body = {
            "root_prompt": rec.get("stats", {}).get("root_prompt", ""),
            "reviewer_note": note,
            "steps": [
                {
                    "msg_id": mid,
                    "original_thinking": _orig_thinking(by_id[mid])[:8000],
                    "previous_rewrite": (
                        (targets[mid].get("edited_text") or "").strip()
                        if targets[mid].get("rewrite") == "edited"
                        else prev_rw.get(mid)
                    ),
                    "reviewer_rejected_previous": targets[mid].get("rewrite") == "rejected",
                    "reviewer_edited_previous": targets[mid].get("rewrite") == "edited",
                    "reviewer_comment": (targets[mid].get("comment") or "").strip(),
                }
                for mid in rw_targets[bi : bi + batch_size]
            ],
        }
        out = chat_json(
            REFINE_REWRITE_SYSTEM,
            REFINE_REWRITE_SCHEMA + "\n\n待修订：\n" + json.dumps(body, ensure_ascii=False),
            temperature=float(rcfg.get("temperature", 0.0)),
            prompt_version=PROMPT_VERSION + ".refine_rewrite",
        )
        for rw in (out or {}).get("rewrites", []):
            if not isinstance(rw, dict) or rw.get("msg_id") not in by_id:
                continue
            mid, text = rw["msg_id"], (rw.get("rewritten") or "").strip()
            if text:
                new_rw[mid] = text
            if (rw.get("reply") or "").strip():
                replies[mid] = rw["reply"].strip()

    sample = _sample_steps(asst, int(rcfg.get("max_steps_per_session", 40)))
    judge_max = int(rcfg.get("judge_max_think_chars", 3000))
    payload = {
        "session": {
            "session_id": rec.get("session_id"),
            "project_group": rec.get("project_group"),
            "label": rec.get("label"),
            "root_prompt": rec.get("stats", {}).get("root_prompt", ""),
            "total_steps": len(asst),
        },
        "previous_review": {
            "verdict": prev.get("verdict"),
            "confidence": prev.get("confidence"),
            "reasons": prev.get("reasons", []),
            "bad_step_ids": prev.get("bad_step_ids", []),
        },
        "reviewer_feedback": {
            "note": note,
            "steps": [
                {
                    "msg_id": mid,
                    "comment": (it.get("comment") or "").strip(),
                    "rewrite_rejected": it.get("rewrite") == "rejected",
                    "step_decision": it.get("step"),
                    "thinking_head": _orig_thinking(by_id[mid])[:600],
                }
                for mid, it in targets.items()
            ],
        },
        "steps": [_step(m, judge_max) for m in sample],
    }
    verdict = chat_json(
        REFINE_JUDGE_SYSTEM,
        REFINE_JUDGE_SCHEMA + "\n\n待复审数据：\n" + json.dumps(payload, ensure_ascii=False),
        temperature=float(rcfg.get("temperature", 0.0)),
        prompt_version=PROMPT_VERSION + ".refine_judge",
    )
    if not isinstance(verdict, dict):
        raise RuntimeError("复审返回结构异常")
    confidence = float(verdict.get("confidence") or 0.0)
    bad = [b for b in (verdict.get("bad_step_ids") or []) if b in by_id]

    merged = [
        {"msg_id": rw["msg_id"], "rewritten": new_rw.get(rw["msg_id"], rw["rewritten"])}
        for rw in (prev.get("thinking_rewrites") or [])
    ]
    merged += [{"msg_id": mid, "rewritten": t} for mid, t in new_rw.items() if mid not in prev_rw]
    return {
        "kind": "refine",
        "verdict": _normalize_verdict(verdict.get("verdict"), confidence, rcfg),
        "confidence": confidence,
        "reasons": verdict.get("reasons", []),
        "bad_step_ids": bad,
        "thinking_rewrites": merged,
        "changed_msg_ids": sorted(new_rw),
        "replies": replies,
        "judge_reply": (verdict.get("reply") or "").strip(),
        "feedback": {
            "note": note,
            "items": {mid: {k: v for k, v in it.items() if k in ("comment", "rewrite", "step")} for mid, it in targets.items()},
        },
        "reviewed_steps": len(sample),
        "total_steps": len(asst),
        "model": cfg["llm"].get("model"),
        "prompt_version": PROMPT_VERSION,
    }


def run_review(
    session_ids: Optional[List[str]] = None,
    force: bool = False,
    log=print,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    if not cfg.get("review", {}).get("enabled", False):
        log("[review] 注意：review.enabled=false（仍会执行，但不参与后续流程的话需自行开启）")
    gold_dir = stage_dirs(cfg)["gold"]
    index = load_index(gold_dir / "index.json")
    labels = set(cfg.get("review", {}).get("labels", ["positive", "pending"]))
    targets = session_ids or [sid for sid, e in index["sessions"].items() if e.get("label") in labels]
    reviewed = skipped = failed = 0
    for sid in targets:
        entry = index["sessions"].get(sid)
        if not entry:
            log(f"[review] 会话不存在: {sid}")
            continue
        ann_path = Path(entry["annotation_file"])
        existing = read_json(ann_path, default={}) or {}
        if existing.get("model_review") and not force:
            skipped += 1
            continue
        try:
            rnd = review_session(entry, cfg=cfg, log=log)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"[review] 失败 {sid}: {exc}")
            continue
        ann = read_json(ann_path, default={}) or {}
        curation.ensure_rounds(ann)
        stale_ids = [mid for mid, it in curation.human(ann)["items"].items() if it.get("rewrite")]
        curation.append_round(ann, rnd, reset_ids=stale_ids)
        write_json_atomic(ann_path, ann)
        reviewed += 1
        log(
            f"[review] {sid[:8]} -> {rnd['verdict']} (conf {rnd['confidence']:.2f}, "
            f"改写 {len(rnd['thinking_rewrites'])} 处)"
        )
    return {"reviewed": reviewed, "skipped": skipped, "failed": failed}


def run_refine(session_id: str, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    index = load_index(stage_dirs(cfg)["gold"] / "index.json")
    entry = index["sessions"].get(session_id)
    if not entry:
        raise RuntimeError(f"会话不存在: {session_id}")
    ann_path = Path(entry["annotation_file"])
    rnd = refine_session(entry, read_json(ann_path, default={}) or {}, cfg=cfg, log=log)
    ann = read_json(ann_path, default={}) or {}
    curation.ensure_rounds(ann)
    curation.append_round(ann, rnd, reset_ids=list(rnd["feedback"]["items"]))
    write_json_atomic(ann_path, ann)
    log(
        f"[refine] {session_id[:8]} -> 第 {ann['review_rounds'][-1]['round']} 轮 {rnd['verdict']} "
        f"(conf {rnd['confidence']:.2f}，重写 {len(rnd['changed_msg_ids'])} 步)"
    )
    return {"round": ann["review_rounds"][-1]['round'], "verdict": rnd["verdict"], "changed": rnd["changed_msg_ids"]}
