"""人工复核层：模型评审轮次 + 人工逐条处置 + 最终结论，并给 stage3 计算「有效结论」。

标注文件（02_gold/annotations/<sid>.json）里与复核相关的字段：

- model_review   ：最新一轮模型评审（兼容旧逻辑）
- review_rounds  ：所有模型轮次，只追加。round / kind(initial|refine) / verdict / thinking_rewrites /
                   feedback（本轮收到的人工意见）/ changed_msg_ids / replies / judge_reply
- human_review   ：人工工作区与结论
    items     {msg_id: {rewrite: accepted|rejected|edited|None, edited_text, step: exclude|keep|None, comment}}
    note      给模型的整体意见
    decision  approved|rejected|None；review_key / round 记录结论针对的数据版本与轮次
    history   结论变更记录
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common.store import now_iso

REWRITE_STATES = ("accepted", "rejected", "edited")
STEP_STATES = ("exclude", "keep")
DECISIONS = ("approved", "rejected")

EFFECTIVE_ZH = {
    "approved": "人工通过",
    "rejected": "人工驳回",
    "auto_gold": "模型通过",
    "auto_reject": "模型剔除",
    "held": "待人工复核",
    "stale": "结论过期待确认",
    "unreviewed": "未评审",
    "finalized": "定稿",
}
IN_SFT = {"approved", "auto_gold", "unreviewed", "finalized"}


def ensure_rounds(ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    """旧标注只有 model_review：补成第 1 轮。"""
    rounds = ann.get("review_rounds")
    if isinstance(rounds, list) and rounds:
        return rounds
    mr = ann.get("model_review")
    if not mr:
        ann["review_rounds"] = []
        return ann["review_rounds"]
    ann["review_rounds"] = [{"round": 1, "kind": "initial", "at": ann.get("reviewed_at"), **mr}]
    return ann["review_rounds"]


def latest_round(ann: Dict[str, Any]) -> Dict[str, Any]:
    rounds = ensure_rounds(ann)
    return rounds[-1] if rounds else {}


def human(ann: Dict[str, Any]) -> Dict[str, Any]:
    hr = ann.get("human_review")
    if not isinstance(hr, dict):
        hr = {}
        ann["human_review"] = hr
    hr.setdefault("items", {})
    hr.setdefault("note", "")
    hr.setdefault("decision", None)
    hr.setdefault("finalized", False)
    hr.setdefault("history", [])
    return hr


def is_stale(ann: Dict[str, Any]) -> bool:
    hr = ann.get("human_review") or {}
    if not hr.get("decision"):
        return False
    saved_key = hr.get("review_key")
    cur_key = ann.get("review_key")
    if not saved_key or not cur_key:
        return False
    # 比较源文件指纹 + 回合流签名，算法版本升级本身不误废除人工结论
    return ":".join(str(saved_key).split(":")[:3]) != ":".join(str(cur_key).split(":")[:3])


def resolve(ann: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """给出会话的有效结论、生效的 thinking 改写、排除的步骤与人工强制保留的步骤。"""
    rcfg = cfg.get("review", {})
    hold = bool(rcfg.get("hold_needs_review", True))
    conf_min = float(rcfg.get("judge_confidence_min", 0.7))
    rnd = {} if ann.get("review_rounds_stale") else latest_round(ann)
    hr = ann.get("human_review") or {}
    items: Dict[str, Dict[str, Any]] = hr.get("items") or {}
    bad = set(rnd.get("bad_step_ids") or [])
    model_rw = {rw["msg_id"]: rw["rewritten"] for rw in (rnd.get("thinking_rewrites") or []) if rw.get("msg_id")}

    def _rewrites(default_accept: bool) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for mid, text in model_rw.items():
            st = (items.get(mid) or {}).get("rewrite")
            if st == "rejected":
                continue
            if st == "edited" and (items[mid].get("edited_text") or "").strip():
                out[mid] = items[mid]["edited_text"].strip()
            elif st == "accepted" or (st is None and default_accept):
                out[mid] = text
        for mid, it in items.items():
            if mid not in model_rw and it.get("rewrite") == "edited" and (it.get("edited_text") or "").strip():
                out[mid] = it["edited_text"].strip()
        return out

    def _excluded() -> List[str]:
        ex = {mid for mid, it in items.items() if it.get("step") == "exclude"}
        ex |= {mid for mid in bad if (items.get(mid) or {}).get("step") != "keep"}
        return sorted(ex)

    def _kept() -> List[str]:
        kp = {
            mid
            for mid, it in items.items()
            if it.get("step") == "keep" or (it.get("rewrite") == "edited" and (it.get("edited_text") or "").strip())
        }
        return sorted(kp)

    decision = hr.get("decision")
    fresh = bool(decision) and not is_stale(ann)
    if fresh and decision == "approved" and hr.get("finalized"):
        return {"state": "finalized", "rewrites": _rewrites(True), "excluded": _excluded(), "kept": _kept()}
    if ann.get("review_rounds_stale") and not fresh:
        return {"state": "stale", "rewrites": {}, "excluded": [], "kept": []}
    if fresh:
        state = "approved" if decision == "approved" else "rejected"
        if state == "rejected":
            return {"state": "rejected", "rewrites": {}, "excluded": _excluded(), "kept": []}
        if ann.get("review_rounds_stale") or hr.get("ignore_model_review"):
            # 人工在新版本上直接通过：仅应用人工手改与人工排除项，不套用旧模型建议
            manual_only_rw = {
                mid: it["edited_text"].strip()
                for mid, it in items.items()
                if it.get("rewrite") == "edited" and (it.get("edited_text") or "").strip()
            }
            manual_ex = sorted(mid for mid, it in items.items() if it.get("step") == "exclude")
            return {"state": state, "rewrites": manual_only_rw, "excluded": manual_ex, "kept": _kept()}
        return {"state": state, "rewrites": _rewrites(True), "excluded": _excluded(), "kept": _kept()}
    if decision:
        return {"state": "stale", "rewrites": {}, "excluded": [], "kept": []}
    if not rnd:
        # 尚未跑 review.py 时，优先使用 stage2 judge 产出的 model_report 作为模型结论
        rep = ann.get("model_report") or {}
        rep_verdict = rep.get("verdict")
        if rep.get("needs_wholesale_rewrite") or rep_verdict == "discard":
            return {"state": "auto_reject", "rewrites": {}, "excluded": _excluded(), "kept": []}
        if rep_verdict == "needs_human":
            return {"state": "held" if hold else "auto_gold", "rewrites": _rewrites(False), "excluded": _excluded(), "kept": _kept()}
        if rep_verdict == "trainable" and float(rep.get("confidence") or 0) >= conf_min:
            return {"state": "auto_gold", "rewrites": _rewrites(True), "excluded": _excluded(), "kept": _kept()}
        # 即使尚未进行模型评审，人工手改或排除的单步也立即生效
        return {"state": "unreviewed", "rewrites": _rewrites(False), "excluded": _excluded(), "kept": _kept()}
    verdict = rnd.get("verdict")
    if verdict == "reject":
        return {"state": "auto_reject", "rewrites": {}, "excluded": _excluded(), "kept": []}
    if verdict == "gold" and float(rnd.get("confidence") or 0) >= conf_min:
        return {"state": "auto_gold", "rewrites": _rewrites(True), "excluded": _excluded(), "kept": _kept()}
    if hold:
        return {"state": "held", "rewrites": _rewrites(False), "excluded": _excluded(), "kept": _kept()}
    return {"state": "auto_gold", "rewrites": _rewrites(True), "excluded": _excluded(), "kept": _kept()}


def progress(ann: Dict[str, Any]) -> Dict[str, int]:
    """当前轮次里人工处置进度。"""
    rnd = latest_round(ann)
    items = (ann.get("human_review") or {}).get("items") or {}
    rw_ids = [rw["msg_id"] for rw in (rnd.get("thinking_rewrites") or []) if rw.get("msg_id")]
    bad = rnd.get("bad_step_ids") or []
    return {
        "rewrites": len(rw_ids),
        "rewrites_done": sum(1 for m in rw_ids if (items.get(m) or {}).get("rewrite")),
        "rewrites_accepted": sum(1 for m in rw_ids if (items.get(m) or {}).get("rewrite") == "accepted"),
        "rewrites_rejected": sum(1 for m in rw_ids if (items.get(m) or {}).get("rewrite") == "rejected"),
        "rewrites_edited": sum(1 for m in rw_ids if (items.get(m) or {}).get("rewrite") == "edited"),
        "bad_steps": len(bad),
        "bad_done": sum(1 for m in bad if (items.get(m) or {}).get("step")),
        "manual_edits": sum(1 for m, it in items.items() if m not in rw_ids and it.get("rewrite") == "edited"),
        "manual_excludes": sum(1 for m, it in items.items() if m not in bad and it.get("step") == "exclude"),
        "comments": sum(1 for it in items.values() if (it.get("comment") or "").strip()),
    }


def update_items(ann: Dict[str, Any], patch: Dict[str, Dict[str, Any]], note: Optional[str]) -> None:
    hr = human(ann)
    for mid, p in (patch or {}).items():
        cur = dict(hr["items"].get(mid) or {})
        for k, v in (p or {}).items():
            if k == "rewrite" and v not in (*REWRITE_STATES, None):
                raise ValueError(f"rewrite 状态非法: {v}")
            if k == "step" and v not in (*STEP_STATES, None):
                raise ValueError(f"step 状态非法: {v}")
            if k not in ("rewrite", "edited_text", "step", "comment"):
                continue
            cur[k] = v
        cur = {k: v for k, v in cur.items() if v not in (None, "")}
        if cur:
            hr["items"][mid] = cur
        else:
            hr["items"].pop(mid, None)
    if note is not None:
        hr["note"] = note
    hr["updated_at"] = now_iso()


def decide(ann: Dict[str, Any], decision: Optional[str], note: str = "") -> None:
    if decision not in (*DECISIONS, None):
        raise ValueError(f"decision 非法: {decision}")
    hr = human(ann)
    rnd = latest_round(ann)
    was_stale = bool(ann.get("review_rounds_stale") or is_stale(ann))
    if decision == "approved" and rnd and not was_stale:
        for rw in rnd.get("thinking_rewrites") or []:
            it = hr["items"].setdefault(rw["msg_id"], {})
            it.setdefault("rewrite", "accepted")
        for mid in rnd.get("bad_step_ids") or []:
            hr["items"].setdefault(mid, {}).setdefault("step", "exclude")
    hr["decision"] = decision
    hr["decided_at"] = now_iso() if decision else None
    hr["review_key"] = ann.get("review_key") if decision else None
    hr["round"] = rnd.get("round") if (decision and rnd) else None
    hr["ignore_model_review"] = bool(decision == "approved" and was_stale)
    if decision:
        ann.pop("review_rounds_stale", None)
    hr["history"].append({"decision": decision or "reopened", "at": now_iso(), "round": rnd.get("round"), "note": note})


def finalize(ann: Dict[str, Any], note: str = "") -> None:
    """定稿 = 通过 + 未处置改写全部默认采纳 + 锁定状态。"""
    decide(ann, "approved", note)
    hr = human(ann)
    hr["finalized"] = True
    hr["finalized_at"] = now_iso()
    hr["history"].append({"decision": "finalized", "at": now_iso(),
                          "round": (latest_round(ann) or {}).get("round"), "note": note})


def unfinalize(ann: Dict[str, Any], note: str = "") -> None:
    hr = human(ann)
    hr["finalized"] = False
    hr["finalized_at"] = None
    hr["history"].append({"decision": "unfinalized", "at": now_iso(),
                          "round": (latest_round(ann) or {}).get("round"), "note": note})


def feedback_targets(ann: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """带意见重审时需要模型重新处理的条目：有批注，或改写被拒。"""
    items = (ann.get("human_review") or {}).get("items") or {}
    return {
        mid: it for mid, it in items.items()
        if (it.get("comment") or "").strip() or it.get("rewrite") == "rejected"
    }


def append_round(ann: Dict[str, Any], rnd: Dict[str, Any], reset_ids: Optional[List[str]] = None) -> None:
    """追加一轮模型结果；人工结论作废，被重做的条目清空处置与批注（批注已记入本轮 feedback）。"""
    rounds = ensure_rounds(ann)
    rnd = {"round": len(rounds) + 1, "at": now_iso(), **rnd}
    rounds.append(rnd)
    ann.pop("review_rounds_stale", None)
    ann["model_review"] = {k: v for k, v in rnd.items() if k not in ("round", "at", "kind", "feedback", "replies", "judge_reply", "changed_msg_ids")}
    ann["reviewed_at"] = rnd["at"]
    hr = human(ann)
    for mid in reset_ids or []:
        it = {k: v for k, v in (hr["items"].get(mid) or {}).items() if k == "step"}
        if it:
            hr["items"][mid] = it
        else:
            hr["items"].pop(mid, None)
    if rnd.get("kind") == "refine" and (hr.get("note") or "").strip() == ((rnd.get("feedback") or {}).get("note") or "").strip():
        hr["note"] = ""
    if hr.get("finalized"):
        hr["finalized"] = False
        hr["finalized_at"] = None
        hr["history"].append({"decision": "unfinalized", "at": rnd["at"], "round": rnd["round"], "note": "模型新一轮结果，定稿自动解除"})
    if hr.get("decision"):
        hr["history"].append({"decision": "reopened", "at": rnd["at"], "round": rnd["round"], "note": "模型新一轮评审，原结论作废"})
    hr["decision"] = None
    hr["decided_at"] = None
    hr["review_key"] = None
    hr["round"] = None
    hr["ignore_model_review"] = False
