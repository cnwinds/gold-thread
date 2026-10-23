"""Stage 2：原始日志 -> 黄金会话（状态折叠 / 弯路剪枝 / CoT 去噪 / 静默执行）+ 天然 DPO 对。

核心特性：
1. 完整保留提纯前后血缘快照（`raw_thinking`, `raw_text`, `fold_details`, `pruned_details`），支持前端三态 Diff 审计；
2. 双态质量评估：同时记录原始质量（`raw_quality`，用于诊断库）与提纯后质量（`quality`）；
3. 增量指纹同时校验 `raw_file` 与 `meta_file`，保证 Stage 1 回合调整后 Stage 2 自动感知。
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import judge
from .common import pilog
from .common.config import load_config, section_hash, stage_dirs
from .common.format import to_sharegpt
from .common.quality import (
    LOW_LEVEL_MISTAKE_OUTPUT_REGEX,
    NEXT_STEP_OOPS_REGEX,
    analyze_step_quality,
    compact_thinking,
    evaluate_effective_step,
)
from .common.store import (
    dump_index,
    fingerprint,
    fingerprint_equal,
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    safe_name,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)

STAGE = "stage2"

_POLLING_RE = re.compile(
    r"(nohup\s+[^\n&]+>\s*(/tmp/[^\s;&]+)\s+2>&1\s*&)\s*\n?\s*echo\s+started;\s*sleep\s+(\d+);\s*tail\s+(-\d+)\s+\2"
)


def normalize_polling_command(cmd: str) -> str:
    def _repl(m: re.Match) -> str:
        bg, log_file, wait_sec, tail_arg = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        loops = max(5, wait_sec // 2)
        return (
            f"{bg} BG_PID=$!; echo started pid=$BG_PID; "
            f"for _i in $(seq 1 {loops}); do kill -0 $BG_PID 2>/dev/null || break; sleep 2; done; "
            f"tail {tail_arg} {log_file}"
        )

    return _POLLING_RE.sub(_repl, cmd)


def _units(flow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for m in flow:
        if m["role"] in ("system", "user"):
            units.append({"type": m["role"], "msg": copy.deepcopy(m), "tools": []})
        elif m["role"] == "assistant":
            units.append({"type": "assistant_turn", "msg": copy.deepcopy(m), "tools": []})
        elif m["role"] == "tool" and units and units[-1]["type"] == "assistant_turn":
            units[-1]["tools"].append(copy.deepcopy(m))
        elif m["role"] == "tool":
            units.append({"type": "orphan_tool", "msg": copy.deepcopy(m), "tools": []})
    return units


def purify(
    flow: List[Dict[str, Any]],
    gcfg: Dict[str, Any],
    session_cwd: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """返回 (gold_messages, pruned_records, stats)。pruned_records 用于审计与 DPO。"""
    units = _units(flow)
    pruned: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    folded = 0
    dropped = 0
    think_compressed = 0
    think_backfilled = 0
    think_synthesized = 0
    silent_cleared = 0
    polling_normalized = 0

    i = 0
    while i < len(units):
        cur = units[i]
        if cur["type"] != "assistant_turn":
            kept.append(cur)
            i += 1
            continue

        msg = cur["msg"]
        msg.setdefault("raw_thinking", msg.get("thinking") or "")
        msg.setdefault("raw_text", msg.get("text") or "")
        tcs = msg.get("tool_calls", [])
        trs = cur["tools"]

        if i + 1 < len(units) and units[i + 1]["type"] == "assistant_turn":
            nxt = units[i + 1]
            nxt_msg = nxt["msg"]
            nxt_msg.setdefault("raw_thinking", nxt_msg.get("thinking") or "")
            nxt_msg.setdefault("raw_text", nxt_msg.get("text") or "")
            nxt_tcs = nxt_msg.get("tool_calls", [])

            # A. 状态折叠：write(path) + edit(path) 精确合并
            if (
                gcfg.get("state_folding", True)
                and len(tcs) == 1
                and len(nxt_tcs) == 1
                and tcs[0]["name"] == "write"
                and nxt_tcs[0]["name"] == "edit"
                and tcs[0].get("arguments", {}).get("path") == nxt_tcs[0].get("arguments", {}).get("path")
            ):
                orig_content = tcs[0]["arguments"].get("content", "")
                content = orig_content
                edits = nxt_tcs[0]["arguments"].get("edits", [])
                if not edits and "oldText" in nxt_tcs[0]["arguments"]:
                    edits = [nxt_tcs[0]["arguments"]]
                edits = edits[: int(gcfg.get("max_fold_edits", 8))]
                ok = bool(edits)
                for ed in edits:
                    old_t = ed.get("oldText", "")
                    new_t = ed.get("newText", "")
                    if old_t and old_t in content:
                        content = content.replace(old_t, new_t, 1)
                    else:
                        ok = False
                        break
                if ok:
                    original = copy.deepcopy(msg)
                    tcs[0]["arguments"]["content"] = content
                    msg["folded_from"] = nxt_msg["id"]
                    msg["fold_details"] = {
                        "path": tcs[0]["arguments"].get("path", ""),
                        "folded_msg_id": nxt_msg["id"],
                        "edits_count": len(edits),
                        "before_content": orig_content[:4000],
                        "after_content": content[:4000],
                        "edit_thinking": (nxt_msg.get("thinking") or "")[:800],
                    }
                    msg["flags"] = sorted(set(msg.get("flags", []) + ["state_folded"]))
                    pruned.append(
                        {
                            "reason": "state_folding_bug_fix",
                            "has_dpo": True,
                            "context_units": copy.deepcopy(kept),
                            "chosen_msg": copy.deepcopy(msg),
                            "rejected_msg": original,
                            "dropped_msg_id": nxt_msg["id"],
                            "file_path": tcs[0]["arguments"].get("path", ""),
                            "before_snippet": orig_content[:2000],
                            "after_snippet": content[:2000],
                        }
                    )
                    kept.append(cur)
                    folded += 1
                    i += 2
                    continue

            # B. 弯路剪枝：低级命令错误 / 下一步自认手滑
            if gcfg.get("detour_pruning", True):
                out_text = "\n".join(t.get("content", "") for t in trs)
                has_error = any(t.get("isError") for t in trs) or bool(
                    LOW_LEVEL_MISTAKE_OUTPUT_REGEX.search(out_text)
                )
                next_admits = bool(NEXT_STEP_OOPS_REGEX.search((nxt_msg.get("thinking") or "").strip()))
                if (has_error or next_admits) and nxt_tcs:
                    cleaned = copy.deepcopy(nxt)
                    raw_nxt_think = (cleaned["msg"].get("thinking") or "").strip()
                    cleaned_think = re.sub(
                        r"^(Oops[^.!?\n]*[.!?\n]+|It didn't recognize because[^.!?\n]*[.!?\n]+|"
                        r"`zip` not available[.!?\n]*)",
                        "",
                        raw_nxt_think,
                        flags=re.IGNORECASE,
                    ).strip()
                    if not cleaned_think and msg.get("thinking"):
                        cleaned_think = compact_thinking(msg["thinking"], 1200)
                    cleaned["msg"]["raw_thinking"] = cleaned["msg"].get("raw_thinking") or raw_nxt_think or (msg.get("thinking") or "")
                    cleaned["msg"]["thinking"] = cleaned_think
                    cleaned["msg"]["pruned_detour"] = msg["id"]
                    cleaned["msg"]["pruned_details"] = {
                        "dropped_msg_id": msg["id"],
                        "trigger": "next_step_oops" if next_admits and not has_error else "tool_error",
                        "rejected_thinking": (msg.get("thinking") or "")[:1200],
                        "rejected_tools": [
                            {"name": tc.get("name"), "args": json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:300]}
                            for tc in tcs
                        ],
                        "error_output": out_text[:800],
                    }
                    cleaned["msg"]["flags"] = sorted(
                        set(cleaned["msg"].get("flags", []) + ["detour_pruned"])
                    )
                    pruned.append(
                        {
                            "reason": "detour_pruned_error_or_oops",
                            "has_dpo": True,
                            "context_units": copy.deepcopy(kept),
                            "chosen_msg": copy.deepcopy(cleaned["msg"]),
                            "rejected_msg": copy.deepcopy(msg),
                            "dropped_msg_id": msg["id"],
                            "error_output": out_text[:1000],
                        }
                    )
                    units[i + 1] = cleaned
                    dropped += 1
                    i += 1
                    continue

        # C. 空 thinking：报错剪除并记录审计；成功则用对外播报文本内化为 thinking 或合成意图
        if not msg.get("thinking") and tcs:
            if any(t.get("isError") for t in trs):
                out_text = "\n".join(t.get("content", "") for t in trs)
                pruned.append(
                    {
                        "reason": "empty_thinking_error_dropped",
                        "has_dpo": False,
                        "context_units": [],
                        "chosen_msg": None,
                        "rejected_msg": copy.deepcopy(msg),
                        "dropped_msg_id": msg["id"],
                        "error_output": out_text[:1000],
                    }
                )
                dropped += 1
                i += 1
                continue
            if msg.get("text"):
                msg["thinking"] = msg["text"].strip()
                msg["text"] = ""
                msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_backfilled"]))
                think_backfilled += 1
            elif gcfg.get("fix_missing_thinking", True):
                parts = []
                for tc in tcs:
                    args = tc.get("arguments", {}) if isinstance(tc.get("arguments"), dict) else {}
                    name = tc.get("name", "tool")
                    if name == "bash" and args.get("command"):
                        parts.append(f"执行命令：{str(args['command'])[:160]}")
                    elif name in ("write", "edit") and args.get("path"):
                        parts.append(f"{'写入' if name == 'write' else '修改'}文件 {args['path']}")
                    elif name == "read" and args.get("path"):
                        parts.append(f"读取文件 {args['path']}")
                    else:
                        parts.append(f"调用 {name}")
                msg["thinking"] = "；".join(parts) + "。"
                msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_synthesized"]))
                think_synthesized += 1

        # D. CoT 去噪（保留 raw_thinking 供模型重写与人工对照）
        max_chars = int(gcfg.get("max_thinking_chars", 2400))
        target = int(gcfg.get("target_thinking_chars", 1600))
        if len(msg.get("thinking", "")) > max_chars:
            msg["raw_thinking"] = msg.get("raw_thinking") or msg["thinking"]
            msg["thinking"] = compact_thinking(msg["thinking"], target)
            msg["script_thinking"] = msg["thinking"]
            msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_compressed"]))
            think_compressed += 1

        # E. 静默执行：中间工具步清空对外文本
        if gcfg.get("silent_intermediate_steps", True) and tcs and msg.get("text"):
            msg["raw_text"] = msg.get("raw_text") or msg["text"]
            msg["text"] = ""
            msg["flags"] = sorted(set(msg.get("flags", []) + ["silent_executed"]))
            silent_cleared += 1

        # F. 轮询命令规范化
        for tc in tcs:
            if tc.get("name") == "bash" and isinstance(tc.get("arguments"), dict):
                cmd = tc["arguments"].get("command")
                if isinstance(cmd, str):
                    norm_cmd = normalize_polling_command(cmd)
                    if norm_cmd != cmd:
                        tc["arguments"]["command"] = norm_cmd
                        msg["flags"] = sorted(set(msg.get("flags", []) + ["polling_normalized"]))
                        polling_normalized += 1

        kept.append(cur)
        i += 1

    gold_messages: List[Dict[str, Any]] = []
    for u in kept:
        if u["type"] == "assistant_turn":
            u["msg"].pop("flag_pruned_tools", None)
        gold_messages.append(u["msg"])
        gold_messages.extend(u["tools"])

    stats = {
        "units_in": len([u for u in units if u["type"] == "assistant_turn"]),
        "units_out": len([u for u in gold_messages if u["role"] == "assistant"]),
        "folded": folded,
        "pruned_detours": dropped,
        "thinking_compressed": think_compressed,
        "thinking_backfilled": think_backfilled,
        "thinking_synthesized": think_synthesized,
        "silent_cleared": silent_cleared,
        "polling_normalized": polling_normalized,
    }
    return gold_messages, pruned, stats


# ---------------------------------------------------------------------------
# judge 执行器：模型给决定，程序确定性执行并留血缘
# ---------------------------------------------------------------------------

def _can_fold(prev_msg: Dict[str, Any], msg: Dict[str, Any]) -> bool:
    ptcs = prev_msg.get("tool_calls") or []
    tcs = msg.get("tool_calls") or []
    if len(ptcs) != 1 or len(tcs) != 1:
        return False
    a, b = ptcs[0], tcs[0]
    if a.get("name") != "write" or b.get("name") != "edit":
        return False
    pa = (a.get("arguments") or {}).get("path")
    pb = (b.get("arguments") or {}).get("path")
    return bool(pa and pa == pb)


def _do_fold(prev_msg: Dict[str, Any], msg: Dict[str, Any], max_edits: int = 8) -> bool:
    tcs = prev_msg.get("tool_calls") or []
    if not tcs:
        return False
    args = tcs[0].setdefault("arguments", {})
    content = args.get("content", "")
    eargs = (msg.get("tool_calls") or [{}])[0].get("arguments") or {}
    edits = eargs.get("edits") or ([eargs] if eargs.get("oldText") else [])
    edits = edits[:max_edits]
    if not edits:
        return False
    new_content = content
    for ed in edits:
        old_t, new_t = ed.get("oldText", ""), ed.get("newText", "")
        if not old_t or old_t not in new_content:
            return False
        new_content = new_content.replace(old_t, new_t, 1)
    args["content"] = new_content
    prev_msg["folded_from"] = msg.get("id")
    prev_msg["fold_details"] = {
        "path": args.get("path", ""),
        "folded_msg_id": msg.get("id"),
        "edits_count": len(edits),
        "before_content": content[:4000],
        "after_content": new_content[:4000],
        "edit_thinking": (msg.get("thinking") or "")[:800],
    }
    prev_msg["flags"] = sorted(set(prev_msg.get("flags", []) + ["state_folded"]))
    return True


def _rewrite_ok(new: str, msg: Dict[str, Any]) -> bool:
    old = (msg.get("raw_thinking") or msg.get("thinking") or "").strip()
    new = (new or "").strip()
    if len(new) < 10:
        return False
    cap = max(int(len(old) * 1.2), 400)
    return len(new) <= cap


def purify_by_judge(
    flow: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    session_cwd: str = "",
    turn_marks: Optional[List[Dict[str, Any]]] = None,
    log=print,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """按模型决定执行删/折/改/清/排；再接缝审核。返回 (gold, pruned, stats, audit)。"""
    gcfg = cfg.get("gold", {})
    units = _units(flow)
    for u in units:
        if u["type"] == "assistant_turn":
            m = u["msg"]
            m.setdefault("raw_thinking", m.get("thinking") or "")
            m.setdefault("raw_text", m.get("text") or "")
            for tc in m.get("tool_calls") or []:
                if tc.get("name") == "bash" and isinstance(tc.get("arguments"), dict):
                    cmd = tc["arguments"].get("command")
                    if isinstance(cmd, str):
                        norm = normalize_polling_command(cmd)
                        if norm != cmd:
                            tc["arguments"]["command"] = norm
                            m["flags"] = sorted(set(m.get("flags", []) + ["polling_normalized"]))

    flat: List[Dict[str, Any]] = []
    for u in units:
        flat.append(u["msg"])
        flat.extend(u["tools"])
    # 手册阶段 2 第 3 条「中间步骤强制静默执行」是机械规则，由程序直接执行
    if gcfg.get("silent_intermediate_steps", True):
        asst_units = [u for u in units if u["type"] == "assistant_turn"]
        silent_n = 0
        for u in asst_units:
            m = u["msg"]
            if (m.get("tool_calls") or []) and (m.get("text") or "").strip():
                m["raw_text"] = m.get("raw_text") or m.get("text")
                m["text"] = ""
                m["flags"] = sorted(set(m.get("flags", []) + ["silent_executed"]))
                silent_n += 1
        log(f"[judge] 机械静默执行 {silent_n} 步")

    decisions = judge.decide_steps(flat, cfg=cfg, log=log)
    log(f"[judge] 步级决定 {len(decisions)} 条（模型）")

    asst_total = sum(1 for u in units if u["type"] == "assistant_turn")
    prune_budget = int(asst_total * judge.MAX_STEPS_KEPT)
    kept: List[Tuple[Dict[str, Any], int]] = []
    pruned: List[Dict[str, Any]] = []
    pruned_n = 0
    folded = compressed = backfilled = excluded = 0
    polling_norm = 0

    for idx, u in enumerate(units):
        if u["type"] != "assistant_turn":
            kept.append((u, idx))
            continue
        msg = u["msg"]
        mid = msg.get("id")
        if any(f == "polling_normalized" for f in (msg.get("flags") or [])):
            polling_norm += 1
        d = decisions.get(mid) or {}
        act = d.get("action", "keep")

        if act == "prune":
            # 不变量：首步不可删；剪除量不超过预算
            if idx <= 1 or pruned_n >= prune_budget:
                kept.append((u, idx))
                continue
            pruned.append({
                "reason": f"judge::{d.get('reason') or 'prune'}",
                "has_dpo": True,
                "unit_index": idx,
                "rejected_msg": copy.deepcopy(msg),
                "dropped_msg_id": mid,
                "error_output": "\n".join(t.get("content", "") for t in (msg.get("tool_results") or []))[:800],
                "by": "judge",
            })
            pruned_n += 1
            continue

        if act == "fold":
            prev = kept[-1] if kept else None
            prev_msg = prev[0]["msg"] if prev and prev[0]["type"] == "assistant_turn" else None
            if prev_msg is not None and _can_fold(prev_msg, msg) and _do_fold(prev_msg, msg):
                pruned.append({
                    "reason": f"judge::{d.get('reason') or 'fold'}",
                    "has_dpo": False,
                    "unit_index": idx,
                    "rejected_msg": copy.deepcopy(msg),
                    "dropped_msg_id": mid,
                    "by": "judge",
                })
                folded += 1
                continue
            kept.append((u, idx))
            continue

        if act == "rewrite":
            new = d.get("rewritten") or ""
            if _rewrite_ok(new, msg):
                old = (msg.get("raw_thinking") or msg.get("thinking") or "").strip()
                msg["raw_thinking"] = msg.get("raw_thinking") or old
                msg["thinking"] = new.strip()
                msg["judge_reason"] = d.get("reason")
                if len(new.strip()) < len(old) * 0.9:
                    msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_compressed"]))
                    compressed += 1
                else:
                    msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_backfilled"]))
                    backfilled += 1


        if act == "exclude":
            msg["excluded"] = True
            msg["exclude_reason"] = d.get("reason") or "judge_exclude"
            excluded += 1

        kept.append((u, idx))

    # 剪枝记录补 chosen：被删步之后第一个保留的助手步（即"当时应该走的那一步"）
    kept_asst_idx = [i for (u, i) in kept if u["type"] == "assistant_turn"]
    for pd in pruned:
        if not pd.get("has_dpo"):
            continue
        nxt = next((i for i in kept_asst_idx if i > pd["unit_index"]), None)
        if nxt is None:
            pd["has_dpo"] = False
            continue
        chosen = next(u["msg"] for (u, i) in kept if i == nxt)
        pd["chosen_msg"] = copy.deepcopy(chosen)

    gold_messages: List[Dict[str, Any]] = []
    for u, _ in kept:
        gold_messages.append(u["msg"])
        gold_messages.extend(u["tools"])

    # ---- 接缝审核（模型）----
    seams = judge.extract_seams(gold_messages, pruned, turn_marks)
    audit = {"seams_total": len(seams), "seams_fixed": 0, "seam_issues": []}
    if seams:
        verdicts = judge.audit_seams(seams, cfg=cfg, log=log)
        by_id = {m.get("id"): m for m in gold_messages if m.get("role") == "assistant"}
        for sid, v in verdicts.items():
            if v.get("smooth", True):
                continue
            tgt = by_id.get(v.get("patch_msg_id") or "")
            patch = v.get("patched_thinking") or ""
            if tgt is not None and patch and _rewrite_ok(patch, tgt):
                tgt["raw_thinking"] = tgt.get("raw_thinking") or tgt.get("thinking")
                tgt["thinking"] = patch
                tgt["flags"] = sorted(set(tgt.get("flags", []) + ["seam_repaired"]))
                audit["seams_fixed"] += 1
            audit["seam_issues"].append({"seam_id": sid, "issue": v.get("issue") or "",
                                         "applied": bool(tgt is not None and patch)})
        log(f"[judge] 接缝 {len(seams)} 处，模型判定不平滑 {sum(1 for v in verdicts.values() if not v.get('smooth', True))} 处，已修 {audit['seams_fixed']} 处")

    stats = {
        "units_in": asst_total,
        "units_out": sum(1 for m in gold_messages if m.get("role") == "assistant"),
        "folded": folded,
        "pruned_detours": pruned_n,
        "thinking_compressed": compressed,
        "thinking_backfilled": backfilled,
        "thinking_synthesized": 0,
        "silent_cleared": sum(1 for m in gold_messages
                              if m.get("role") == "assistant" and "silent_executed" in (m.get("flags") or [])),
        "polling_normalized": polling_norm,
        "excluded": excluded,
        "decisions": len(decisions),
        "decided_by": "model",
    }
    return gold_messages, pruned, stats, audit


def _dpo_from_pruned(
    pruned: List[Dict[str, Any]],
    sys_msg: Optional[Dict[str, Any]],
    root_msg: Optional[Dict[str, Any]],
    session_id: str,
    project: str,
    model: str,
    window_turns: int,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    def conv(units, strip=True):
        out: List[Dict[str, str]] = []
        if sys_msg:
            out.extend(to_sharegpt(sys_msg))
        if root_msg:
            out.extend(to_sharegpt(root_msg))
        for u in units[-window_turns:]:
            if u.get("type") in ("system", "user"):
                continue
            if root_msg and u["msg"].get("id") == root_msg.get("id"):
                continue
            out.extend(to_sharegpt(u["msg"], strip_thinking=strip))
            for tr in u.get("tools", []):
                out.extend(to_sharegpt(tr))
        return out

    for i, pd in enumerate(pruned):
        if not pd.get("has_dpo", True) or not pd.get("chosen_msg") or not pd.get("rejected_msg"):
            continue
        convs = conv(pd["context_units"])
        if not convs:
            continue
        chosen = to_sharegpt(pd["chosen_msg"])
        rejected = to_sharegpt(pd["rejected_msg"])
        if not chosen or not rejected:
            continue
        if chosen[0].get("value") == rejected[0].get("value"):
            continue
        pairs.append(
            {
                "sample_id": f"dpo::{session_id}::pruned{i}::{pd['rejected_msg']['id']}",
                "conversations": convs,
                "chosen": chosen[0],
                "rejected": rejected[0],
                "metadata": {
                    "source": "stage2_pruned",
                    "reason": pd["reason"],
                    "session_id": session_id,
                    "project_group": project,
                    "model": model,
                    "chosen_msg_id": pd["chosen_msg"]["id"],
                    "rejected_msg_id": pd["rejected_msg"]["id"],
                },
            }
        )
    return pairs


def run(force: bool = False, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    raw_index = load_index(dirs["raw"] / "index.json")
    if not raw_index["sessions"]:
        raise RuntimeError("stage1 未产出数据，请先运行 stage1")

    gold_root = dirs["gold"]
    sessions_out = gold_root / "sessions"
    rejects_out = gold_root / "rejects"
    annotations_out = gold_root / "annotations"
    for d in (sessions_out, rejects_out, annotations_out):
        d.mkdir(parents=True, exist_ok=True)

    cfgh = section_hash(
        cfg, ["gold", "review", "synth.window_turns", "synth.max_tool_output_chars", "labeling", "turns"]
    )
    state_path = dirs["state"] / "stage2.json"
    state = read_json(state_path, default={}) or {}
    prev_files = state.get("files", {}) if state.get("config_hash") == cfgh else {}
    prev_index = load_index(gold_root / "index.json")
    gcfg = cfg.get("gold", {})
    min_steps = int(gcfg.get("min_gold_steps", 10))
    window_turns = int(cfg["synth"].get("window_turns", 8))
    max_tool_chars = int(cfg["synth"].get("max_tool_output_chars", 12000))

    sessions_index: Dict[str, Any] = {}
    state_files: Dict[str, Any] = {}
    all_dpo: List[Dict[str, Any]] = []
    all_diag: List[Dict[str, Any]] = []
    written = reused = 0
    counts = Counter()

    for sid, entry in sorted(raw_index["sessions"].items()):
        label = entry.get("label", "pending")
        counts[label] += 1
        raw_file = Path(entry["raw_file"])
        if not raw_file.exists():
            log(f"[stage2] 缺失原始文件，跳过 {sid}")
            continue
        fp = entry.get("fingerprint") or fingerprint(raw_file)
        meta_path = Path(entry.get("meta_file", "")) if entry.get("meta_file") else None
        meta_fp = fingerprint(meta_path) if meta_path and meta_path.exists() else None
        prev = prev_files.get(sid)
        prev_entry = (prev_index.get("sessions") or {}).get(sid)

        unchanged = (
            not force
            and prev
            and fingerprint_equal(prev.get("fingerprint"), fp)
            and fingerprint_equal(prev.get("meta_fingerprint"), meta_fp)
            and prev.get("flow_version") == pilog.FLOW_VERSION
            and prev_entry
            and prev_entry.get("label") == label
            and all(Path(p).exists() for p in prev.get("outputs", []))
        )
        if unchanged:
            sessions_index[sid] = prev_entry
            state_files[sid] = prev
            reused += 1
            for pair in read_jsonl(Path(prev_entry.get("dpo_file", ""))) if prev_entry.get("dpo_file") else []:
                all_dpo.append(pair)
            for diag in read_jsonl(Path(prev_entry.get("diag_file", ""))) if prev_entry.get("diag_file") else []:
                all_diag.append(diag)
            continue

        session = pilog.parse_session(raw_file, max_tool_output_chars=max_tool_chars)
        raw_meta = read_json(meta_path, default={}) if meta_path and meta_path.exists() else {}
        annotations = (raw_meta or {}).get("turn_annotations") or None
        manual_turn_ov = (raw_meta or {}).get("manual_turn_overrides") or None
        flow, marks, flow_stats = pilog.build_flow(session, cfg, annotations, manual_turn_ov)
        asst_all = [m for m in flow if m["role"] == "assistant"]
        raw_quality_by_id = {}
        for m in asst_all:
            raw_quality_by_id[m["id"]] = analyze_step_quality(
                m.get("thinking", ""),
                m.get("text", ""),
                m.get("tool_calls", []),
                m.get("tool_results", []),
                entry.get("cwd", ""),
                max_think_chars=4500,
            )

        model_audit = {}
        if judge.enabled(cfg, label):
            gold_msgs, pruned, purify_stats, model_audit = purify_by_judge(
                flow, cfg, entry.get("cwd", ""), turn_marks=marks, log=log)
        else:
            gold_msgs, pruned, purify_stats = purify(flow, gcfg, entry.get("cwd", ""))
        project = entry.get("project_group", "unknown")

        # 模型质量报告 + 高光分析（模型负责"值不值得训练"，程序据此门控）
        quality_report: Dict[str, Any] = {}
        if judge.enabled(cfg):
            quality_report = judge.audit_quality(gold_msgs, purify_stats, cfg=cfg, log=log)
            if quality_report:
                log(
                    f"[judge] 质量报告 {sid[:8]}: {quality_report['verdict']}"
                    f"（高光 {len(quality_report.get('highlights') or [])} 步"
                    f"{'，建议整条重写→按不建议训练处理' if quality_report.get('needs_wholesale_rewrite') else ''}）"
                )

        # 病态步诊断（保留原始流中的全部）
        diag_path = rejects_out / f"{sid}.diagnostics.jsonl"
        diag_rows = []
        for m in asst_all:
            q = raw_quality_by_id[m["id"]]
            if not q["pathologies"]:
                continue
            diag_rows.append(
                {
                    "diagnostic_id": f"diag::{sid}::{m['id']}",
                    "sample_id": f"diag::{sid}::{m['id']}",
                    "session_id": sid,
                    "label": label,
                    "project_group": project,
                    "model": m.get("model", ""),
                    "msg_id": m["id"],
                    "quality_score": q["score"],
                    "pathologies": q["pathologies"],
                    "metrics": q["metrics"],
                    "thinking": m.get("thinking", ""),
                    "tool_calls": [tc["name"] for tc in m.get("tool_calls", [])],
                    "timestamp": m.get("timestamp", ""),
                }
            )
        write_jsonl_atomic(diag_path, diag_rows)
        all_diag.extend(diag_rows)
        dominant = max((entry.get("models") or {"unknown": 1}).items(), key=lambda kv: kv[1])[0]

        # 附加双态质量（raw_quality + 提纯后 quality）与血缘字段
        clean_after_purify = 0
        for m in gold_msgs:
            if m["role"] == "assistant":
                raw_q = raw_quality_by_id.get(m["id"]) or analyze_step_quality(
                    m.get("raw_thinking") or m.get("thinking", ""),
                    m.get("raw_text") or m.get("text", ""),
                    m.get("tool_calls", []),
                    m.get("tool_results", []),
                    entry.get("cwd", ""),
                    4500,
                )
                m["raw_quality"] = raw_q
                eff_q = evaluate_effective_step(m, entry.get("cwd", ""), max_think_chars=4500)
                m["quality"] = eff_q
                m["raw_msg_id"] = m["id"]
                m["is_clean"] = eff_q["is_clean"]
                if eff_q["is_clean"]:
                    clean_after_purify += 1

        gold_asst = [m for m in gold_msgs if m["role"] == "assistant"]
        sys_msg = next((m for m in flow if m["role"] == "system"), None)
        root_msg = next((m for m in flow if m["role"] == "user"), None)

        status = "gold" if (label == "positive" and len(gold_asst) >= min_steps) else (
            "raw_only" if label == "pending" else "negative_source"
        )
        if label == "positive" and quality_report:
            if quality_report.get("verdict") == "discard":
                status = "discard_recommended"
            elif quality_report.get("needs_wholesale_rewrite"):
                status = "discard_recommended"
        reasons = []
        if label == "positive" and len(gold_asst) < min_steps:
            reasons.append(f"too_few_steps({len(gold_asst)}<{min_steps})")
            status = "rejected"
        if not flow_stats.get("has_root"):
            reasons.append("no_root_task")
        if quality_report.get("verdict") == "discard":
            reasons.append(f"model_discard:{quality_report.get('reason', '')}")
        if quality_report.get("needs_wholesale_rewrite"):
            reasons.append("model_needs_wholesale_rewrite")

        proj_dir = safe_name(project)
        gold_path = sessions_out / proj_dir / f"{sid}.gold.jsonl"
        dpo_path = rejects_out / f"{sid}.dpo.jsonl"
        reject_path = rejects_out / f"{sid}.pruned.jsonl"

        record = {
            "gold_id": f"gold::{sid}",
            "session_id": sid,
            "label": label,
            "project_group": project,
            "cwd": entry.get("cwd", ""),
            "dominant_model": dominant,
            "source": {
                "raw_file": str(raw_file),
                "session_file": entry.get("source", {}).get("rel_path", ""),
                "fingerprint": fp,
            },
            "stats": {
                **purify_stats,
                "raw_steps": len(asst_all),
                "gold_steps": len(gold_asst),
                "clean_gold_steps": clean_after_purify,
                "turns_total": flow_stats["turns_total"],
                "turns_kept": flow_stats["turns_kept"],
                "turns_direct": flow_stats.get("turns_direct", flow_stats["turns_kept"]),
                "turns_folded": flow_stats.get("turns_folded", 0),
                "turns_dropped": flow_stats.get("turns_dropped", 0),
                "dropped_turns": flow_stats["dropped"],
                "root_prompt": flow_stats["root_prompt"],
                "raw_root_prompt": flow_stats.get("raw_root_prompt", flow_stats["root_prompt"]),
                "constraints": flow_stats["constraints"],
            },
            "messages": gold_msgs,
            "model_report": quality_report or None,
            "model_audit": model_audit or None,
        }
        write_jsonl_atomic(gold_path, [record])

        dpo_pairs = _dpo_from_pruned(
            pruned, sys_msg, root_msg, sid, project, dominant, window_turns
        )
        write_jsonl_atomic(dpo_path, dpo_pairs)
        write_jsonl_atomic(
            reject_path,
            [
                {
                    "session_id": sid,
                    "reason": pd["reason"],
                    "has_dpo": pd.get("has_dpo", True),
                    "dropped_msg_id": pd.get("dropped_msg_id"),
                    "rejected_msg_id": pd["rejected_msg"]["id"] if pd.get("rejected_msg") else None,
                    "rejected_thinking": (pd["rejected_msg"].get("thinking") or "")[:2000] if pd.get("rejected_msg") else "",
                    "rejected_tool_calls": [tc["name"] for tc in pd["rejected_msg"].get("tool_calls", [])] if pd.get("rejected_msg") else [],
                    "chosen_msg_id": pd["chosen_msg"]["id"] if pd.get("chosen_msg") else None,
                    "chosen_thinking": (pd["chosen_msg"].get("thinking") or "")[:2000] if pd.get("chosen_msg") else "",
                    "file_path": pd.get("file_path", ""),
                    "before_snippet": pd.get("before_snippet", ""),
                    "after_snippet": pd.get("after_snippet", ""),
                    "error_output": pd.get("error_output", ""),
                }
                for pd in pruned
            ],
        )
        all_dpo.extend(dpo_pairs)

        ann_path = annotations_out / f"{sid}.json"
        prev_ann = read_json(ann_path, default={}) or {}
        flow_signature = sha256_text(
            json.dumps(
                {
                    "root": flow_stats.get("root_prompt"),
                    "constraints": flow_stats.get("constraints"),
                    "marks": [
                        (m.get("turn_index"), m.get("kind"), m.get("action"), m.get("dropped"))
                        for m in marks
                    ],
                    "assistant_ids": [m.get("id") for m in gold_asst],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )[:16]
        review_key = f"{fp.get('size')}:{fp.get('tail_sha256')}:{flow_signature}:{pilog.FLOW_VERSION}"
        # 原始会话或人工回合流发生变化时，旧评审必须作废；仅算法版本变化不作废人工结论
        prev_key_prefix = ":".join(str(prev_ann.get("review_key") or "").split(":")[:3])
        cur_key_prefix = ":".join(review_key.split(":")[:3])
        same_raw_content = bool(prev_ann.get("review_key") and prev_key_prefix == cur_key_prefix)
        keep_review = prev_ann.get("model_review") if same_raw_content else None

        annotation = {
            "session_id": sid,
            "model_report": quality_report or None,
            "model_audit": model_audit or None,
            "label": label,
            "status": status,
            "reasons": reasons,
            "script_score": round(
                sum(m["quality"]["score"] for m in gold_asst) / len(gold_asst), 2
            ) if gold_asst else 0.0,
            "model_review": keep_review,
            "review_key": review_key,
            "reviewed_at": prev_ann.get("reviewed_at") if keep_review else None,
        }
        for k in ("review_rounds", "human_review"):
            if prev_ann.get(k):
                annotation[k] = prev_ann[k]
        if same_raw_content and isinstance(annotation.get("human_review"), dict) and annotation["human_review"].get("decision"):
            annotation["human_review"]["review_key"] = review_key
        if not keep_review and annotation.get("review_rounds"):
            annotation["review_rounds_stale"] = True
        write_json_atomic(ann_path, annotation)

        sessions_index[sid] = {
            "session_id": sid,
            "gold_id": record["gold_id"],
            "label": label,
            "status": status,
            "project_group": project,
            "dominant_model": dominant,
            "cwd": entry.get("cwd", ""),
            "root_prompt": flow_stats["root_prompt"],
            "stats": record["stats"],
            "dpo_count": len(dpo_pairs),
            "pathologies": entry.get("pathologies", {}),
            "gold_file": str(gold_path),
            "dpo_file": str(dpo_path),
            "diag_file": str(diag_path),
            "reject_file": str(reject_path),
            "annotation_file": str(ann_path),
            "fingerprint": fp,
            "last_run": now_iso(),
        }
        state_files[sid] = {
            "fingerprint": fp,
            "meta_fingerprint": meta_fp,
            "flow_version": pilog.FLOW_VERSION,
            "label": label,
            "outputs": [str(gold_path), str(dpo_path), str(reject_path), str(diag_path), str(ann_path)],
        }
        written += 1
        log(
            f"[stage2] {label:<8} {project:<22} {sid[:8]}  "
            f"steps {len(asst_all)}->{len(gold_asst)} (干净 {clean_after_purify})  "
            f"折叠 {purify_stats['folded']}  剪枝 {purify_stats['pruned_detours']}  "
            f"压缩 {purify_stats['thinking_compressed']}  DPO {len(dpo_pairs)}"
        )

    for sid in list((prev_index.get("sessions") or {}).keys()):
        if sid not in sessions_index:
            state_files.pop(sid, None)

    write_jsonl_atomic(gold_root / "dpo_pairs.jsonl", all_dpo)
    write_jsonl_atomic(gold_root / "diagnostics.jsonl", all_diag)
    dump_index(
        gold_root / "index.json",
        sessions_index,
        {
            "stage": STAGE,
            "config_hash": cfgh,
            "counts": dict(counts),
            "dpo_pairs": len(all_dpo),
            "diagnostics": len(all_diag),
        },
    )
    write_json_atomic(
        state_path,
        {"stage": STAGE, "version": 2, "config_hash": cfgh, "last_run": now_iso(), "files": state_files},
    )

    summary = {
        "stage": STAGE,
        "written": written,
        "reused": reused,
        "counts": dict(counts),
        "dpo_pairs": len(all_dpo),
        "diagnostics": len(all_diag),
        "gold_steps": sum(v["stats"]["gold_steps"] for v in sessions_index.values()),
    }
    log(f"[stage2] 完成：写入 {written} / 复用 {reused}，黄金步数 {summary['gold_steps']}，DPO 对 {len(all_dpo)}")
    return summary
