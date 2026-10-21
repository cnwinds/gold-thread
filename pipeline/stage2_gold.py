"""Stage 2：原始日志 -> 黄金会话（状态折叠 / 弯路剪枝 / CoT 去噪 / 静默执行）+ 天然 DPO 对。"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import pilog
from .common.config import config_hash, load_config, section_hash, stage_dirs
from .common.format import to_sharegpt
from .common.quality import (
    LOW_LEVEL_MISTAKE_OUTPUT_REGEX,
    NEXT_STEP_OOPS_REGEX,
    analyze_step_quality,
    compact_thinking,
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
    """返回 (gold_messages, pruned_records, stats)。pruned_records 用于 DPO。"""
    units = _units(flow)
    pruned: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    folded = 0
    dropped = 0
    think_compressed = 0

    i = 0
    while i < len(units):
        cur = units[i]
        if cur["type"] != "assistant_turn":
            kept.append(cur)
            i += 1
            continue

        msg = cur["msg"]
        tcs = msg.get("tool_calls", [])
        trs = cur["tools"]

        if i + 1 < len(units) and units[i + 1]["type"] == "assistant_turn":
            nxt = units[i + 1]
            nxt_msg = nxt["msg"]
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
                content = tcs[0]["arguments"].get("content", "")
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
                    msg["flags"] = sorted(set(msg.get("flags", []) + ["state_folded"]))
                    pruned.append(
                        {
                            "reason": "state_folding_bug_fix",
                            "context_units": copy.deepcopy(kept),
                            "chosen_msg": copy.deepcopy(msg),
                            "rejected_msg": original,
                            "dropped_msg_id": nxt_msg["id"],
                        }
                    )
                    kept.append(cur)
                    folded += 1
                    i += 2
                    continue

            # B. 弯路剪枝：最低级命令错误 / 下一步自认手滑
            if gcfg.get("detour_pruning", True):
                out_text = "\n".join(t.get("content", "") for t in trs)
                has_error = any(t.get("isError") for t in trs) or bool(
                    LOW_LEVEL_MISTAKE_OUTPUT_REGEX.search(out_text)
                )
                next_admits = bool(NEXT_STEP_OOPS_REGEX.search((nxt_msg.get("thinking") or "").strip()))
                if (has_error or next_admits) and nxt_tcs:
                    cleaned = copy.deepcopy(nxt)
                    cleaned_think = re.sub(
                        r"^(Oops[^.!?\n]*[.!?\n]+|It didn't recognize because[^.!?\n]*[.!?\n]+|"
                        r"`zip` not available[.!?\n]*)",
                        "",
                        (cleaned["msg"].get("thinking") or "").strip(),
                        flags=re.IGNORECASE,
                    ).strip()
                    if not cleaned_think and msg.get("thinking"):
                        cleaned_think = compact_thinking(msg["thinking"], 1200)
                    cleaned["msg"]["thinking"] = cleaned_think
                    cleaned["msg"]["pruned_detour"] = msg["id"]
                    cleaned["msg"]["flags"] = sorted(
                        set(cleaned["msg"].get("flags", []) + ["detour_pruned"])
                    )
                    pruned.append(
                        {
                            "reason": "detour_pruned_error_or_oops",
                            "context_units": copy.deepcopy(kept),
                            "chosen_msg": copy.deepcopy(cleaned["msg"]),
                            "rejected_msg": copy.deepcopy(msg),
                            "dropped_msg_id": msg["id"],
                        }
                    )
                    units[i + 1] = cleaned
                    dropped += 1
                    i += 1
                    continue

        # C. 空 thinking：报错剪除；成功则用对外播报文本内化为 thinking
        if not msg.get("thinking") and tcs:
            if any(t.get("isError") for t in trs):
                dropped += 1
                i += 1
                continue
            if msg.get("text"):
                msg["thinking"] = msg["text"].strip()
                msg["text"] = ""
                msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_backfilled"]))
            elif gcfg.get("fix_missing_thinking", True):
                # 用工具参数构造最小可读意图（比空 thinking 更有信息量），并打标
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

        # D. CoT 去噪
        max_chars = int(gcfg.get("max_thinking_chars", 2400))
        target = int(gcfg.get("target_thinking_chars", 1600))
        if len(msg.get("thinking", "")) > max_chars:
            msg["thinking"] = compact_thinking(msg["thinking"], target)
            msg["flags"] = sorted(set(msg.get("flags", []) + ["thinking_compressed"]))
            think_compressed += 1

        # E. 静默执行：中间工具步清空对外文本
        if gcfg.get("silent_intermediate_steps", True) and tcs and msg.get("text"):
            msg["text"] = ""

        # F. 轮询命令规范化
        for tc in tcs:
            if tc.get("name") == "bash" and isinstance(tc.get("arguments"), dict):
                cmd = tc["arguments"].get("command")
                if isinstance(cmd, str):
                    tc["arguments"]["command"] = normalize_polling_command(cmd)

        kept.append(cur)
        i += 1

    gold_messages: List[Dict[str, Any]] = []
    for u in kept:
        if u["type"] == "assistant_turn":
            u["msg"]["flag_pruned_tools"] = None
        gold_messages.append(u["msg"])
        gold_messages.extend(u["tools"])

    stats = {
        "units_in": len([u for u in units if u["type"] == "assistant_turn"]),
        "units_out": len([u for u in gold_messages if u["role"] == "assistant"]),
        "folded": folded,
        "pruned_detours": dropped,
        "thinking_compressed": think_compressed,
    }
    return gold_messages, pruned, stats


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
        cfg, ["gold", "review", "synth.window_turns", "synth.max_tool_output_chars", "labeling"]
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
        raw_meta = read_json(Path(entry.get("meta_file", "")), default={}) or {}
        annotations = raw_meta.get("turn_annotations") or None
        flow, marks, flow_stats = pilog.build_flow(session, cfg, annotations)
        asst_all = [m for m in flow if m["role"] == "assistant"]
        quality_by_id = {}
        for m in asst_all:
            quality_by_id[m["id"]] = analyze_step_quality(
                m.get("thinking", ""),
                m.get("text", ""),
                m.get("tool_calls", []),
                m.get("tool_results", []),
                entry.get("cwd", ""),
                max_think_chars=4500,
            )

        gold_msgs, pruned, purify_stats = purify(flow, gcfg, entry.get("cwd", ""))
        project = entry.get("project_group", "unknown")

        # 病态步诊断（保留原始流中的全部）
        diag_path = rejects_out / f"{sid}.diagnostics.jsonl"
        diag_rows = []
        for m in asst_all:
            q = quality_by_id[m["id"]]
            if not q["pathologies"]:
                continue
            diag_rows.append(
                {
                    "diagnostic_id": f"diag::{sid}::{m['id']}",
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
                }
            )
        write_jsonl_atomic(diag_path, diag_rows)
        all_diag.extend(diag_rows)
        dominant = max(entry.get("models", {"unknown": 1}).items(), key=lambda kv: kv[1])[0]

        # 附加质量与血缘字段
        for m in gold_msgs:
            if m["role"] == "assistant":
                q = quality_by_id.get(m["id"]) or analyze_step_quality(
                    m.get("thinking", ""), m.get("text", ""), m.get("tool_calls", []),
                    m.get("tool_results", []), entry.get("cwd", ""), 4500,
                )
                m["quality"] = q
                m["raw_msg_id"] = m["id"]
                m["is_clean"] = q["is_clean"]

        gold_asst = [m for m in gold_msgs if m["role"] == "assistant"]
        sys_msg = next((m for m in flow if m["role"] == "system"), None)
        root_msg = next((m for m in flow if m["role"] == "user"), None)

        status = "gold" if (label == "positive" and len(gold_asst) >= min_steps) else (
            "raw_only" if label == "pending" else "negative_source"
        )
        reasons = []
        if label == "positive" and len(gold_asst) < min_steps:
            reasons.append(f"too_few_steps({len(gold_asst)}<{min_steps})")
            status = "rejected"
        if not flow_stats.get("has_root"):
            reasons.append("no_root_task")

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
                "turns_total": flow_stats["turns_total"],
                "turns_kept": flow_stats["turns_kept"],
                "dropped_turns": flow_stats["dropped"],
                "root_prompt": flow_stats["root_prompt"],
                "constraints": flow_stats["constraints"],
            },
            "messages": gold_msgs,
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
                    "dropped_msg_id": pd.get("dropped_msg_id"),
                    "rejected_msg_id": pd["rejected_msg"]["id"],
                    "rejected_thinking": pd["rejected_msg"].get("thinking", "")[:2000],
                    "rejected_tool_calls": [tc["name"] for tc in pd["rejected_msg"].get("tool_calls", [])],
                    "chosen_msg_id": pd["chosen_msg"]["id"],
                }
                for pd in pruned
            ],
        )
        all_dpo.extend(dpo_pairs)

        ann_path = annotations_out / f"{sid}.json"
        prev_ann = read_json(ann_path, default={}) or {}
        review_key = f"{fp.get('size')}:{fp.get('tail_sha256')}:{pilog.FLOW_VERSION}"
        keep_review = prev_ann.get("model_review") if prev_ann.get("review_key") == review_key else None
        annotation = {
            "session_id": sid,
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
        # 复核历史与人工结论跨数据版本保留；版本变了由 curation.is_stale 标记过期
        for k in ("review_rounds", "human_review"):
            if prev_ann.get(k):
                annotation[k] = prev_ann[k]
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
            f"steps {len(asst_all)}->{len(gold_asst)}  折叠 {purify_stats['folded']}  剪枝 {purify_stats['pruned_detours']}  DPO {len(dpo_pairs)}"
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
        {"stage": STAGE, "version": 1, "config_hash": cfgh, "last_run": now_iso(), "files": state_files},
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