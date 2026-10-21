"""Stage 3：从黄金会话合成训练数据（单步窗口 / 完整轨迹 / DPO / 诊断 / 切分）。"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import curation
from .common.config import config_hash, load_config, stage_dirs
from .common.format import conversation, to_openai, to_sharegpt
from .common.quality import analyze_step_quality, compact_thinking
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
        q = m.get("quality") or {}
        bad = [
            p for p in q.get("pathologies", [])
            if p in ("over_rumination_length", "over_rumination_doubt_loop", "mental_hex_checkmark")
        ]
        if not bad:
            continue
        original = m.get("thinking", "")
        if len(original) < min_chars:
            continue
        compacted = compact_thinking(original, 1500)
        if len(compacted) >= len(original) * 0.75:
            continue
        history = msgs[max(0, idx - window_turns) : idx]
        while history and history[0]["role"] == "tool":
            history = history[1:]
        convs: List[Dict[str, str]] = []
        if sys_msg:
            convs.extend(to_sharegpt(sys_msg))
        if root_msg and not any(h.get("id") == root_msg.get("id") for h in history):
            convs.extend(to_sharegpt(root_msg))
        for h in history:
            if h["role"] == "system":
                continue
            convs.extend(to_sharegpt(h, strip_thinking=True))
        if not convs:
            continue
        pairs.append(
            {
                "sample_id": f"dpo_neg::{sid}::{m['id']}",
                "conversations": convs,
                "chosen": to_sharegpt(m, override_thinking=compacted)[0],
                "rejected": to_sharegpt(m)[0],
                "metadata": {
                    "source": "negative_step",
                    "session_id": sid,
                    "project_group": project,
                    "split": split,
                    "model": model,
                    "pathologies": bad,
                    "rejected_think_len": len(original),
                    "chosen_think_len": len(compacted),
                },
            }
        )
    return pairs


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
    window_turns = int(scfg.get("window_turns", 8))
    sft_labels = set(scfg.get("sft_labels", ["positive"]))
    weights = scfg.get("weights", {})
    max_think_target = int(cfg["gold"].get("target_thinking_chars", 1600))

    records = gold_index["sessions"]
    split_map = _pick_split([r.get("project_group", "unknown") for r in records.values()], cfg)

    gold_trajs: List[Dict[str, Any]] = []
    step_windows: List[Dict[str, Any]] = []
    dpo_pairs: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    # 1) stage2 产出的天然 DPO 对
    if dcfg.get("from_pruned_steps", True):
        for pair in read_jsonl(dirs["gold"] / "dpo_pairs.jsonl"):
            pair["metadata"]["split"] = split_map.get(pair["metadata"].get("project_group"), "train")
            dpo_pairs.append(pair)

    stage_counts = Counter()
    review_counts = Counter()
    project_counts = Counter()
    model_counts = Counter()

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
        asst = [m for m in msgs if m["role"] == "assistant"]
        model = rec.get("dominant_model") or entry.get("dominant_model") or "unknown"
        sys_msg = next((m for m in msgs if m["role"] == "system"), None)
        root_msg = next((m for m in msgs if m["role"] == "user"), None)

        # 复核门控：人工结论优先；待人工复核/结论过期的会话扣住不进 SFT；只用采纳的改写
        annotation = read_json(Path(entry.get("annotation_file", "")), default={}) or {}
        effective = curation.resolve(annotation, cfg)
        review_counts[effective["state"]] += 1
        rewrites = effective["rewrites"]
        excluded = set(effective["excluded"])
        blocked = effective["state"] not in curation.IN_SFT
        if blocked and label in sft_labels:
            log(f"[stage3] {curation.EFFECTIVE_ZH[effective['state']]}，不进 SFT: {sid}")

        # 诊断库（全部标签）
        for diag in read_jsonl(Path(entry.get("diag_file", ""))) if entry.get("diag_file") else []:
            diag["split"] = split
            diagnostics.append(diag)

        if label not in sft_labels or blocked:
            # 反例/被驳回的会话：仍然收割病态步 DPO
            if label == "negative" and dcfg.get("from_negative_steps", True):
                dpo_pairs.extend(
                    _negative_dpo(msgs, sys_msg, root_msg, dcfg, window_turns, sid, project, split, model)
                )
            continue

        model_counts[model] += 1

        # 2) 完整黄金轨迹
        meta = {
            "sample_id": f"traj::{sid}",
            "session_id": sid,
            "project_group": project,
            "split": split,
            "label": label,
            "model": model,
            "assistant_steps": len(asst),
            "source_gold": str(gold_file),
        }
        gold_trajs.append(
            {
                "sample_id": f"traj::{sid}",
                "conversations": conversation(msgs, "sharegpt", strip_thinking=False, override=rewrites),
                "tools": _tools_json(msgs),
                "metadata": meta,
            }
        )

        # 3) 单步决策窗口
        asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        total = len(asst_idx)
        for rank, idx in enumerate(asst_idx):
            m = msgs[idx]
            q = m.get("quality") or analyze_step_quality(
                m.get("thinking", ""), m.get("text", ""), m.get("tool_calls", []),
                m.get("tool_results", []), rec.get("cwd", ""), 4500,
            )
            if not q.get("is_clean") or not (m.get("thinking") or "").strip() or m["id"] in excluded:
                continue
            # 合成补白的 thinking 不进单步 SFT（除非模型评审已重写）
            if "thinking_synthesized" in (m.get("flags") or []) and m["id"] not in rewrites:
                continue

            history = msgs[max(0, idx - window_turns) : idx]
            while history and history[0]["role"] == "tool":
                history = history[1:]

            anchored: List[Dict[str, Any]] = []
            if sys_msg and (not history or history[0]["role"] != "system"):
                anchored.append(sys_msg)
            if root_msg and not any(h.get("id") == root_msg.get("id") for h in history):
                anchored.append(root_msg)
            full_history = anchored + [h for h in history if h["role"] != "system"]
            # 历史窗口本身以 system 开头时会被上面的过滤剔除，这里补回，保证 system 永远第一
            if sys_msg and (not full_history or full_history[0].get("id") != sys_msg.get("id")):
                full_history.insert(0, sys_msg)

            cleaned: List[Dict[str, Any]] = []
            for h in full_history:
                if h["role"] == "tool" and (not cleaned or cleaned[-1]["role"] not in ("assistant", "tool")):
                    continue
                cleaned.append(h)
            if not cleaned:
                continue

            if rank < 3:
                stage_name, weight = "opening", int(weights.get("opening", 3))
            elif rank >= max(3, total - 3):
                stage_name, weight = "closing", int(weights.get("closing", 3))
            elif len(m.get("tool_calls", [])) >= 2 or "script_verification" in q.get("strengths", []) or any(
                tc["name"] in ("write", "edit") for tc in m.get("tool_calls", [])
            ):
                stage_name, weight = "key_decision", int(weights.get("key_decision", 2))
            else:
                stage_name, weight = "routine", int(weights.get("routine", 1))

            thinking = m.get("thinking", "")
            if m["id"] in rewrites:
                thinking = rewrites[m["id"]]
            elif len(thinking) > int(cfg["gold"].get("max_thinking_chars", 2400)):
                thinking = compact_thinking(thinking, max_think_target)

            step_id = f"step::{sid}::{m['id']}"
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
                "msg_id": m["id"],
                "step_rank": rank + 1,
                "total_steps": total,
                "stage": stage_name,
                "sample_weight": weight,
                "quality_score": q.get("score"),
                "strengths": q.get("strengths", []),
            }
            stage_counts[stage_name] += 1
            project_counts[project] += 1

            target_sgpt = to_sharegpt(m, override_thinking=thinking)[0]
            step_windows.append(
                {
                    "sample_id": step_id,
                    "conversations": prompt_sgpt + [target_sgpt],
                    "tools": _tools_json(msgs),
                    "metadata": step_meta,
                }
            )
            step_windows.append(
                {
                    "sample_id": step_id,
                    "messages": prompt_oai + [to_openai(m, override_thinking=thinking)],
                    "tools": json.loads(_tools_json(msgs)),
                    "metadata": step_meta,
                }
            )


    # 5) 全局打乱（跨项目混训，避免优化器动量偏向单一游戏）
    if scfg.get("shuffle", True):
        rng = random.Random(int(scfg.get("seed", 42)))
        rng.shuffle(gold_trajs)
        sgpt = [w for w in step_windows if "conversations" in w]
        oai = [w for w in step_windows if "messages" in w]
        rng.shuffle(sgpt)
        rng.shuffle(oai)
    else:
        sgpt = [w for w in step_windows if "conversations" in w]
        oai = [w for w in step_windows if "messages" in w]
        rng = None

    # 6) 落盘
    out = {
        "sft_gold_trajectories_sharegpt.jsonl": gold_trajs,
        "sft_step_windows_sharegpt.jsonl": sgpt,
        "sft_step_windows_openai.jsonl": oai,
        "dpo_preference_pairs.jsonl": dpo_pairs,
        "negative_diagnostics.jsonl": diagnostics,
    }
    counts = {}
    for name, rows in out.items():
        counts[name] = write_jsonl_atomic(train_root / name, rows)

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
    write_json_atomic(train_root / "dataset_info.json", dataset_info)

    stats = {
        "generated_at": now_iso(),
        "config_hash": config_hash(cfg),
        "counts": counts,
        "split": split_map,
        "by_project": dict(project_counts),
        "by_stage": dict(stage_counts),
        "by_model": dict(model_counts),
        "weighted_steps": sum(int(w["metadata"].get("sample_weight", 1)) for w in sgpt),
        "dpo_by_source": dict(Counter(p["metadata"].get("source", "?") for p in dpo_pairs)),
        "diagnostics_by_pathology": dict(
            Counter(p.split(":")[0] for d in diagnostics for p in d.get("pathologies", []))
        ),
        "label_counts": dict(Counter(r.get("label", "?") for r in records.values())),
        "review_states": dict(review_counts),
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