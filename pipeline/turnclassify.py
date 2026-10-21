"""模型参与：会话回合语义分类（根任务 / 任务追问 / 环境准备 / 噪音 / 催促 / 纠偏）。

正则只负责明显的机械噪音，语义边界交给模型；模型输出标注，脚本确定性应用。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .common.config import load_config
from .common.llm import chat_json

PROMPT_VERSION = "turnclassify.v1"

SYSTEM = """你是工程分析 Agent 训练数据的标注员。给定一条 pi 会话的全部用户回合（含各自的助手步数），
你需要判定每个回合的性质，找出真正的「根任务」，以便后续脚本剥离噪音、折叠催促、内化纠偏。

判定标准：
- root：第一个明确给出破解目标的回合（通常是 .apk/.xapk 路径 + 破解要求）。整个会话只能有一个 root。
- task：root 之后新的实质任务或追问（例如分析玩法数值、还原某个函数、总结报告、备份成果）。
- setup：安装工具、配模型、配 API key、连模拟器/设备、装驱动、排环境报错。属于任务前置准备，不进训练。
- noise：闲聊、硬件咨询、测试连通性（hi）、粘贴 API 报错、磁盘清理、与目标无关的问题。
- continuation：纯过程推进（继续/可以/好的/补全结构/继续A 等）。
- correction：对 Agent 纪律的纠偏（不要参考其他目录、不要偷看别的实验目录等）。

注意：
1. 回合文本里可能包含被粘贴的报错，不要因此把真正的任务判成 noise；
2. 只根据用户意图判定，不要因为助手回复多就提高重要性；
3. root 之前的一切回合都标记 drop=true（即使看起来像任务准备）；
4. 只输出 JSON。"""

SCHEMA = """输出 JSON：
{
  "root_turn": 整数（回合 index）或 null,
  "turns": [
    {"index": 整数, "kind": "root|task|setup|noise|continuation|correction", "drop": 布尔, "why": "不超过 20 字"}
  ],
  "notes": "可选，一句话总结"
}"""

KINDS = {"root", "task", "setup", "noise", "continuation", "correction"}


def turn_summaries(session: Dict[str, Any], max_chars: int = 800) -> List[Dict[str, Any]]:
    """把消息流切成 user 回合摘要，供模型分类。"""
    turns: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for m in session["messages"]:
        if m["role"] == "user":
            current = {"text": m["content"], "steps": 0, "tools": [], "error_results": 0}
            blocks.append(current)
        elif current is not None and m["role"] == "assistant":
            current["steps"] += 1
            for tc in m.get("tool_calls", []):
                if tc["name"] not in current["tools"]:
                    current["tools"].append(tc["name"])
            current["error_results"] += sum(1 for tr in m.get("tool_results", []) if tr.get("isError"))
    for i, b in enumerate(blocks):
        turns.append(
            {
                "index": i,
                "user_text": b["text"][:max_chars],
                "assistant_steps": b["steps"],
                "tools": b["tools"][:6],
            }
        )
    return turns


def classify_session(
    session: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    log=print,
) -> Optional[Dict[str, Any]]:
    cfg = cfg or load_config()
    turns = turn_summaries(session, int(cfg["turns"].get("model_max_turn_chars", 800)))
    if not turns:
        return None
    payload = {
        "cwd": session.get("cwd", ""),
        "turns": turns,
        "hint": "root 通常是包含 .apk/.xapk 且要求破解/还原/分析的那一条；找不到则 null。",
    }
    result = chat_json(
        SYSTEM,
        SCHEMA + "\n\n会话数据：\n" + json.dumps(payload, ensure_ascii=False),
        prompt_version=PROMPT_VERSION,
    )
    if not isinstance(result, dict) or not isinstance(result.get("turns"), list):
        log("[turnclassify] 模型返回结构异常，回退到正则判定")
        return None

    valid: Dict[int, Dict[str, Any]] = {}
    for t in result["turns"]:
        try:
            idx = int(t.get("index"))
        except (TypeError, ValueError):
            continue
        kind = t.get("kind")
        if kind not in KINDS or idx < 0 or idx >= len(turns):
            continue
        valid[idx] = {
            "index": idx,
            "kind": kind,
            "drop": bool(t.get("drop")),
            "why": str(t.get("why", ""))[:120],
        }
    if not valid:
        return None

    root = result.get("root_turn")
    try:
        root = int(root) if root is not None else None
    except (TypeError, ValueError):
        root = None
    if root is not None and (root < 0 or root >= len(turns)):
        root = None
    if root is None:
        roots = [i for i, v in valid.items() if v["kind"] == "root"]
        root = min(roots) if roots else None
    # root 必须是真正有助手执行的回合；否则向后找第一个 root/task
    if root is not None and turns[root]["assistant_steps"] == 0:
        root = None
        for cand in sorted(valid):
            if valid[cand]["kind"] in ("root", "task") and turns[cand]["assistant_steps"] > 0:
                root = cand
                valid[cand] = {**valid[cand], "kind": "root", "drop": False}
                break
    if root is not None and turns[root]["assistant_steps"] > 0:
        valid[root] = {**valid.get(root, {"index": root}), "index": root, "kind": "root", "drop": False}

    return {
        "root_turn": root,
        "turns": [valid[i] for i in sorted(valid)],
        "notes": str(result.get("notes", ""))[:400],
        "model": cfg["llm"].get("model"),
        "prompt_version": PROMPT_VERSION,
    }