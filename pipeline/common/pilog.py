"""pi 会话日志解析与回合级清洗（stage1 / stage2 共用，保证判定一致）。"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 回合成流算法版本：改动 build_flow / 清洗逻辑时递增，触发 stage1-3 重建
FLOW_VERSION = 6

DEFAULT_SYSTEM_PROMPT = (
    "你是一名顶尖的移动应用与游戏工程分析专家 Agent。在执行解包、反编译、资产恢复、协议与配置表解密任务时，必须遵守以下工程纪律：\n"
    "1. 严格在当前工作目录内独立完成研究与产出，严禁读取、参考或拷贝同级其他实验目录的文件；\n"
    "2. 禁止在思维链中心算十六进制偏移、XOR 或 IEEE-754 浮点数并自行打勾确认，一切字节偏移与结构假设必须通过 `python -c` 脚本实测验证；\n"
    "3. 优先并发调用工具与复用成熟开源分析工具链（如 jadx、gdre_tools、Il2CppDumper、radare2 等），遇到连续 2 次搜索 404 立即止损切换方案；\n"
    "4. 阶段性关键发现（魔数、密钥、结构体偏移）及时落盘记录，收尾编写报告前必须先完成全量文件计数与 SHA256/明文抽样核验。"
)

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the workspace shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Optional timeout in milliseconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Start line number"},
                    "limit": {"type": "integer", "description": "Number of lines"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to edit"},
                    "oldText": {"type": "string", "description": "Exact text to replace"},
                    "newText": {"type": "string", "description": "Replacement text"},
                    "edits": {"type": "array", "description": "Multiple edits", "items": {"type": "object"}},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "codemode",
            "description": "Run JavaScript that calls other tools.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "JavaScript source"}},
                "required": ["code"],
            },
        },
    },
]


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def parse_session(path: Path, max_tool_output_chars: int = 12000) -> Dict[str, Any]:
    """把 pi `.jsonl` 解析为结构化消息流（忠实还原：应用 context_edit 删除/替换）。"""
    path = Path(path)
    session_id = path.stem
    cwd = ""
    session_ts = ""
    current_model = "unknown"
    current_provider = "unknown"
    thinking_level = "default"

    events: List[Dict[str, Any]] = []
    deleted_ids: set[str] = set()
    replacements: Dict[str, Any] = {}

    line_count = 0
    bad_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            events.append(obj)
            if obj.get("type") == "context_edit":
                target = obj.get("targetId")
                if target:
                    if obj.get("replacement") is None:
                        deleted_ids.add(target)
                    else:
                        replacements[target] = obj.get("replacement")

    messages: List[Dict[str, Any]] = []
    models: Counter = Counter()
    providers: Counter = Counter()
    compactions = 0
    custom_notices = 0
    last_assistant: Optional[Dict[str, Any]] = None

    for ev in events:
        etype = ev.get("type")
        eid = ev.get("id", "")

        if etype in ("session", "session_info"):
            if etype == "session":
                session_id = ev.get("id", session_id)
                cwd = ev.get("cwd", cwd)
                session_ts = ev.get("timestamp", session_ts)
            continue
        if etype == "model_change":
            current_model = ev.get("modelId") or current_model
            current_provider = ev.get("provider") or current_provider
            continue
        if etype == "thinking_level_change":
            thinking_level = ev.get("thinkingLevel") or thinking_level
            continue
        if etype == "compaction":
            compactions += 1
            continue
        if etype == "custom_message":
            custom_notices += 1
            continue
        if etype != "message":
            continue
        if eid in deleted_ids:
            continue

        msg = replacements.get(eid, ev.get("message", {}))
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content", [])
        ts = ev.get("timestamp", "")

        if role == "user":
            text = _text_from_content(content).strip()
            if not text:
                continue
            messages.append(
                {
                    "id": eid or f"user_{len(messages)}",
                    "role": "user",
                    "content": text,
                    "timestamp": ts,
                }
            )
            last_assistant = None
            continue

        if role == "system":
            text = _text_from_content(content).strip()
            if text:
                messages.append(
                    {"id": eid or f"sys_{len(messages)}", "role": "system", "content": text, "timestamp": ts}
                )
            continue

        if role == "assistant":
            model = msg.get("model") or current_model
            provider = msg.get("provider") or current_provider
            error_message = msg.get("errorMessage")
            if error_message and not content:
                continue

            think_parts, text_parts, tool_calls = [], [], []
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    ctype = item.get("type")
                    if ctype == "thinking":
                        t = item.get("thinking", "")
                        if t:
                            think_parts.append(t)
                    elif ctype == "text":
                        t = item.get("text", "")
                        if t:
                            text_parts.append(t)
                    elif ctype == "toolCall":
                        tool_calls.append(
                            {
                                "id": item.get("id") or f"call_{eid}_{len(tool_calls)}",
                                "name": item.get("name", "unknown"),
                                "arguments": item.get("arguments", {}) or {},
                            }
                        )
            elif isinstance(content, str):
                text_parts.append(content)

            thinking = "\n".join(think_parts).strip()
            text = "\n".join(text_parts).strip()
            if not thinking and not text and not tool_calls:
                continue

            entry = {
                "id": eid or f"asst_{len(messages)}",
                "role": "assistant",
                "model": model,
                "provider": provider,
                "thinking_level": thinking_level,
                "thinking": thinking,
                "text": text,
                "tool_calls": tool_calls,
                "tool_results": [],
                "stop_reason": msg.get("stopReason", ""),
                "timestamp": ts,
            }
            messages.append(entry)
            models[model] += 1
            providers[provider] += 1
            last_assistant = entry
            continue

        if role == "toolResult":
            tr_texts = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            tr_texts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            tr_texts.append("[Image Output]")
            elif isinstance(content, str):
                tr_texts.append(content)
            raw_text = "\n".join(tr_texts)
            record = {
                "id": eid or f"tr_{len(messages)}",
                "role": "tool",
                "tool_call_id": msg.get("toolCallId", ""),
                "name": msg.get("toolName", "tool"),
                "isError": bool(msg.get("isError", False)),
                "content": truncate_middle(raw_text, max_tool_output_chars),
                "raw_len": len(raw_text),
                "timestamp": ts,
            }
            messages.append(record)
            if last_assistant is not None:
                last_assistant["tool_results"].append(record)

    return {
        "session_id": session_id,
        "cwd": cwd,
        "timestamp": session_ts,
        "line_count": line_count,
        "bad_lines": bad_lines,
        "messages": messages,
        "models": dict(models),
        "providers": dict(providers),
        "platform_counts": {"compactions": compactions, "custom_notices": custom_notices},
    }


def truncate_middle(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n\n... [truncated {omitted} chars] ...\n\n{text[-tail:]}"


class Flows:
    """编译后的回合过滤规则（避免每回合重复编译正则）。"""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        tcfg = cfg.get("turns", {})

        def comp(patterns):
            return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in (patterns or []) if p]

        self.root = comp(tcfg.get("root_task_patterns"))
        self.irrelevant = comp(tcfg.get("irrelevant_patterns"))
        self.whitelist = comp(tcfg.get("irrelevant_whitelist_patterns"))
        self.continuation = comp(tcfg.get("continuation_patterns"))
        self.correction = comp(tcfg.get("correction_patterns"))
        self.fallback_root = bool(tcfg.get("fallback_first_turn_as_root", True))
        self.fold_await = bool(tcfg.get("fold_await_user_messages", True))
        self.normalize_root = bool(tcfg.get("normalize_root_prompt", True))
        self.root_norms: Dict[str, str] = dict(tcfg.get("root_normalizations") or {})

    def _any(self, patterns, text: str) -> bool:
        return any(p.search(text) for p in patterns)

    def is_root_task(self, text: str) -> bool:
        return self._any(self.root, text)

    def is_noise(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) <= 1:
            return True
        if self.is_root_task(t):
            return False
        if self.whitelist and self._any(self.whitelist, t):
            return False
        if ("Error: 400" in t or "Error: 502" in t or "Error: 503" in t) and "继续" in t and len(t) > 60:
            return True
        return self._any(self.irrelevant, t) if self.irrelevant else False

    def is_continuation(self, text: str) -> bool:
        return self._any(self.continuation, (text or "").strip()) if self.continuation else False

    def is_correction(self, text: str) -> bool:
        return self._any(self.correction, (text or "").strip()) if self.correction else False

    def normalize_root_text(self, text: str, cwd: str = "") -> str:
        if not self.normalize_root or not self.root_norms:
            return text
        probe = f"{text}\n{cwd}".lower()
        for key, target in self.root_norms.items():
            if key.lower() in probe and target:
                return target
        return text


API_ERROR_MARKERS = re.compile(
    r"Error:\s*(400|401|403|404|429|500|502|503)|invalid_request_error|rate limit|"
    r"APIConnectionError|ECONNREFUSED|fetch failed",
    re.IGNORECASE,
)


def iter_blocks(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """按 user 消息切块（每块 = 1 条 user + 其后所有 assistant/tool）。"""
    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            if current:
                blocks.append(current)
            current = [copy.deepcopy(m)]
        elif current:
            current.append(copy.deepcopy(m))
    if current:
        blocks.append(current)
    return blocks


def _steps(block: List[Dict[str, Any]]) -> int:
    return sum(1 for m in block[1:] if m["role"] == "assistant")


def detect_root_regex(blocks: List[List[Dict[str, Any]]], rules: "Flows") -> Optional[int]:
    """正则找根任务：先找明确破解指令，再退回「首个有实质执行的非噪音回合」。"""
    for idx, block in enumerate(blocks):
        if _steps(block) > 0 and rules.is_root_task(block[0]["content"].strip()):
            return idx
    if not rules.fallback_root:
        return None
    for idx, block in enumerate(blocks):
        text = block[0]["content"].strip()
        if _steps(block) < 3:
            continue
        if API_ERROR_MARKERS.search(text):
            continue
        if rules.is_noise(text):
            continue
        return idx
    return None


def build_flow(
    session: Dict[str, Any],
    cfg: Dict[str, Any],
    annotations: Optional[Dict[str, Any]] = None,
    manual_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """回合成流：剥离前置/中途噪音、折叠催促、内化纠偏，支持人工覆盖。

    优先级：manual_overrides > annotations (model) > regex rules。
    返回 (flow_messages, turn_marks, stats)
    """
    rules = Flows(cfg)
    session_id = session["session_id"]
    cwd = session.get("cwd", "")
    blocks = iter_blocks(session["messages"])
    manual_overrides = manual_overrides or {}
    manual_turns: Dict[int, Dict[str, Any]] = {}
    for k, v in (manual_overrides.get("turns") or {}).items():
        try:
            manual_turns[int(k)] = v if isinstance(v, dict) else {"kind": str(v)}
        except (TypeError, ValueError):
            continue

    ann_map: Dict[int, Dict[str, Any]] = {}
    root_idx: Optional[int] = None
    if annotations:
        for t in annotations.get("turns", []):
            try:
                ann_map[int(t["index"])] = t
            except (KeyError, TypeError, ValueError):
                continue
        root_idx = annotations.get("root_turn")
        try:
            root_idx = int(root_idx) if root_idx is not None else None
        except (TypeError, ValueError):
            root_idx = None
        if root_idx is not None and (root_idx < 0 or root_idx >= len(blocks) or _steps(blocks[root_idx]) == 0):
            root_idx = None

    # 若人工显式指定了某个回合为 root，以人工指定为准
    manual_roots = [idx for idx, mt in sorted(manual_turns.items()) if mt.get("kind") == "root" and 0 <= idx < len(blocks)]
    if manual_roots:
        root_idx = manual_roots[0]
    elif root_idx is not None and manual_turns.get(root_idx, {}).get("kind") in ("setup", "noise"):
        # 人工把原 root 改成了噪音/准备，重新向后找首个非噪音回合
        root_idx = None

    if root_idx is None:
        root_idx = detect_root_regex(blocks, rules)

    flow: List[Dict[str, Any]] = [
        {
            "id": f"sys_{session_id}",
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
            "timestamp": session.get("timestamp", ""),
        }
    ]
    marks: List[Dict[str, Any]] = []
    constraints: List[str] = []
    raw_root_prompt = ""
    root_prompt = ""
    drop_counts: Counter = Counter()
    dropped_steps_total = 0

    def _pop_trailing_await() -> bool:
        if (
            rules.fold_await
            and len(flow) > 1
            and flow[-1]["role"] == "assistant"
            and not flow[-1].get("tool_calls")
        ):
            flow.pop()
            return True
        return False

    def _append_steps_from_folded(block_msgs: List[Dict[str, Any]], turn_idx: int, fold_type: str) -> None:
        first_asst = True
        for m in block_msgs:
            m["turn_index"] = turn_idx
            if m["role"] == "assistant" and first_asst:
                m["folded_turn_index"] = turn_idx
                m["folded_turn_type"] = fold_type
                first_asst = False
            flow.append(m)

    for idx, block in enumerate(blocks):
        user = block[0]
        text = user["content"].strip()
        asst = [m for m in block[1:] if m["role"] == "assistant"]
        man = manual_turns.get(idx)
        ann = ann_map.get(idx, {})
        source = "manual" if man else ("model" if ann else "rule")
        eff_kind = (man or {}).get("kind") or ann.get("kind")

        mark: Dict[str, Any] = {
            "turn_index": idx,
            "user_id": user["id"],
            "text": text[:1200],
            "text_preview": text[:160],
            "kind": eff_kind or "task",
            "action": "kept",
            "dropped": False,
            "folded": False,
            "reason": (man or {}).get("why") or ann.get("why", ""),
            "assistant_steps": len(asst),
            "kept_steps": len(asst),
            "popped_await_msg": False,
            "source": source,
        }

        if not asst:
            mark.update(action="dropped", dropped=True, kept_steps=0, kind=eff_kind or "noise", reason=mark["reason"] or "empty_turn")
            drop_counts["empty_turn"] += 1
            marks.append(mark)
            continue

        if not man and idx + 1 < len(blocks) and len(asst) <= 3 and blocks[idx + 1][0]["content"].strip() == text:
            if idx == root_idx:
                root_idx = idx + 1
                mark.update(action="dropped", dropped=True, kept_steps=0, kind="setup", reason="root_retry_moved_to_next")
            else:
                mark.update(action="dropped", dropped=True, kept_steps=0, kind="noise", reason="duplicate_retry_turn")
            drop_counts["duplicate_retry_turn"] += 1
            dropped_steps_total += len(asst)
            marks.append(mark)
            continue

        drop = bool((man or {}).get("drop")) if man and "drop" in man else bool(ann.get("drop"))
        if man and man.get("kind") in ("root", "task", "continuation", "correction"):
            drop = False
        elif man and man.get("kind") in ("setup", "noise"):
            drop = True

        if root_idx is not None and idx < root_idx and not (man and man.get("kind") in ("task", "root")):
            mark.update(
                action="dropped",
                dropped=True,
                kept_steps=0,
                reason=mark["reason"] or "pre_root_setup",
                kind=eff_kind if eff_kind in ("setup", "noise") else "setup",
            )
            drop_counts["pre_root"] += 1
            dropped_steps_total += len(asst)
            marks.append(mark)
            continue

        if root_idx is not None and idx == root_idx:
            raw_root_prompt = text
            custom_root = (manual_overrides.get("root_prompt") or "").strip()
            root_prompt = custom_root or rules.normalize_root_text(text, cwd)
            user["raw_content"] = text
            user["content"] = root_prompt
            mark.update(
                kind="root",
                action="kept",
                dropped=False,
                folded=False,
                reason=mark["reason"] or ("root_normalized" if root_prompt != text else "root_task"),
                normalized_prompt=root_prompt if root_prompt != text else None,
            )
            for m in block:
                m["turn_index"] = idx
                flow.append(m)
            marks.append(mark)
            continue

        if root_idx is None and rules.is_noise(text) and eff_kind is None:
            mark.update(action="dropped", dropped=True, kept_steps=0, reason="noise_no_root", kind="noise")
            drop_counts["noise_no_root"] += 1
            dropped_steps_total += len(asst)
            marks.append(mark)
            continue

        if eff_kind == "correction" and source == "model" and not rules.is_correction(text) and len(text) > 200:
            eff_kind = "task"

        if eff_kind == "correction" or (eff_kind is None and rules.is_correction(text)):
            constraints.append(text)
            popped = _pop_trailing_await()
            _append_steps_from_folded(block[1:], idx, "correction")
            mark.update(
                kind="correction",
                action="internalized",
                dropped=False,
                folded=True,
                popped_await_msg=popped,
                reason=mark["reason"] or "correction_internalized",
            )
            drop_counts["correction_internalized"] += 1
            marks.append(mark)
            continue

        if eff_kind == "continuation" or (eff_kind is None and rules.is_continuation(text)):
            popped = _pop_trailing_await()
            _append_steps_from_folded(block[1:], idx, "continuation")
            mark.update(
                kind="continuation",
                action="folded",
                dropped=False,
                folded=True,
                popped_await_msg=popped,
                reason=mark["reason"] or "folded_into_mainline",
            )
            drop_counts["continuation_folded"] += 1
            marks.append(mark)
            continue

        if drop or eff_kind in ("noise", "setup"):
            k = eff_kind if eff_kind in ("noise", "setup") else "noise"
            mark.update(
                kind=k,
                action="dropped",
                dropped=True,
                kept_steps=0,
                reason=mark["reason"] or f"{k}_dropped",
            )
            drop_counts[f"{k}_dropped"] += 1
            dropped_steps_total += len(asst)
            marks.append(mark)
            continue

        if eff_kind is None and rules.is_noise(text):
            mark.update(
                kind="noise",
                action="dropped",
                dropped=True,
                kept_steps=0,
                reason="mid_session_noise",
            )
            drop_counts["mid_session_noise"] += 1
            dropped_steps_total += len(asst)
            marks.append(mark)
            continue

        mark.update(kind=eff_kind or "task", action="kept", dropped=False, folded=False)
        for m in block:
            m["turn_index"] = idx
            flow.append(m)
        marks.append(mark)

    if constraints:
        flow[0]["content"] = flow[0]["content"] + "\n5. " + "；".join(
            c.strip() for c in constraints
        )

    stats = {
        "turns_total": len(blocks),
        "turns_kept": sum(1 for m in marks if not m["dropped"]),
        "turns_direct": sum(1 for m in marks if m["action"] == "kept"),
        "turns_folded": sum(1 for m in marks if m["action"] in ("folded", "internalized")),
        "turns_dropped": sum(1 for m in marks if m["action"] == "dropped"),
        "dropped_steps": dropped_steps_total,
        "dropped": dict(drop_counts),
        "root_prompt": root_prompt,
        "raw_root_prompt": raw_root_prompt,
        "root_normalized": bool(root_prompt and raw_root_prompt and root_prompt != raw_root_prompt),
        "root_index": root_idx,
        "constraints": constraints,
        "has_root": bool(root_prompt),
        "model_classified": bool(annotations),
        "manual_overridden": bool(manual_turns or manual_overrides.get("root_prompt")),
        "flow_messages": len(flow),
    }
    return flow, marks, stats
