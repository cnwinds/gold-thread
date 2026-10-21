"""训练样本格式化：openai messages / sharegpt conversations。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def to_openai(
    m: Dict[str, Any],
    override_thinking: Optional[str] = None,
    strip_thinking: bool = False,
) -> Dict[str, Any]:
    role = m["role"]
    if role in ("system", "user"):
        return {"role": role, "content": m["content"]}
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.get("tool_call_id", ""),
            "name": m.get("name", "tool"),
            "content": m.get("content", ""),
        }
    if role == "assistant":
        thinking = "" if strip_thinking else (
            override_thinking if override_thinking is not None else m.get("thinking", "")
        )
        text = m.get("text", "")
        content = f"<think\n{thinking}\n</think>\n\n{text}".strip() if thinking else text
        out: Dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            out["reasoning_content"] = thinking
        if m.get("tool_calls"):
            out["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for tc in m["tool_calls"]
            ]
        return out
    return {"role": "user", "content": str(m)}


def to_sharegpt(
    m: Dict[str, Any],
    override_thinking: Optional[str] = None,
    strip_thinking: bool = False,
) -> List[Dict[str, str]]:
    role = m["role"]
    if role == "system":
        return [{"from": "system", "value": m["content"]}]
    if role == "user":
        return [{"from": "human", "value": m["content"]}]
    if role == "tool":
        return [{"from": "observation", "value": m.get("content", "")}]
    if role != "assistant":
        return []

    thinking = "" if strip_thinking else (
        override_thinking if override_thinking is not None else m.get("thinking", "")
    )
    text = m.get("text", "")
    prefix = f"<think\n{thinking}\n</think>\n\n" if thinking else ""
    if m.get("tool_calls"):
        blocks = [
            "<tool_call>\n"
            + json.dumps({"name": tc["name"], "arguments": tc.get("arguments", {})}, ensure_ascii=False)
            + "\n</tool_call>"
            for tc in m["tool_calls"]
        ]
        return [{"from": "gpt", "value": f"{prefix}{text}\n" + "\n".join(blocks).strip()}]
    return [{"from": "gpt", "value": f"{prefix}{text}".strip()}]


def conversation(
    messages: List[Dict[str, Any]],
    fmt: str = "sharegpt",
    strip_thinking: bool = True,
    override: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    override = override or {}
    out: List[Dict[str, Any]] = []
    for m in messages:
        if fmt == "sharegpt":
            out.extend(
                to_sharegpt(
                    m,
                    override_thinking=override.get(m.get("id")),
                    strip_thinking=strip_thinking,
                )
            )
        else:
            msg = to_openai(
                m,
                override_thinking=override.get(m.get("id")),
                strip_thinking=strip_thinking,
            )
            out.append(msg)
    return out
