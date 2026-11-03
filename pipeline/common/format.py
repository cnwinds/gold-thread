"""训练样本格式化：openai messages / sharegpt conversations。

工具调用渲染风格（与 tools/check_toolcall_contract.py 的命名保持一致）：
- hermes-json: <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>       (qwen3 / 8B 线)
- coder-xml:   <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>  (qwen3_8 / 27B 线)

三方一致铁律（见 DATASET_BEST_PRACTICES.md §5）：数据渲染格式 == 训练 chat template == 推理 --tool-call-parser。
基座/中立数据用 openai 结构化 tool_calls（to_openai），风格由渲染层决定（本模块的 tool_style）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 工具调用风格注册表（SSoT：渲染/检测/转换只用这里定义的两种实现）
# ---------------------------------------------------------------------------
TOOL_CALL_STYLES = ("hermes-json", "coder-xml")


def _render_tc_hermes(name: str, arguments: Dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        + "\n</tool_call>"
    )


def _render_tc_coder_xml(name: str, arguments: Dict[str, Any]) -> str:
    params = "".join(
        f"<parameter={k}>\n{(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}\n</parameter>\n"
        for k, v in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


TOOL_CALL_RENDERERS = {
    "hermes-json": _render_tc_hermes,
    "coder-xml": _render_tc_coder_xml,
}

_HERMES_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_XML_FUNC_RE = re.compile(r"<tool_call>\s*<function=([^>\n]*)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
# 渲染器在值两侧各加一个换行。只剥这一层，值自身的首尾空格和换行要留下。
# 用 \s* 会把 write 正文结尾的换行、命令行结尾的空格一起吃掉。
_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]*)>\n?(.*?)\n?</parameter>", re.S)
_ANY_TC_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _parse_tc_block(block: str) -> Optional[Dict[str, Any]]:
    """把一个 <tool_call>…</tool_call> 块解析成 {name, arguments}；两种风格都认。失败返回 None。"""
    m = _HERMES_BLOCK_RE.fullmatch(block.strip()) or re.fullmatch(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", block.strip(), re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return {
                "name": obj.get("name"),
                "arguments": obj.get("arguments", {}),
            }
        except Exception:
            return None
    m = re.fullmatch(
        r"<tool_call>\s*<function=([^>\n]*)>\s*(.*?)\s*</function>\s*</tool_call>",
        block.strip(),
        re.S,
    )
    if m:
        name = m.group(1).strip()
        args: Dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            raw = pm.group(2)
            try:
                args[key] = json.loads(raw)  # 数字/数组/对象/布尔还原
            except Exception:
                args[key] = raw
        return {"name": name, "arguments": args}
    return None


def convert_toolcall_text(text: str, target_style: str) -> tuple[str, int, int]:
    """把文本里所有可解析的 <tool_call> 块重渲染为 target_style。
    返回 (新文本, 转换块数, 不可解析块数——保持原样)。
    注意：只应作用于 assistant 侧文本（gpt/chosen/rejected），不要对 observation 内容使用。"""
    if target_style not in TOOL_CALL_RENDERERS:
        raise ValueError(f"unknown tool_style: {target_style!r} (expected {TOOL_CALL_STYLES})")
    render = TOOL_CALL_RENDERERS[target_style]
    converted = unparsed = 0

    def _sub(m: re.Match) -> str:
        nonlocal converted, unparsed
        tc = _parse_tc_block(m.group(0))
        if tc is None or not tc.get("name"):
            unparsed += 1
            return m.group(0)
        converted += 1
        return render(tc["name"], tc.get("arguments", {}))

    return _ANY_TC_RE.sub(_sub, text), converted, unparsed


def detect_tool_style(text: str) -> Optional[str]:
    """探测文本中 <tool_call> 块的风格；混合/无法识别返回 None。"""
    styles = set()
    for m in _ANY_TC_RE.finditer(text):
        block = m.group(0)
        if _HERMES_BLOCK_RE.fullmatch(block.strip()):
            styles.add("hermes-json")
        elif _XML_FUNC_RE.fullmatch(block.strip()):
            styles.add("coder-xml")
        else:
            return None
    if not styles:
        return None
    return styles.pop() if len(styles) == 1 else None


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
        thinking = (thinking or "").strip()
        text = (m.get("text") or "").strip()
        content = f"<think>\n{thinking}\n</think>\n\n{text}".strip() if thinking else text
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
    tool_style: str = "hermes-json",
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
    thinking = (thinking or "").strip()
    text = (m.get("text") or "").strip()
    prefix = f"<think>\n{thinking}\n</think>\n\n" if thinking else ""
    if m.get("tool_calls"):
        render = TOOL_CALL_RENDERERS[tool_style]
        blocks = [render(tc["name"], tc.get("arguments", {})) for tc in m["tool_calls"]]
        body = f"{text}\n" if text else ""
        return [{"from": "gpt", "value": f"{prefix}{body}" + "\n".join(blocks).strip()}]
    return [{"from": "gpt", "value": f"{prefix}{text}".strip()}]


# 一个导出习惯 = 数据风格 + 训练 template + 推理 parser。三处必须一起变。
# id 与 tools/export_toolstyle.py 的 --profile 保持一致；别名只为界面和命令行省事。
DEFAULT_EXPORT_PROFILE = "Qwen3.8-27B"

EXPORT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Qwen3.8-27B": {
        "label": "Qwen XML",
        "habit": "Qwen3-Coder / Qwen3.8-27B",
        "summary": "Qwen3-Coder 与 Qwen3.8-27B 的工具调用是整段 XML。打包时只改写 assistant 侧的调用，并把 27B 配置的 template 锁成 qwen3_8。8B 的 yaml 留在仓库里，不进这个包。",
        "style": "coder-xml",
        "template": "qwen3_8",
        "parser": "qwen3_coder",
        "pack_8b_yaml": False,
        "default": True,
        "example": (
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\n"
            "ls -la\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    },
    "Qwen3.8-8B": {
        "label": "Hermes JSON",
        "habit": "Qwen3 / 8B",
        "summary": "Qwen3 与 8B 在 <tool_call> 里写 JSON。选这个格式时，27B 配置的 template 会锁成 qwen3，和数据同一套协议。8B 配置会一起打进包。",
        "style": "hermes-json",
        "template": "qwen3",
        "parser": "hermes",
        "pack_8b_yaml": True,
        "default": False,
        "example": (
            "<tool_call>\n"
            '{"name": "bash", "arguments": {"command": "ls -la"}}\n'
            "</tool_call>"
        ),
    },
}

_PROFILE_ALIASES = {
    "qwen-xml": "Qwen3.8-27B",
    "coder-xml": "Qwen3.8-27B",
    "hermes-json": "Qwen3.8-8B",
    "hermes": "Qwen3.8-8B",
}


def resolve_export_profile(name: Optional[str] = None) -> Dict[str, Any]:
    """返回带 id 的 profile。空名字用 Qwen XML（27B 的习惯）。"""
    key = (name or "").strip() or DEFAULT_EXPORT_PROFILE
    key = _PROFILE_ALIASES.get(key, key)
    src = EXPORT_PROFILES.get(key)
    if not src:
        known = ", ".join([*EXPORT_PROFILES, *_PROFILE_ALIASES])
        raise ValueError(f"未知导出格式: {name}（可选：{known}）")
    return {"id": key, **src}


def export_profiles_public() -> List[Dict[str, Any]]:
    """给导出界面的字段。不含打包内部细节以外的可展示项。"""
    out: List[Dict[str, Any]] = []
    for pid, src in EXPORT_PROFILES.items():
        out.append({
            "id": pid,
            "label": src["label"],
            "habit": src["habit"],
            "summary": src["summary"],
            "style": src["style"],
            "template": src["template"],
            "parser": src["parser"],
            "example": src["example"],
            "pack_8b_yaml": src["pack_8b_yaml"],
            "default": src["default"],
        })
    return out


def profile_includes_yaml(name: str, profile: Dict[str, Any]) -> bool:
    """Qwen XML 不带 *qwen3_8b.yaml：那些文件的 template 是 qwen3。"""
    if "qwen3_8b" in name and not profile.get("pack_8b_yaml"):
        return False
    return True


def convert_sharegpt_row(row: Dict[str, Any], style: str) -> tuple[int, int]:
    """把一行里 assistant 侧的 <tool_call> 重渲染为 style。

    只动 from=gpt 的消息，以及 DPO 的 chosen/rejected。
    observation / human / system 里出现的同名文本是工具输出，保持原样。
    返回 (转换块数, 无法识别的块数)。
    """
    converted = unparsed = 0

    def _take(holder: Dict[str, Any]) -> None:
        nonlocal converted, unparsed
        value = holder.get("value")
        if not isinstance(value, str) or "<tool_call>" not in value:
            return
        new, c, u = convert_toolcall_text(value, style)
        holder["value"] = new
        converted += c
        unparsed += u

    convs = row.get("conversations")
    if isinstance(convs, list):
        for msg in convs:
            if isinstance(msg, dict) and msg.get("from") == "gpt":
                _take(msg)
    for key in ("chosen", "rejected"):
        obj = row.get(key)
        if isinstance(obj, dict) and obj.get("from") == "gpt":
            _take(obj)
        elif isinstance(obj, str) and "<tool_call>" in obj:
            new, c, u = convert_toolcall_text(obj, style)
            row[key] = new
            converted += c
            unparsed += u
    return converted, unparsed


def conversation(
    messages: List[Dict[str, Any]],
    fmt: str = "sharegpt",
    strip_thinking: bool = True,
    override: Optional[Dict[str, str]] = None,
    tool_style: str = "hermes-json",
) -> List[Dict[str, Any]]:
    override = override or {}
    out: List[Dict[str, Any]] = []
    for m in messages:
        mid = m.get("id")
        ov = override.get(mid) if mid else None
        if fmt == "sharegpt":
            out.extend(
                to_sharegpt(
                    m,
                    override_thinking=ov,
                    strip_thinking=strip_thinking,
                    tool_style=tool_style,
                )
            )
        else:
            msg = to_openai(
                m,
                override_thinking=ov,
                strip_thinking=strip_thinking,
            )
            out.append(msg)
    return out
