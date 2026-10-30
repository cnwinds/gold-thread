#!/usr/bin/env python3
"""训练前「工具调用契约」自检：数据集里的 tool_call 写法必须和 chat template / 推理端 parser 一致。

为什么需要它（2026-10 27B 事故）：
    数据集里的工具调用是 hermes JSON 风格：
        <tool_call>
        {"name": "bash", "arguments": {"command": "ls"}}
        </tool_call>
    而 27B 用的 `template: qwen3_8` 会在 system prompt 里注入 coder XML 风格指令：
        <tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>
    两边打架 → 模型推理时随机二选一 → 一半工具调用服务端 parser 解析不了，直接停工。
    8B 用的 `template: qwen3` 注入的是 JSON 指令，所以 8B 没出问题。

用法：
    # 1) 让脚本自己从模型目录读 chat_template 判断应该用哪种风格
    python check_toolcall_contract.py --model /data/models/Qwen3.8-27B

    # 2) 手动指定（没有模型目录时）
    python check_toolcall_contract.py --style hermes-json
    python check_toolcall_contract.py --style coder-xml

    # 3) 直接给一份 chat_template 文本
    python check_toolcall_contract.py --chat-template-file chat_template.jinja

退出码：0 一致；1 不一致或有非法块；2 参数/文件问题。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 两种主流工具调用渲染
JSON_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
XML_BLOCK_RE = re.compile(r"<tool_call>\s*(<function=.*?</function>)\s*</tool_call>", re.S)
XML_STRAY_RE = re.compile(r"<function=|<parameter=")

STYLES = ("hermes-json", "coder-xml")

# chat_template 里能识别的特征串
XML_TEMPLATE_HINTS = ("<function=", "<parameter=", "must be nested within <tool_call>")
JSON_TEMPLATE_HINTS = ("return a json object", '"name": "example_function_name"', "json object with function name")


def detect_style_from_template(text: str) -> str | None:
    xml = any(h in text for h in XML_TEMPLATE_HINTS)
    js = any(h in text.lower() for h in (h.lower() for h in JSON_TEMPLATE_HINTS))
    if xml and not js:
        return "coder-xml"
    if js and not xml:
        return "hermes-json"
    return None  # 两种都命中或都没命中，让人来判


def load_template(model_dir: Path | None, template_file: Path | None) -> str:
    if template_file:
        return template_file.read_text(encoding="utf-8")
    if not model_dir:
        return ""
    for name in ("chat_template.jinja", "chat_template.json"):
        p = model_dir / name
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            if name.endswith(".json"):
                try:
                    obj = json.loads(raw)
                    return obj.get("chat_template") or obj.get("default", "")
                except json.JSONDecodeError:
                    pass
            return raw
    cfg = model_dir / "tokenizer_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("chat_template") or ""
        except json.JSONDecodeError:
            return ""
    return ""


def tool_names(tools_field) -> set[str]:
    if not tools_field:
        return set()
    if isinstance(tools_field, str):
        try:
            tools_field = json.loads(tools_field)
        except json.JSONDecodeError:
            return set()
    out = set()
    for t in tools_field or []:
        fn = (t or {}).get("function") or {}
        if fn.get("name"):
            out.add(fn["name"])
    return out


def assistant_texts(row: dict):
    """产出 (位置标签, 助手文本)；兼容 sharegpt / openai 两种格式。"""
    for c in row.get("conversations") or []:
        if c.get("from") == "gpt":
            yield "gpt", c.get("value") or ""
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        # openai 格式：tool_calls 是结构化的，不会踩文本硬编码的坑
        if m.get("tool_calls"):
            yield "openai-tool_calls", ""
            continue
        yield "assistant", m.get("content") or ""


def scan_file(path: Path, style: str) -> dict:
    stats = {"rows": 0, "texts": 0, "ok": 0, "bad": [], "wrong_style": 0, "stray_xml": 0}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        stats["rows"] += 1
        names = tool_names(row.get("tools"))
        for where, text in assistant_texts(row):
            if not text.strip():
                continue
            stats["texts"] += 1
            if XML_STRAY_RE.search(text):
                stats["stray_xml"] += 1
                stats["bad"].append((line_no, where, "出现 <function=/<parameter=> 裸 XML", text[:160]))
            # DPO 文件的 chosen/rejected 用字符串承载整轮对话
            for key in ("chosen", "rejected"):
                pass
            blocks = JSON_BLOCK_RE.findall(text)
            xml_blocks = XML_BLOCK_RE.findall(text)
            if style == "hermes-json":
                if xml_blocks:
                    stats["wrong_style"] += 1
                    stats["bad"].append((line_no, where, "应是 hermes JSON，实际是 coder XML", xml_blocks[0][:160]))
                for b in blocks:
                    try:
                        obj = json.loads(b)
                    except json.JSONDecodeError as exc:
                        stats["bad"].append((line_no, where, f"JSON 解析失败: {exc}", b[:160]))
                        continue
                    if not isinstance(obj.get("name"), str):
                        stats["bad"].append((line_no, where, "缺 name", b[:160]))
                    elif names and obj["name"] not in names:
                        stats["bad"].append((line_no, where, f"name 不在 tools 里: {obj['name']}", b[:160]))
                    if not isinstance(obj.get("arguments"), dict):
                        stats["bad"].append((line_no, where, "arguments 不是 object", b[:160]))
                        continue
                    stats["ok"] += 1
            else:  # coder-xml
                if blocks:
                    stats["wrong_style"] += 1
                    stats["bad"].append((line_no, where, "应是 coder XML，实际是 hermes JSON", blocks[0][:160]))
                for b in xml_blocks:
                    if "<parameter=" not in b:
                        stats["bad"].append((line_no, where, "XML 块没有 <parameter=>", b[:160]))
                        continue
                    stats["ok"] += 1
    return stats


def dpo_style_split(path: Path) -> tuple[int, int]:
    """DPO 里 chosen/rejected 的「有调用/无调用」是否一致（不一致=在教格式偏好而不是质量偏好）。"""
    diff = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        c, j = row.get("chosen"), row.get("rejected")
        if c is None or j is None:
            continue
        c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        j = j if isinstance(j, str) else json.dumps(j, ensure_ascii=False)
        total += 1
        if ("<tool_call>" in c) != ("<tool_call>" in j):
            diff += 1
    return total, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "data" / "03_train"))
    ap.add_argument("--model", help="模型目录（读 tokenizer_config.json 的 chat_template）")
    ap.add_argument("--chat-template-file")
    ap.add_argument("--style", choices=STYLES, help="数据集应当使用的风格；不给则从模板推断")
    ap.add_argument("--glob", default="*.jsonl")
    args = ap.parse_args()

    tpl = load_template(Path(args.model) if args.model else None,
                        Path(args.chat_template_file) if args.chat_template_file else None)
    detected = detect_style_from_template(tpl) if tpl else None
    style = args.style or detected
    if style is None:
        print("[err] 无法从模板推断风格（模板里没有明确指令）。请用 --style 指定。", file=sys.stderr)
        return 2
    print(f"数据集应使用的工具调用风格：{style}"
          + (f"（来自模板推断，长度 {len(tpl)} 字符）" if detected and not args.style else "（手动指定）"))
    if args.style and detected and args.style != detected:
        print(f"[warn] 模板实际推断为 {detected}，与你指定的 {args.style} 不一致 —— 这正是事故形态。")

    root = Path(args.dir)
    files = [p for p in sorted(root.glob(args.glob))
             if "diagnostics" not in p.name and "logs" not in p.parts]
    if not files:
        print(f"[err] {root} 下没有匹配 {args.glob} 的文件", file=sys.stderr)
        return 2

    total_bad = 0
    print(f"\n{'文件':<52}{'行数':>7}{'调用块':>8}{'OK':>7}{'错风格':>8}{'裸XML':>7}")
    for p in files:
        st = scan_file(p, style)
        total_bad += len(st["bad"]) + st["wrong_style"]
        print(f"{p.name:<52}{st['rows']:>7}{st['texts']:>8}{st['ok']:>7}{st['wrong_style']:>8}{st['stray_xml']:>7}")
        for item in st["bad"][:3]:
            print(f"    L{item[0]} [{item[1]}] {item[2]}: {item[3]!r}")

    print("\nDPO chosen/rejected 一致性：")
    for p in files:
        if "dpo" not in p.name:
            continue
        total, diff = dpo_style_split(p)
        if total:
            flag = "ok" if diff == 0 else f"注意：{diff} 对只有一侧带 tool_call（可能在教格式偏好）"
            print(f"  {p.name:<52} {total:>5} 对  {flag}")

    print("\n结论：" + ("通过" if total_bad == 0 else f"发现 {total_bad} 处不一致，训练前必须先统一"))
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())