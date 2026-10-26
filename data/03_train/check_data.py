#!/usr/bin/env python3
"""训练前自检：数据结构 + 对话模板渲染（防 think 标签重复注入）。

用法：
    python check_data.py --model /data/models/Qwen3.8-27B --template qwen3
    python check_data.py --drop-tools          # 框架不支持 tools 列时先剥离
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THINK_RE = re.compile(r"</?think>")
TOOL_RE = re.compile(r"<tool_call>")


def load(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--model")
    ap.add_argument("--template", default="qwen")
    ap.add_argument("--drop-tools", action="store_true")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.dir)

    if args.drop_tools:
        n = 0
        for p in root.glob("*.jsonl"):
            if "diagnostics" in p.name:
                continue
            rows = list(load(p))
            for r in rows:
                r.pop("tools", None)
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        print(f"[ok] 已剥离 tools 列：{n} 个文件")
        return 0

    # 1) 结构检查
    files = sorted(p for p in root.glob("*.jsonl") if "diagnostics" not in p.name)
    for p in files:
        rows = list(load(p))
        bad = [r for r in rows if not r.get("conversations") and not r.get("messages")]
        print(f"  {p.name:<48} {len(rows):>6} 条  结构异常 {len(bad)}")
    train = next((root / x for x in (
        "sft_step_windows_sharegpt.train.v2.jsonl",
        "sft_step_windows_sharegpt.train.jsonl") if (root / x).exists()), None)
    if train is None:
        print("[err] 找不到训练集（v2 / v1 均无）", file=sys.stderr)
        return 1
    print(f"  目标数据集: {train.name}")

    # 2) 目标步检查
    rows = list(load(train))[: args.samples]
    problems = 0
    for r in rows:
        conv = r["conversations"]
        if conv[0]["from"] != "system" or conv[1]["from"] != "human":
            problems += 1
            continue
        if conv[-1]["from"] != "gpt":
            problems += 1
            continue
        if len(THINK_RE.findall(conv[-1]["value"])) != 2:
            problems += 1
    print(f"\n目标步结构（抽样 {len(rows)}）：异常 {problems} 条")

    # 3) 模板渲染检查
    if not args.model:
        print("\n（跳过模板渲染检查：未提供 --model）")
        return 0 if problems == 0 else 1
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[err] 需要 transformers：pip install transformers", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    msgs_pool = []
    for r in rows[:5]:
        conv = r["conversations"]
        msgs = []
        for c in conv[:-1]:
            role = {"system": "system", "human": "user", "gpt": "assistant",
                    "observation": "tool", "tool": "tool"}.get(c["from"], "user")
            msgs.append({"role": role, "content": c["value"]})
        msgs.append({"role": "assistant", "content": conv[-1]["value"]})
        msgs_pool.append((r["sample_id"], msgs))

    print("\n模板渲染检查：")
    for sid, msgs in msgs_pool:
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {sid[:14]} 模板渲染失败: {exc}")
            continue
        opens = text.count(" thinking")
        closes = text.count("<" + "/think>")
        tools = TOOL_RE.findall(text)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        flag = "ok" if opens == 1 and closes == 1 else "检查"
        print(f"  [{flag}] {sid[:14]} tokens≈{n_tok:<6}  thinking×{opens} ×{closes} <tool_call>×{len(tools)}")
        if opens != 1 or closes != 1:
            print("         ↑ think 标记不等于 1 对：模板可能自带 thinking 注入，"
                  "请改用不注入的模板（或把数据里的 thinking 去掉）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
