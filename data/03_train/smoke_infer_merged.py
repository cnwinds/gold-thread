#!/usr/bin/env python3
"""合并后模型冒烟：用 transformers 直接续写 eval 窗口，检查 think + 工具调用。

在交付包里跑（解压目录内，数据默认取 ./data/sft.eval.lf.jsonl）：
    python smoke_infer_merged.py --model /data/models/Qwen3.8-27B-merged --n 3

工具调用同时认两种风格：Qwen XML（<function=name><parameter=k>v）与 Hermes JSON。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLE = {"system": "system", "human": "user", "gpt": "assistant",
        "observation": "tool", "tool": "tool"}

# 交付包里优先用规整后的 lf 文件；老工作区回落到 stage3 原名。
_DATA_CANDIDATES = (
    "./data/sft.eval.lf.jsonl",
    "./data/sft.eval.jsonl",
    "/data/goldthread/03_train/sft_step_windows_sharegpt.eval.v2.jsonl",
)

_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.\-]+)\s*>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def default_data() -> str:
    for path in _DATA_CANDIDATES:
        if Path(path).is_file():
            return path
    return _DATA_CANDIDATES[-1]


def detect_tool_calls(text: str) -> tuple[int, int]:
    """返回 (工具调用块数, 可解析块数)。Qwen XML 与 Hermes JSON 都算。"""
    total = valid = 0
    for name in _XML_CALL.findall(text):
        total += 1
        if name:
            valid += 1
    for blob in _JSON_CALL.findall(text):
        total += 1
        try:
            payload = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        if "name" in payload and ("arguments" in payload or "parameters" in payload):
            valid += 1
    return total, valid


def to_messages(row: dict) -> list[dict]:
    """lf（messages）与 ShareGPT（conversations）两种行都接受，丢掉最后一条回答。"""
    conv = row.get("conversations")
    if conv:
        return [{"role": ROLE.get(c["from"], "user"), "content": c["value"]} for c in conv[:-1]]
    return [{"role": m["role"], "content": m["content"]} for m in (row.get("messages") or [])[:-1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3.8-27B-merged",
                    help="合并后的模型目录（SFT 或 DPO 合并产物）")
    ap.add_argument("--data", default="", help=f"eval 数据（默认 {_DATA_CANDIDATES[0]}）")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="用 Qwen3 推荐采样参数（temp .6 / top_p .95 / top_k 20）")
    ap.add_argument("--adapter", default="", help="可选：叠加 LoRA 适配器（如 DPO 产物）")
    args = ap.parse_args()
    data_path = args.data or default_data()
    if not Path(data_path).is_file():
        raise SystemExit(f"eval 数据不存在：{data_path}（用 --data 指定）")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"[info] 已叠加 adapter: {args.adapter}")

    print(f"[info] model={args.model}\n[info] data={data_path}")
    rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()][: args.n]
    ok = 0
    for idx, row in enumerate(rows):
        msgs = to_messages(row)
        if not msgs:
            print(f"\n=== row{idx} 跳过：没有可用的消息")
            continue
        tools = row.get("tools")
        if isinstance(tools, str):
            tools = json.loads(tools)
        text = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                          pad_token_id=tok.pad_token_id or tok.eos_token_id,
                          eos_token_id=tok.eos_token_id)
            if args.sample:
                gen_kw.update(temperature=0.6, top_p=0.95, top_k=20)
            out = model.generate(**ids, **gen_kw)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)

        has_think = "<"+"think>" in gen and "<"+"/think>" in gen
        calls, valid = detect_tool_calls(gen)
        passed = has_think and valid > 0
        ok += passed
        sample_id = str(row.get("sample_id") or f"row{idx}")
        print(f"\n=== {sample_id[:32]} | prompt_tokens={ids['input_ids'].shape[1]} "
              f"| new_tokens={out.shape[1]-ids['input_ids'].shape[1]} think_close={gen.count('<'+'/think>')} "
              f"| think={has_think} tool_calls={calls} valid={valid} -> {'PASS' if passed else 'FAIL'}")
        print(gen[:600].replace("\n", "\n  "))
    print(f"\n汇总：{ok}/{len(rows)} 通过")


if __name__ == "__main__":
    main()