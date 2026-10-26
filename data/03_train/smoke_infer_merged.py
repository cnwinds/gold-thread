#!/usr/bin/env python3
"""合并后模型冒烟：用 transformers 直接续写 eval 窗口，检查 think + <tool_call>。

用法：
    python smoke_infer_merged.py --model /data/models/qwen3-8b-merged --n 3
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/data/goldthread/03_train/sft_step_windows_sharegpt.eval.v2.jsonl"
ROLE = {"system": "system", "human": "user", "gpt": "assistant",
        "observation": "tool", "tool": "tool"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen3-8b-merged")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="用 Qwen3 推荐采样参数（temp .6 / top_p .95 / top_k 20）")
    ap.add_argument("--adapter", default="", help="可选：叠加 LoRA 适配器（如 DPO 产物）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"[info] 已叠加 adapter: {args.adapter}")

    rows = [json.loads(line) for line in open(args.data)][: args.n]
    ok = 0
    for row in rows:
        conv = row["conversations"]
        msgs = [{"role": ROLE.get(c["from"], "user"), "content": c["value"]} for c in conv[:-1]]
        tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools")
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
        calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gen, re.S)
        valid = 0
        for c in calls:
            try:
                d = json.loads(c)
                if "name" in d and ("arguments" in d or "parameters" in d):
                    valid += 1
            except Exception:  # noqa: BLE001
                pass
        passed = has_think and valid > 0
        ok += passed
        print(f"\n=== {row['sample_id'][:32]} | prompt_tokens={ids['input_ids'].shape[1]} "
              f"| new_tokens={out.shape[1]-ids['input_ids'].shape[1]} think_close={gen.count('<'+'/think>')} "
              f"| think={has_think} tool_calls={len(calls)} valid_json={valid} -> {'PASS' if passed else 'FAIL'}")
        print(gen[:600].replace("\n", "\n  "))
    print(f"\n汇总：{ok}/{len(rows)} 通过")


if __name__ == "__main__":
    main()