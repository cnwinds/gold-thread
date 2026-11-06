#!/usr/bin/env python3
"""合并后模型冒烟：用 transformers 续写验证集窗口，检查思考是否闭合、工具调用是否合法。

    python tools/smoke_infer.py --model /cloud/cloud-ssd1/models/<合并目录> --n 5

工具调用按 Qwen XML 判定（与数据和 template: qwen3_8 一致）：
    <tool_call>\n<function=bash>\n<parameter=command>\n...\n</parameter>\n</function>\n</tool_call>
必须采样解码（temp 0.6 / top_p 0.95 / top_k 20）；贪心解码会卡在长思考里。
退出码：通过条数 ≥ 一半为 0，否则 1。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "sft.eval.lf.jsonl"
ROLE = {"system": "system", "human": "user", "gpt": "assistant", "observation": "tool", "tool": "tool"}
THINK_CLOSE = "<" + "/think>"
XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def parse_calls(gen: str) -> tuple[int, int]:
    """返回 (tool_call 块数, 合法调用数)。合法 = 有函数名且至少一个参数。"""
    blocks = gen.count("<tool_call>")
    valid = sum(1 for _, body in XML_CALL.findall(gen) if XML_PARAM.search(body))
    return blocks, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="可选：叠加 LoRA adapter")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()][: args.n]
    passed = 0
    for row in rows:
        conv = row["conversations"]
        msgs = [{"role": ROLE.get(c["from"], "user"), "content": c["value"]} for c in conv[:-1]]
        tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools")
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.6, top_p=0.95, top_k=20,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        # 模板会把开标签放进 prompt，生成里只看闭标签
        think_ok = gen.count(THINK_CLOSE) == 1
        blocks, valid = parse_calls(gen)
        ok = think_ok and valid > 0
        passed += ok
        new_tokens = out.shape[1] - ids["input_ids"].shape[1]
        print(f"\n=== {str(row.get('sample_id', ''))[:40]} | prompt={ids['input_ids'].shape[1]} new={new_tokens} "
              f"| think_close={think_ok} tool_call={blocks} valid={valid} -> {'PASS' if ok else 'FAIL'}")
        print("  " + gen[:600].replace("\n", "\n  "))
        if not ok and new_tokens >= args.max_new_tokens:
            print("  （撞到 max-new-tokens 上限，可能只是思考太长，不一定是格式问题）")
    print(f"\n[smoke] 汇总：{passed}/{len(rows)} 通过")
    return 0 if passed * 2 >= len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
