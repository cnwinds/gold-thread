#!/usr/bin/env python3
"""长上下文验收：确认 LoRA 合并后的模型在各档长度下仍能正常工作。

用法：
    python verify_longctx.py --model /data/models/Qwen3.8-27B-merged \
        --lengths 8192,32768,65536,131072,262144

判断标准：不报错、首 token 延迟合理、生成内容与前文相关（人工看前 200 字）。
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("[err] 需要 torch + transformers", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    filler = "以下是逆向分析记录，包含偏移量、密钥与工具输出。\n"
    print(f"模型 max_position_embeddings = {model.config.max_position_embeddings}")

    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        try:
            unit = len(tok(filler, add_special_tokens=False)["input_ids"])
            body = filler * max(1, (L - 200) // max(1, unit))
            prompt = body + "\n问题：上文中出现的第一个工具名是什么？请用一句话回答。"
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            n = ids.shape[1]
            if n > model.config.max_position_embeddings:
                print(f"  [skip] {L}: 实际 {n} tokens 超过模型上限")
                continue
            ids = ids.to(model.device)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            text = tok.decode(out[0][n:], skip_special_tokens=True)[:160].replace("\n", " ")
            print(f"  [ok] {L:>7} (实际 {n:>7} tok) 首轮 {dt:>6.1f}s 峰值 {peak:>5.1f}GB | {text}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [oom] {L}: 显存不足（单卡 80GB 的合理上限，需要 TP/量化 KV）")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [err] {L}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
