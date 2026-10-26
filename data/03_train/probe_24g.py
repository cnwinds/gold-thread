#!/usr/bin/env python3
"""在训练机上模拟小显存推理：报告 bf16 / 4bit / 8bit 三种加载方式能否在限额内跑起来。

用法：
    python probe_24g.py --model /data/models/Qwen3.8-27B-merged --limit-gb 22

输出每种方式的：成功与否、峰值显存、首 token 延迟、报错摘要。
据此决定 24G 推理机走哪条路（4bit 直载 / GPTQ / GGUF）。
"""
from __future__ import annotations

import argparse
import gc
import sys
import time


def try_load(path: str, mode: str, limit_gb: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"trust_remote_code": True, "device_map": {"": 0},
          "max_memory": {0: f"{limit_gb:.0f}GiB"}}
    if mode == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif mode == "8bit":
        kw["load_in_8bit"] = True
    elif mode == "4bit":
        kw["load_in_4bit"] = True

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    load_s = time.time() - t0

    msgs = [{"role": "user", "content": "用一句话说明你会怎么开始分析一个 xapk 包。"}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = tok("用一句话说明你会怎么开始分析一个 xapk 包。", return_tensors="pt").input_ids
    ids = ids.to(model.device)

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=32, do_sample=False)
    gen_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:120].replace("\n", " ")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return True, load_s, gen_s, peak, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit-gb", type=float, default=22.0)
    ap.add_argument("--modes", default="4bit,8bit,bf16")
    args = ap.parse_args()
    try:
        import torch
    except ImportError:
        print("[err] 需要 torch", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)} | 模拟限额 {args.limit_gb:.0f} GB")
    print(f"模型: {args.model}\n")
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"--- 尝试 {mode} ---")
        try:
            ok, load_s, gen_s, peak, text = try_load(args.model, mode, args.limit_gb)
            verdict = "可用" if peak <= args.limit_gb else "超限"
            print(f"  [{verdict}] 加载 {load_s:.0f}s 生成 {gen_s:.1f}s 峰值 {peak:.1f}GB")
            print(f"  输出: {text}")
        except torch.cuda.OutOfMemoryError:
            print("  [OOM] 显存不足")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {type(exc).__name__}: {str(exc)[:200]}")
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        print()
    print("判定建议：")
    print("  · 4bit 可用 -> 24G 机器直接 load_in_4bit，上下文 4K~16K")
    print("  · 仅 8bit 可用 -> 24G 不够，走 GGUF（llama.cpp）")
    print("  · 全部失败 -> 换 48G/80G，或 CPU offload（1~3 tok/s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
