#!/usr/bin/env python3
"""通过 OpenAI 兼容 API 验收长上下文（适配 vLLM / llama-server）。

用法：
    python verify_longctx_api.py --base-url http://localhost:8000/v1         --model Qwen3.8-27B-merged --lengths 8192,32768,65536,131072,262144

每档构造"长前文 + 尾部提问"，记录 HTTP 状态、TTFT、总耗时与回答摘要。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(url: str, body: dict, timeout: float = 900.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
        return r.status, raw, time.time() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536")
    ap.add_argument("--chars-per-token", type=float, default=2.2)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    filler = "以下是逆向分析记录，包含工具输出与偏移量说明。\n"
    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        repeat = max(1, int((L - 300) * args.chars_per_token / len(filler)))
        prompt = filler * repeat + "\n问题：上文反复出现的字段名是什么？只回一个词。"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        }
        code, raw, dt = post(url, body)
        if code == 200:
            try:
                d = json.loads(raw)
                msg = (d["choices"][0]["message"].get("content") or "").strip()[:80]
                usage = d.get("usage") or {}
                print(f"  [ok] {L:>7} tok 目标 | {dt:>7.1f}s | prompt_tokens={usage.get('prompt_tokens')} | {msg}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {L:>7} | 解析失败 {exc} | {raw[:120]}")
        else:
            print(f"  [fail] {L:>7} | HTTP {code} | {raw[:160]}")
    print("\n判定：五档全部 ok 才算通过；OOM 时降 --max-model-len 或加 --kv-cache-dtype fp8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
