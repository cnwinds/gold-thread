#!/usr/bin/env python3
"""判定「历史里的思考内容到底有没有进 prompt」——客户端字段名 vs 服务端模板。

背景争议：
  pi 会把历史 assistant 的 thinking 回传（`convertMessages` 里写 `assistantMsg[signature] = thinking`，
  signature 就是响应里 reasoning 字段的名字，我们的 vLLM 端点是 `reasoning`）。
  但**服务端会不会把它渲染进 prompt**，取决于字段名和 chat template：
    - Qwen3 系模板只认 `message.reasoning_content`；
    - 有的实现只认 `reasoning`，有的两个都认（还会归一化）。

做法（受控对照，2 分钟内出结论）：向端点发 4 个只差一个字段的请求，比较 `usage.prompt_tokens`：

    1) 基线          : assistant 只有 content
    2) reasoning_content = 长文
    3) reasoning      = 同样的长文
    4) 对照          : 同样的长文塞进 content（验证「长文本身的 token 数」）

判定：
    X 的 delta ≈ 对照组 delta  → 该字段会被渲染进 prompt（历史思考被带上）
    X 的 delta ≈ 0             → 该字段被丢弃（历史思考不进 prompt）

这也顺带解释了「同一个模型换个客户端，长度/行为就不一样」的现象。

用法：
    python tools/probe_history_thinking.py --base-url http://host:8000/v1 \
        --model qwen38-dpo --api-key gt-2026
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog while the parser scans the binary blob. " * 40


def post(base: str, key: str, model: str, messages, timeout: float = 120.0) -> int:
    body = {"model": model, "messages": messages, "max_tokens": 1, "temperature": 0}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        o = json.load(r)
    return int(o["usage"]["prompt_tokens"])


def base_msgs(assistant_extra: dict | None = None, assistant_content: str = "hello"):
    a = {"role": "assistant", "content": assistant_content}
    if assistant_extra:
        a.update(assistant_extra)
    return [{"role": "user", "content": "hi"}, a, {"role": "user", "content": "again"}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    try:
        b = post(args.base_url, args.api_key, args.model, base_msgs())
        rc = post(args.base_url, args.api_key, args.model, base_msgs({"reasoning_content": FILLER}))
        r = post(args.base_url, args.api_key, args.model, base_msgs({"reasoning": FILLER}))
        ctl = post(args.base_url, args.api_key, args.model, base_msgs(assistant_content="hello" + FILLER))
    except Exception as exc:  # noqa: BLE001
        print(f"[err] 请求失败：{type(exc).__name__}: {exc}")
        return 2

    print(f"端点 {args.base_url}  模型 {args.model}")
    print(f"  基线（无 reasoning）        prompt_tokens = {b}")
    print(f"  长文塞进 content（对照组）  prompt_tokens = {ctl}   Δ = {ctl - b:>5}  ← 长文本身的 token 数")
    verdict = {}
    for name, n in (("reasoning_content", rc), ("reasoning", r)):
        d = n - b
        ok = d > (ctl - b) * 0.5
        verdict[name] = ok
        print(f"  {name:<18}长文    prompt_tokens = {n}   Δ = {d:>5}  → "
              f"{'✅ 会渲染进 prompt（历史思考被带上）' if ok else '❌ 被丢弃（历史思考不带上）'}")

    rc_ok, r_ok = verdict["reasoning_content"], verdict["reasoning"]
    print("\n结论：")
    if rc_ok and not r_ok:
        print("  只认 reasoning_content，不认 reasoning。")
        print("  pi 用 vLLM 时会按响应的字段名回传（实测 thinkingSignature='reasoning'）→ 历史思考默认**不带**。")
        print("  想让它带上：让 pi 用 reasoning_content 作为签名（或服务端把 reasoning 归一化）。")
    elif rc_ok and r_ok:
        print("  两个字段都认（服务端会归一化）→ 历史思考**会被带上**，训练侧必须保留历史思考。")
    elif (not rc_ok) and r_ok:
        print("  只认 reasoning → 历史思考**会被带上**（pi 正好用这个字段），训练侧必须保留历史思考。")
    else:
        print("  两个字段都被丢弃 → 历史思考不进 prompt，训练侧剥离历史思考是对的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())