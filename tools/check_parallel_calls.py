#!/usr/bin/env python3
"""并行工具调用体检：数据侧统计 + 推理侧探测。

背景：训练数据里存在「一步里连发 2~3 个 <tool_call>」的并行调用样本。训练后能不能在 vLLM 上
还原成 OpenAI 的 `tool_calls: [0,1,2]`，取决于三件事，缺一不可：

    1) 服务端 `--enable-auto-tool-choice --tool-call-parser hermes`（或 qwen3_coder）
       —— parser 必须能把一个 assistant 消息里的**多个** <tool_call> 块拆成多个调用；
    2) 没有把 `</tool_call>` 之类配成 stop 字符串（否则第一个调用之后就截断）；
    3) 客户端不设 `parallel_tool_calls: false`，且流式时按 `index` 合并分片。

用法：
    # 数据侧：每一步的调用数分布（train / eval / 每个项目）
    python tools/check_parallel_calls.py --data

    # 推理侧：固定 prompt 打 N 次，看模型是否真的并发（服务端 parser 是否拆得开）
    python tools/check_parallel_calls.py --probe --base-url http://host:8000/v1 \
        --model qwen38-goldthread --api-key gt-2026 --runs 5

    # 推理侧顺带验证流式（并行调用在流式下是 index 0/1/2 递进的多个分片）
    python tools/check_parallel_calls.py --probe --stream --base-url ...
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.common import seqlength as sl

DATA = ROOT / "data" / "03_train"

# 训练集里真实出现过的并行场景，用来探测「能不能并发」
PROBE_PROMPT = (
    "在 /d/crack/demo 目录下：\n"
    "1) 看有哪些文件；2) 看磁盘剩余空间；3) 确认 jadx 是否可用。\n"
    "三条命令互不依赖，请一次性并行发出。"
)
PROBE_TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the workspace shell.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                     "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
]


def calls_per_step(row: Dict[str, Any]) -> List[int]:
    if row.get("conversations"):
        return [c.get("value", "").count("<tool_call>")
                for c in row["conversations"] if c.get("from") == "gpt"]
    out = []
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        out.append(len(m.get("tool_calls") or []) or (m.get("content") or "").count("<tool_call>"))
    return out


def dataset_report(files: List[str]) -> int:
    """数据侧：并行步占比 + train/eval 分布是否一致。"""
    summary = {}
    for fn in files:
        p = DATA / fn
        if not p.exists():
            print(f"[warn] 缺少 {p}", file=sys.stderr)
            continue
        dist = collections.Counter()
        per_proj: Dict[str, List[int]] = {}
        for r in sl.load_jsonl(p):
            meta = r.get("metadata") or {}
            proj = meta.get("project_group") or meta.get("project") or "?"
            bucket = per_proj.setdefault(proj, [0, 0])
            for n in calls_per_step(r):
                dist[n] += 1
                bucket[0] += 1
                bucket[1] += 1 if n >= 2 else 0
        total = sum(dist.values())
        multi = sum(v for k, v in dist.items() if k >= 2)
        summary[fn] = (total, multi)
        print(f"\n=== {fn}   assistant 步 {total}")
        print("  每步调用数分布: " + ", ".join(f"{k}个={v}" for k, v in sorted(dist.items())))
        print(f"  ≥2 并行: {multi} 步 ({multi / max(1, total) * 100:.1f}%)")
        for proj, (tt, mm) in sorted(per_proj.items()):
            print(f"    {proj:<26} 步 {tt:>5}  并行 {mm:>4} ({mm / max(1, tt) * 100:.1f}%)")

    train = next((v for k, v in summary.items() if ".train." in k), None)
    ev = next((v for k, v in summary.items() if ".eval." in k), None)
    if train and ev:
        tr = train[1] / max(1, train[0])
        er = ev[1] / max(1, ev[0])
        print(f"\ntrain 并行率 {tr * 100:.1f}%  vs  eval 并行率 {er * 100:.1f}%", end="  ")
        if er > tr * 1.5:
            print("→ ⚠️ eval 明显更依赖并行调用；train 供给不足，训练后 eval 上容易掉，"
                  "建议对并行步上采样或调高这些步的 sample_weight")
        else:
            print("→ 分布基本一致")
    return 0


def probe(base: str, model: str, key: str, runs: int, stream: bool, temperature: float) -> int:
    """推理侧：固定 prompt 打 N 次，看服务端能否把多个调用拆出来。"""
    url = base.rstrip("/") + "/chat/completions"
    counts: List[int] = []
    print(f"探测 {url}  model={model}  runs={runs}  stream={stream}")
    for i in range(runs):
        body = {"model": model, "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "tools": PROBE_TOOLS, "temperature": temperature, "max_tokens": 1024, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if not stream:
                    o = json.load(r)
                    ch = o["choices"][0]
                    tcs = ch["message"].get("tool_calls") or []
                    names = [t["function"]["name"] for t in tcs]
                    print(f"  run{i + 1}: finish={ch['finish_reason']:<10} tool_calls={len(tcs)} {names}")
                    counts.append(len(tcs))
                else:
                    idx, fin = set(), None
                    for line in r:
                        s = line.decode("utf-8").strip()
                        if not s.startswith("data:"):
                            continue
                        p = s[5:].strip()
                        if p == "[DONE]":
                            break
                        o = json.loads(p)
                        for ch in o.get("choices") or []:
                            if ch.get("finish_reason"):
                                fin = ch["finish_reason"]
                            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                                idx.add(tc.get("index", 0))
                    print(f"  run{i + 1}: finish={fin:<10} tool_calls(按index去重)={len(idx)}")
                    counts.append(len(idx))
        except Exception as exc:  # noqa: BLE001
            print(f"  run{i + 1}: 失败 {type(exc).__name__}: {exc}")
    if not counts:
        return 1
    dist = collections.Counter(counts)
    print("\n调用数分布: " + ", ".join(f"{k}个={v}" for k, v in sorted(dist.items())))
    if max(counts) >= 2:
        print("✅ 服务端能输出并行工具调用（parser 把多个块拆成了 multiple tool_calls）")
        return 0
    print("⚠️ 只出现单个 tool_call。依次排查：\n"
          "   1) 服务端有没有 --enable-auto-tool-choice --tool-call-parser hermes/qwen3_coder\n"
          "   2) 有没有把 </tool_call> 配成 stop 字符串\n"
          "   3) 模型是否真的学过并行（跑 --data 看训练集并行率；太低就上采样并行步）\n"
          "   4) 客户端是否设了 parallel_tool_calls: false")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="数据侧统计")
    ap.add_argument("--files", nargs="+",
                    default=["sft.train.jsonl", "sft.eval.jsonl", "sft_step_windows_openai.train.jsonl"])
    ap.add_argument("--probe", action="store_true", help="推理侧探测")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=False)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    rc = 0
    if args.data or not args.probe:
        rc |= dataset_report(args.files)
    if args.probe:
        if not args.model:
            print("[err] --probe 需要 --model", file=sys.stderr)
            return 2
        rc |= probe(args.base_url, args.model, args.api_key, args.runs, args.stream, args.temperature)
    return rc


if __name__ == "__main__":
    sys.exit(main())