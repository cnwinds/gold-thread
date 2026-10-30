#!/usr/bin/env python3
"""思考形态一致性自检：历史步该不该带 ` thinking`，同一批训练集里不能「有的带有的不带」。

背景（实测 pi + vLLM）：
  - 轮内多步 tool 调用：思考 100% 带进下一次调用；
  - 跨轮：思考也带（除非该轮回复是空白且无 tool_calls —— pi 会把整条 assistant 消息丢掉）；
  - Qwen3 模板只认 `message.reasoning_content`；pi 回传的字段名取决于响应里的 reasoning 字段。

所以数据集只有两种合法形态：
    A. 历史剥离思考（`strip_history_thinking: true`）→ 历史带  thinking 比例应为 0%
    B. 历史保留思考（`false`）                       → 应接近 100%
混在一起（同一个 dataset 列表里既有 A 又有 B）是**错的**，模型会学到「有时想有时不想」。

另外目标步必须带思考（空思考 = 训练信号缺陷）。

用法：
    python tools/check_thinking_shape.py                       # 扫默认几个文件
    python tools/check_thinking_shape.py --files sft.train.jsonl sft.eval.jsonl
    python tools/check_thinking_shape.py --expect strip        # 强制要求 A 形态
退出码：0 一致；1 混形态/目标步缺思考；2 参数问题。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "03_train"
OPEN = chr(60) + "think" + chr(62)
CLOSE = chr(60) + "/think" + chr(62)


def scan(path: Path) -> dict:
    hist = hist_th = 0
    tgt = tgt_th = 0
    empty_targets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # 每个 assistant 轮 -> (是否带思考, 思考正文)
        turns: list[tuple[bool, str]] = []
        conv = row.get("conversations")
        if conv:
            for c in conv:
                if c.get("from") != "gpt":
                    continue
                v = c.get("value") or ""
                has = OPEN in v
                body = v.split(OPEN, 1)[1].split(CLOSE, 1)[0] if (has and CLOSE in v) else ""
                turns.append((has, body))
        else:
            # openai 格式：思考可能放在独立的 reasoning_content 字段里
            for m in row.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                rc = m.get("reasoning_content") or ""
                content = m.get("content") or ""
                has = bool(rc.strip()) or OPEN in content
                body = rc or (content.split(OPEN, 1)[1].split(CLOSE, 1)[0] if (OPEN in content and CLOSE in content) else "")
                turns.append((has, body))
        if not turns:
            continue
        for has, _ in turns[:-1]:
            hist += 1
            hist_th += 1 if has else 0
        tgt += 1
        has, body = turns[-1]
        if has:
            tgt_th += 1
            if len(body.strip()) < 20:
                empty_targets.append(row.get("sample_id"))
        else:
            empty_targets.append(row.get("sample_id"))
    return {"hist": hist, "hist_th": hist_th, "tgt": tgt, "tgt_th": tgt_th,
            "empty_targets": empty_targets}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--files", nargs="+",
                    default=["sft.train.jsonl", "sft.eval.jsonl", "sft_step_windows_openai.train.jsonl"])
    ap.add_argument("--expect", choices=["auto", "strip", "keep"], default="auto",
                    help="auto=按每个文件自行判定是否混形态；strip=要求历史无思考；keep=要求历史有思考")
    args = ap.parse_args()

    root = Path(args.dir)
    rows = []
    for fn in args.files:
        p = root / fn
        if not p.exists():
            print(f"[warn] 缺少 {p}", file=sys.stderr)
            continue
        rows.append((fn, scan(p)))

    if not rows:
        print("[err] 没有可检查的文件", file=sys.stderr)
        return 2

    print(f"{'文件':<40}{'历史步':>8}{'带思考':>8}{'比例':>8}{'目标步':>8}{'带思考':>8}")
    shapes = set()
    bad = 0
    for fn, st in rows:
        ratio = st["hist_th"] / st["hist"] if st["hist"] else 0.0
        shape = "keep" if ratio > 0.9 else ("strip" if ratio < 0.1 else "mixed")
        shapes.add(shape)
        print(f"{fn:<40}{st['hist']:>8}{st['hist_th']:>8}{ratio * 100:>7.1f}%{st['tgt']:>8}{st['tgt_th']:>8}")
        if shape == "mixed":
            bad += 1
            print(f"    ⚠️ 混形态：历史思考比例 {ratio * 100:.0f}%，既不是 0% 也不是 ~100%")
        if args.expect != "auto" and shape != args.expect:
            bad += 1
            print(f"    ⚠️ 期望 {args.expect} 形态，实际 {shape}")
        miss = st["tgt"] - st["tgt_th"]
        if miss:
            bad += 1
            print(f"    ⚠️ 目标步缺思考 {miss} 条（训练信号缺陷），例 {st['empty_targets'][:2]}")

    if len(shapes) > 1:
        bad += 1
        print(f"\n⚠️ 同一批里出现多种形态 {sorted(shapes)}：若这些文件会被同时挂进一个 `dataset:`，"
              "模型会学到「有时想有时不想」。要么统一开关，要么分开训练阶段。")

    print("\n结论：" + ("形态一致" if bad == 0 else f"发现 {bad} 处问题"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())