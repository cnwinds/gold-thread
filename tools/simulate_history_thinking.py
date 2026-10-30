#!/usr/bin/env python3
"""翻 `synth.strip_history_thinking` 之前先算账：把现有窗口的「历史思考」补回去，看长度和超限怎么变。

原理：现有窗口的历史步是剥离思考渲染的，但 `data/02_gold/sessions/*/*.gold.jsonl` 里保留了每步的
thinking。按 `metadata.msg_id` 在会话里定位目标步，把前面 k 个 assistant 步的 thinking 按
`to_sharegpt` 的同一格式补回历史，即可精确重算长度（不需要重跑管线、不需要 LLM）。

用法：
    # 现状 vs 打开历史思考 vs 顺带把工具输出压到 3000 字符
    python tools/simulate_history_thinking.py --limit 12288 --caps 12000 4000 3000

    # 只算「打开历史思考」+ 当前 cap
    python tools/simulate_history_thinking.py --limit 12288
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.common import seqlength as sl
from pipeline.common.format import to_sharegpt
from pipeline.common.pilog import truncate_middle


def load_sessions(sessions_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in glob.glob(str(sessions_dir / "*" / "*.gold.jsonl")):
        d = json.loads(Path(f).read_text(encoding="utf-8").splitlines()[0])
        out[d.get("session_id", "")] = d
    return out


def thinking_of_history(row: Dict[str, Any], sess: Dict[str, Any]) -> Tuple[List[str], bool]:
    """按 msg_id 对齐，返回 (历史 assistant 步的 thinking 列表, 是否精确命中)。"""
    meta = row.get("metadata") or {}
    hist = [c["value"] for c in row["conversations"][:-1] if c.get("from") == "gpt"]
    if not sess:
        return [""] * len(hist), False
    ass = [m for m in sess.get("messages") or [] if m.get("role") == "assistant"]
    ids = [m.get("id") for m in ass]
    mid = meta.get("msg_id")
    if mid not in ids:
        return [""] * len(hist), False
    k = ids.index(mid)
    # 先严格校验：窗口历史的最后 len(hist) 步是否就是会话里的前 k 步
    exact = True
    if len(hist) <= k:
        for j, val in zip(range(k - len(hist), k), hist):
            rendered = to_sharegpt(ass[j], strip_thinking=True)
            if not rendered or rendered[0]["value"] != val:
                exact = False
                break
    else:
        exact = False
    window = ass[max(0, k - len(hist)):k] if hist else []
    ths = [(m.get("thinking") or "") for m in window]
    if len(ths) < len(hist):  # 补齐（开头几步可能不存在）
        ths = [""] * (len(hist) - len(ths)) + ths
    return ths, exact


def rebuild_turns(row: Dict[str, Any], ths: List[str], cap: Optional[int]) -> List[Dict[str, str]]:
    """按开关语义重建轮次：历史 gpt 补回  thinking，observation 可截断。"""
    out: List[Dict[str, str]] = []
    ai = 0
    for c in row["conversations"]:
        v = c["value"]
        if c["from"] == "gpt" and ai < len(row["conversations"]) - 1:
            th = (ths[ai] if ai < len(ths) else "").strip()
            comm = v.find("")
            head = v[:comm].strip() if comm > 0 else ""
            body = v[comm:].strip() if comm > 0 else v
            th = th or head
            v = (f"<think\n{th}\n\n\n\n{body}" if th else body)
            ai += 1
        elif c["from"] == "observation" and cap:
            v = truncate_middle(v, cap)
        out.append({"from": c["from"], "value": v})
    return out


def measure(rows, tok, prefix, ths_map, with_history_thinking: bool, cap: Optional[int],
            history_cap: Optional[int] = None) -> List[int]:
    lens = []
    for r in rows:
        ths = ths_map[id(r)][0] if with_history_thinking else [""] * 8
        if with_history_thinking and history_cap:
            ths = [truncate_middle(t, history_cap) if t and len(t) > history_cap else t for t in ths]
        turns = rebuild_turns(r, ths, cap)
        lens.append(sl.count_tokens(tok, sl.render(prefix, turns)))
    return lens


def summary(name: str, lens: List[int], limit: int) -> str:
    q = sl.quantiles(lens)
    over = q["over_12k"] if limit == 12288 else sum(1 for x in lens if x > limit)
    return (f"  {name:<34} p50={q['p50']:>6} p90={q['p90']:>6} p99={q['p99']:>6} max={q['max']:>6}"
            f"  超{limit//1024}K={over:>3}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/03_train/sft.train.jsonl"))
    ap.add_argument("--sessions", default=str(ROOT / "data/02_gold/sessions"))
    ap.add_argument("--limit", type=int, default=12288)
    ap.add_argument("--caps", nargs="*", type=int, default=None,
                    help="工具输出截断字符数列表（对照 synth.max_tool_output_chars）")
    ap.add_argument("--history-caps", nargs="*", type=int, default=None,
                    help="历史思考压缩到多少字符（0/省略=不压缩；对照新开关 history_thinking_max_chars）")
    ap.add_argument("--log-dir", default=str(ROOT / "data/03_train/logs"))
    args = ap.parse_args()

    tok, tok_src = sl.find_tokenizer()
    prefix, pfx = sl.prefix_from_logs(Path(args.log_dir))
    rows = sl.load_jsonl(Path(args.file))
    sess = load_sessions(Path(args.sessions))

    ths_map = {}
    exact = 0
    for r in rows:
        sid = (r.get("metadata") or {}).get("session_id")
        ths, ok = thinking_of_history(r, sess.get(sid) or {})
        ths_map[id(r)] = (ths, ok)
        exact += 1 if ok else 0

    print(f"数据集 {Path(args.file).name}（{len(rows)} 条）")
    print(f"tokenizer {tok_src} | 模板前缀 {pfx}")
    print(f"历史思考可精确对齐 {exact}/{len(rows)} 条（其余按会话内相邻步近似补齐）\n")

    caps = args.caps if args.caps is not None else [None]
    for cap in caps:
        label = f"cap={cap if cap else '现状 12000'}"
        print(f"工具输出截断 {label}")
        base = measure(rows, tok, prefix, ths_map, False, cap)
        print(summary("历史剥离思考（现状）", base, args.limit))
        hcaps = args.history_caps if args.history_caps is not None else [None]
        for hc in hcaps:
            label2 = "历史保留思考" + (f"（思考压到 {hc} 字符）" if hc else "（不压缩）")
            withth = measure(rows, tok, prefix, ths_map, True, cap, hc)
            print(summary(label2, withth, args.limit))
            d = [b - a for a, b in zip(base, withth)]
            print(f"  {'增量':<34} 平均 +{sum(d)/len(d):.0f} tokens/条，"
                  f"最大 +{max(d)}，中位 +{sorted(d)[len(d)//2]}")
        print()

    print(f"判定线 {args.limit}：把「历史保留思考」那行的超限数降到 0，就说明该 cap 够用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())