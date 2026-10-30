#!/usr/bin/env python3
"""训练前序列长度体检：按 LLaMA-Factory 的 ChatML 渲染把每条样本拼成真实训练序列，统计 token 长度。

实际测量逻辑在 `pipeline/common/seqlength.py`（handoff 文档也用同一套口径，避免两处数字打架）。

仍然要在这里量的原因：cutoff_len 一旦小于样本长度，LLaMA-Factory 会**从右侧截断**，
被截掉的正好是目标步（最后一个 assistant）；叠加 `mask_history: true` 就是「标签全被切光、
loss=0」的假训练——loss 曲线好看，能力没学到。

用法：
    # 1) 自动找 tokenizer + 从训练日志抠模板前缀（推荐）
    python tools/check_seq_len.py --kind sft --limit 12288 \
        --log-dir data/03_train/logs \
        --files sft_step_windows_sharegpt.train.v2.jsonl

    # 2) 显式指定模板前缀 / tokenizer / 模型目录
    python tools/check_seq_len.py --kind sft --limit 12288 \
        --prefix-file /tmp/prefix.txt --tokenizer /data/models/Qwen3.8-27B/tokenizer.json

    # 3) DPO：prompt+chosen 与 prompt+rejected 取大者
    python tools/check_seq_len.py --kind dpo --limit 8192 --log-dir data/03_train/logs \
        --files dpo_preference_pairs.train.v2w.jsonl

退出码：0 全部在 limit 内；1 有超限；2 参数/文件问题。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.common import seqlength as sl

DATA = ROOT / "data" / "03_train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sft", "dpo"], required=True)
    ap.add_argument("--limit", type=int, required=True, help="例如 12288(SFT) / 8192(DPO)")
    ap.add_argument("--files", nargs="+", required=True, help="相对 --dir 的文件名")
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--tokenizer", help="tokenizer.json 或含它的模型目录")
    ap.add_argument("--model", help="模型目录（会找 tokenizer.json）")
    ap.add_argument("--log-dir", help="训练日志目录，用来抠真实模板前缀（推荐）")
    ap.add_argument("--template-from-log", help="指定某一份日志")
    ap.add_argument("--prefix-file", help="直接给模板前缀文本")
    ap.add_argument("--samples", type=int, default=0, help=">0 时只测前 N 条")
    ap.add_argument("--verify", action="store_true",
                    help="用训练日志的 ground truth（真 tokenizer 数出的 input_ids）验收当前 tokenizer")
    ap.add_argument("--via-api", metavar="BASE_URL",
                    help="不上训练机：用推理端点的 tokenizer 当计数器（如 http://106.75.44.136:8000/v1）")
    ap.add_argument("--api-model", default="qwen38-dpo", help="与 --via-api 搭配的模型名")
    ap.add_argument("--api-key", default="gt-2026", help="与 --via-api 搭配的 api key")
    ap.add_argument("--api-cache", default=str(ROOT / ".cache" / "api_token_counts.json"))
    args = ap.parse_args()

    if args.via_api:
        tok = sl.ApiTokenizer(args.via_api, args.api_model, args.api_key,
                              cache_path=Path(args.api_cache))
        tok_src = f"api({args.via_api} / {args.api_model}, mode={tok.probe()})"
        if not tok_src.endswith("unavailable)"):
            print("⚠️ API 口径只在其服务的是**同一份模型/分词器**时准确；不同模型实测偏 3~8%。")
    else:
        tok, tok_src = sl.find_tokenizer(args.tokenizer, args.model)
    if args.prefix_file:
        prefix, pfx_src = Path(args.prefix_file).read_text(encoding="utf-8"), f"prefix-file:{args.prefix_file}"
    elif args.template_from_log:
        prefix = sl.prefix_from_log(Path(args.template_from_log)) or ""
        pfx_src = f"log:{args.template_from_log}"
    elif args.log_dir:
        prefix, pfx_src = sl.prefix_from_logs(Path(args.log_dir))
        if not prefix:
            pfx_src += "（未抠到，长度会偏小）"
    else:
        prefix, pfx_src = "", "none（未含模板前缀，长度偏小 ≈686 tok/条）"

    print(f"tokenizer : {tok_src}")
    print(f"模板前缀  : {pfx_src}")
    print(f"           {len(prefix)} 字符 / {sl.prefix_tokens(tok, prefix)} tokens（每条样本固定开销）")

    if args.verify or args.log_dir or args.template_from_log:
        log_dir = Path(args.log_dir) if args.log_dir else (Path(args.template_from_log).parent if args.template_from_log else None)
        logs = sorted(log_dir.glob("*.log")) if log_dir else []
        seen = 0
        for lp in logs:
            v = sl.verify_tokenizer(tok, lp)
            if not v:
                continue
            seen += 1
            flag = "OK" if v["ok"] else "偏差大，可能不是同一份 tokenizer"
            print(f"ground truth 校验 {v['file']}: 日志 {v['log_tokens']} vs 本口径 {v['tok_tokens']}"
                  f"（偏差 {v['delta']:+d} / {v['delta_pct']:+.3f}%）→ {flag}")
            if seen >= 3:
                break
        if not seen:
            print("ground truth 校验：没找到可用的 training example（日志里没有 input_ids/inputs）")

    root = Path(args.dir)
    worst = 0
    for fn in args.files:
        p = root / fn
        if not p.exists():
            print(f"[warn] 找不到 {p}", file=sys.stderr)
            continue
        rows = sl.load_jsonl(p)
        if args.samples:
            rows = rows[: args.samples]
        lens = (sl.sft_lengths(rows, tok, prefix) if args.kind == "sft"
                else sl.dpo_lengths(rows, tok, prefix))
        q = sl.quantiles([n for n, _ in lens])
        over = [(n, i) for n, i in lens if n > args.limit]
        worst += len(over)
        print(f"\n=== {fn}  n={q['n']}  上限 {args.limit}")
        print(f"  min={min(n for n, _ in lens)}  p50={q['p50']}  p90={q['p90']}  p99={q['p99']}  max={q['max']}"
              f"  平均 {q['mean']}")
        print(f"  超限 {len(over)} 条 ({len(over)/q['n']*100:.1f}%)"
              f"  |  >8K {q['over_8k']}  >12K {q['over_12k']}  >16K {q['over_16k']}"
              f"  |  距上限 <5% 的边界样本 {sum(1 for n, _ in lens if args.limit*0.95 < n <= args.limit)} 条")
        for n, i in sorted(over, reverse=True)[:8]:
            print(f"    {n:>7} tok  超出 {n - args.limit:>5}   {i}")

    print("\n结论：" + ("全部在限制内" if worst == 0 else f"{worst} 条超限，需要裁剪或调 cutoff_len"))
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())