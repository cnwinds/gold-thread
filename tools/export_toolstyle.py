#!/usr/bin/env python3
"""按模型导出训练数据：把 03_train 的中立/基线数据集转换为目标模型的工具调用格式。

背景（2026-10 27B 事故复盘，见 DATASET_BEST_PRACTICES.md §5）：
    同一份金线数据，不同模型要求不同的工具调用格式：
      - Qwen3.8-27B（template: qwen3_8）要求 coder-xml
      - Qwen3.8-8B  （template: qwen3）  要求 hermes-json
    03_train 的 sharegpt 产物默认是 hermes-json（与 to_sharegpt 的历史行为一致）。
    本工具把它们转换成目标模型的格式，并输出 FORMAT_CONTRACT.json 记录
    {风格, template, parser} 三元组——三方一致的另外两方由训练/部署配置落实。

设计原则：**只转换 assistant 侧文本**（sharegpt 的 `from=gpt`、DPO 的 chosen/rejected），
绝不碰 observation/human/system——工具输出里可能恰好含有 <tool_call> 字样，那是数据不是格式。
openai 结构化文件（messages + tool_calls）风格中立，直接原样复制即可（渲染由 chat template 负责）。

用法：
    # 按 profile 导出（推荐，profile 内置 style/template/parser 三元组）
    python tools/export_toolstyle.py --profile Qwen3.8-27B \
        --files sft_step_windows_sharegpt.train.v2.jsonl sft_step_windows_sharegpt.eval.v2.jsonl

    # 手动指定风格
    python tools/export_toolstyle.py --style coder-xml --files dpo_preference_pairs.train.v2w.jsonl

    # 查看可用 profile
    python tools/export_toolstyle.py --list-profiles

导出后自检（三方一致的验收）：
    python tools/check_toolcall_contract.py --style coder-xml --dir data/03_train/exports/<profile>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.format import (  # noqa: E402
    EXPORT_PROFILES,
    TOOL_CALL_STYLES,
    convert_sharegpt_row,
    detect_tool_style,
    resolve_export_profile,
)
from pipeline.common.store import read_jsonl, write_jsonl_atomic  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO / "data" / "03_train"

# 与交付 zip 共用同一份注册表（pipeline.common.format.EXPORT_PROFILES）。
PROFILES = EXPORT_PROFILES

ASSISTANT_KEYS = ("gpt",)  # sharegpt 里 assistant 侧的 from 标签


def _style_report(rows: list[dict]) -> dict:
    """统计行内 assistant 侧 <tool_call> 块的风格分布（验收用）。"""
    hermes = xml = mixed = none = 0
    for row in rows:
        texts: list[str] = []
        for m in row.get("conversations") or []:
            if isinstance(m, dict) and m.get("from") in ASSISTANT_KEYS:
                texts.append(m.get("value") or "")
        for key in ("chosen", "rejected"):
            obj = row.get(key)
            if isinstance(obj, dict):
                texts.append(obj.get("value") or "")
        seen = set()
        unrecognized = False
        for t in texts:
            if "<tool_call>" not in t:
                continue
            st = detect_tool_style(t)
            if st is None:
                unrecognized = True
                break
            seen.add(st)
        if unrecognized or len(seen) > 1:
            mixed += 1
        elif seen == {"hermes-json"}:
            hermes += 1
        elif seen == {"coder-xml"}:
            xml += 1
        else:
            none += 1
    return {"pure_hermes_rows": hermes, "pure_xml_rows": xml, "mixed_or_unparsed_rows": mixed, "no_toolcall_rows": none}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", help=f"目标模型 profile：{', '.join(PROFILES)}")
    ap.add_argument("--style", choices=TOOL_CALL_STYLES, help="手动指定风格（不给 profile 时用）")
    ap.add_argument("--template", help="配套训练 template（手动模式记录进契约清单）")
    ap.add_argument("--parser", help="配套推理 parser（手动模式记录进契约清单）")
    ap.add_argument("--files", nargs="+", help="03_train 下的数据文件（相对或绝对路径）")
    ap.add_argument("--out-dir", help="导出目录（默认 data/03_train/exports/<name>/）")
    ap.add_argument("--openai-files", nargs="*", default=[], help="风格中立的 openai 结构化文件，原样复制")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args()

    if args.list_profiles:
        for name, p in PROFILES.items():
            print(f"{name:16} style={p['style']:12} template={p['template']:8} parser={p['parser']}")
        return 0

    if args.profile:
        try:
            prof = resolve_export_profile(args.profile)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        style, template, parser = prof["style"], prof["template"], prof["parser"]
        prof_name = prof["id"]
    elif args.style:
        style, prof_name = args.style, args.style
        template, parser = args.template or "?", args.parser or "?"
    else:
        print("必须给 --profile 或 --style", file=sys.stderr)
        return 2

    if not args.files:
        print("必须给 --files", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else TRAIN_DIR / "exports" / prof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": prof_name,
        "contract": {"tool_style": style, "train_template": template, "serve_tool_parser": parser},
        "note": "三方一致：tool_style == train_template 注入格式 == serve_tool_parser；改任一方需同步其余两方",
        "files": {},
        "openai_passthrough": [],
    }

    print(f"[export] profile={prof_name} style={style} -> {out_dir}")
    print(f"{'source':50} {'rows':>6} {'conv':>6} {'unparsed':>8} {'after':>28}")
    print("-" * 105)
    ok = True
    for f in args.files:
        src = Path(f)
        if not src.is_absolute():
            src = TRAIN_DIR / f
        if not src.exists():
            print(f"{f:50}  MISSING", file=sys.stderr)
            ok = False
            continue
        rows = read_jsonl(src)
        c_total = u_total = 0
        for row in rows:
            c, u = convert_sharegpt_row(row, style)
            c_total += c
            u_total += u
        dst = out_dir / src.name
        write_jsonl_atomic(dst, rows)
        report = _style_report(rows)
        after = f"xml={report['pure_xml_rows']} hermes={report['pure_hermes_rows']} mixed={report['mixed_or_unparsed_rows']}"
        print(f"{src.name:50} {len(rows):6} {c_total:6} {u_total:8} {after:>28}")
        if u_total:
            ok = False
        manifest["files"][src.name] = {
            "rows": len(rows),
            "blocks_converted": c_total,
            "blocks_unparsed": u_total,
            "style_report": report,
        }

    for f in args.openai_files:
        src = Path(f)
        if not src.is_absolute():
            src = TRAIN_DIR / f
        if not src.exists():
            print(f"{f:50}  MISSING", file=sys.stderr)
            ok = False
            continue
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())
        manifest["openai_passthrough"].append(src.name)
        print(f"{src.name:50} {'-':>6} {'-':>6} {'-':>8} {'copied(style-neutral)':>28}")

    (out_dir / "FORMAT_CONTRACT.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[contract] {out_dir / 'FORMAT_CONTRACT.json'}")
    if not ok:
        print("[warn] 存在未解析的 <tool_call> 块或缺失文件——请人工检查后再用于训练", file=sys.stderr)
        return 1
    print("[done] 全部块已转换；验收：python tools/check_toolcall_contract.py "
          f"--style {style} --dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
