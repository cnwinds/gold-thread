"""judge 模型对照测试：同一输入下比较各模型 4 类判断任务的表现。

    python -m pipeline.judge_bench --models glm-5.3-flash,MiniMax-M3.1-Flash,deepseek-v4.1-flash-1017
    python -m pipeline.judge_bench --models glm-5.3-flash --session 01a158b3

评测维度（质量优先）：
  ok      是否返回可解析 JSON（失败=该任务对此会话不可用）
  n       有效决定/发现数量（漏判则偏低）
  ms      单任务总耗时
  sample  首条输出摘要，人工判断是否切题

注意：这里的"n 越多越好"只适用于 decide/seams/dpo（发现问题数），
quality 的 n 指 problems 数，同样越多越细。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from . import judge
from .common import pilog
from .common.config import load_config


def _find_raw(cfg: Dict[str, Any], sid: str) -> Path:
    hits = glob.glob(str(Path(cfg["paths"]["data_dir"]).parent / "data" / "**" / f"{sid}*.jsonl"),
                     recursive=True) if False else []
    for pat in ("data/01_raw/*/*/*.jsonl", "data/**/*.jsonl"):
        for f in glob.glob(pat, recursive=True):
            if Path(f).stem.startswith(sid):
                return Path(f)
    raise SystemExit(f"找不到会话原始文件: {sid}")


def _load_session(cfg: Dict[str, Any], sid: str):
    raw = _find_raw(cfg, sid)
    flow, marks, stats = pilog.build_flow(pilog.parse_session(raw), cfg)
    return raw, flow, stats


def _dpo_pairs(n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open("data/03_train/dpo_preference_pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            if len(out) >= n:
                break
            out.append(json.loads(line))
    return out


def _real_seams(sid: str, limit: int = 4) -> List[Dict[str, Any]]:
    """从既有黄金产物里还原真实接缝（被剪步 + 前后相邻步），保证模型间可比。"""
    golds = glob.glob(f"data/02_gold/sessions/*/{sid}*.gold.jsonl")
    if not golds:
        return []
    rec = json.loads(open(golds[0], encoding="utf-8").readline())
    msgs = rec.get("messages") or []
    asst = [m for m in msgs if m.get("role") == "assistant"]
    by_id = {m.get("id"): i for i, m in enumerate(asst)}

    def ctx(i):
        m = asst[i] if 0 <= i < len(asst) else None
        if not m:
            return {}
        return {"msg_id": m.get("id"), "thinking": (m.get("thinking") or "")[:500],
                "text": (m.get("text") or "")[:160],
                "tools": [tc.get("name") for tc in (m.get("tool_calls") or [])]}

    seams: List[Dict[str, Any]] = []
    for f in glob.glob(f"data/02_gold/rejects/{sid}*.pruned.jsonl"):
        for line in open(f, encoding="utf-8"):
            if len(seams) >= limit:
                break
            d = json.loads(line)
            rid = d.get("dropped_msg_id")
            after = next((i for i, m in enumerate(asst) if m.get("pruned_detour") == rid), None)
            if after is None:
                continue
            seams.append({
                "seam_id": f"prune::{rid}",
                "type": "detour_pruned",
                "removed": {"msg_id": rid, "reason": d.get("reason"),
                            "thinking": (d.get("rejected_thinking") or "")[:500],
                            "tools": d.get("rejected_tool_calls") or [],
                            "error_output": (d.get("error_output") or "")[:220]},
                "before": ctx(after - 1),
                "after": ctx(after),
            })
    return seams


def run_one(model: str, sid: str, cfg: Dict[str, Any], dpo_n: int) -> List[Dict[str, Any]]:
    cfg = {**cfg, "judge": {**cfg["judge"], "task_models": {t: model for t in ("steps", "seams", "quality", "dpo")},
                            "model": model}}
    raw, flow, stats = _load_session(cfg, sid)
    rows: List[Dict[str, Any]] = []

    # 1) 步级判断
    t = time.time()
    dec = judge.decide_steps(flow, cfg=cfg, log=lambda *_: None)
    acts: Dict[str, int] = {}
    for d in dec.values():
        acts[d["action"]] = acts.get(d["action"], 0) + 1
    rows.append({"task": "steps", "n": len(dec), "ms": int((time.time() - t) * 1000),
                 "ok": bool(dec), "detail": json.dumps(acts, ensure_ascii=False),
                 "sample": next((f'{d["action"]}:{d["reason"]}' for d in dec.values()), "")})

    # 2) 接缝审核：用该会话真实的剪枝记录构造接缝（三模型输入完全一致）
    seams = _real_seams(sid)
    t = time.time()
    sv = judge.audit_seams(seams, cfg=cfg, log=lambda *_: None)
    bad = [v for v in sv.values() if not v.get("smooth", True)]
    rows.append({"task": "seams", "n": len(bad), "ms": int((time.time() - t) * 1000),
                 "ok": bool(seams) and bool(sv), "detail": f"接缝 {len(seams)} 处",
                 "sample": (bad[0].get("issue", "") if bad else "全部平滑")})

    # 3) 质量报告 + 高光
    t = time.time()
    rep = judge.audit_quality(flow, stats, cfg=cfg, log=lambda *_: None)
    rows.append({"task": "quality", "n": len(rep.get("problems") or []), "ms": int((time.time() - t) * 1000),
                 "ok": bool(rep.get("verdict")),
                 "detail": f'{rep.get("verdict")}/{rep.get("confidence")} 高光{len(rep.get("highlights") or [])} 块失败{rep.get("chunks_failed")}',
                 "sample": (rep.get("reason") or "")[:110]})

    # 4) DPO 复核
    pairs = _dpo_pairs(dpo_n)
    t = time.time()
    dv = judge.review_dpo(pairs, cfg=cfg, log=lambda *_: None)
    failed = [v for v in dv.values() if v.get("verdict") == "review_failed"]
    rows.append({"task": "dpo", "n": len(dv) - len(failed), "ms": int((time.time() - t) * 1000),
                 "ok": len(failed) < len(pairs),
                 "detail": f"{len(pairs)} 对，失败 {len(failed)}",
                 "sample": next((f'{v["verdict"]}:{v["reason"]}' for v in dv.values()),
                                "全部 good")})
    return rows


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glm-5.3-flash,MiniMax-M3.1-Flash,deepseek-v4.1-flash-1017")
    ap.add_argument("--session", default="01a158b3")
    ap.add_argument("--dpo", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = load_config()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"会话: {args.session} | 模型: {', '.join(models)}\n")
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for m in models:
        try:
            all_rows[m] = run_one(m, args.session, cfg, args.dpo)
        except Exception as exc:  # noqa: BLE001
            print(f"[{m}] 失败: {exc}")
            all_rows[m] = []
    print(f'{"模型":<26}{"任务":<9}{"ok":<4}{"n":>4}{"秒":>7}  详情 / 摘要')
    print("-" * 118)
    for m in models:
        for r in all_rows.get(m) or []:
            print(f'{m:<26}{r["task"]:<9}{"✓" if r["ok"] else "✗":<4}{r["n"]:>4}{r["ms"]/1000:>7.0f}  '
                  f'{r["detail"]} | {r["sample"][:56]}')
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())