"""纯确定性构建、硬性质量门禁 (Gate) 与交付打包工具。"""
from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline import package, stage1_extract, stage2_gold, stage3_synth
from pipeline.common.config import load_config

from . import eval_tools, ledger


def _offline_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg or load_config())
    out.setdefault("turns", {})["use_model"] = False
    out.setdefault("judge", {})["enabled"] = False
    return out


def gate_check(
    target: str | Path = "latest",
    *,
    style: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    ev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行硬性质量门禁断言；全部通过 passed=True，否则返回具体 violations。"""
    cfg = cfg or load_config()
    if ev is None:
        ev = eval_tools.eval_pack(target, cfg=cfg, expected_style=style)

    lg = ev.get("length_gate") or {}
    lg_not_skipped = (
        isinstance(lg, dict)
        and "_skip" not in lg
        and not any(isinstance(v, dict) and v.get("skipped") for v in lg.values())
    )
    lg_zip_present = ev["resolved_kind"] != "zip" or ("sft.train" in lg and "sft.eval" in lg)
    lg_counts_ok = (
        ev["sft_train_rows"] >= ev["sft_train_unique"]
        and ev["sft_eval_rows"] == ev["sft_eval_unique"]
    )

    checks: List[Dict[str, Any]] = [
        {
            "name": "sft_non_empty",
            "passed": bool(ev["sft_train_rows"] > 0 and ev["sft_eval_rows"] > 0),
            "detail": f"train={ev['sft_train_rows']}, eval={ev['sft_eval_rows']}",
        },
        {
            "name": "zero_premature_stop",
            "passed": ev["premature_stop_targets"] == 0,
            "detail": f"premature_stop_targets={ev['premature_stop_targets']}",
        },
        {
            "name": "zero_prompt_tag_leaks",
            "passed": ev["prompt_tag_leaks"] == 0,
            "detail": f"prompt_tag_leaks={ev['prompt_tag_leaks']}",
        },
        {
            "name": "toolcall_contract_100pct",
            "passed": (ev["wrong_style"] == 0 and ev["stray_xml"] == 0 and ev["bad_blocks"] == 0),
            "detail": (
                f"wrong_style={ev['wrong_style']}, stray_xml={ev['stray_xml']}, "
                f"bad_blocks={ev['bad_blocks']}"
            ),
        },
        {
            "name": "target_thinking_present",
            "passed": (ev["missing_target_thinking"] == 0 and ev["thinking_shape"] != "mixed"),
            "detail": (
                f"missing_target_thinking={ev['missing_target_thinking']}, "
                f"thinking_shape={ev['thinking_shape']}"
            ),
        },
        {
            "name": "canonical_project_groups",
            "passed": ev["non_canonical_project_count"] == 0,
            "detail": f"non_canonical_project_count={ev['non_canonical_project_count']}",
        },
        {
            "name": "valid_dpo_pairs",
            "passed": ev["dpo_invalid_pairs"] == 0,
            "detail": f"dpo_invalid_pairs={ev['dpo_invalid_pairs']}",
        },
        {
            "name": "length_gate_effective",
            "passed": bool(lg_not_skipped and lg_zip_present and lg_counts_ok),
            "detail": (
                f"skipped={not lg_not_skipped}, train={ev['sft_train_rows']}/{ev['sft_train_unique']}, "
                f"eval={ev['sft_eval_rows']}/{ev['sft_eval_unique']}"
            ),
        },
    ]

    violations: List[Dict[str, Any]] = []
    for chk in checks:
        if not chk["passed"]:
            violations.append(
                {
                    "severity": "error",
                    "code": chk["name"],
                    "message": f"门禁未通过 {chk['name']}: {chk['detail']}",
                }
            )
    for v in ev.get("violations") or []:
        if v.get("severity") == "error" and not any(x["code"] == v.get("code") for x in violations):
            violations.append(v)

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "target": str(target),
        "resolved_kind": ev["resolved_kind"],
        "resolved_path": ev["resolved_path"],
        "build_id": ev.get("build_id"),
        "package_hash": ev.get("package_hash"),
        "expected_style": ev.get("expected_style"),
        "checks": checks,
        "violations": violations,
        "summary": {
            "sft_train_rows": ev["sft_train_rows"],
            "sft_eval_rows": ev["sft_eval_rows"],
            "dpo_train_rows": ev["dpo_train_rows"],
            "dpo_eval_rows": ev["dpo_eval_rows"],
            "project_count": ev["project_count"],
        },
    }


def build_pipeline(
    *,
    force: bool = False,
    scan_raw: bool = False,
    dry_run: bool = False,
    run_gate: bool = True,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """纯确定性执行管线构建（零隐式 LLM 调用），默认紧接执行 gate_check。"""
    t0 = time.monotonic()
    offline_cfg = _offline_cfg(cfg)

    if dry_run:
        preview = stage3_synth.draft_preview(offline_cfg, persist_split=False)
        rows = preview.get("rows") or []
        sft_train_n = sum(1 for r in rows if r.get("kind") == "sft" and r.get("included") and r.get("split") == "train")
        sft_eval_n = sum(1 for r in rows if r.get("kind") == "sft" and r.get("included") and r.get("split") == "eval")
        dpo_train_n = sum(1 for r in rows if r.get("kind") == "dpo" and r.get("included") and r.get("split") == "train")
        dpo_eval_n = sum(1 for r in rows if r.get("kind") == "dpo" and r.get("included") and r.get("split") == "eval")
        return {
            "ok": True,
            "dry_run": True,
            "plan_hash": preview.get("plan_hash"),
            "dirty": preview.get("dirty"),
            "last_build": preview.get("last_build"),
            "split": preview.get("split"),
            "summary": preview.get("summary"),
            "gate_preview": {
                "sft_train_rows": sft_train_n,
                "sft_eval_rows": sft_eval_n,
                "dpo_train_rows": dpo_train_n,
                "dpo_eval_rows": dpo_eval_n,
                "sft_non_empty": bool(sft_train_n > 0 and sft_eval_n > 0),
                "pending_evaluations": int((preview.get("summary") or {}).get("pending_evaluations") or 0),
            },
        }

    before_summary = ledger.corpus_snapshot_summary(offline_cfg)
    r1: Optional[Dict[str, Any]] = None
    if scan_raw:
        r1 = stage1_extract.run(force=force, log=lambda *_: None, cfg=offline_cfg)
    r2 = stage2_gold.run(force=force, log=lambda *_: None, cfg=offline_cfg)
    r3 = stage3_synth.run(force=True, log=lambda *_: None, cfg=offline_cfg)
    build_id = str(r3.get("build_id") or "")

    gate_res = gate_check(build_id, cfg=offline_cfg) if run_gate else None
    after_summary = ledger.corpus_snapshot_summary(offline_cfg)
    ok = gate_res is None or bool(gate_res["passed"])

    ev = ledger.record_operation(
        "build.run",
        reason=reason,
        actor=actor,
        params={
            "force": bool(force),
            "scan_raw": bool(scan_raw),
            "run_gate": bool(run_gate),
        },
        affected={
            "build_id": build_id,
            "projects": after_summary.get("projects") or [],
        },
        before_summary=before_summary,
        after_summary=after_summary,
        status="ok" if ok else "gate_failed",
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=offline_cfg,
    )
    return {
        "ok": ok,
        "dry_run": False,
        "build_id": build_id,
        "build_dir": r3.get("build_dir"),
        "stage1": r1,
        "stage2": r2,
        "stage3": {
            "build_id": build_id,
            "sft_windows": r3.get("sft_step_windows_sharegpt.jsonl", 0),
            "sft_train": r3.get("sft_step_windows_sharegpt.train.jsonl", 0),
            "sft_eval": r3.get("sft_step_windows_sharegpt.eval.jsonl", 0),
            "dpo_total": r3.get("dpo_preference_pairs.jsonl", 0),
            "dpo_train": r3.get("dpo_preference_pairs.train.jsonl", 0),
            "dpo_eval": r3.get("dpo_preference_pairs.eval.jsonl", 0),
        },
        "gate": gate_res,
        "op_id": ev["op_id"],
    }


def pack_delivery(
    *,
    build_id: Optional[str] = None,
    profile: Optional[str] = None,
    copy_to: Optional[str | Path] = None,
    run_gate: bool = True,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从不可变 Build 打包交付 zip，默认执行 gate_check，可选 copy_to 同步拷贝。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    before_summary = ledger.corpus_snapshot_summary(cfg)

    res = package.build_delivery_zip_from_build(
        cfg,
        build_id=build_id,
        profile=profile,
        log=lambda *_: None,
    )
    zip_path = Path(res["zip_path"])

    gate_res = gate_check(str(zip_path), cfg=cfg) if run_gate else None
    copied_to: Optional[str] = None
    if copy_to:
        dest = Path(copy_to)
        if dest.is_dir() or str(copy_to).endswith(("/", "\\")) or dest.suffix.lower() != ".zip":
            dest.mkdir(parents=True, exist_ok=True)
            target_file = dest / zip_path.name
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            target_file = dest
        shutil.copy2(zip_path, target_file)
        copied_to = str(target_file.resolve())

    after_summary = ledger.corpus_snapshot_summary(cfg)
    ok = gate_res is None or bool(gate_res["passed"])

    ev = ledger.record_operation(
        "pack.delivery",
        reason=reason,
        actor=actor,
        params={
            "build_id": res.get("build_id"),
            "profile": (res.get("format") or {}).get("profile"),
            "copy_to": copied_to,
            "run_gate": bool(run_gate),
        },
        affected={
            "build_id": res.get("build_id"),
            "pack": res.get("package_hash"),
        },
        before_summary=before_summary,
        after_summary=after_summary,
        status="ok" if ok else "gate_failed",
        duration_ms=int((time.monotonic() - t0) * 1000),
        cfg=cfg,
    )
    return {
        **res,
        "ok": ok,
        "copied_to": copied_to,
        "gate": gate_res,
        "op_id": ev["op_id"],
    }
