"""纯确定性构建、硬性质量门禁 (Gate) 与交付打包工具。"""
from __future__ import annotations

import copy
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pipeline import package, stage1_extract, stage2_gold, stage3_synth
from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import replace_atomic

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
            "passed": bool(
                ev["sft_train_rows"] > 0
                and ev["sft_eval_rows"] > 0
                and ev.get("missing_assistant_targets", 0) == 0
            ),
            "detail": (
                f"train={ev['sft_train_rows']}, eval={ev['sft_eval_rows']}, "
                f"missing_assistant_targets={ev.get('missing_assistant_targets', 0)}"
            ),
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
    require_reviewed_dpo: bool = False,
    reason: str = "",
    actor: str = "agent",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """纯确定性执行管线构建（零隐式 LLM 调用），默认紧接执行 gate_check。"""
    t0 = time.monotonic()
    offline_cfg = _offline_cfg(cfg)

    preview_for_dpo = stage3_synth.draft_preview(offline_cfg, persist_split=False)
    p_rows = preview_for_dpo.get("rows") or []
    dpo_rows_all = [r for r in p_rows if r.get("kind") == "dpo"]
    vstore = stage3_synth.read_json(stage_dirs(offline_cfg)["state"] / "dpo_verdicts.json", default={}) or {}
    dpo_audit = {
        "total_candidates": len(dpo_rows_all),
        "explicit_verdicts": sum(1 for r in dpo_rows_all if r.get("sample_id") in vstore),
        "cached_reviews": sum(
            1 for r in dpo_rows_all
            if r.get("sample_id") not in vstore and r.get("gate_status") in ("good", "swap", "drop", "passed")
        ),
        "bypassed_unreviewed": sum(1 for r in dpo_rows_all if r.get("gate_status") == "bypassed"),
        "pending_evaluations": int((preview_for_dpo.get("summary") or {}).get("pending_evaluations") or 0),
    }

    if dry_run:
        sft_train_n = sum(1 for r in p_rows if r.get("kind") == "sft" and r.get("included") and r.get("split") == "train")
        sft_eval_n = sum(1 for r in p_rows if r.get("kind") == "sft" and r.get("included") and r.get("split") == "eval")
        dpo_train_n = sum(1 for r in p_rows if r.get("kind") == "dpo" and r.get("included") and r.get("split") == "train")
        dpo_eval_n = sum(1 for r in p_rows if r.get("kind") == "dpo" and r.get("included") and r.get("split") == "eval")
        strict_ok = (not require_reviewed_dpo) or (dpo_audit["bypassed_unreviewed"] == 0 and dpo_audit["pending_evaluations"] == 0)
        return {
            "ok": strict_ok,
            "dry_run": True,
            "plan_hash": preview_for_dpo.get("plan_hash"),
            "dirty": preview_for_dpo.get("dirty"),
            "last_build": preview_for_dpo.get("last_build"),
            "split": preview_for_dpo.get("split"),
            "summary": preview_for_dpo.get("summary"),
            "dpo_audit": dpo_audit,
            "gate_preview": {
                "sft_train_rows": sft_train_n,
                "sft_eval_rows": sft_eval_n,
                "dpo_train_rows": dpo_train_n,
                "dpo_eval_rows": dpo_eval_n,
                "sft_non_empty": bool(sft_train_n > 0 and sft_eval_n > 0),
                "pending_evaluations": dpo_audit["pending_evaluations"],
                "bypassed_unreviewed": dpo_audit["bypassed_unreviewed"],
                "dpo_audit": dpo_audit,
            },
        }

    if require_reviewed_dpo and (dpo_audit["bypassed_unreviewed"] > 0 or dpo_audit["pending_evaluations"] > 0):
        raise ValueError(
            f"严格 DPO 审核模式下仍有未显式复核项: bypassed={dpo_audit['bypassed_unreviewed']}, "
            f"pending={dpo_audit['pending_evaluations']}"
        )

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
    parent_pack: Optional[str] = None,
    copy_to: Optional[str | Path] = None,
    run_gate: bool = True,
    reason: str = "",
    actor: str = "agent",
    log: Optional[Callable[[str], None]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """暂存区打包 -> 硬门禁校验 -> 仅通过后原子发布至 delivery/ 并执行 copy_to（幂等保护已有合格包）。"""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    before_summary = ledger.corpus_snapshot_summary(cfg)
    emit = log or (lambda *_: None)

    delivery_root = stage_dirs(cfg)["root"] / "04_release" / "delivery"
    delivery_root.mkdir(parents=True, exist_ok=True)
    staging_dir = delivery_root / f".staging-{uuid.uuid4().hex[:10]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        res = package.build_delivery_zip_from_build(
            cfg,
            build_id=build_id,
            profile=profile,
            parent_pack=parent_pack,
            log=emit,
            delivery_dir=staging_dir,
            reference_delivery_dir=delivery_root,
        )
        staged_zip = Path(res["zip_path"])
        gate_res = gate_check(str(staged_zip), cfg=cfg) if run_gate else None
        ok = gate_res is None or bool(gate_res["passed"])

        final_zip = delivery_root / staged_zip.name
        copied_to: Optional[str] = None

        if ok:
            if final_zip.is_file() and parent_pack is None:
                try:
                    with zipfile.ZipFile(final_zip, "r") as zf:
                        has_valid_existing = len(zf.namelist()) > 0
                except zipfile.BadZipFile:
                    has_valid_existing = False
            else:
                has_valid_existing = False

            if not has_valid_existing:
                replace_atomic(staged_zip, final_zip)
            staged_profile = staging_dir / "export_profile.json"
            if staged_profile.is_file():
                replace_atomic(staged_profile, delivery_root / "export_profile.json")
            res["zip_path"] = str(final_zip)
            res["size_bytes"] = final_zip.stat().st_size

            if copy_to:
                dest = Path(copy_to)
                if dest.is_dir() or str(copy_to).endswith(("/", "\\")) or dest.suffix.lower() != ".zip":
                    dest.mkdir(parents=True, exist_ok=True)
                    target_file = dest / final_zip.name
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    target_file = dest
                shutil.copy2(final_zip, target_file)
                copied_to = str(target_file.resolve())
        else:
            res["zip_path"] = None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    after_summary = ledger.corpus_snapshot_summary(cfg)
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
            "pack": res.get("package_hash") if ok else None,
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
