"""纯确定性构建、硬性质量门禁 (Gate) 与交付打包工具。"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pipeline import package, stage1_extract, stage2_gold, stage3_synth
from pipeline.common.config import load_config, stage_dirs

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

    if dry_run:
        if scan_raw:
            blocker = "raw_scan_requested"
        elif force:
            blocker = "forced_rebuild"
        else:
            blocker = stage2_gold.reuse_blocker(offline_cfg)
        preview = {} if blocker else stage3_synth.draft_preview(offline_cfg, persist_split=False)
        if blocker:
            dpo_audit = dict.fromkeys(stage3_synth.dpo_review_audit(0, []))
            dpo_audit.update(readiness="undetermined", reason=blocker)
        else:
            dpo_audit = preview["dpo_audit"]
        row_counts = {
            f"{kind}_{split}_rows": None if blocker else sum(
                row["kind"] == kind and row["included"] and row["split"] == split
                for row in preview["rows"]
            )
            for kind in ("sft", "dpo") for split in ("train", "eval")
        }
        return {
            "ok": not require_reviewed_dpo or dpo_audit["readiness"] == "ready",
            "dry_run": True,
            **{key: preview.get(key) for key in ("plan_hash", "dirty", "last_build", "split", "summary")},
            "dpo_audit": dpo_audit,
            "gate_preview": {
                **row_counts,
                "sft_non_empty": None if blocker else bool(row_counts["sft_train_rows"] and row_counts["sft_eval_rows"]),
                "readiness": dpo_audit["readiness"],
                "pending_evaluations": dpo_audit["pending_evaluations"],
                "bypassed_unreviewed": dpo_audit["bypassed_unreviewed"],
                "dpo_audit": dpo_audit,
            },
        }

    before_summary = ledger.corpus_snapshot_summary(offline_cfg)
    r1: Optional[Dict[str, Any]] = None
    if scan_raw:
        r1 = stage1_extract.run(force=force, log=lambda *_: None, cfg=offline_cfg)
    r2 = stage2_gold.run(force=force, log=lambda *_: None, cfg=offline_cfg)
    params = {
        "force": bool(force), "scan_raw": bool(scan_raw), "run_gate": bool(run_gate),
        "require_reviewed_dpo": bool(require_reviewed_dpo),
    }
    try:
        r3 = stage3_synth.run(
            force=True, log=lambda *_: None, cfg=offline_cfg, require_reviewed_dpo=require_reviewed_dpo,
        )
    except stage3_synth.DPOReviewRequired:
        # Stage 2 refreshed the reviewable pool even though no Build was written.
        ledger.record_operation(
            "build.run", reason=reason, actor=actor, params=params,
            before_summary=before_summary, after_summary=ledger.corpus_snapshot_summary(offline_cfg),
            status="dpo_review_required", duration_ms=int((time.monotonic() - t0) * 1000), cfg=offline_cfg,
        )
        raise
    build_id = str(r3.get("build_id") or "")

    gate_res = gate_check(build_id, cfg=offline_cfg) if run_gate else None
    after_summary = ledger.corpus_snapshot_summary(offline_cfg)
    ok = gate_res is None or bool(gate_res["passed"])

    ev = ledger.record_operation(
        "build.run",
        reason=reason,
        actor=actor,
        params=params,
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
        "dpo_audit": r3.get("dpo_audit"),
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
    # Packing does not modify the corpus or the latest Build. Use this as a
    # fallback if the final snapshot cannot be read.
    after_summary = before_summary
    emit = log or (lambda *_: None)

    delivery_root = stage_dirs(cfg)["root"] / "04_release" / "delivery"
    staging_dir: Optional[Path] = None
    res: Dict[str, Any] = {}
    copied_to: Optional[str] = None
    published = False
    failure_status = "refused"

    def record(status: str, *, pack: Optional[str] = None, error: Optional[Exception] = None) -> Dict[str, Any]:
        resolved_build = res.get("build_id") or build_id
        params = {
            "build_id": resolved_build,
            "profile": (res.get("format") or {}).get("profile") or profile,
            "copy_to": copied_to,
            "run_gate": bool(run_gate),
        }
        if error is not None:
            params.update(
                parent_pack=parent_pack,
                requested_copy_to=str(copy_to) if copy_to is not None else None,
                candidate_pack=res.get("package_hash"),
                error=str(error),
            )
        return ledger.record_operation(
            "pack.delivery", reason=reason, actor=actor, params=params,
            affected={"build_id": resolved_build, "pack": pack},
            before_summary=before_summary, after_summary=after_summary,
            status=status, duration_ms=int((time.monotonic() - t0) * 1000), cfg=cfg,
        )

    try:
        delivery_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=delivery_root))
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
        final_zip = delivery_root / staged_zip.name
        reuse = final_zip.exists() or final_zip.is_symlink()
        gate_target = final_zip if reuse else staged_zip
        gate_res = gate_check(str(gate_target), cfg=cfg) if run_gate else None
        ok = gate_res is None or bool(gate_res["passed"])

        if ok:
            if reuse:
                package.require_matching_delivery_zip(staged_zip, final_zip)
                res["reused_existing"] = True
            else:
                package.publish_new_delivery_zip(staged_zip, final_zip)
                published = True
            outcome = "published" if published else "reused"
            failure_status = f"{outcome}_metadata_failed"
            staged_profile = staging_dir / "export_profile.json"
            if staged_profile.is_file():
                staged_profile.replace(delivery_root / "export_profile.json")
            res["zip_path"] = str(final_zip)
            res["size_bytes"] = final_zip.stat().st_size
            if gate_res is not None:
                # A fresh publication is the same file as the one checked in staging.
                gate_res["target"] = str(final_zip)
                gate_res["resolved_path"] = str(final_zip.resolve())

            if copy_to:
                failure_status = f"{outcome}_copy_failed"
                dest = Path(copy_to)
                if dest.is_dir() or str(copy_to).endswith(("/", "\\")) or dest.suffix.lower() != ".zip":
                    dest.mkdir(parents=True, exist_ok=True)
                    target_file = dest / final_zip.name
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    target_file = dest
                if not (target_file.exists() and target_file.samefile(final_zip)):
                    if target_file.parent.resolve() == delivery_root.resolve() and target_file.name.lower().startswith("goldthread-train-"):
                        raise ValueError(f"copy_to cannot replace another delivery ZIP: {target_file}")
                    with tempfile.NamedTemporaryFile(dir=target_file.parent, suffix=".tmp", delete=False) as tmp:
                        copy_tmp = Path(tmp.name)
                    try:
                        shutil.copy2(final_zip, copy_tmp)
                        copy_tmp.replace(target_file)
                    except BaseException as exc:
                        try:
                            copy_tmp.unlink(missing_ok=True)
                        except OSError as cleanup_error:
                            exc.add_note(f"Could not remove temporary delivery copy: {cleanup_error}")
                        raise
                copied_to = str(target_file.resolve())
                failure_status = f"{outcome}_metadata_failed"
        else:
            res["zip_path"] = None
        after_summary = ledger.corpus_snapshot_summary(cfg)
    except Exception as exc:
        try:
            # Lineage treats affected.pack as a publication. An occupied path
            # alone (including a refused conflict) does not establish one.
            record(failure_status, pack=res.get("package_hash") if published else None, error=exc)
        except Exception as ledger_error:
            exc.add_note(f"Could not record delivery failure in ledger: {ledger_error}")
        raise
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

    ev = record("ok" if ok else "gate_failed", pack=res.get("package_hash") if ok else None)
    return {
        **res,
        "ok": ok,
        "copied_to": copied_to,
        "gate": gate_res,
        "op_id": ev["op_id"],
    }
