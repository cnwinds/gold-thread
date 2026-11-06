"""数据集草稿的服务端决策账本。

浏览器筛选不是入集事实；人工排除/恢复按逻辑 sample_id 持久化，Stage 3
与草稿预览都通过这里解析同一个有效决策。
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .common.config import config_hash, load_config, stage_dirs
from .common.store import load_index, now_iso, read_json, sha256_file, write_json_atomic

DRAFT_VERSION = 1
DATASET_PLAN_VERSION = 2
# 本地服务由 HTTP 线程和后台 Job 共用；锁必须覆盖完整读—改—写事务。
DRAFT_LOCK = threading.RLock()


def draft_path(cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or load_config()
    return stage_dirs(cfg)["state"] / "dataset_draft.json"


def _empty() -> Dict[str, Any]:
    return {
        "version": DRAFT_VERSION,
        "updated_at": None,
        "decisions": {},
        "split_plan": {"unit": "project_group", "assignments": {}, "pinned": []},
    }


def load_draft(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = read_json(draft_path(cfg), default={}) or {}
    if not isinstance(raw, dict):
        raw = {}
    out = _empty()
    decisions = raw.get("decisions")
    out["decisions"] = dict(decisions) if isinstance(decisions, dict) else {}
    plan = raw.get("split_plan")
    plan = dict(plan) if isinstance(plan, dict) else {}
    assignments = plan.get("assignments")
    pinned = plan.get("pinned")
    out["split_plan"] = {
        "unit": "project_group",
        "eval_ratio": plan.get("eval_ratio"),
        "seed": plan.get("seed"),
        "assignments": {
            str(k): v for k, v in (assignments or {}).items() if v in ("train", "eval")
        } if isinstance(assignments, dict) else {},
        "pinned": sorted({str(p) for p in (pinned or []) if str(p)}) if isinstance(pinned, list) else [],
    }
    out["updated_at"] = raw.get("updated_at")
    last_build = raw.get("last_build")
    out["last_build"] = dict(last_build) if isinstance(last_build, dict) else None
    return out


def save_draft(draft: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
    with DRAFT_LOCK:
        draft["version"] = DRAFT_VERSION
        draft["updated_at"] = now_iso()
        write_json_atomic(draft_path(cfg), draft)


def _project_hash(seed: int, project: str) -> int:
    digest = hashlib.sha256(f"{seed}:{project}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def resolve_split_plan(
    projects: Iterable[str],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    persist: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with DRAFT_LOCK:
        return _resolve_split_plan_unlocked(projects, cfg, persist=persist)


def _resolve_split_plan_unlocked(
    projects: Iterable[str],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    persist: bool = True,
) -> Dict[str, Any]:
    """返回持久化项目切分；已有项目不因新增项目或普通重建而移动。"""
    cfg = cfg or load_config()
    draft = load_draft(cfg)
    plan = draft["split_plan"]
    before = copy.deepcopy(plan)
    scfg = cfg.get("synth", {})
    if plan.get("eval_ratio") is None:
        plan["eval_ratio"] = float(scfg.get("eval_ratio", 0.0))
    if plan.get("seed") is None:
        plan["seed"] = int(scfg.get("seed", 42))
    ratio = float(plan["eval_ratio"])
    seed = int(plan["seed"])
    unique = sorted({str(p or "unknown") for p in projects})
    assignments = plan["assignments"]

    if not assignments:
        # 一次性继承现有工作区，避免引入草稿功能本身改变当前项目口径。
        stats = read_json(stage_dirs(cfg)["train"] / "stats.json", default={}) or {}
        legacy = stats.get("split") or {}
        if isinstance(legacy, dict):
            assignments.update({p: legacy[p] for p in unique if legacy.get(p) in ("train", "eval")})
        explicit = {str(p) for p in (scfg.get("eval_projects") or [])}
        if explicit:
            assignments.update({p: ("eval" if p in explicit else "train") for p in unique})

    missing = [p for p in unique if p not in assignments]
    if missing and not assignments:
        if ratio > 0 and len(unique) >= 2:
            n_eval = max(1, round(len(unique) * ratio))
            eval_set = set(sorted(unique, key=lambda p: (_project_hash(seed, p), p))[:n_eval])
        else:
            eval_set = set()
        assignments.update({p: ("eval" if p in eval_set else "train") for p in unique})
    else:
        for project in missing:
            bucket = _project_hash(seed, project) / float(1 << 64)
            assignments[project] = "eval" if ratio > 0 and bucket < ratio else "train"

    # 不保留已不在黄金库中的映射，重新出现时按同一 seed 重新分配。
    plan["assignments"] = {p: assignments[p] for p in unique}
    known = set(unique)
    plan["pinned"] = [p for p in plan["pinned"] if p in known]
    if persist and plan != before:
        save_draft(draft, cfg)
    return plan


def source_revisions(
    cfg: Dict[str, Any],
    *,
    include_review_cache: bool = True,
    include_aux_outputs: bool = False,
) -> Dict[str, Any]:
    """一次读取构建输入哈希，供指纹与 Build 清单共用。"""
    gold_index = load_index(stage_dirs(cfg)["gold"] / "index.json")
    sources = []
    for sid, entry in sorted((gold_index.get("sessions") or {}).items()):
        item = {"session_id": sid}
        for field in ("gold_file", "annotation_file", "diag_file"):
            path = Path(entry.get(field) or "")
            item[field] = sha256_file(path) if path.is_file() else ""
        sources.append(item)
    input_paths = {
        "dpo_pairs": stage_dirs(cfg)["gold"] / "dpo_pairs.jsonl",
        "dpo_exclusions": stage_dirs(cfg)["state"] / "dpo_exclusions.json",
    }
    if include_review_cache:
        input_paths["dpo_review_cache"] = stage_dirs(cfg)["state"] / "dpo_review_cache.json"
    stage3_inputs = {
        name: sha256_file(path) if path.is_file() else ""
        for name, path in input_paths.items()
    }
    aux_outputs = {}
    if include_aux_outputs:
        train_root = stage_dirs(cfg)["train"]
        aux_paths = [train_root / "handoff.md", *sorted(train_root.glob("*.yaml"))]
        aux_outputs = {
            path.name: sha256_file(path) for path in aux_paths if path.is_file()
        }
    return {"sources": sources, "stage3_inputs": stage3_inputs, "aux_outputs": aux_outputs}


def plan_fingerprint(
    cfg: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    include_review_cache: bool = True,
    include_aux_outputs: bool = False,
    revisions: Optional[Dict[str, Any]] = None,
) -> str:
    """构建指纹；可复用同一次 source_revisions 读取结果。"""
    revisions = revisions or source_revisions(
        cfg,
        include_review_cache=include_review_cache,
        include_aux_outputs=include_aux_outputs,
    )
    payload = {
        "plan_version": DATASET_PLAN_VERSION,
        "config_hash": config_hash(cfg),
        "split_plan": draft.get("split_plan") or {},
        "decisions": {
            sample_id: {
                "action": value.get("action"),
                "reason": value.get("reason", ""),
            }
            for sample_id, value in sorted((draft.get("decisions") or {}).items())
            if isinstance(value, dict)
        },
        "sources": revisions["sources"],
        "stage3_inputs": revisions["stage3_inputs"],
    }
    if include_aux_outputs:
        payload["aux_outputs"] = revisions.get("aux_outputs") or {}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def mark_built(cfg: Dict[str, Any], plan_hash: str, build_dir: Optional[Path] = None) -> None:
    with DRAFT_LOCK:
        draft = load_draft(cfg)
        draft["last_build"] = {
            "plan_hash": plan_hash,
            "build_id": plan_hash,
            "built_at": now_iso(),
            "build_dir": str(build_dir) if build_dir else "",
        }
        save_draft(draft, cfg)


def set_project_splits(
    updates: Dict[str, str], cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with DRAFT_LOCK:
        return _set_project_splits_unlocked(updates, cfg)


def _set_project_splits_unlocked(
    updates: Dict[str, str], cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    draft = load_draft(cfg)
    plan = draft["split_plan"]
    assignments = plan["assignments"]
    pinned = set(plan["pinned"])
    ratio = float(plan["eval_ratio"])
    seed = int(plan["seed"])
    for project, split in updates.items():
        if project not in assignments:
            raise ValueError(f"项目不在当前黄金库中: {project}")
        if split not in ("train", "eval", "auto"):
            raise ValueError("split 必须是 train|eval|auto")
        if split == "auto":
            pinned.discard(project)
            bucket = _project_hash(seed, project) / float(1 << 64)
            assignments[project] = "eval" if ratio > 0 and bucket < ratio else "train"
        else:
            assignments[project] = split
            pinned.add(project)
    plan["pinned"] = sorted(pinned)
    save_draft(draft, cfg)
    return plan


class SplitPlanConflict(ValueError):
    pass


def split_plan_hash(plan: Dict[str, Any]) -> str:
    payload = {
        "unit": "project_group",
        "eval_ratio": float(plan["eval_ratio"]),
        "seed": int(plan["seed"]),
        "assignments": plan.get("assignments") or {},
        "pinned": sorted(plan.get("pinned") or []),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def replan_split(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    eval_ratio: float,
    seed: int,
    dry_run: bool = False,
    base_plan: Optional[Dict[str, Any]] = None,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with DRAFT_LOCK:
        return _replan_split_unlocked(
            cfg,
            eval_ratio=eval_ratio,
            seed=seed,
            dry_run=dry_run,
            base_plan=base_plan,
            expected_hash=expected_hash,
        )


def _replan_split_unlocked(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    eval_ratio: float,
    seed: int,
    dry_run: bool = False,
    base_plan: Optional[Dict[str, Any]] = None,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    if not 0.0 <= eval_ratio <= 1.0:
        raise ValueError("eval_ratio 必须在 0 到 1 之间")
    cfg = cfg or load_config()
    draft = load_draft(cfg)
    if not dry_run:
        gold_index = load_index(stage_dirs(cfg)["gold"] / "index.json")
        records = gold_index.get("sessions") or {}
        draft["split_plan"] = _resolve_split_plan_unlocked(
            [r.get("project_group", "unknown") for r in records.values()], cfg, persist=False
        )
    elif base_plan is not None:
        draft["split_plan"] = copy.deepcopy(base_plan)
    plan = draft["split_plan"]
    if expected_hash and split_plan_hash(plan) != expected_hash:
        raise SplitPlanConflict("切分计划已变化，请重新预览后再确认")
    before = dict(plan["assignments"])
    projects = sorted(plan["assignments"])
    pinned = set(plan["pinned"])
    fixed = {p: plan["assignments"][p] for p in projects if p in pinned}
    desired = 0
    if eval_ratio > 0 and len(projects) >= 2:
        desired = max(1, round(len(projects) * eval_ratio))
    remaining = max(0, desired - sum(v == "eval" for v in fixed.values()))
    movable = [p for p in projects if p not in pinned]
    eval_movable = set(sorted(movable, key=lambda p: (_project_hash(seed, p), p))[:remaining])
    plan["eval_ratio"] = float(eval_ratio)
    plan["seed"] = int(seed)
    plan["assignments"] = {
        p: fixed.get(p) or ("eval" if p in eval_movable else "train") for p in projects
    }
    changes = {
        p: {"before": before[p], "after": plan["assignments"][p]}
        for p in projects if before[p] != plan["assignments"][p]
    }
    result = {"split_plan": copy.deepcopy(plan), "changes": changes}
    if not dry_run:
        save_draft(draft, cfg)
    return result


def decision_for(sample_id: str, draft: Dict[str, Any]) -> Dict[str, Any]:
    value = (draft.get("decisions") or {}).get(sample_id) or {}
    if not isinstance(value, dict) or value.get("action") not in ("exclude", "auto"):
        return {"action": "auto", "reason": ""}
    return {
        "action": value["action"],
        "reason": str(value.get("reason") or ""),
        "by": str(value.get("by") or "user"),
        "at": value.get("at"),
    }


def dpo_decision_for(
    sample_id: str,
    draft: Dict[str, Any],
    legacy: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """通用账本优先；显式 auto 可以撤销旧 DPO 排除账本。"""
    if sample_id in (draft.get("decisions") or {}):
        return decision_for(sample_id, draft)
    old = legacy.get(sample_id)
    if old is not None:
        if not isinstance(old, dict):
            old = {"reason": str(old)}
        return {
            "action": "exclude",
            "reason": str(old.get("reason") or "旧 DPO 排除账本"),
            "by": str(old.get("by") or "legacy"),
            "at": old.get("at"),
        }
    return {"action": "auto", "reason": ""}


def update_decisions(
    updates: Iterable[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with DRAFT_LOCK:
        return _update_decisions_unlocked(updates, cfg)


def _update_decisions_unlocked(
    updates: Iterable[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    draft = load_draft(cfg)
    decisions = draft["decisions"]
    changed = 0
    for item in updates:
        sample_id = str(item.get("sample_id") or "").strip()
        action = str(item.get("action") or "").strip()
        if not sample_id:
            raise ValueError("sample_id 不能为空")
        if action not in ("exclude", "auto"):
            raise ValueError("action 必须是 exclude|auto")
        reason = str(item.get("reason") or "").strip()
        if action == "exclude" and not reason:
            raise ValueError("排除样本必须填写原因")
        by = str(item.get("by") or "user")
        old = decisions.get(sample_id) or {}
        if all(old.get(key) == value for key, value in (("action", action), ("reason", reason), ("by", by))):
            continue
        decisions[sample_id] = {
            "action": action,
            "reason": reason,
            "by": by,
            "at": now_iso(),
        }
        changed += 1
    if changed:
        save_draft(draft, cfg)
    return {"changed": changed, "decisions": len(decisions), "updated_at": draft["updated_at"]}
