"""虚拟数据集实体与演进关系图（Dataset Lineage DAG + CAS 四层时间旅行查询引擎）。"""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline import cas_store, curation, dataset_build, dataset_draft, stage2_gold
from pipeline.common import pilog
from pipeline.common.config import load_config, stage_dirs
from pipeline.common.store import load_index, read_json, read_jsonl, sha256_file

from . import eval_tools, inspect_tools, ledger, report_tools

_ZIP_PREFIXES = ("goldthread-train-", "goldthread-delivery-")


def normalize_pack_id(ref: Optional[str]) -> str:
    """将路径、zip 文件名或短哈希统一归一化为 package_hash（或 'workspace'）。"""
    if not ref:
        return ""
    raw = str(ref).strip()
    if raw.lower() in ("workspace", "draft", "working-tree"):
        return "workspace"
    name = Path(raw).name
    stem = name[:-4] if name.lower().endswith(".zip") else name
    for prefix in _ZIP_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _delivery_dir(cfg: Dict[str, Any]) -> Path:
    return stage_dirs(cfg)["root"] / "04_release" / "delivery"


def _list_delivery_zips(cfg: Dict[str, Any]) -> List[Path]:
    ddir = _delivery_dir(cfg)
    if not ddir.exists():
        return []
    found: List[Path] = []
    for prefix in _ZIP_PREFIXES:
        found.extend(p for p in ddir.glob(f"{prefix}*.zip") if p.is_file())
    return sorted({p.resolve() for p in found}, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_zip_json(zp: Path, candidates: Tuple[str, ...]) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            names = set(zf.namelist())
            for arc in candidates:
                if arc in names:
                    data = json.loads(zf.read(arc).decode("utf-8"))
                    return data if isinstance(data, dict) else {}
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError):
        pass
    return {}


def _read_zip_jsonl(zp: Path, rel_name: str) -> List[Dict[str, Any]]:
    arcs = (
        f"goldthread-train/data/{rel_name}",
        f"data/{rel_name}",
        rel_name,
    )
    rows: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            names = set(zf.namelist())
            for arc in arcs:
                if arc in names:
                    text = zf.read(arc).decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                            if isinstance(obj, dict):
                                rows.append(obj)
                        except json.JSONDecodeError:
                            continue
                    return rows
    except (OSError, zipfile.BadZipFile, KeyError):
        pass
    return rows


def _resolve_pack_zip(dataset_id: str, cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    target = normalize_pack_id(dataset_id)
    for zp in _list_delivery_zips(cfg):
        man = _read_zip_json(zp, ("goldthread-train/data/MANIFEST.json", "data/MANIFEST.json", "MANIFEST.json"))
        pkg_h = normalize_pack_id(man.get("package_hash") or zp.name)
        if pkg_h == target or zp.name == dataset_id or pkg_h.startswith(target):
            return zp, man
    raise FileNotFoundError(f"未找到数据集交付包: {dataset_id}")


def _session_snapshot_status(
    sid: str,
    cas_entry: Dict[str, Any],
    raw_sessions: Dict[str, Any],
    gold_sessions: Dict[str, Any],
) -> str:
    """判定历史包中的某条会话相较当前工作区的状态：exact | workspace_modified | workspace_deleted。"""
    if sid not in raw_sessions and sid not in gold_sessions:
        return "workspace_deleted"
    gold_e = gold_sessions.get(sid) or {}
    expected_gold = (
        (cas_entry.get("files") or {}).get("gold_file")
        or (cas_entry.get("expected_revisions") or {}).get("gold_file")
        or ""
    )
    cur_gold_path = gold_e.get("gold_file") or ""
    if not cur_gold_path or not Path(cur_gold_path).is_file():
        return "workspace_deleted" if sid not in raw_sessions else "workspace_modified"
    if expected_gold and sha256_file(Path(cur_gold_path)) != expected_gold:
        return "workspace_modified"
    return "exact"


def _pack_node_summary(zp: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    man = _read_zip_json(zp, ("goldthread-train/data/MANIFEST.json", "data/MANIFEST.json", "MANIFEST.json"))
    len_stats = _read_zip_json(
        zp,
        ("goldthread-train/data/length_stats.json", "data/length_stats.json", "length_stats.json"),
    )
    pkg_h = normalize_pack_id(man.get("package_hash") or zp.name)
    build_id = str(man.get("build_id") or "").strip()
    counts = man.get("counts") if isinstance(man.get("counts"), dict) else {}
    fmt_obj = (
        man.get("format_contract")
        if isinstance(man.get("format_contract"), dict)
        else (man.get("format") if isinstance(man.get("format"), dict) else {})
    )

    projects_set: set[str] = set()
    sessions_count = 0
    cas_object_count = 0
    cas_fidelity = "none"
    if build_id:
        try:
            b_dir, _b_man = dataset_build.resolve_build(cfg, build_id)
            b_stats = read_json(b_dir / "stats.json", default={}) or {}
            projects_set.update(str(k) for k in (b_stats.get("by_project") or {}).keys() if k)
            cas_man = cas_store.load_build_cas_manifest(build_id, cfg=cfg, readonly=True)
            sessions_map = cas_man.get("sessions") or {}
            sessions_count = len(sessions_map)
            cas_object_count = int(cas_man.get("object_count") or 0)
            cas_fidelity = str(cas_man.get("fidelity") or "backfilled")
            if not projects_set:
                projects_set.update(
                    str(v.get("project_group"))
                    for v in sessions_map.values()
                    if isinstance(v, dict) and v.get("project_group")
                )
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

    if not sessions_count:
        # 无关联 Build 目录的早期独立包，从包内 jsonl metadata 提取项目与会话规模
        sids_seen: set[str] = set()
        for fn in ("sft.train.lf.jsonl", "sft.eval.lf.jsonl"):
            for row in _read_zip_jsonl(zp, fn):
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                if meta.get("session_id"):
                    sids_seen.add(str(meta["session_id"]))
                if meta.get("project_group"):
                    projects_set.add(str(meta["project_group"]))
        sessions_count = len(sids_seen)

    lg = man.get("length_gate")
    lg_ok = bool(isinstance(lg, dict) and lg and "_skip" not in lg)
    has_explicit_parent = "parent_pack" in man
    parent_pack = normalize_pack_id(man.get("parent_pack")) or None

    return {
        "id": pkg_h,
        "kind": "pack",
        "name": zp.name,
        "package_hash": pkg_h,
        "parent_pack": parent_pack,
        "has_explicit_parent": has_explicit_parent,
        "build_id": build_id or None,
        "generated_at": str(man.get("generated_at") or ""),
        "size_bytes": zp.stat().st_size,
        "profile": str(fmt_obj.get("profile") or ""),
        "format_label": str(fmt_obj.get("label") or fmt_obj.get("tool_call") or ""),
        "sft_train": int(counts.get("sft.train.lf.jsonl") or 0),
        "sft_eval": int(counts.get("sft.eval.lf.jsonl") or 0),
        "dpo_train": int(counts.get("dpo.train.lf.jsonl") or 0),
        "dpo_eval": int(counts.get("dpo.eval.lf.jsonl") or 0),
        "projects_count": len(projects_set),
        "sessions_count": sessions_count,
        "cas_object_count": cas_object_count,
        "cas_fidelity": cas_fidelity,
        "length_gate_ok": lg_ok,
        "length": len_stats or None,
        "is_head": False,
    }


def _find_pack_ledger_index(all_ledger_asc: List[Dict[str, Any]], pack_id: Optional[str]) -> Optional[int]:
    """在正序账本中查找某次交付包生成的物理位置下标（取最后一次匹配）。"""
    if not pack_id:
        return None
    hit: Optional[int] = None
    for i, ev in enumerate(all_ledger_asc):
        if ev.get("tool") not in ("pack.delivery", "web.package"):
            continue
        aff = ev.get("affected") if isinstance(ev.get("affected"), dict) else {}
        if normalize_pack_id(aff.get("pack")) == pack_id:
            hit = i
    return hit


def _deterministic_config_digest(cfg: Dict[str, Any]) -> str:
    """提取影响数据清洗与训练切片产物的确定性配置摘要（排除远程隐式模型开关）。"""
    turns_cfg = copy.deepcopy(cfg.get("turns") or {})
    if isinstance(turns_cfg, dict):
        turns_cfg.pop("use_model", None)
    payload = {
        "admission": cfg.get("admission") or {},
        "turns": turns_cfg,
        "gold": cfg.get("gold") or {},
        "synth": cfg.get("synth") or {},
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _has_draft_diverged_from_build(
    head_bid: str,
    draft: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """精确比对当前工作区实质性数据输入、配置与草稿决策是否偏离 head_pack 所绑定的 Build。"""
    dirs = stage_dirs(cfg)
    latest_b = read_json(dirs["root"] / "04_release" / "builds" / "LATEST_BUILD.json", default={}) or {}
    latest_bid = str(latest_b.get("build_id") or "").strip()
    if latest_bid and latest_bid != head_bid:
        return True, "build_ahead_of_pack"
    try:
        b_dir, b_man = dataset_build.resolve_build(cfg, head_bid)
    except (FileNotFoundError, ValueError, RuntimeError):
        return False, ""

    try:
        b_cfg = dataset_build.load_build_config(b_dir, cfg)
        if _deterministic_config_digest(cfg) != _deterministic_config_digest(b_cfg):
            return True, "synth_config_changed"
    except RuntimeError:
        pass

    cur_rev = dataset_draft.source_revisions(cfg, include_aux_outputs=False)
    bld_rev = b_man.get("source_revisions") or {}
    if cur_rev.get("sources") != bld_rev.get("sources"):
        return True, "session_sources_changed"

    cur_inputs = {k: v for k, v in (cur_rev.get("stage3_inputs") or {}).items() if v}
    bld_inputs = {k: v for k, v in (bld_rev.get("stage3_inputs") or {}).items() if v}
    if cur_inputs != bld_inputs:
        return True, "stage3_inputs_changed"

    for field in ("decisions", "scope_decisions", "split_plan"):
        if (draft.get(field) or {}) != (b_man.get(field) or {}):
            return True, f"{field}_changed"
    return False, ""


def _compute_workspace_status(
    nodes_desc: List[Dict[str, Any]],
    all_ledger_asc: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    dirs = stage_dirs(cfg)
    raw_idx = load_index(dirs["raw"] / "index.json")
    gold_idx = load_index(dirs["gold"] / "index.json")
    raw_sessions = raw_idx.get("sessions") or {}
    gold_sessions = gold_idx.get("sessions") or {}

    stages_state = read_json(dirs["state"] / "stages.json", default={}) or {}
    s1_at = str((stages_state.get("stage1") or {}).get("last_run") or "")
    s2_at = str((stages_state.get("stage2") or {}).get("last_run") or "")
    s3_at = str((stages_state.get("stage3") or {}).get("last_run") or "")
    sync_dirty = bool((s1_at and s2_at and s1_at > s2_at) or (s2_at and s3_at and s2_at > s3_at))

    draft = dataset_draft.load_draft(cfg)
    current_plan_hash = (
        dataset_draft.plan_fingerprint(cfg, draft, include_aux_outputs=True)
        if gold_sessions
        else ""
    )

    head = nodes_desc[0] if nodes_desc else None
    reasons: List[str] = []
    unpurified = sorted(set(raw_sessions.keys()) - set(gold_sessions.keys()))

    if head is None:
        if raw_sessions or gold_sessions:
            reasons.append("no_delivery_pack")
    else:
        if unpurified:
            reasons.append("unpurified_raw_sessions")
        if sync_dirty:
            reasons.append("pipeline_unsynced")
        head_bid = str(head.get("build_id") or "")
        if head_bid:
            diverged, div_reason = _has_draft_diverged_from_build(head_bid, draft, cfg)
            if diverged:
                reasons.append(div_reason)

    head_ts = str((head or {}).get("generated_at") or "")
    head_idx = _find_pack_ledger_index(all_ledger_asc, head["id"] if head else None)
    if head_idx is not None:
        tail_slice = all_ledger_asc[head_idx + 1:]
    else:
        tail_slice = [
            ev for ev in all_ledger_asc
            if not head_ts or str(ev.get("timestamp") or "") > head_ts
        ]
    pending_ops = [
        {
            "op_id": ev.get("op_id"),
            "timestamp": ev.get("timestamp"),
            "actor": ev.get("actor"),
            "tool": ev.get("tool"),
            "reason": ev.get("reason"),
        }
        for ev in tail_slice
        if ev.get("tool") not in ("report.publish", "pack.delivery", "web.package")
    ]

    train_stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    t_counts = train_stats.get("counts") or {}
    projects = sorted(
        {
            str(e.get("project_group") or "")
            for e in (*raw_sessions.values(), *gold_sessions.values())
            if isinstance(e, dict) and e.get("project_group")
        }
    )

    return {
        "dirty": bool(reasons),
        "reasons": reasons,
        "head_pack": head["id"] if head else None,
        "current_plan_hash": current_plan_hash or None,
        "unpurified_count": len(unpurified),
        "counts": {
            "raw_sessions": len(raw_sessions),
            "gold_sessions": len(gold_sessions),
            "projects_count": len(projects),
            "sft_train": int(t_counts.get("sft_step_windows_sharegpt.train.jsonl") or 0),
            "sft_eval": int(t_counts.get("sft_step_windows_sharegpt.eval.jsonl") or 0),
            "dpo_train": int(t_counts.get("dpo_preference_pairs.train.jsonl") or 0),
            "dpo_eval": int(t_counts.get("dpo_preference_pairs.eval.jsonl") or 0),
        },
        "pending_ops": pending_ops,
    }


def build_dataset_graph(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构建完整的数据集演进关系图（DAG）：包节点 + 报告/账本桥接边 + 动态工作空间状态。"""
    cfg = cfg or load_config()
    zips = _list_delivery_zips(cfg)
    nodes_desc = [_pack_node_summary(zp, cfg) for zp in zips]
    nodes_desc.sort(key=lambda n: (n.get("generated_at") or "", n["id"]), reverse=True)
    if nodes_desc:
        nodes_desc[0]["is_head"] = True

    node_by_id = {n["id"]: n for n in nodes_desc}
    nodes_asc = list(reversed(nodes_desc))

    # 读取全部评估报告并解析其 before_pack -> after_pack 关系
    report_index = report_tools.list_reports(cfg=cfg)
    full_reports: List[Dict[str, Any]] = []
    report_parent_hint: Dict[str, str] = {}
    for r_idx in report_index:
        rid = str(r_idx.get("report_id") or "")
        if not rid:
            continue
        try:
            full_r = report_tools.get_report(rid, cfg=cfg)
        except (FileNotFoundError, ValueError):
            full_r = dict(r_idx)
        b_id = normalize_pack_id(full_r.get("before_pack"))
        a_id = normalize_pack_id(full_r.get("after_pack"))
        full_r["_before_id"] = b_id
        full_r["_after_id"] = a_id
        full_reports.append(full_r)
        if b_id and a_id and a_id in node_by_id and b_id in node_by_id and a_id not in report_parent_hint:
            report_parent_hint[a_id] = b_id

    # 读取操作账本（按时间正序）
    all_ledger_desc = ledger.read_ledger(cfg=cfg, limit=1000)
    all_ledger_asc = list(reversed(all_ledger_desc))

    # 为每个节点解析父节点（区分显式根 parent_pack:null 与旧包无字段，并保证全局无环 DAG）
    parent_map: Dict[str, Tuple[Optional[str], bool]] = {}
    for idx, node in enumerate(nodes_asc):
        nid = node["id"]
        explicit_p = node.get("parent_pack")
        has_exp = bool(node.get("has_explicit_parent"))
        inferred = False
        parent_id: Optional[str] = None
        if explicit_p and explicit_p in node_by_id and explicit_p != nid:
            parent_id = explicit_p
        elif nid in report_parent_hint and report_parent_hint[nid] != nid:
            parent_id = report_parent_hint[nid]
        elif has_exp and explicit_p is None:
            # MANIFEST 明确记录 parent_pack 为 null：确认为根节点，严禁按时间反推父包
            parent_id = None
        elif idx > 0:
            parent_id = nodes_asc[idx - 1]["id"]
            inferred = True
        parent_map[nid] = (parent_id, inferred)

    # DAG 无环校验：若任何父指针链形成环路，优先切断推断边或打破回边，确保至少存在根节点
    for nid in [n["id"] for n in nodes_asc]:
        seen_chain: List[str] = [nid]
        cur_p, _ = parent_map.get(nid, (None, False))
        while cur_p:
            if cur_p in seen_chain:
                cycle_nodes = seen_chain[seen_chain.index(cur_p):]
                victim = next((cn for cn in cycle_nodes if parent_map.get(cn, (None, False))[1]), cycle_nodes[0])
                parent_map[victim] = (None, False)
                break
            seen_chain.append(cur_p)
            cur_p, _ = parent_map.get(cur_p, (None, False))

    edges: List[Dict[str, Any]] = []
    for node in nodes_asc:
        nid = node["id"]
        parent_id, inferred = parent_map.get(nid, (None, False))
        node["resolved_parent"] = parent_id
        if not parent_id:
            continue

        parent_node = node_by_id[parent_id]
        t_start = str(parent_node.get("generated_at") or "")
        t_end = str(node.get("generated_at") or "")
        p_idx = _find_pack_ledger_index(all_ledger_asc, parent_id)
        c_idx = _find_pack_ledger_index(all_ledger_asc, nid)

        edge_reports = []
        for fr in full_reports:
            if fr.get("_before_id") == parent_id and fr.get("_after_id") == nid:
                diff = fr.get("diff_summary") if isinstance(fr.get("diff_summary"), dict) else {}
                edge_reports.append(
                    {
                        "report_id": fr.get("report_id"),
                        "title": fr.get("title"),
                        "summary": fr.get("summary"),
                        "created_at": fr.get("created_at"),
                        "deltas": diff.get("deltas") or {},
                    }
                )

        edge_ops = []
        for l_idx, ev in enumerate(all_ledger_asc):
            aff = ev.get("affected") if isinstance(ev.get("affected"), dict) else {}
            prm = ev.get("params") if isinstance(ev.get("params"), dict) else {}
            ev_ts = str(ev.get("timestamp") or "")
            ev_pack = normalize_pack_id(aff.get("pack") or prm.get("after_pack"))
            ev_bid = str(aff.get("build_id") or prm.get("build_id") or "")
            direct_hit = bool(
                (ev_pack and ev_pack == nid)
                or (node.get("build_id") and ev_bid == node.get("build_id"))
            )
            if p_idx is not None and c_idx is not None and p_idx < c_idx:
                in_window = (p_idx < l_idx <= c_idx) and ev.get("tool") != "report.publish"
            else:
                in_window = bool(
                    t_start and t_end and (t_start < ev_ts <= t_end)
                    and ev.get("tool") != "report.publish"
                )
            if direct_hit or in_window:
                edge_ops.append(
                    {
                        "op_id": ev.get("op_id"),
                        "timestamp": ev_ts,
                        "actor": ev.get("actor"),
                        "tool": ev.get("tool"),
                        "reason": ev.get("reason"),
                    }
                )

        edges.append(
            {
                "id": f"{parent_id}->{nid}",
                "from": parent_id,
                "to": nid,
                "kind": "pack",
                "inferred": inferred,
                "reports": edge_reports,
                "ledger_ops": edge_ops,
            }
        )

    workspace = _compute_workspace_status(nodes_desc, all_ledger_asc, cfg)
    if workspace["dirty"] and workspace.get("head_pack"):
        edges.append(
            {
                "id": f"{workspace['head_pack']}->workspace",
                "from": workspace["head_pack"],
                "to": "workspace",
                "kind": "draft",
                "inferred": False,
                "reports": [],
                "ledger_ops": workspace["pending_ops"],
            }
        )

    return {
        "nodes": nodes_asc,
        "head_pack": nodes_desc[0]["id"] if nodes_desc else None,
        "edges": edges,
        "workspace": workspace,
        "cas_total_objects": cas_store.count_cas_objects(cfg),
    }


def _compute_dataset_breakdown(
    *,
    zp: Optional[Path],
    build_id: Optional[str],
    is_workspace: bool,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """构建单个虚拟数据集自身的项目切分表、会话阶段占比、DPO 来源结构与病态分布。"""
    stats_obj: Dict[str, Any] = {}
    if is_workspace:
        stats_obj = read_json(stage_dirs(cfg)["train"] / "stats.json", default={}) or {}
    elif build_id:
        try:
            b_dir, _ = dataset_build.resolve_build(cfg, build_id)
            stats_obj = read_json(b_dir / "stats.json", default={}) or {}
        except (FileNotFoundError, ValueError, RuntimeError):
            stats_obj = {}

    split_map = dict(stats_obj.get("split") or {})
    by_stage = dict(stats_obj.get("by_stage") or {})
    dpo_by_source = dict(stats_obj.get("dpo_by_source") or {})
    pathologies = dict(stats_obj.get("diagnostics_by_pathology") or {})
    proj_sft: Dict[str, int] = {}
    proj_dpo: Dict[str, int] = {}

    if zp is not None:
        for sp_name, fn in (("train", "sft.train.lf.jsonl"), ("eval", "sft.eval.lf.jsonl")):
            for r in _read_zip_jsonl(zp, fn):
                meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                pg = str(meta.get("project_group") or r.get("project_group") or "unknown")
                proj_sft[pg] = proj_sft.get(pg, 0) + 1
                split_map.setdefault(pg, sp_name)
                if not stats_obj and meta.get("stage"):
                    st_k = str(meta["stage"])
                    by_stage[st_k] = by_stage.get(st_k, 0) + 1
        for sp_name, fn in (("train", "dpo.train.lf.jsonl"), ("eval", "dpo.eval.lf.jsonl")):
            for r in _read_zip_jsonl(zp, fn):
                meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                pg = str(meta.get("project_group") or r.get("project_group") or "unknown")
                proj_dpo[pg] = proj_dpo.get(pg, 0) + 1
                split_map.setdefault(pg, sp_name)
                if not stats_obj:
                    src_k = str(meta.get("source") or r.get("source") or "unknown")
                    dpo_by_source[src_k] = dpo_by_source.get(src_k, 0) + 1
    else:
        for pg, cnt in (stats_obj.get("by_project") or {}).items():
            proj_sft[str(pg)] = int(cnt or 0)
        train_dir = stage_dirs(cfg)["train"]
        for fn in ("dpo_preference_pairs.train.jsonl", "dpo_preference_pairs.eval.jsonl"):
            for r in read_jsonl(train_dir / fn):
                pg = str((r.get("metadata") or {}).get("project_group") or r.get("project_group") or "unknown")
                proj_dpo[pg] = proj_dpo.get(pg, 0) + 1

    all_projs = sorted(
        set(proj_sft.keys()) | set(proj_dpo.keys()) | set(split_map.keys()),
        key=lambda p: (-(proj_sft.get(p, 0) + proj_dpo.get(p, 0)), p),
    )
    projects_rows = [
        {
            "project": p,
            "split": str(split_map.get(p) or "train"),
            "sft_rows": int(proj_sft.get(p, 0)),
            "dpo_rows": int(proj_dpo.get(p, 0)),
        }
        for p in all_projs
    ]
    return {
        "projects": projects_rows,
        "by_stage": by_stage,
        "dpo_by_source": dpo_by_source,
        "pathologies": pathologies,
    }


def get_virtual_dataset_overview(
    dataset_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    include_gate: bool = False,
) -> Dict[str, Any]:
    """返回单个虚拟数据集（正式 hash 包或 'workspace'）的概览、门禁与演进上下文。"""
    cfg = cfg or load_config()
    norm_id = normalize_pack_id(dataset_id)
    graph = build_dataset_graph(cfg)

    if norm_id == "workspace":
        ws = graph["workspace"]
        return {
            "id": "workspace",
            "kind": "draft",
            "readonly": False,
            "workspace": ws,
            "parent_pack": ws.get("head_pack"),
            "ledger_ops": ws.get("pending_ops") or [],
            "reports": [],
            "breakdown": _compute_dataset_breakdown(zp=None, build_id=None, is_workspace=True, cfg=cfg),
        }

    zp, man = _resolve_pack_zip(norm_id, cfg)
    node = next((n for n in graph["nodes"] if n["id"] == norm_id), None) or _pack_node_summary(zp, cfg)
    incoming_edge = next((e for e in graph["edges"] if e["to"] == norm_id), None)
    gate = build_gate_summary_safe(zp, cfg) if include_gate else None
    breakdown = _compute_dataset_breakdown(
        zp=zp,
        build_id=node.get("build_id"),
        is_workspace=False,
        cfg=cfg,
    )

    return {
        **node,
        "readonly": True,
        "manifest": {
            "package": man.get("package") or zp.name,
            "package_hash": node["id"],
            "parent_pack": node.get("resolved_parent") or node.get("parent_pack"),
            "build_id": node.get("build_id"),
            "config_hash": man.get("config_hash"),
            "generated_at": node.get("generated_at"),
            "format": man.get("format_contract") or man.get("format") or {},
            "counts": man.get("counts") or {},
            "length_gate": man.get("length_gate") or {},
        },
        "gate": gate,
        "incoming_edge": incoming_edge,
        "breakdown": breakdown,
    }


def build_gate_summary_safe(zp: Path, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from . import build_tools

    try:
        return build_tools.gate_check(str(zp), cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "error": str(exc), "checks": [], "violations": []}


def list_virtual_dataset_train(
    dataset_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    kind: str = "all",
    split: str = "",
    project: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """读取虚拟数据集冻结的训练切片（SFT / DPO），支持筛选与分页。"""
    cfg = cfg or load_config()
    norm_id = normalize_pack_id(dataset_id)
    rows_all: List[Dict[str, Any]] = []
    projects_set: set[str] = set()

    if norm_id == "workspace":
        train_dir = stage_dirs(cfg)["train"]
        file_specs = [
            ("sft_step_windows_sharegpt.train.jsonl", "sft", "train"),
            ("sft_step_windows_sharegpt.eval.jsonl", "sft", "eval"),
            ("dpo_preference_pairs.train.jsonl", "dpo", "train"),
            ("dpo_preference_pairs.eval.jsonl", "dpo", "eval"),
        ]
        for fname, k_tag, sp_tag in file_specs:
            for idx, r in enumerate(read_jsonl(train_dir / fname)):
                rows_all.append(_compact_train_row(r, k_tag, sp_tag, f"{fname}:{idx}"))
    else:
        zp, _man = _resolve_pack_zip(norm_id, cfg)
        file_specs = [
            ("sft.train.lf.jsonl", "sft", "train"),
            ("sft.eval.lf.jsonl", "sft", "eval"),
            ("dpo.train.lf.jsonl", "dpo", "train"),
            ("dpo.eval.lf.jsonl", "dpo", "eval"),
        ]
        for fname, k_tag, sp_tag in file_specs:
            for idx, r in enumerate(_read_zip_jsonl(zp, fname)):
                rows_all.append(_compact_train_row(r, k_tag, sp_tag, f"{fname}:{idx}"))

    for item in rows_all:
        if item.get("project_group"):
            projects_set.add(item["project_group"])

    q_low = q.strip().lower()
    filtered: List[Dict[str, Any]] = []
    for item in rows_all:
        if kind in ("sft", "dpo") and item["kind"] != kind:
            continue
        if split in ("train", "eval") and item["split"] != split:
            continue
        if project and item["project_group"] != project:
            continue
        if q_low and q_low not in item["_search_blob"]:
            continue
        out_item = {k: v for k, v in item.items() if k != "_search_blob"}
        filtered.append(out_item)

    sliced = filtered[max(0, offset): max(0, offset) + max(1, min(limit, 200))]
    return {
        "dataset_id": norm_id,
        "total": len(filtered),
        "offset": max(0, offset),
        "limit": limit,
        "projects": sorted(projects_set),
        "rows": sliced,
    }


def _compact_train_row(row: Dict[str, Any], kind: str, split: str, fallback_id: str) -> Dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    sid = str(meta.get("session_id") or row.get("session_id") or "")
    proj = str(meta.get("project_group") or row.get("project_group") or "")
    sample_id = str(row.get("sample_id") or meta.get("sample_id") or fallback_id)
    stage = str(meta.get("stage") or meta.get("source") or row.get("source") or "")
    weight = meta.get("sample_weight") or row.get("sample_weight") or 1.0

    preview = ""
    char_len = 0
    if kind == "sft":
        convs = row.get("conversations") or []
        char_len = sum(len(str(c.get("value") or "")) for c in convs if isinstance(c, dict))
        last_gpt = next(
            (str(c.get("value") or "") for c in reversed(convs) if isinstance(c, dict) and c.get("from") in ("gpt", "assistant")),
            "",
        )
        preview = last_gpt.replace("\n", " ")[:180]
    else:
        chosen = row.get("chosen")
        chosen_s = chosen.get("value", "") if isinstance(chosen, dict) else str(chosen or "")
        rejected = row.get("rejected")
        rej_s = rejected.get("value", "") if isinstance(rejected, dict) else str(rejected or "")
        char_len = len(chosen_s) + len(rej_s) + len(json.dumps(row.get("conversations") or [], ensure_ascii=False))
        preview = f"Chosen: {chosen_s.replace(chr(10), ' ')[:110]}"

    return {
        "sample_id": sample_id,
        "kind": kind,
        "split": split,
        "session_id": sid,
        "project_group": proj,
        "stage": stage,
        "sample_weight": weight,
        "approx_chars": char_len,
        "preview": preview,
        "raw_row": row,
        "_search_blob": f"{sample_id} {sid} {proj} {stage} {preview}".lower(),
    }


def _load_pack_cas_and_build(dataset_id: str, cfg: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any], Dict[str, Any]]:
    zp, man = _resolve_pack_zip(dataset_id, cfg)
    pkg_h = normalize_pack_id(man.get("package_hash") or zp.name)
    build_id = str(man.get("build_id") or "").strip() or None
    if build_id:
        try:
            b_dir, _b_man = dataset_build.resolve_build(cfg, build_id)
            b_stats = read_json(b_dir / "stats.json", default={}) or {}
            cas_man = cas_store.load_build_cas_manifest(build_id, cfg=cfg, readonly=True)
            return pkg_h, build_id, cas_man, b_stats
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

    # 无关联 Build 目录的早期包：从 zip 内切片反解参与会话列表
    sessions_fallback: Dict[str, Any] = {}
    for fn in ("sft.train.lf.jsonl", "sft.eval.lf.jsonl", "dpo.train.lf.jsonl", "dpo.eval.lf.jsonl"):
        for r in _read_zip_jsonl(zp, fn):
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            sid = str(meta.get("session_id") or "")
            if not sid:
                continue
            ent = sessions_fallback.setdefault(
                sid,
                {
                    "session_id": sid,
                    "project_group": str(meta.get("project_group") or "unknown"),
                    "dominant_model": str(meta.get("model") or "unknown"),
                    "cwd": "",
                    "origin": "pi",
                    "raw_stats": {},
                    "gold_stats": {},
                    "dpo_count": 0,
                    "files": {},
                },
            )
            if fn.startswith("dpo."):
                ent["dpo_count"] = int(ent.get("dpo_count") or 0) + 1
    return pkg_h, None, {"version": 1, "fidelity": "none", "object_count": 0, "sessions": sessions_fallback}, {}


def list_virtual_dataset_gold(
    dataset_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    project: str = "",
    q: str = "",
) -> Dict[str, Any]:
    """读取虚拟数据集包含的黄金会话列表（历史包通过 CAS 精确隔离范围并标注工作区漂移状态）。"""
    cfg = cfg or load_config()
    norm_id = normalize_pack_id(dataset_id)
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})

    rows: List[Dict[str, Any]] = []
    projects_set: set[str] = set()
    q_low = q.strip().lower()

    draft = dataset_draft.load_draft(cfg)
    if norm_id == "workspace":
        t_stats = read_json(dirs["train"] / "stats.json", default={}) or {}
        funnels = t_stats.get("session_funnels") or {}
        split_map = t_stats.get("split") or {}
        ws_sft_counts: Dict[str, int] = {}
        for fn in ("sft_step_windows_sharegpt.train.jsonl", "sft_step_windows_sharegpt.eval.jsonl"):
            for r in read_jsonl(dirs["train"] / fn):
                rsid = str((r.get("metadata") or {}).get("session_id") or r.get("session_id") or "")
                if rsid:
                    ws_sft_counts[rsid] = ws_sft_counts.get(rsid, 0) + 1
        ws_dpo_counts: Dict[str, int] = {}
        for fn in ("dpo_preference_pairs.train.jsonl", "dpo_preference_pairs.eval.jsonl"):
            for r in read_jsonl(dirs["train"] / fn):
                rsid = str((r.get("metadata") or {}).get("session_id") or r.get("session_id") or "")
                if rsid:
                    ws_dpo_counts[rsid] = ws_dpo_counts.get(rsid, 0) + 1
        has_ws_train_files = bool(ws_sft_counts or ws_dpo_counts)

        for sid, ge in sorted(gold_sessions.items()):
            pg = str(ge.get("project_group") or "unknown")
            projects_set.add(pg)
            g_stats = ge.get("stats") or {}
            fnl = (funnels.get(sid) or {}).get("funnel") or {}
            usage = dataset_draft.session_kind_exclusions(draft, sid, pg)
            sft_off = bool(usage.get("sft_excluded"))
            dpo_off = bool(usage.get("dpo_excluded"))
            admitted = fnl.get("admitted_steps")
            sft_win = 0 if sft_off else (ws_sft_counts.get(sid, 0) if has_ws_train_files else int(0 if admitted is None else admitted))
            dpo_n = 0 if dpo_off else (ws_dpo_counts.get(sid, 0) if has_ws_train_files else int(ge.get("dpo_count") or 0))
            item = {
                "session_id": sid,
                "project_group": pg,
                "dominant_model": str(ge.get("dominant_model") or "unknown"),
                "split": str(split_map.get(pg) or "train"),
                "raw_steps": int(g_stats.get("raw_steps") or g_stats.get("units_in") or 0),
                "gold_steps": int(g_stats.get("gold_steps") or g_stats.get("units_out") or 0),
                "sft_windows": sft_win,
                "dpo_count": dpo_n,
                "sft_excluded": sft_off,
                "dpo_excluded": dpo_off,
                "cas_gold_sha256": "",
                "snapshot_status": "workspace",
                "readonly": False,
            }
            if project and pg != project:
                continue
            if q_low and q_low not in f"{sid} {pg} {item['dominant_model']}".lower():
                continue
            rows.append(item)
        return {"dataset_id": "workspace", "readonly": False, "total": len(rows), "projects": sorted(projects_set), "sessions": rows}

    zp, _man = _resolve_pack_zip(norm_id, cfg)
    pkg_h, build_id, cas_man, b_stats = _load_pack_cas_and_build(norm_id, cfg)
    split_map = b_stats.get("split") or {}

    # 直接统计该交付包 zip 内实际写入的每会话 SFT 唯一窗口数与 DPO 对数（避免 0 or gold_steps 假值回退）
    pack_sft_unique: Dict[str, set[str]] = {}
    pack_sft_rows: Dict[str, int] = {}
    for fn in ("sft.train.lf.jsonl", "sft.eval.lf.jsonl"):
        for idx, r in enumerate(_read_zip_jsonl(zp, fn)):
            meta = r.get("metadata") if isinstance(row := r, dict) and isinstance(r.get("metadata"), dict) else {}
            rsid = str(meta.get("session_id") or row.get("session_id") or "")
            if not rsid:
                continue
            pack_sft_rows[rsid] = pack_sft_rows.get(rsid, 0) + 1
            raw_smp_id = str(row.get("sample_id") or meta.get("sample_id") or f"{fn}:{idx}")
            logical_id = str(meta.get("origin_sample_id") or raw_smp_id.split("#w", 1)[0])
            pack_sft_unique.setdefault(rsid, set()).add(logical_id)

    pack_dpo_rows: Dict[str, int] = {}
    for fn in ("dpo.train.lf.jsonl", "dpo.eval.lf.jsonl"):
        for r in _read_zip_jsonl(zp, fn):
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            rsid = str(meta.get("session_id") or r.get("session_id") or "")
            if rsid:
                pack_dpo_rows[rsid] = pack_dpo_rows.get(rsid, 0) + 1

    build_draft_like = {"decisions": {}, "scope_decisions": {}, "split_plan": {}}
    if build_id:
        try:
            _bd, b_manifest = dataset_build.resolve_build(cfg, build_id)
            build_draft_like["scope_decisions"] = b_manifest.get("scope_decisions") or {}
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

    for sid, ce in sorted((cas_man.get("sessions") or {}).items()):
        pg = str(ce.get("project_group") or "unknown")
        projects_set.add(pg)
        g_stats = ce.get("gold_stats") or {}
        files = ce.get("files") or {}
        status = _session_snapshot_status(sid, ce, raw_sessions, gold_sessions)

        # 若历史回填时 gold_stats 为空但 CAS 有 gold_file，从 CAS 直接补算步数
        gold_steps = int(g_stats.get("gold_steps") or g_stats.get("units_out") or 0)
        raw_steps = int(g_stats.get("raw_steps") or g_stats.get("units_in") or 0)
        if not gold_steps and files.get("gold_file"):
            g_rows = cas_store.read_cas_jsonl(files.get("gold_file"), cfg=cfg)
            if g_rows:
                gold_steps = sum(1 for m in (g_rows[0].get("messages") or []) if m.get("role") == "assistant")

        b_usage = dataset_draft.session_kind_exclusions(build_draft_like, sid, pg)
        sft_uniq_cnt = len(pack_sft_unique.get(sid) or ())
        item = {
            "session_id": sid,
            "project_group": pg,
            "dominant_model": str(ce.get("dominant_model") or "unknown"),
            "split": str(split_map.get(pg) or "train"),
            "raw_steps": raw_steps,
            "gold_steps": gold_steps,
            "sft_windows": sft_uniq_cnt,
            "sft_lf_rows": int(pack_sft_rows.get(sid, 0)),
            "dpo_count": int(pack_dpo_rows.get(sid, 0)),
            "sft_excluded": bool(b_usage.get("sft_excluded")),
            "dpo_excluded": bool(b_usage.get("dpo_excluded")),
            "cas_gold_sha256": str(files.get("gold_file") or ""),
            "cas_raw_sha256": str(files.get("raw_file") or ""),
            "snapshot_status": status,
            "readonly": True,
        }
        if project and pg != project:
            continue
        if q_low and q_low not in f"{sid} {pg} {item['dominant_model']}".lower():
            continue
        rows.append(item)

    return {
        "dataset_id": pkg_h,
        "build_id": build_id,
        "readonly": True,
        "total": len(rows),
        "projects": sorted(projects_set),
        "sessions": rows,
    }


def list_virtual_dataset_raw(
    dataset_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    project: str = "",
    q: str = "",
) -> Dict[str, Any]:
    """读取归属该虚拟数据集的原始会话副本列表与回合清洗统计。"""
    cfg = cfg or load_config()
    norm_id = normalize_pack_id(dataset_id)
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})

    rows: List[Dict[str, Any]] = []
    projects_set: set[str] = set()
    q_low = q.strip().lower()

    if norm_id == "workspace":
        for sid, re_ent in sorted(raw_sessions.items()):
            pg = str(re_ent.get("project_group") or "unknown")
            projects_set.add(pg)
            r_stats = re_ent.get("stats") or {}
            models = re_ent.get("models") or {}
            dom = max(models.items(), key=lambda kv: int(kv[1] or 0))[0] if isinstance(models, dict) and models else "unknown"
            item = {
                "session_id": sid,
                "project_group": pg,
                "dominant_model": dom,
                "origin": str(re_ent.get("origin") or "pi"),
                "cwd": str(re_ent.get("cwd") or ""),
                "analyzed": sid in gold_sessions,
                "assistant_steps": int(r_stats.get("assistant_steps") or 0),
                "turns_total": int(r_stats.get("turns_total") or 0),
                "turns_kept": int(r_stats.get("turns_kept") or 0),
                "turns_folded": int(r_stats.get("turns_folded") or 0),
                "turns_dropped": int(r_stats.get("turns_dropped") or 0),
                "cas_raw_sha256": "",
                "snapshot_status": "workspace",
                "readonly": False,
            }
            if project and pg != project:
                continue
            if q_low and q_low not in f"{sid} {pg} {dom} {item['cwd']}".lower():
                continue
            rows.append(item)
        return {"dataset_id": "workspace", "readonly": False, "total": len(rows), "projects": sorted(projects_set), "sessions": rows}

    pkg_h, build_id, cas_man, _b_stats = _load_pack_cas_and_build(norm_id, cfg)
    for sid, ce in sorted((cas_man.get("sessions") or {}).items()):
        pg = str(ce.get("project_group") or "unknown")
        projects_set.add(pg)
        r_stats = ce.get("raw_stats") or {}
        g_stats = ce.get("gold_stats") or {}
        files = ce.get("files") or {}
        status = _session_snapshot_status(sid, ce, raw_sessions, gold_sessions)
        item = {
            "session_id": sid,
            "project_group": pg,
            "dominant_model": str(ce.get("dominant_model") or "unknown"),
            "origin": str(ce.get("origin") or "pi"),
            "cwd": str(ce.get("cwd") or ""),
            "analyzed": True,
            "assistant_steps": int(r_stats.get("assistant_steps") or g_stats.get("raw_steps") or 0),
            "turns_total": int(r_stats.get("turns_total") or g_stats.get("turns_total") or 0),
            "turns_kept": int(r_stats.get("turns_kept") or g_stats.get("turns_kept") or 0),
            "turns_folded": int(r_stats.get("turns_folded") or g_stats.get("turns_folded") or 0),
            "turns_dropped": int(r_stats.get("turns_dropped") or g_stats.get("turns_dropped") or 0),
            "cas_raw_sha256": str(files.get("raw_file") or ""),
            "cas_meta_sha256": str(files.get("meta_file") or ""),
            "snapshot_status": status,
            "readonly": True,
        }
        if project and pg != project:
            continue
        if q_low and q_low not in f"{sid} {pg} {item['dominant_model']} {item['cwd']}".lower():
            continue
        rows.append(item)

    return {
        "dataset_id": pkg_h,
        "build_id": build_id,
        "readonly": True,
        "total": len(rows),
        "projects": sorted(projects_set),
        "sessions": rows,
    }


def _compact_args(args: Any, max_len: int = 120) -> str:
    if isinstance(args, dict):
        for k in ("command", "path", "file_path", "pattern", "query"):
            if k in args and args[k]:
                s = f"{k}={args[k]}"
                return s if len(s) <= max_len else s[: max_len - 1] + "…"
        raw = json.dumps(args, ensure_ascii=False)
        return raw if len(raw) <= max_len else raw[: max_len - 1] + "…"
    s = str(args or "")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def get_virtual_dataset_session_lineage(
    dataset_id: str,
    sid_or_prefix: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """按指定虚拟数据集快照读取单会话三态血缘（历史包严格走 CAS 只读还原，100% 免疫工作区删改）。"""
    cfg = cfg or load_config()
    norm_id = normalize_pack_id(dataset_id)
    dirs = stage_dirs(cfg)
    raw_sessions = (load_index(dirs["raw"] / "index.json").get("sessions") or {})
    gold_sessions = (load_index(dirs["gold"] / "index.json").get("sessions") or {})

    if norm_id == "workspace":
        pool = set(raw_sessions) | set(gold_sessions)
        full_sid = inspect_tools._resolve_session_id(sid_or_prefix, pool)  # noqa: SLF001
        raw_e = raw_sessions.get(full_sid) or {}
        gold_e = gold_sessions.get(full_sid) or {}
        raw_path = Path(raw_e["raw_file"]) if raw_e.get("raw_file") and Path(raw_e["raw_file"]).is_file() else None
        raw_meta = (
            read_json(Path(raw_e["meta_file"]), default={})
            if raw_e.get("meta_file") and Path(raw_e["meta_file"]).is_file()
            else dict(raw_e)
        ) or {}
        g_rows = read_jsonl(Path(gold_e["gold_file"])) if gold_e.get("gold_file") and Path(gold_e["gold_file"]).is_file() else []
        gold_rec = g_rows[0] if g_rows else {}
        ann = (
            copy.deepcopy(read_json(Path(gold_e["annotation_file"]), default={}) or {})
            if gold_e.get("annotation_file") and Path(gold_e["annotation_file"]).is_file()
            else {}
        )
        pruned_rows = (
            read_jsonl(Path(gold_e["reject_file"]))
            if gold_e.get("reject_file") and Path(gold_e["reject_file"]).is_file()
            else []
        )
        dpo_rows = (
            read_jsonl(Path(gold_e["dpo_file"]))
            if gold_e.get("dpo_file") and Path(gold_e["dpo_file"]).is_file()
            else []
        )
        sd_entry = (stage2_gold.load_step_decisions(cfg).get(full_sid) or {})
        project_group = str(gold_e.get("project_group") or raw_e.get("project_group") or "unknown")
        dominant_model = str(gold_e.get("dominant_model") or "unknown")
        cwd = str(gold_e.get("cwd") or raw_e.get("cwd") or "")
        snapshot_status = "workspace"
        readonly = False
        cas_files: Dict[str, str] = {}
        effective_cfg = cfg
        cas_fidelity = "workspace"
    else:
        pkg_h, _build_id, cas_man, _b_stats = _load_pack_cas_and_build(norm_id, cfg)
        effective_cfg = cfg
        if _build_id:
            try:
                _bdir, _ = dataset_build.resolve_build(cfg, _build_id)
                effective_cfg = dataset_build.load_build_config(_bdir, cfg)
            except (FileNotFoundError, ValueError, RuntimeError):
                effective_cfg = cfg
        sess_map = cas_man.get("sessions") or {}
        full_sid = inspect_tools._resolve_session_id(sid_or_prefix, set(sess_map.keys()))  # noqa: SLF001
        ce = sess_map[full_sid]
        cas_files = dict(ce.get("files") or {})
        raw_path = cas_store.get_cas_path(cas_files.get("raw_file"), cfg=cfg)
        raw_meta = cas_store.read_cas_json(cas_files.get("meta_file"), cfg=cfg, default={}) or {}
        g_rows = cas_store.read_cas_jsonl(cas_files.get("gold_file"), cfg=cfg)
        gold_rec = g_rows[0] if g_rows else {}
        ann = copy.deepcopy(cas_store.read_cas_json(cas_files.get("annotation_file"), cfg=cfg, default={}) or {})
        pruned_rows = cas_store.read_cas_jsonl(cas_files.get("reject_file"), cfg=cfg)
        dpo_rows = cas_store.read_cas_jsonl(cas_files.get("dpo_file"), cfg=cfg)
        sd_entry = ce.get("step_decisions") if isinstance(ce.get("step_decisions"), dict) else {}
        project_group = str(ce.get("project_group") or "unknown")
        dominant_model = str(ce.get("dominant_model") or "unknown")
        cwd = str(ce.get("cwd") or "")
        snapshot_status = _session_snapshot_status(full_sid, ce, raw_sessions, gold_sessions)
        readonly = True
        norm_id = pkg_h
        cas_fidelity = str(cas_man.get("fidelity") or "none")

    if ann:
        curation.ensure_rounds(ann)

    raw_flow: List[Dict[str, Any]] = []
    raw_marks: List[Dict[str, Any]] = []
    if raw_path and raw_path.is_file():
        max_tool_chars = int((effective_cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
        parsed = pilog.parse_session(raw_path, max_tool_output_chars=max_tool_chars)
        raw_flow, raw_marks, _ = pilog.build_flow(
            parsed,
            effective_cfg,
            raw_meta.get("turn_annotations"),
            raw_meta.get("manual_turn_overrides"),
        )
    else:
        raw_marks = list(raw_meta.get("turns") or [])

    turns_out = [
        {
            "turn_index": t.get("turn_index"),
            "kind": t.get("kind") or "",
            "action": t.get("action") or ("dropped" if t.get("dropped") else ("folded" if t.get("folded") else "kept")),
            "dropped": bool(t.get("dropped")),
            "folded": bool(t.get("folded")),
            "assistant_steps": int(t.get("assistant_steps") or 0),
            "reason": t.get("reason") or "",
            "text": str(t.get("text") or t.get("text_preview") or ""),
        }
        for t in raw_marks
        if isinstance(t, dict)
    ]

    sd_steps = sd_entry.get("steps") if isinstance(sd_entry.get("steps"), dict) else {}
    eff = curation.resolve(ann, cfg) if ann else {"state": "raw_only", "excluded": [], "rewrites": {}}
    eff_excluded = set(eff.get("excluded") or [])
    eff_rewrites = eff.get("rewrites") or {}

    raw_asst = [m for m in raw_flow if m.get("role") == "assistant"]
    gold_asst = [m for m in (gold_rec.get("messages") or []) if m.get("role") == "assistant"]
    if not raw_asst:
        raw_asst = gold_asst

    gold_by_id = {str(m.get("id") or ""): m for m in gold_asst if m.get("id")}
    pruned_by_id = {
        str(pr.get("dropped_msg_id")): pr
        for pr in pruned_rows
        if isinstance(pr, dict) and pr.get("dropped_msg_id")
    }

    steps_out: List[Dict[str, Any]] = []
    for idx, rm in enumerate(raw_asst):
        mid = str(rm.get("id") or f"step_{idx}")
        gm = gold_by_id.get(mid)
        pr = pruned_by_id.get(mid)
        sd_step = sd_steps.get(mid) if isinstance(sd_steps.get(mid), dict) else {}
        raw_thinking = str(rm.get("raw_thinking") if "raw_thinking" in rm else (rm.get("thinking") or ""))
        if gm and not raw_thinking and gm.get("raw_thinking"):
            raw_thinking = str(gm.get("raw_thinking") or "")

        if gm:
            gold_thinking = str(eff_rewrites.get(mid) or gm.get("thinking") or "")
            flags = list(gm.get("flags") or [])
            is_excluded = bool(gm.get("excluded")) or (mid in eff_excluded) or (sd_step.get("action") == "exclude")
            changed = bool(raw_thinking.strip() != gold_thinking.strip())
            state = "excluded" if is_excluded else ("rewritten" if (changed or sd_step.get("action") == "rewrite") else "kept")
            text_val = str(gm.get("text") or "")
        else:
            gold_thinking = ""
            flags = []
            is_excluded = sd_step.get("action") == "exclude"
            changed = False
            text_val = str(rm.get("text") or "")
            state = "pruned" if (pr or gold_asst) else "kept"

        src_msg = gm or rm
        tcs = [
            {"name": str(tc.get("name") or ""), "args_preview": _compact_args(tc.get("arguments"))}
            for tc in (src_msg.get("tool_calls") or [])
            if isinstance(tc, dict)
        ]
        steps_out.append(
            {
                "state": state,
                "msg_id": mid,
                "step_index": idx,
                "tool_calls": tcs,
                "raw_thinking": raw_thinking,
                "gold_thinking": gold_thinking,
                "changed": changed,
                "text": text_val,
                "flags": flags,
                "excluded": is_excluded,
                "in_gold": bool(gm),
                "reason": str(sd_step.get("reason") or (pr or {}).get("reason") or ""),
            }
        )

    ws_draft = dataset_draft.load_draft(cfg)
    ws_in_corpus = (full_sid in raw_sessions) or (full_sid in gold_sessions)
    ws_usage = dataset_draft.session_kind_exclusions(ws_draft, full_sid, project_group) if ws_in_corpus else {}

    mrep = ann.get("model_report") or gold_rec.get("model_report") or None
    ledger_events = ledger.read_ledger(cfg=cfg, sid=full_sid, limit=20)
    return {
        "dataset_id": norm_id,
        "session_id": full_sid,
        "readonly": readonly,
        "snapshot_status": snapshot_status,
        "cas_fidelity": cas_fidelity,
        "cas_files": cas_files,
        "workspace_usage": {
            "in_workspace": ws_in_corpus,
            "sft_excluded": bool(ws_usage.get("sft_excluded")),
            "dpo_excluded": bool(ws_usage.get("dpo_excluded")),
        },
        "meta": {
            "session_id": full_sid,
            "project_group": project_group,
            "dominant_model": dominant_model,
            "cwd": cwd,
        },
        "gold_stats": dict(gold_rec.get("stats") or {}),
        "model_report": mrep,
        "gold_messages": list(gold_rec.get("messages") or []),
        "raw_messages": raw_flow,
        "turns": turns_out,
        "steps": steps_out,
        "dpo_count": len(dpo_rows),
        "ledger_events": ledger_events,
    }
