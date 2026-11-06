"""任务中心：Job 的展示元数据 + tasks.json 持久化。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from .common.config import load_config, stage_dirs
from .common.store import now_iso, read_json, write_json_atomic

MAX_TASKS = 50
LIST_HOURS = 24
# 任务详情可回看的执行日志上限（只留尾部，避免 tasks.json 膨胀）
MAX_TASK_LOGS = 2000


def tasks_path(cfg: Optional[Dict[str, Any]] = None) -> Any:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["state"] / "tasks.json"


def _load_store(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = read_json(tasks_path(cfg), default={}) or {}
    if not isinstance(data, dict):
        data = {}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    return {"tasks": tasks}


def _save_store(store: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
    tasks: List[Dict[str, Any]] = list(store.get("tasks") or [])
    running = [t for t in tasks if t.get("status") == "running" and not t.get("dismissed")]
    rest = [t for t in tasks if t.get("status") != "running" or t.get("dismissed")]
    rest.sort(key=lambda t: t.get("finished_at") or t.get("created_at") or "", reverse=True)
    capped = running + rest[: max(0, MAX_TASKS - len(running))]
    write_json_atomic(tasks_path(cfg), {"tasks": capped})


def _sid8(sid: str) -> str:
    s = str(sid or "")
    return s[:8] if len(s) >= 8 else s


def clip_logs(logs: Optional[List[str]], limit: int = MAX_TASK_LOGS) -> List[str]:
    """裁剪任务日志尾部，供落盘与详情回放。"""
    if not logs:
        return []
    if limit <= 0:
        return []
    if len(logs) <= limit:
        return list(logs)
    return list(logs[-limit:])


def _sids_from_job(stage: str, extra: Dict[str, Any], result: Any) -> List[str]:
    if stage == "delete":
        return []
    if isinstance(result, dict):
        if result.get("sessions"):
            raw = result["sessions"]
            if isinstance(raw, list):
                return [str(x) for x in raw if x]
    if extra.get("sessions"):
        return [str(x) for x in extra["sessions"] if x]
    if stage == "collect" and isinstance(result, dict) and result.get("sessions"):
        return [str(x) for x in result["sessions"]]
    if stage in ("analyze", "classify") and extra.get("sessions"):
        return [str(extra["sessions"][0])] if extra["sessions"] else []
    return []


def describe_task(stage: str, force: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extra = extra or {}
    kind = stage
    affects: List[str] = ["overview"]
    links: Dict[str, Any] = {}
    title = stage
    subtitle = ""

    if stage == "collect":
        n = len(extra.get("ids") or [])
        src = extra.get("source") or "pi"
        if extra.get("analyze"):
            title = f"采集并分析 {n} 条"
            affects = ["overview", "raw", "gold"]
            links = {"open_tab": "gold", "source": src}
        else:
            title = f"采集 {n} 条"
            affects = ["overview", "raw"]
            links = {"open_tab": "raw", "source": src}
        subtitle = f"{src} · {n} 条" if n else src
    elif stage == "analyze":
        sids = extra.get("sessions") or []
        sid = str(sids[0]) if sids else "?"
        title = f"分析会话 {_sid8(sid)}"
        subtitle = sid
        affects = ["overview", "gold"]
        links = {"open_tab": "gold", "sids": sids}
    elif stage == "classify":
        sids = extra.get("sessions") or []
        sid = str(sids[0]) if sids else "?"
        title = f"回合分类 {_sid8(sid)}"
        subtitle = sid
        affects = ["overview", "raw"]
        links = {"open_tab": "raw", "sids": sids}
    elif stage == "stage1":
        title = "扫描并提取原始会话"
        affects = ["overview", "raw"]
        links = {"open_tab": "raw"}
    elif stage == "stage2":
        title = "黄金提纯" + ("（全量）" if force else "（增量）")
        affects = ["overview", "gold"]
        links = {"open_tab": "gold"}
    elif stage == "dataset_evaluate":
        title = "DPO 模型评估"
        affects = ["overview", "train"]
        links = {"open_tab": "train"}
    elif stage == "stage3":
        title = "训练集合成" + ("（全量）" if force else "（增量）")
        affects = ["overview", "train"]
        links = {"open_tab": "train"}
    elif stage == "all":
        title = "全流程" + ("（全量重建）" if force else "")
        affects = ["overview", "raw", "gold", "train"]
        links = {"open_tab": "gold"}
    elif stage == "delete":
        sids = extra.get("sessions") or []
        sid = str(sids[0]) if sids else "?"
        title = f"删除会话 {_sid8(sid)}"
        subtitle = sid
        affects = ["overview", "raw", "gold", "train"]
        links = {}
    elif stage == "package":
        build_id = str(extra.get("build_id") or "")
        title = f"交付打包 {build_id[:8]}" if build_id else "交付打包"
        affects = ["overview", "train"]
        links = {"open_tab": "runs", "kind": "package"}
    elif stage == "review":
        title = "模型评审"
        affects = ["overview", "gold", "train"]
        links = {"open_tab": "gold"}
    elif stage == "refine":
        title = "带意见重审"
        affects = ["overview", "gold", "train"]
        links = {"open_tab": "gold"}
    else:
        title = stage

    return {
        "kind": kind,
        "title": title.strip(),
        "subtitle": subtitle,
        "affects": affects,
        "links": links,
    }


def task_record_from_job(job: Any) -> Dict[str, Any]:
    meta = describe_task(job.stage, job.force, job.extra)
    sids = _sids_from_job(job.stage, job.extra, job.result)
    if sids and "sids" not in meta["links"]:
        meta["links"] = {**meta["links"], "sids": sids}
    logs = clip_logs(getattr(job, "logs", None) or [])
    rec = {
        "id": job.id,
        "stage": job.stage,
        "force": job.force,
        "extra": job.extra,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "kind": meta["kind"],
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "affects": meta["affects"],
        "links": meta["links"],
        "logs": logs,
        "log_count": len(logs),
    }
    return rec


def register_job(job: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = task_record_from_job(job)
    store = _load_store(cfg)
    tasks = [t for t in store["tasks"] if t.get("id") != job.id]
    tasks.insert(0, rec)
    store["tasks"] = tasks
    _save_store(store, cfg)
    return rec


def sync_job(job: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = task_record_from_job(job)
    store = _load_store(cfg)
    found = False
    tasks: List[Dict[str, Any]] = []
    for t in store["tasks"]:
        if t.get("id") == job.id:
            merged = {**t, **rec}
            if t.get("dismissed"):
                merged["dismissed"] = True
            tasks.append(merged)
            found = True
        else:
            tasks.append(t)
    if not found:
        tasks.insert(0, rec)
    store["tasks"] = tasks
    _save_store(store, cfg)
    return rec


def reconcile_stale(active_ids: Set[str], cfg: Optional[Dict[str, Any]] = None) -> int:
    """内存中无对应 Job 的 running 任务标为 stale。"""
    store = _load_store(cfg)
    n = 0
    for t in store["tasks"]:
        if t.get("status") == "running" and t.get("id") not in active_ids:
            t["status"] = "stale"
            t["error"] = t.get("error") or "服务已重启或任务已中断"
            t["finished_at"] = t.get("finished_at") or now_iso()
            n += 1
    if n:
        _save_store(store, cfg)
    return n


def dismiss_task(task_id: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    store = _load_store(cfg)
    ok = False
    for t in store["tasks"]:
        if t.get("id") == task_id:
            t["dismissed"] = True
            ok = True
    if ok:
        _save_store(store, cfg)
    return ok


def get_task(task_id: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    store = _load_store(cfg)
    for t in store["tasks"]:
        if t.get("id") == task_id and not t.get("dismissed"):
            return dict(t)
    return None


def _public_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """列表接口不含完整 logs，只带 log_count。"""
    item = {k: v for k, v in task.items() if k != "logs"}
    stored = task.get("logs")
    if isinstance(stored, list):
        item["log_count"] = len(stored)
    else:
        item["log_count"] = int(task.get("log_count") or 0)
    return item


def list_tasks(
    cfg: Optional[Dict[str, Any]] = None,
    limit: int = 50,
    hours: float = LIST_HOURS,
) -> Dict[str, Any]:
    store = _load_store(cfg)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: List[Dict[str, Any]] = []
    running_n = 0
    for t in store["tasks"]:
        if t.get("dismissed"):
            continue
        st = t.get("status")
        if st == "running":
            running_n += 1
            out.append(t)
            continue
        ts = t.get("finished_at") or t.get("created_at") or ""
        try:
            dt = datetime.fromisoformat(str(ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            dt = cutoff
        if dt >= cutoff or st == "stale":
            out.append(t)
    running = [t for t in out if t.get("status") == "running"]
    other = [t for t in out if t.get("status") != "running"]
    other.sort(key=lambda x: x.get("finished_at") or x.get("created_at") or "", reverse=True)
    merged = running + other
    return {"running": running_n, "tasks": [_public_task(t) for t in merged[:limit]]}


def enrich_with_live_job(task: Dict[str, Any], job: Any) -> Dict[str, Any]:
    if not job:
        return task
    live = task_record_from_job(job)
    merged = {**task, **live}
    if task.get("dismissed"):
        merged["dismissed"] = True
    merged["live"] = True
    # 列表场景不需要把完整 logs 塞进每行
    merged.pop("logs", None)
    return merged
