"""本地 Web 控制台：FastAPI + 静态单页（无构建步骤）。

启动： python serve.py  ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import curation
from pipeline.common import pilog
from pipeline.common.config import (
    CONFIG_PATH,
    config_hash,
    load_config,
    save_config_text,
    stage_dirs,
)
from pipeline.common.llm import test_model
from pipeline.common.store import (
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

app = FastAPI(title="GoldThread 数据集控制台", version="1.0")


# ---------------------------------------------------------------------------
# 作业管理
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, stage: str, force: bool, extra: Optional[Dict[str, Any]] = None):
        self.id = uuid.uuid4().hex[:12]
        self.stage = stage
        self.force = force
        self.extra = extra or {}
        self.status = "running"
        self.logs: List[str] = []
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = now_iso()
        self.finished_at: Optional[str] = None

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.logs.append(line)
        if len(self.logs) > 5000:
            del self.logs[:1000]

    def as_dict(self, with_logs: bool = True, offset: int = 0) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "stage": self.stage,
            "force": self.force,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "log_count": len(self.logs),
        }
        if with_logs:
            d["logs"] = self.logs[offset:]
            d["offset"] = offset
        return d


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _run_job(job: Job) -> None:
    cfg = load_config()
    try:
        job.log(f"[{job.stage}] 开始（force={job.force}）")
        if job.stage == "stage1":
            from pipeline.stage1_extract import run
            job.result = run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "stage2":
            from pipeline.stage2_gold import run
            job.result = run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "stage3":
            from pipeline.stage3_synth import run
            job.result = run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "all":
            from pipeline.stage1_extract import run as r1
            from pipeline.stage2_gold import run as r2
            from pipeline.stage3_synth import run as r3
            job.result = {
                "stage1": r1(force=job.force, log=job.log, cfg=cfg),
                "stage2": r2(force=job.force, log=job.log, cfg=cfg),
                "stage3": r3(force=job.force, log=job.log, cfg=cfg),
            }
        elif job.stage == "review":
            from pipeline.review import run_review
            job.result = run_review(
                session_ids=job.extra.get("sessions"), force=job.force, log=job.log, cfg=cfg
            )
        elif job.stage == "refine":
            from pipeline.review import run_refine
            job.result = run_refine((job.extra.get("sessions") or [None])[0], log=job.log, cfg=cfg)
        elif job.stage == "classify":
            from pipeline.turnclassify import classify_session
            sid = (job.extra.get("sessions") or [None])[0]
            dirs = stage_dirs(cfg)
            index = load_index(dirs["raw"] / "index.json")
            entry = index["sessions"].get(sid)
            if not entry:
                raise RuntimeError(f"会话不存在: {sid}")
            session = pilog.parse_session(Path(entry["raw_file"]), max_tool_output_chars=12000)
            ann = classify_session(session, cfg=cfg, log=job.log)
            meta = read_json(Path(entry["meta_file"]), default={}) or {}
            meta["turn_annotations"] = ann
            write_json_atomic(Path(entry["meta_file"]), meta)
            job.result = ann
        else:
            raise RuntimeError(f"未知阶段: {job.stage}")
        job.status = "done"
        job.log(f"[{job.stage}] 完成")
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{exc}"
        job.log(f"[{job.stage}] 失败: {exc}")
        job.log(traceback.format_exc()[-2000:])
    finally:
        job.finished_at = now_iso()
        _persist_job(job)


def _persist_job(job: Job) -> None:
    cfg = load_config()
    path = stage_dirs(cfg)["state"] / "runs.json"
    hist = read_json(path, default=[]) or []
    hist.insert(
        0,
        {
            "id": job.id,
            "stage": job.stage,
            "force": job.force,
            "status": job.status,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "result": job.result,
        },
    )
    write_json_atomic(path, hist[:100])


def _start(stage: str, force: bool, extra: Optional[Dict[str, Any]] = None) -> Job:
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.status == "running" and (j.stage == stage or stage == "all" or j.stage == "all"):
                raise HTTPException(409, f"已有作业在运行: {j.stage} ({j.id})")
        job = Job(stage, force, extra)
        JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

class ConfigBody(BaseModel):
    yaml: str


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    cfg = load_config()
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    return {"path": str(CONFIG_PATH), "yaml": text, "parsed": {k: v for k, v in cfg.items() if not k.startswith("_")}}


@app.post("/api/config/validate")
def validate_config(body: ConfigBody) -> Dict[str, Any]:
    import yaml as yaml_lib

    try:
        parsed = yaml_lib.safe_load(body.yaml) or {}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "keys": sorted(parsed.keys())}


@app.put("/api/config")
def put_config(body: ConfigBody) -> Dict[str, Any]:
    import yaml as yaml_lib

    try:
        parsed = yaml_lib.safe_load(body.yaml) or {}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"YAML 解析失败: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(400, "配置根节点必须是对象")
    save_config_text(body.yaml)
    return {"ok": True, "path": str(CONFIG_PATH), "config_hash": config_hash(load_config())}


@app.post("/api/llm/test")
def llm_test(body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model = (body or {}).get("model")
    try:
        return {"ok": True, **test_model(model)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

class RunBody(BaseModel):
    stage: str
    force: bool = False
    sessions: Optional[List[str]] = None


@app.post("/api/run")
def run_stage(body: RunBody) -> Dict[str, Any]:
    if body.stage not in ("stage1", "stage2", "stage3", "all", "review", "refine", "classify"):
        raise HTTPException(400, "stage 必须是 stage1|stage2|stage3|all|review|refine")
    job = _start(body.stage, body.force, {"sessions": body.sessions})
    return {"job_id": job.id}


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    with JOBS_LOCK:
        return {"jobs": [j.as_dict(with_logs=False) for j in JOBS.values()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, offset: int = 0) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "作业不存在")
    return job.as_dict(with_logs=True, offset=offset)


@app.get("/api/runs")
def run_history() -> Dict[str, Any]:
    cfg = load_config()
    return {"runs": read_json(stage_dirs(cfg)["state"] / "runs.json", default=[]) or []}


# ---------------------------------------------------------------------------
# 概览
# ---------------------------------------------------------------------------

@app.get("/api/overview")
def overview() -> Dict[str, Any]:
    cfg = load_config()
    dirs = stage_dirs(cfg)
    raw_index = load_index(dirs["raw"] / "index.json")
    gold_index = load_index(dirs["gold"] / "index.json")
    train_stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    runs = read_json(dirs["state"] / "runs.json", default=[]) or []
    last_review = next((r for r in runs if r.get("stage") == "review"), None)
    reviewed = 0
    review_states: Counter = Counter()
    for v in gold_index["sessions"].values():
        ann = read_json(Path(v.get("annotation_file", "")), default={}) or {}
        if ann.get("model_review"):
            reviewed += 1
        review_states[curation.resolve(ann, cfg)["state"]] += 1
    state = {
        s: read_json(dirs["state"] / f"{s}.json", default={}) or {}
        for s in ("stage1", "stage2")
    }

    labels = Counter()
    projects: Dict[str, Dict[str, Any]] = {}
    models = Counter()
    for sid, e in raw_index["sessions"].items():
        label = e.get("label", "pending")
        labels[label] += 1
        proj = e.get("project_group", "unknown")
        p = projects.setdefault(
            proj, {"project": proj, "positive": 0, "negative": 0, "pending": 0, "steps": 0, "sessions": 0}
        )
        p[label] = p.get(label, 0) + 1
        p["sessions"] += 1
        p["steps"] += int(e.get("stats", {}).get("assistant_steps", 0))
        for m, c in (e.get("models") or {}).items():
            models[m] += c

    gold_steps = sum(int(v.get("stats", {}).get("gold_steps", 0)) for v in gold_index["sessions"].values())
    raw_steps = sum(int(v.get("stats", {}).get("assistant_steps", 0)) for v in raw_index["sessions"].values())
    return {
        "generated_at": now_iso(),
        "config": {
            "path": str(CONFIG_PATH),
            "hash": config_hash(cfg),
            "model": cfg["llm"]["model"],
            "use_model_turns": cfg["turns"].get("use_model", False),
            "review_enabled": cfg["review"].get("enabled", False),
            "sessions_dir": cfg["paths"]["sessions_dir"],
            "data_dir": cfg["paths"]["data_dir"],
        },
        "stages": {
            "stage1": {
                "last_run": state["stage1"].get("last_run"),
                "sessions": len(raw_index["sessions"]),
                "generated_at": raw_index.get("generated_at"),
            },
            "stage2": {
                "last_run": state["stage2"].get("last_run"),
                "sessions": len(gold_index["sessions"]),
                "generated_at": gold_index.get("generated_at"),
            },
            "stage3": {"last_run": train_stats.get("generated_at"), "counts": train_stats.get("counts", {})},
            "review": {
                "last_run": (last_review or {}).get("finished_at"),
                "reviewed": reviewed,
                "states": dict(review_states),
            },
        },
        "labels": dict(labels),
        "raw_steps": raw_steps,
        "gold_steps": gold_steps,
        "projects": sorted(projects.values(), key=lambda x: -x["sessions"]),
        "models": dict(models),
        "train": train_stats,
        "skipped": raw_index.get("skipped", {}),
        "folder_stats": raw_index.get("folder_stats", {}),
    }


# ---------------------------------------------------------------------------
# 原始数据
# ---------------------------------------------------------------------------

@app.get("/api/raw/sessions")
def raw_sessions(
    label: str = "",
    project: str = "",
    q: str = "",
    limit: int = 500,
) -> Dict[str, Any]:
    cfg = load_config()
    index = load_index(stage_dirs(cfg)["raw"] / "index.json")
    rows = []
    for sid, e in index["sessions"].items():
        if label and e.get("label") != label:
            continue
        if project and e.get("project_group") != project:
            continue
        if q:
            blob = json.dumps(
                [sid, e.get("project_group"), e.get("cwd"), e.get("stats", {}).get("root_prompt"), e.get("label_reason")],
                ensure_ascii=False,
            )
            if q.lower() not in blob.lower():
                continue
        rows.append(
            {
                "session_id": sid,
                "label": e.get("label"),
                "label_reason": e.get("label_reason"),
                "project_group": e.get("project_group"),
                "cwd": e.get("cwd"),
                "timestamp": e.get("timestamp"),
                "models": e.get("models"),
                "stats": e.get("stats"),
                "pathologies": e.get("pathologies"),
                "annotation": "model" if e.get("turn_annotations") else "rule",
            }
        )
    rows.sort(key=lambda r: (r["label"], r["project_group"], r["session_id"]))
    return {"total": len(rows), "sessions": rows[:limit]}


@app.get("/api/raw/session/{sid}")
def raw_session(sid: str) -> Dict[str, Any]:
    cfg = load_config()
    index = load_index(stage_dirs(cfg)["raw"] / "index.json")
    e = index["sessions"].get(sid)
    if not e:
        raise HTTPException(404, "会话不存在")
    meta = read_json(Path(e["meta_file"]), default={}) or {}
    session = pilog.parse_session(Path(e["raw_file"]), max_tool_output_chars=12000)
    ann = meta.get("turn_annotations")
    flow, marks, stats = pilog.build_flow(session, cfg, ann)
    return {
        "meta": {k: v for k, v in meta.items() if k not in ("turn_annotations",)},
        "turn_annotations": ann,
        "flow_stats": stats,
        "turns": marks,
        "messages": flow,
    }


class RelabelBody(BaseModel):
    label: str


@app.post("/api/raw/session/{sid}/relabel")
def relabel(sid: str, body: RelabelBody) -> Dict[str, Any]:
    if body.label not in ("positive", "negative", "pending"):
        raise HTTPException(400, "label 必须是 positive|negative|pending")
    cfg = load_config()
    path = stage_dirs(cfg)["state"] / "labels.json"
    overrides = read_json(path, default={}) or {}
    overrides[sid] = body.label
    write_json_atomic(path, overrides)
    return {"ok": True, "session_id": sid, "label": body.label, "note": "重新运行 stage1 生效"}


@app.get("/api/raw/session/{sid}/gold")
def raw_to_gold(sid: str) -> Dict[str, Any]:
    cfg = load_config()
    idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
    e = idx["sessions"].get(sid)
    if not e:
        raise HTTPException(404, "黄金数据不存在，请先运行 stage2")
    return {"session": e}


# ---------------------------------------------------------------------------
# 黄金数据
# ---------------------------------------------------------------------------

@app.get("/api/gold/sessions")
def gold_sessions(label: str = "", project: str = "", status: str = "", q: str = "") -> Dict[str, Any]:
    cfg = load_config()
    idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
    rows = []
    for sid, e in idx["sessions"].items():
        if label and e.get("label") != label:
            continue
        if project and e.get("project_group") != project:
            continue
        if status and e.get("status") != status:
            continue
        if q and q.lower() not in json.dumps(e, ensure_ascii=False).lower():
            continue
        ann = read_json(Path(e.get("annotation_file", "")), default={}) or {}
        eff = curation.resolve(ann, cfg)
        rows.append(
            {
                **{k: v for k, v in e.items() if k != "annotation_file"},
                "review_status": (ann.get("model_review") or {}).get("verdict"),
                "script_status": ann.get("status"),
                "review_confidence": (ann.get("model_review") or {}).get("confidence"),
                "rewrites": len((ann.get("model_review") or {}).get("thinking_rewrites") or []),
                "effective": eff["state"],
                "review_round": curation.latest_round(ann).get("round"),
                "review_progress": curation.progress(ann),
            }
        )
    rows.sort(key=lambda r: (r.get("label", ""), r.get("project_group", ""), r["session_id"]))
    return {"total": len(rows), "sessions": rows}


@app.get("/api/gold/session/{sid}")
def gold_session(sid: str) -> Dict[str, Any]:
    cfg = load_config()
    idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
    e = idx["sessions"].get(sid)
    if not e:
        raise HTTPException(404, "黄金数据不存在")
    rows = read_jsonl(Path(e["gold_file"]))
    if not rows:
        raise HTTPException(404, "黄金文件为空")
    ann = copy.deepcopy(read_json(Path(e["annotation_file"]), default={}) or {})
    curation.ensure_rounds(ann)
    return {
        "record": rows[0],
        "annotation": ann,
        "effective": curation.resolve(ann, cfg)["state"],
        "progress": curation.progress(ann),
        "pruned": read_jsonl(Path(e["reject_file"])) if e.get("reject_file") else [],
        "dpo": read_jsonl(Path(e["dpo_file"])) if e.get("dpo_file") else [],
    }


@app.post("/api/gold/session/{sid}/review")
def review_one(sid: str) -> Dict[str, Any]:
    job = _start("review", False, {"sessions": [sid]})
    return {"job_id": job.id}


class ReviewItemsBody(BaseModel):
    items: Dict[str, Dict[str, Any]] = {}
    note: Optional[str] = None


class ReviewDecisionBody(BaseModel):
    decision: Optional[str] = None
    note: str = ""


def _gold_entry(sid: str) -> Dict[str, Any]:
    cfg = load_config()
    e = load_index(stage_dirs(cfg)["gold"] / "index.json")["sessions"].get(sid)
    if not e:
        raise HTTPException(404, "黄金数据不存在")
    return e


def _load_ann(sid: str) -> tuple:
    e = _gold_entry(sid)
    path = Path(e["annotation_file"])
    return path, read_json(path, default={}) or {}


@app.put("/api/gold/session/{sid}/review/items")
def save_review_items(sid: str, body: ReviewItemsBody) -> Dict[str, Any]:
    path, ann = _load_ann(sid)
    try:
        curation.update_items(ann, body.items, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_json_atomic(path, ann)
    return {"ok": True, "progress": curation.progress(ann)}


@app.post("/api/gold/session/{sid}/review/decision")
def review_decision(sid: str, body: ReviewDecisionBody) -> Dict[str, Any]:
    path, ann = _load_ann(sid)
    if not curation.latest_round(ann):
        raise HTTPException(409, "该会话还没有模型评审")
    if body.decision == "approved" and (ann.get("review_rounds_stale") or curation.is_stale(ann)):
        raise HTTPException(409, "会话内容已变化，请先让模型重新评审")
    try:
        curation.decide(ann, body.decision, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_json_atomic(path, ann)
    eff = curation.resolve(ann, load_config())
    return {"ok": True, "effective": eff["state"], "note": "重新运行 stage3 后生效"}


@app.post("/api/gold/session/{sid}/review/refine")
def review_refine(sid: str, body: ReviewItemsBody) -> Dict[str, Any]:
    path, ann = _load_ann(sid)
    try:
        curation.update_items(ann, body.items, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not curation.feedback_targets(ann) and not (ann.get("human_review") or {}).get("note", "").strip():
        raise HTTPException(400, "没有可供模型修订的意见：请先批注步骤、拒绝改写或填写整体意见")
    write_json_atomic(path, ann)
    job = _start("refine", False, {"sessions": [sid]})
    return {"job_id": job.id}


# ---------------------------------------------------------------------------
# 训练数据
# ---------------------------------------------------------------------------

@app.get("/api/train/datasets")
def datasets() -> Dict[str, Any]:
    cfg = load_config()
    root = stage_dirs(cfg)["train"]
    stats = read_json(root / "stats.json", default={}) or {}
    out = []
    for p in sorted(root.glob("*.jsonl")):
        out.append({"name": p.name, "size": p.stat().st_size, "rows": stats.get("counts", {}).get(p.name)})
    return {"dir": str(root), "datasets": out, "stats": stats, "dataset_info": read_json(root / "dataset_info.json", default={})}


@app.get("/api/train/dataset/{name}")
def dataset_rows(name: str, offset: int = 0, limit: int = Query(50, le=500), q: str = "") -> Dict[str, Any]:
    if "/" in name or "\\" in name or not name.endswith(".jsonl"):
        raise HTTPException(400, "非法文件名")
    cfg = load_config()
    path = stage_dirs(cfg)["train"] / name
    if not path.exists():
        raise HTTPException(404, "数据集不存在")
    rows = read_jsonl(path)
    if q:
        rows = [r for r in rows if q.lower() in json.dumps(r, ensure_ascii=False).lower()]
    page = rows[offset : offset + limit]
    return {
        "name": name,
        "total": len(rows),
        "offset": offset,
        "rows": [
            {
                "index": offset + i,
                "sample_id": r.get("sample_id"),
                "metadata": r.get("metadata", {}),
                "preview": _preview(r),
            }
            for i, r in enumerate(page)
        ],
    }


@app.get("/api/train/dataset/{name}/{index}")
def dataset_row(name: str, index: int) -> Dict[str, Any]:
    if "/" in name or "\\" in name or not name.endswith(".jsonl"):
        raise HTTPException(400, "非法文件名")
    cfg = load_config()
    rows = read_jsonl(stage_dirs(cfg)["train"] / name)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "行不存在")
    return rows[index]


def _preview(row: Dict[str, Any]) -> Dict[str, Any]:
    msgs = row.get("conversations") or row.get("messages") or []
    first_user = next(
        (m.get("value") or m.get("content") for m in msgs if (m.get("from") or m.get("role")) in ("human", "user")),
        "",
    )
    last = msgs[-1] if msgs else {}
    target = str(last.get("value") or last.get("content") or "")
    if " thinking" in target:
        target = re.sub(r"^[\s\S]*?\s*", "", target, count=1)
    return {
        "turns": len(msgs),
        "user": str(first_user)[:240],
        "target": target[:240],
    }


# ---------------------------------------------------------------------------
# 静态页面
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": now_iso()}