"""本地 Web 控制台：FastAPI + 静态单页（无构建步骤）。

启动： python serve.py  ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import copy
import io
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import curation, stage1_extract, stage2_gold, stage3_synth
from pipeline import steps as stepcat
from pipeline.common import pilog
from pipeline.common.config import (
    CONFIG_PATH,
    config_hash,
    load_config,
    save_config_text,
    stage_dirs,
)
from pipeline.common.llm import load_pi_models, test_model

try:
    from ruamel.yaml import YAML as _RuamelYAML
    _HAS_RUAMEL = True
except ImportError:  # noqa: BLE001
    _HAS_RUAMEL = False
from pipeline.common.store import (
    load_index,
    now_iso,
    read_json,
    read_jsonl,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

app = FastAPI(title="GoldThread 数据集控制台", version="2.0")


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
        self.progress: Dict[str, Any] = {}
        self.done_steps: List[str] = []

    def log(self, msg: str) -> None:
        # 结构化进度：pipeline 通过 `@@STEP {json}` 上报，不进可见日志
        if msg.startswith("@@STEP "):
            try:
                p = json.loads(msg[7:])
                p["at"] = now_iso()
                self.progress = p
                sid = p.get("id")
                if sid and sid not in self.done_steps:
                    d, t = p.get("done"), p.get("total")
                    if d is not None and t is not None and t > 0 and d >= t:
                        self.done_steps.append(sid)
            except Exception:  # noqa: BLE001
                pass
            return
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
            "progress": self.progress,
            "done_steps": self.done_steps,
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
            job.result = stage1_extract.run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "stage2":
            job.result = stage2_gold.run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "stage3":
            job.result = stage3_synth.run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "all":
            r1 = stage1_extract.run(force=job.force, log=job.log, cfg=cfg)
            r2 = stage2_gold.run(force=job.force, log=job.log, cfg=cfg)
            r3 = stage3_synth.run(force=job.force, log=job.log, cfg=cfg)
            job.result = {"stage1": r1, "stage2": r2, "stage3": r3}
        elif job.stage == "review":
            from pipeline.review import run_review
            res = run_review(
                session_ids=job.extra.get("sessions"), force=job.force, log=job.log, cfg=cfg
            )
            # 评审完成后自动刷新 stage3 统计
            try:
                stage3_synth.run(force=False, log=job.log, cfg=cfg)
            except Exception:  # noqa: BLE001
                pass
            job.result = res
        elif job.stage == "refine":
            from pipeline.review import run_refine
            res = run_refine((job.extra.get("sessions") or [None])[0], log=job.log, cfg=cfg)
            try:
                stage3_synth.run(force=False, log=job.log, cfg=cfg)
            except Exception:  # noqa: BLE001
                pass
            job.result = res
        elif job.stage == "classify":
            from pipeline.turnclassify import classify_session
            sid = (job.extra.get("sessions") or [None])[0]
            dirs = stage_dirs(cfg)
            index_path = dirs["raw"] / "index.json"
            index = load_index(index_path)
            entry = index["sessions"].get(sid)
            if not entry:
                raise RuntimeError(f"会话不存在: {sid}")
            session = pilog.parse_session(Path(entry["raw_file"]), max_tool_output_chars=12000)
            ann = classify_session(session, cfg=cfg, log=job.log)
            meta_path = Path(entry["meta_file"])
            meta = read_json(meta_path, default={}) or {}
            meta["turn_annotations"] = ann
            write_json_atomic(meta_path, meta)
            # 重算该会话的清洗流并同步到 stage2
            stage1_extract.update_turn_overrides(sid, cfg=cfg)
            stage2_gold.run(force=False, log=job.log, cfg=cfg)
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
    return {
        "path": str(CONFIG_PATH),
        "yaml": text,
        "parsed": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }


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


@app.get("/api/llm/models")
def llm_models() -> Dict[str, Any]:
    try:
        providers = load_pi_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"读取 pi models.json 失败: {exc}") from exc
    out = []
    for pname, pv in providers.items():
        for entry in pv.get("models", []) or []:
            mid = entry.get("id") if isinstance(entry, dict) else entry
            if not mid:
                continue
            out.append({"id": mid, "provider": pname, "reasoning": bool(entry.get("reasoning")) if isinstance(entry, dict) else False})
    return {"models": out, "ruamel": _HAS_RUAMEL}


class ConfigPatchBody(BaseModel):
    values: List[Dict[str, Any]] = []


def _set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = data
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


@app.post("/api/config/structured")
def config_structured(body: ConfigPatchBody) -> Dict[str, Any]:
    """按 path->value 深度合并进当前配置；ruamel 往返保留注释与顺序。"""
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    import yaml as yaml_lib  # noqa: PLC0415

    engine = "ruamel"
    if _HAS_RUAMEL:
        yml = _RuamelYAML()
        yml.preserve_quotes = True
        data = yml.load(text) or {}
        for v in body.values:
            _set_path(data, v["path"], v["value"])
        buf = io.StringIO()
        yml.dump(data, buf)
        new_text = buf.getvalue()
    else:
        engine = "pyyaml"
        data = yaml_lib.safe_load(text) or {}
        for v in body.values:
            _set_path(data, v["path"], v["value"])
        new_text = yaml_lib.safe_dump(data, allow_unicode=True, sort_keys=False)
    parsed = yaml_lib.safe_load(new_text)
    if not isinstance(parsed, dict):
        raise HTTPException(400, "配置根节点必须是对象")
    save_config_text(new_text)
    return {
        "ok": True,
        "engine": engine,
        "config_hash": config_hash(load_config()),
        "yaml": new_text,
        "applied": len(body.values),
    }


# ---------------------------------------------------------------------------
# 运行清单（runbook）：界面与真实执行一一对应
# ---------------------------------------------------------------------------

def _runbook_state() -> Dict[str, Any]:
    """从真实产物推导每一步状态：todo / ready / done / warn。"""
    cfg = load_config()
    dirs = stage_dirs(cfg)
    raw = load_index(dirs["raw"] / "index.json")
    gold = load_index(dirs["gold"] / "index.json")
    train_root = dirs["train"]
    stats = read_json(train_root / "stats.json", default={}) or {}
    info = read_json(train_root / "dataset_info.json", default={}) or {}
    counts = stats.get("counts") or {}
    # load_index 把 extra 字段平铺在顶层（历史数据可能嵌在 extra 里，两种都兼容）
    raw_extra = raw.get("extra") or raw
    raw_sessions = raw.get("sessions") or {}
    gold_sessions = gold.get("sessions") or {}

    st: Dict[str, Dict[str, Any]] = {}

    def put(step_id: str, status: str, value: str = "", hint: str = "") -> None:
        st[step_id] = {"status": status, "value": value, "hint": hint}

    # ---- stage1 ----
    skipped = raw_extra.get("skipped") or {}
    folder_stats = raw_extra.get("folder_stats") or {}
    scanned = sum(folder_stats.values())
    if raw_sessions:
        put("s1.scan", "done", f"扫描 {scanned} 个文件", f"{len(folder_stats)} 个目录")
        put("s1.admit", "done", f"准入 {len(raw_sessions)} 个会话",
            f"忽略 {skipped.get('non_crack_folder', 0)} / 排除 {skipped.get('excluded_folder', 0)}")
        dup = raw_extra.get("duplicates") or {}
        put("s1.dedupe", "done", f"去重 {len(dup)} 个", "同 cwd + 同首条消息的副本")
        with_ann = sum(1 for v in raw_sessions.values() if (v.get("stats") or {}).get("model_classified"))
        put("s1.classify", "done" if with_ann else "ready",
            f"{with_ann} / {len(raw_sessions)} 条已分类", "模型回合分类")
        put("s1.flow", "done", f"{len(raw_sessions)} 条成流",
            f"回合合计 {sum((v.get('stats') or {}).get('turns_total', 0) for v in raw_sessions.values())}")
    else:
        for sid in ("s1.scan", "s1.admit", "s1.dedupe", "s1.classify", "s1.flow"):
            put(sid, "todo", "尚未运行", "点「运行全流程」从采集开始")

    # ---- stage2 ----
    if gold_sessions:
        stats_list = [(v.get("stats") or {}) for v in gold_sessions.values()]
        gold_steps = sum(s.get("gold_steps", 0) for s in stats_list)
        folded = sum(s.get("folded", 0) for s in stats_list)
        pruned = sum(s.get("pruned_detours", 0) for s in stats_list)
        failed_batches = sum(s.get("judge_failed_batches", 0) for s in stats_list)
        decided = sum(s.get("decisions", 0) for s in stats_list)
        silent = sum(s.get("silent_cleared", 0) for s in stats_list)
        verdicts: Dict[str, int] = {}
        needs_rw = 0
        seams_total = seams_fixed = 0
        for v in gold_sessions.values():
            ann = read_json(Path(v.get("annotation_file", "")), default={}) or {}
            rep = ann.get("model_report") or {}
            aud = ann.get("model_audit") or {}
            if rep:
                verdicts[rep.get("verdict", "?")] = verdicts.get(rep.get("verdict", "?"), 0) + 1
                if rep.get("needs_wholesale_rewrite"):
                    needs_rw += 1
            seams_total += int(aud.get("seams_total", 0) or 0)
            seams_fixed += int(aud.get("seams_fixed", 0) or 0)
        put("s2.normalize", "done" if gold_steps else "ready", f"静默 {silent} 步", "机械规则，无需模型")
        put("s2.judge_steps", "warn" if failed_batches else ("done" if decided else "ready"),
            f"{decided} 条决定", f"失败批 {failed_batches}" if failed_batches else "模型逐条判断")
        put("s2.apply", "done" if gold_steps else "ready", f"黄金 {gold_steps} 步",
            f"剪除 {pruned} / 折叠 {folded}")
        put("s2.seams_extract", "done" if seams_total else "ready", f"{seams_total} 处接缝", "程序机械提取")
        put("s2.judge_seams", "done" if seams_total else "ready", f"修复 {seams_fixed} 处",
            f"接缝 {seams_total} 处")
        warn_q = verdicts.get("discard", 0) or needs_rw
        put("s2.judge_quality", "warn" if warn_q else ("done" if verdicts else "ready"),
            f"{sum(verdicts.values())} 条已审",
            " · ".join(f"{k} {n}" for k, n in sorted(verdicts.items())) or "尚未审计")
        dpo_total = int(counts.get("dpo_preference_pairs.jsonl", 0) or 0)
        put("s2.dpo_build", "done" if dpo_total else "ready", f"{dpo_total} 对",
            "剪枝弯路 + 反例坏步")
    else:
        for sid in ("s2.normalize", "s2.judge_steps", "s2.apply", "s2.seams_extract",
                    "s2.judge_seams", "s2.judge_quality", "s2.dpo_build"):
            put(sid, "todo", "等待 stage1", "")

    # ---- human ----
    if gold_sessions:
        review_states: Dict[str, int] = {}
        finalized = 0
        for v in gold_sessions.values():
            ann = read_json(Path(v.get("annotation_file", "")), default={}) or {}
            eff = curation.resolve(ann, cfg)
            review_states[eff["state"]] = review_states.get(eff["state"], 0) + 1
            if (ann.get("human_review") or {}).get("finalized"):
                finalized += 1
        pending = sum(review_states.get(k, 0) for k in ("unreviewed", "held", "stale"))
        put("h.review", "warn" if pending else "done",
            f"待处理 {pending} 条 / 共 {len(gold_sessions)}",
            " · ".join(f"{k} {n}" for k, n in sorted(review_states.items())))
        put("h.finalize", "done" if finalized else "ready", f"已定稿 {finalized} 条",
            "定稿后修改会被拦截，需先撤销")
    else:
        put("h.review", "todo", "等待 stage2", "")
        put("h.finalize", "todo", "等待复核", "")

    # ---- stage3 ----
    win = int(counts.get("sft_step_windows_sharegpt.jsonl", 0) or 0)
    if win:
        drop = stats.get("drop_reasons") or {}
        put("s3.gate", "done",
            f"拦下 {int(drop.get('session_blocked', 0)) + int(drop.get('model_discard', 0))} 步",
            f"model_discard {drop.get('model_discard', 0)} / pathology {drop.get('pathology', 0)}")
        put("s3.windows", "done", f"{win} 条窗口",
            f"train {counts.get('sft_step_windows_sharegpt.train.jsonl', 0)} / "
            f"eval {counts.get('sft_step_windows_sharegpt.eval.jsonl', 0)}")
        put("s3.traj", "done",
            f"{counts.get('sft_gold_trajectories_sharegpt.jsonl', 0)} 条轨迹", "")
        dpo_pairs = int(counts.get("dpo_preference_pairs.jsonl", 0) or 0)
        cache = read_json(Path(dirs["state"]) / "dpo_review_cache.json", default={}) or {}
        rev = stats.get("dpo_review") or {}
        # 缓存会累积历史键，这里按当前偏好对重新计算"已复核数"
        done_rev = 0
        pairs_file = train_root / "dpo_preference_pairs.jsonl"
        if pairs_file.exists():
            from pipeline import judge as _judge  # noqa: PLC0415

            _model = _judge._model(cfg, _judge._jcfg(cfg), "dpo") or ""
            for line in pairs_file.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _judge._vkey(_model, "dpo", p) in cache:
                    done_rev += 1
        failed_rev_all = sum(1 for v in cache.values() if v.get("verdict") == "review_failed")
        failed_rev = sum(1 for v in cache.values() if v.get("verdict") == "review_failed")
        if dpo_pairs and done_rev < dpo_pairs:
            put("s3.dpo_review", "warn", f"{done_rev} / {dpo_pairs} 对已复核",
                "未完成：重跑 stage3 会断点续跑")
        elif failed_rev:
            put("s3.dpo_review", "warn", f"{done_rev}/{dpo_pairs}（失败 {failed_rev}）", "需重跑补齐")
        else:
            put("s3.dpo_review", "done" if dpo_pairs else "ready", f"{done_rev} 对已复核",
                f"调换 {rev.get('swapped', 0)} / 丢弃 {rev.get('dropped', 0)}")
        smap = split_map = stats.get("split") or {}
        put("s3.split", "done", f"{len(smap)} 个项目已切分",
            " · ".join(f"{k}→{v}" for k, v in list(smap.items())[:4]))
        put("s3.manifest", "done", f"{len(info)} 个数据集登记",
            "dataset_info.json / stats.json / *.train|eval")
        ho = train_root / "handoff.md"
        put("s3.handoff", "done" if ho.exists() else "ready",
            "handoff.md 已生成" if ho.exists() else "待生成", "含环境搭建 → 训练 → 评估全流程")
    else:
        for sid in ("s3.gate", "s3.windows", "s3.traj", "s3.dpo_review",
                    "s3.split", "s3.manifest", "s3.handoff"):
            put(sid, "todo", "等待 stage2 与人工复核", "")

    return st


@app.get("/api/runbook")
def runbook() -> Dict[str, Any]:
    cfg = load_config()
    state = _runbook_state()
    running = next((j for j in JOBS.values() if j.status == "running"), None)
    steps_out = []
    for s in stepcat.STEPS:
        item = dict(s)
        info = state.get(s["id"]) or {"status": "todo", "value": "", "hint": ""}
        item.update(info)
        if running:
            prog = running.progress or {}
            if prog.get("id") == s["id"]:
                item["status"] = "running"
                d, t = prog.get("done"), prog.get("total")
                item["value"] = f"{d}/{t}" if (d is not None and t) else (prog.get("msg") or "运行中")
                if prog.get("msg"):
                    item["hint"] = prog["msg"]
            elif s["id"] in running.done_steps:
                item["done_this_run"] = True
        steps_out.append(item)
    phases = []
    for ph in stepcat.PHASES:
        ph_steps = [x for x in steps_out if x["phase"] == ph["id"]]
        phases.append({**ph, "steps": ph_steps,
                       "done": sum(1 for x in ph_steps if x["status"] == "done"),
                       "total": len(ph_steps)})
    total = len(steps_out)
    done = sum(1 for x in steps_out if x["status"] == "done")
    nxt = next((x for x in steps_out if x["status"] in ("running", "warn", "ready", "todo")), None)
    return {
        "phases": phases,
        "total": total,
        "done": done,
        "job": running.as_dict(with_logs=False) if running else None,
        "next": ({"id": nxt["id"], "label": nxt["label"], "phase": nxt["phase"],
                  "owner": nxt["owner"], "status": nxt["status"]} if nxt else None),
        "config_hash": config_hash(cfg),
        "data_dir": str(cfg["paths"]["data_dir"]),
        "handoff": str(stage_dirs(cfg)["train"] / "handoff.md"),
    }


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
        raise HTTPException(400, "stage 必须是 stage1|stage2|stage3|all|review|refine|classify")
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
    latest_review_action_at = ""
    for v in gold_index["sessions"].values():
        ann = read_json(Path(v.get("annotation_file", "")), default={}) or {}
        if ann.get("model_review"):
            reviewed += 1
        eff = curation.resolve(ann, cfg)
        review_states[eff["state"]] += 1
        hr = ann.get("human_review") or {}
        for ts_field in (hr.get("decided_at"), hr.get("updated_at"), ann.get("reviewed_at")):
            if ts_field and str(ts_field) > latest_review_action_at:
                latest_review_action_at = str(ts_field)

    state = {
        s: read_json(dirs["state"] / f"{s}.json", default={}) or {}
        for s in ("stage1", "stage2")
    }

    labels = Counter()
    projects: Dict[str, Dict[str, Any]] = {}
    models = Counter()
    raw_turns_total = 0
    raw_turns_kept = 0
    raw_turns_folded = 0
    raw_turns_dropped = 0
    raw_clean_steps = 0

    for sid, e in raw_index["sessions"].items():
        label = e.get("label", "pending")
        labels[label] += 1
        proj = e.get("project_group", "unknown")
        p = projects.setdefault(
            proj,
            {
                "project": proj,
                "positive": 0,
                "negative": 0,
                "pending": 0,
                "steps": 0,
                "gold_steps": 0,
                "sessions": 0,
            },
        )
        p[label] = p.get(label, 0) + 1
        p["sessions"] += 1
        st = e.get("stats") or {}
        p["steps"] += int(st.get("assistant_steps", 0))
        raw_clean_steps += int(st.get("clean_steps", 0))
        raw_turns_total += int(st.get("turns_total", 0))
        raw_turns_kept += int(st.get("turns_kept", 0))
        raw_turns_folded += int(st.get("turns_folded", 0))
        raw_turns_dropped += int(st.get("turns_dropped", max(0, int(st.get("turns_total", 0)) - int(st.get("turns_kept", 0)))))
        for m, c in (e.get("models") or {}).items():
            models[m] += c

    gold_steps = 0
    gold_folded = 0
    gold_pruned = 0
    gold_compressed = 0
    gold_backfilled = 0
    gold_silent = 0
    for sid, v in gold_index["sessions"].items():
        gst = v.get("stats") or {}
        g_steps = int(gst.get("gold_steps", 0))
        gold_steps += g_steps
        gold_folded += int(gst.get("folded", 0))
        gold_pruned += int(gst.get("pruned_detours", 0))
        gold_compressed += int(gst.get("thinking_compressed", 0))
        gold_backfilled += int(gst.get("thinking_backfilled", 0)) + int(gst.get("thinking_synthesized", 0))
        gold_silent += int(gst.get("silent_cleared", 0))
        proj = v.get("project_group", "unknown")
        if proj in projects:
            projects[proj]["gold_steps"] += g_steps

    raw_steps = sum(int(v.get("stats", {}).get("assistant_steps", 0)) for v in raw_index["sessions"].values())
    s3_at = str(train_stats.get("generated_at") or "")
    s2_at = str(state["stage2"].get("last_run") or "")
    s1_at = str(state["stage1"].get("last_run") or "")
    sync_dirty = bool(
        (s1_at and s2_at and s1_at > s2_at)
        or (s2_at and s3_at and s2_at > s3_at)
        or (latest_review_action_at and s3_at and latest_review_action_at > s3_at)
    )

    return {
        "generated_at": now_iso(),
        "sync_dirty": sync_dirty,
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
                "turns_total": raw_turns_total,
                "turns_kept": raw_turns_kept,
                "turns_folded": raw_turns_folded,
                "turns_dropped": raw_turns_dropped,
                "clean_steps": raw_clean_steps,
            },
            "stage2": {
                "last_run": state["stage2"].get("last_run"),
                "sessions": len(gold_index["sessions"]),
                "generated_at": gold_index.get("generated_at"),
                "folded": gold_folded,
                "pruned": gold_pruned,
                "compressed": gold_compressed,
                "backfilled": gold_backfilled,
                "silent": gold_silent,
            },
            "stage3": {
                "last_run": train_stats.get("generated_at"),
                "counts": train_stats.get("counts", {}),
            },
            "review": {
                "last_run": (last_review or {}).get("finished_at") or latest_review_action_at,
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
        "duplicates": raw_index.get("duplicates", {}),
    }


# ---------------------------------------------------------------------------
# 原始数据 (Stage 1)
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
        ann_mode = "manual" if e.get("stats", {}).get("manual_overridden") else (
            "model" if e.get("turn_annotations") else "rule"
        )
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
                "strengths": e.get("strengths"),
                "annotation": ann_mode,
            }
        )
    rows.sort(key=lambda r: (r["label"], r["project_group"], r["session_id"]))
    return {
        "total": len(rows),
        "sessions": rows[:limit],
        "audit": {
            "skipped": index.get("skipped", {}),
            "folder_stats": index.get("folder_stats", {}),
            "duplicates": index.get("duplicates", {}),
            "rejected_samples": index.get("rejected_samples", []),
            "replay_mode": index.get("replay_mode", False),
        },
    }


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
    manual_ov = meta.get("manual_turn_overrides") or {}
    flow, marks, stats = pilog.build_flow(session, cfg, ann, manual_ov)
    raw_blocks = pilog.iter_blocks(session["messages"])
    blocks_detail = []
    for idx, blk in enumerate(raw_blocks):
        u = blk[0]
        assts = [m for m in blk[1:] if m["role"] == "assistant"]
        blocks_detail.append(
            {
                "turn_index": idx,
                "user_id": u.get("id"),
                "user_text": u.get("content", ""),
                "timestamp": u.get("timestamp", ""),
                "assistant_steps": len(assts),
                "tools": [tc["name"] for m in assts for tc in m.get("tool_calls", [])][:12],
                "first_thinking": (assts[0].get("thinking", "")[:300] if assts else ""),
                "last_reply": (assts[-1].get("text", "")[:300] if assts else ""),
            }
        )
    return {
        "meta": {k: v for k, v in meta.items() if k not in ("turn_annotations",)},
        "turn_annotations": ann,
        "manual_turn_overrides": manual_ov,
        "flow_stats": stats,
        "turns": marks,
        "raw_blocks": blocks_detail,
        "messages": flow,
    }


class RelabelBody(BaseModel):
    label: str
    auto_sync: bool = True


@app.post("/api/raw/session/{sid}/relabel")
def relabel(sid: str, body: RelabelBody) -> Dict[str, Any]:
    if body.label not in ("positive", "negative", "pending"):
        raise HTTPException(400, "label 必须是 positive|negative|pending")
    cfg = load_config()
    try:
        res = stage1_extract.relabel_session(sid, body.label, cfg=cfg)
        if body.auto_sync:
            gold_idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
            if gold_idx.get("sessions"):
                stage3_synth.run(force=False, log=lambda *_: None, cfg=cfg)
        return {**res, "note": "已即时更新标签并同步索引"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


class TurnOverridesBody(BaseModel):
    turns: Optional[Dict[str, Any]] = None
    root_prompt: Optional[str] = None
    reset: bool = False
    recompute_downstream: bool = True


@app.put("/api/raw/session/{sid}/turns")
def save_turn_overrides(sid: str, body: TurnOverridesBody) -> Dict[str, Any]:
    cfg = load_config()
    try:
        res = stage1_extract.update_turn_overrides(
            sid=sid,
            turns_patch=body.turns,
            root_prompt=body.root_prompt,
            reset=body.reset,
            cfg=cfg,
        )
        if body.recompute_downstream:
            gold_idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
            if gold_idx.get("sessions"):
                stage2_gold.run(force=False, log=lambda *_: None, cfg=cfg)
                stage3_synth.run(force=False, log=lambda *_: None, cfg=cfg)
        return res
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/raw/session/{sid}/classify")
def classify_one(sid: str) -> Dict[str, Any]:
    job = _start("classify", True, {"sessions": [sid]})
    return {"job_id": job.id}


@app.get("/api/raw/session/{sid}/gold")
def raw_to_gold(sid: str) -> Dict[str, Any]:
    cfg = load_config()
    idx = load_index(stage_dirs(cfg)["gold"] / "index.json")
    e = idx["sessions"].get(sid)
    if not e:
        raise HTTPException(404, "黄金数据不存在，请先运行 stage2")
    return {"session": e}


# ---------------------------------------------------------------------------
# 黄金数据与人机协同复核 (Stage 2 & Review)
# ---------------------------------------------------------------------------

@app.get("/api/gold/sessions")
def gold_sessions(label: str = "", project: str = "", status: str = "", q: str = "") -> Dict[str, Any]:
    cfg = load_config()
    dirs = stage_dirs(cfg)
    idx = load_index(dirs["gold"] / "index.json")
    train_stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    session_funnels = train_stats.get("session_funnels") or {}
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
        rnd = curation.latest_round(ann)
        funnel_info = (session_funnels.get(sid) or {}).get("funnel") or {}
        rows.append(
            {
                **{k: v for k, v in e.items() if k != "annotation_file"},
                "review_status": rnd.get("verdict") or (ann.get("model_review") or {}).get("verdict"),
                "script_status": ann.get("status"),
                "script_score": ann.get("script_score"),
                "review_confidence": rnd.get("confidence") or (ann.get("model_review") or {}).get("confidence"),
                "rewrites": len(rnd.get("thinking_rewrites") or []),
                "bad_steps": len(rnd.get("bad_step_ids") or []),
                "effective": eff["state"],
                "review_round": rnd.get("round"),
                "review_progress": curation.progress(ann),
                "funnel": funnel_info,
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
    synth_preview = {}
    try:
        synth_preview = stage3_synth.preview_session_synthesis(sid, cfg=cfg)
    except Exception:  # noqa: BLE001
        synth_preview = {}
    return {
        "record": rows[0],
        "annotation": ann,
        "effective": curation.resolve(ann, cfg)["state"],
        "progress": curation.progress(ann),
        "pruned": read_jsonl(Path(e["reject_file"])) if e.get("reject_file") else [],
        "dpo": read_jsonl(Path(e["dpo_file"])) if e.get("dpo_file") else [],
        "diagnostics": read_jsonl(Path(e["diag_file"])) if e.get("diag_file") else [],
        "synthesis_preview": synth_preview,
    }


@app.post("/api/gold/session/{sid}/review")
def review_one(sid: str) -> Dict[str, Any]:
    job = _start("review", True, {"sessions": [sid]})
    return {"job_id": job.id}


class ReviewItemsBody(BaseModel):
    items: Dict[str, Dict[str, Any]] = {}
    note: Optional[str] = None


class ReviewBatchBody(BaseModel):
    action: str


class ReviewDecisionBody(BaseModel):
    decision: Optional[str] = None
    note: str = ""
    auto_sync_stage3: bool = True


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
    cfg = load_config()
    path, ann = _load_ann(sid)
    try:
        curation.update_items(ann, body.items, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_json_atomic(path, ann)
    preview = {}
    try:
        preview = stage3_synth.preview_session_synthesis(sid, cfg=cfg)
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "progress": curation.progress(ann),
        "effective": curation.resolve(ann, cfg)["state"],
        "synthesis_preview": preview,
    }


@app.post("/api/gold/session/{sid}/review/batch")
def batch_review_items(sid: str, body: ReviewBatchBody) -> Dict[str, Any]:
    cfg = load_config()
    path, ann = _load_ann(sid)
    hr = curation.human(ann)
    rnd = curation.latest_round(ann)
    rw_ids = [rw["msg_id"] for rw in (rnd.get("thinking_rewrites") or []) if rw.get("msg_id")]
    bad_ids = list(rnd.get("bad_step_ids") or [])

    act = body.action
    if act == "accept_all_rewrites":
        for mid in rw_ids:
            it = hr["items"].setdefault(mid, {})
            if not it.get("rewrite"):
                it["rewrite"] = "accepted"
    elif act == "accept_all_rewrites_force":
        for mid in rw_ids:
            it = hr["items"].setdefault(mid, {})
            it["rewrite"] = "accepted"
    elif act == "reject_all_rewrites":
        for mid in rw_ids:
            it = hr["items"].setdefault(mid, {})
            if not it.get("rewrite"):
                it["rewrite"] = "rejected"
    elif act == "exclude_all_bad":
        for mid in bad_ids:
            it = hr["items"].setdefault(mid, {})
            if not it.get("step"):
                it["step"] = "exclude"
    elif act == "keep_all_bad":
        for mid in bad_ids:
            it = hr["items"].setdefault(mid, {})
            if not it.get("step"):
                it["step"] = "keep"
    elif act == "reset_all":
        hr["items"] = {}
    else:
        raise HTTPException(400, f"未知批量操作: {act}")

    hr["updated_at"] = now_iso()
    write_json_atomic(path, ann)
    preview = {}
    try:
        preview = stage3_synth.preview_session_synthesis(sid, cfg=cfg)
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "items": hr["items"],
        "progress": curation.progress(ann),
        "effective": curation.resolve(ann, cfg)["state"],
        "synthesis_preview": preview,
    }


@app.post("/api/gold/session/{sid}/review/decision")
def review_decision(sid: str, body: ReviewDecisionBody) -> Dict[str, Any]:
    cfg = load_config()
    path, ann = _load_ann(sid)
    try:
        curation.decide(ann, body.decision, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_json_atomic(path, ann)
    eff = curation.resolve(ann, cfg)
    s3_res = None
    if body.auto_sync_stage3:
        try:
            s3_res = stage3_synth.run(force=False, log=lambda *_: None, cfg=cfg)
        except Exception:  # noqa: BLE001
            s3_res = None
    return {
        "ok": True,
        "effective": eff["state"],
        "stage3_synced": bool(s3_res),
        "note": "已通过并同步更新 Stage 3 训练集" if s3_res else "结论已保存",
    }


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


class FinalizeBody(BaseModel):
    finalized: bool
    note: str = ""


@app.post("/api/gold/session/{sid}/finalize")
def gold_finalize(sid: str, body: FinalizeBody) -> Dict[str, Any]:
    cfg = load_config()
    path, ann = _load_ann(sid)
    try:
        if body.finalized:
            curation.finalize(ann, body.note)
        else:
            curation.unfinalize(ann, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_json_atomic(path, ann)
    eff = curation.resolve(ann, cfg)
    synced = False
    try:
        stage3_synth.run(force=False, log=lambda *_: None, cfg=cfg)
        synced = True
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "effective": eff["state"],
        "finalized": bool((ann.get("human_review") or {}).get("finalized")),
        "stage3_synced": synced,
    }


# ---------------------------------------------------------------------------
# 训练数据检视 (Stage 3)
# ---------------------------------------------------------------------------

_ROW_CACHE: Dict[str, Any] = {}


def _count_lines(path: Path) -> int:
    """按 (mtime, size) 缓存行数，避免每次刷新都全量扫描。"""
    st = path.stat()
    key = str(path)
    ck = f"{int(st.st_mtime)}:{st.st_size}"
    hit = _ROW_CACHE.get(key)
    if hit and hit[0] == ck:
        return hit[1]
    n = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    _ROW_CACHE[key] = (ck, n)
    return n


@app.get("/api/train/handoff")
def train_handoff() -> Dict[str, Any]:
    """返回 handoff.md 内容（不存在时给出提示）。"""
    cfg = load_config()
    p = stage_dirs(cfg)["train"] / "handoff.md"
    if not p.exists():
        raise HTTPException(404, "handoff.md 还没生成，跑一次 stage3 就会产出")
    return {"path": str(p), "markdown": p.read_text(encoding="utf-8")}


@app.get("/api/train/datasets")
def datasets() -> Dict[str, Any]:
    cfg = load_config()
    root = stage_dirs(cfg)["train"]
    stats = read_json(root / "stats.json", default={}) or {}
    counts = stats.get("counts", {}) or {}
    info = read_json(root / "dataset_info.json", default={}) or {}
    registered = {v.get("file_name"): k for k, v in info.items() if isinstance(v, dict)}
    out = []
    for p in sorted(root.glob("*.jsonl")):
        rows = counts.get(p.name)
        if rows is None:
            try:
                rows = _count_lines(p)
            except OSError:
                rows = None
        out.append({
            "name": p.name,
            "size": p.stat().st_size,
            "rows": rows,
            "registered": registered.get(p.name),
        })
    return {
        "dir": str(root),
        "datasets": out,
        "stats": stats,
        "dataset_info": info,
    }


@app.get("/api/train/dataset/{name}")
def dataset_rows(
    name: str,
    offset: int = 0,
    limit: int = Query(50, le=500),
    q: str = "",
    project: str = "",
    stage: str = "",
    split: str = "",
) -> Dict[str, Any]:
    if "/" in name or "\\" in name or not name.endswith(".jsonl"):
        raise HTTPException(400, "非法文件名")
    cfg = load_config()
    path = stage_dirs(cfg)["train"] / name
    if not path.exists():
        raise HTTPException(404, "数据集不存在")
    all_rows = read_jsonl(path)
    filtered: List[tuple[int, Dict[str, Any]]] = []
    for orig_idx, r in enumerate(all_rows):
        meta = r.get("metadata") or r
        if project and meta.get("project_group") != project:
            continue
        if stage and (meta.get("stage") or meta.get("source")) != stage:
            continue
        if split and (meta.get("split") or r.get("split")) != split:
            continue
        if q and q.lower() not in json.dumps(r, ensure_ascii=False).lower():
            continue
        filtered.append((orig_idx, r))

    page = filtered[offset : offset + limit]
    return {
        "name": name,
        "total": len(filtered),
        "unfiltered_total": len(all_rows),
        "offset": offset,
        "rows": [
            {
                "index": orig_idx,
                "sample_id": r.get("sample_id") or r.get("diagnostic_id"),
                "metadata": r.get("metadata") or {
                    "session_id": r.get("session_id"),
                    "project_group": r.get("project_group"),
                    "split": r.get("split"),
                    "model": r.get("model"),
                    "quality_score": r.get("quality_score"),
                    "pathologies": r.get("pathologies", []),
                    "tool_calls": r.get("tool_calls", []),
                },
                "preview": _preview(r, name),
            }
            for orig_idx, r in page
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


_THINK_RE = re.compile(r"^<think>?\s*\n?([\s\S]*?)\n?</think>\s*", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)


def _extract_think_and_body(raw_val: str) -> tuple[str, str, List[str]]:
    val = str(raw_val or "").strip()
    think = ""
    m = _THINK_RE.match(val)
    if m:
        think = m.group(1).strip()
        val = val[m.end() :].strip()
    tools = []
    for tm in _TOOL_CALL_RE.findall(val):
        try:
            obj = json.loads(tm)
            tname = obj.get("name", "tool")
            targs = obj.get("arguments") or {}
            gist = targs.get("command") or targs.get("path") or json.dumps(targs, ensure_ascii=False)
            tools.append(f"{tname}: {str(gist)[:80]}")
        except Exception:  # noqa: BLE001
            tools.append(tm[:80])
    body = _TOOL_CALL_RE.sub("", val).strip()
    return think, body, tools


def _preview(row: Dict[str, Any], dataset_name: str = "") -> Dict[str, Any]:
    # 1) DPO 偏好对
    if "chosen" in row and "rejected" in row:
        c_val = (row.get("chosen") or {}).get("value", "")
        r_val = (row.get("rejected") or {}).get("value", "")
        c_think, c_body, c_tools = _extract_think_and_body(c_val)
        r_think, r_body, r_tools = _extract_think_and_body(r_val)
        convs = row.get("conversations") or []
        first_user = next(
            (m.get("value") for m in convs if m.get("from") in ("human", "user")),
            "",
        )
        return {
            "kind": "dpo",
            "turns": len(convs),
            "user": str(first_user)[:200],
            "chosen_think": (c_think or c_body)[:220],
            "rejected_think": (r_think or r_body)[:220],
            "chosen_tools": c_tools[:2],
            "rejected_tools": r_tools[:2],
            "chosen_len": len(c_val),
            "rejected_len": len(r_val),
        }

    # 2) 病态诊断记录
    if "pathologies" in row and "conversations" not in row and "messages" not in row:
        return {
            "kind": "diagnostic",
            "turns": 1,
            "user": f"[{row.get('model', 'unknown')}] 质量分 {row.get('quality_score', 0)}",
            "target": str(row.get("thinking") or row.get("thinking_preview") or "")[:240],
            "pathologies": row.get("pathologies", []),
            "tools": row.get("tool_calls", []),
            "metrics": row.get("metrics", {}),
        }

    # 3) SFT 单步窗口或完整黄金轨迹
    msgs = row.get("conversations") or row.get("messages") or []
    first_user = next(
        (m.get("value") or m.get("content") for m in msgs if (m.get("from") or m.get("role")) in ("human", "user")),
        "",
    )
    last = msgs[-1] if msgs else {}
    raw_target = str(last.get("value") or last.get("content") or "")
    think, body, tools = _extract_think_and_body(raw_target)
    if not think and last.get("reasoning_content"):
        think = str(last.get("reasoning_content"))
    if not tools and last.get("tool_calls"):
        for tc in last["tool_calls"]:
            fn = tc.get("function") or {}
            tools.append(f"{fn.get('name', 'tool')}: {str(fn.get('arguments', ''))[:80]}")

    return {
        "kind": "sft",
        "turns": len(msgs),
        "user": str(first_user)[:200],
        "think": think[:220],
        "target": (body or think)[:220],
        "tools": tools[:3],
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
