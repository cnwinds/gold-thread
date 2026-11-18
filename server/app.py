"""本地 Web 控制台：FastAPI + 静态单页（无构建步骤）。

启动： python serve.py  ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import (
    curation,
    dataset_build,
    dataset_draft,
    release,
    stage1_extract,
    stage2_gold,
    stage3_synth,
)
from pipeline import steps as stepcat
from pipeline import task_registry
from pipeline.common import pilog
from pipeline.common.config import (
    CONFIG_PATH,
    config_hash,
    load_config,
    save_config_text,
    stage_dirs,
)
from pipeline.common.format import export_profiles_public, resolve_export_profile
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


def _prewarm_tokenizer() -> None:
    """后台预热 length_gate 分词器，缩短首次打包等待。"""
    try:
        from pipeline.length_gate import load_gate_tokenizer

        tok, label = load_gate_tokenizer()
        print(f"[ui] tokenizer prewarm: {'ok' if tok else 'skip'} ({label})")
    except Exception as exc:  # noqa: BLE001
        print(f"[ui] tokenizer prewarm failed: {exc}")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(
        target=_prewarm_tokenizer, daemon=True, name="tok-prewarm"
    ).start()
    yield


app = FastAPI(title="GoldThread 数据集控制台", version="2.0", lifespan=_lifespan)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", "testserver"})


def _allowed_hosts() -> set[str]:
    extra = os.environ.get("GOLDTHREAD_ALLOWED_HOSTS", "")
    out = set(_LOCAL_HOSTS)
    for item in extra.split(","):
        h = item.strip().lower()
        if h:
            out.add(h)
    return out


def _host_only(raw_host: str) -> str:
    val = (raw_host or "").strip().lower()
    if not val:
        return ""
    if val.startswith("["):
        end = val.find("]")
        return val[: end + 1] if end >= 0 else val
    return val.split(":", 1)[0]


@app.middleware("http")
async def _local_security_guard(request, call_next):
    token = os.environ.get("GOLDTHREAD_API_TOKEN", "").strip()
    path = request.url.path
    if token and path.startswith("/api/") and path != "/api/health":
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {token}":
            return JSONResponse({"detail": "未授权：缺少有效 GOLDTHREAD_API_TOKEN"}, status_code=401)
        return await call_next(request)

    allowed = _allowed_hosts()
    if "*" not in allowed:
        req_host = _host_only(request.headers.get("host", ""))
        if req_host and req_host not in allowed:
            return JSONResponse({"detail": f"拒绝非本地 Host 访问: {req_host}"}, status_code=403)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin", "").strip()
            if origin:
                from urllib.parse import urlparse  # noqa: PLC0415

                origin_host = _host_only(urlparse(origin).netloc)
                if origin_host not in allowed:
                    return JSONResponse({"detail": f"拒绝跨源写请求: {origin}"}, status_code=403)
    return await call_next(request)


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
                sid = p.get("id")
                meta = stepcat.STEP_BY_ID.get(sid or "") or {}
                if not p.get("label"):
                    base = meta.get("label") or sid or ""
                    extra = (p.get("msg") or "").strip()
                    p["label"] = f"{base} · {extra}" if base and extra and extra != base else (extra or base)
                self.progress = p
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
        self._persist()

    def _persist(self) -> None:
        callback = getattr(self, "on_persist", None)
        if not callback:
            return
        try:
            callback(self)
        except Exception:  # noqa: BLE001
            pass

    def as_dict(self, with_logs: bool = True, offset: int = 0) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "stage": self.stage,
            "force": self.force,
            "extra": self.extra,
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

# 会改写原始 / 黄金 / 训练工作区的阶段：彼此互斥，冲突时入队串行。
_GOLD_WRITE_STAGES = frozenset(
    {"stage2", "stage3", "dataset_evaluate", "all", "analyze", "classify", "collect"}
)


def _jobs_clash(a_stage: str, b_stage: str) -> bool:
    return (
        a_stage == b_stage
        or a_stage == "all"
        or b_stage == "all"
        or a_stage == "delete"
        or b_stage == "delete"
        or (a_stage in _GOLD_WRITE_STAGES and b_stage in _GOLD_WRITE_STAGES)
        or (a_stage == "package" and b_stage in {"stage3", "all"})
        or (b_stage == "package" and a_stage in {"stage3", "all"})
    )


def _active_blocker(stage: str, *, exclude_id: Optional[str] = None) -> Optional[Job]:
    """返回与 stage 冲突的 running/queued 作业（若有）。"""
    for j in JOBS.values():
        if exclude_id and j.id == exclude_id:
            continue
        if j.status not in ("running", "queued"):
            continue
        if _jobs_clash(stage, j.stage):
            return j
    return None


def _launch_job_thread(job: Job) -> None:
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()


def _evaluate_dpo_or_raise(log, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构建前显式跑完 DPO 评估；仍有 pending/failed 则失败可见。"""
    evaluation = stage3_synth.evaluate_dpo_pending(log=log, cfg=cfg)
    if evaluation.get("pending") or evaluation.get("failed"):
        raise RuntimeError(
            f"DPO 评估未全部完成：剩余 {evaluation.get('pending', 0)}，失败 {evaluation.get('failed', 0)}"
        )
    return evaluation


def _pump_queue() -> None:
    """当前作业结束后，按创建顺序启动下一批互不冲突的排队作业。"""
    to_start: List[Job] = []
    with JOBS_LOCK:
        queued = sorted(
            (j for j in JOBS.values() if j.status == "queued"),
            key=lambda j: j.created_at or "",
        )
        running = [j for j in JOBS.values() if j.status == "running"]
        for q in queued:
            if any(_jobs_clash(q.stage, r.stage) for r in running):
                continue
            if any(_jobs_clash(q.stage, s.stage) for s in to_start):
                continue
            q.status = "running"
            to_start.append(q)
    cfg = load_config() if to_start else None
    for q in to_start:
        q.log(f"[{q.stage}] 出队，开始执行")
        if cfg is not None:
            try:
                task_registry.sync_job(q, cfg)
            except Exception:  # noqa: BLE001
                pass
        _launch_job_thread(q)


def _run_job(job: Job) -> None:
    cfg = load_config()

    def _persist(current: Job, _cfg: Dict[str, Any] = cfg) -> None:
        task_registry.sync_job(current, _cfg)

    job.on_persist = _persist
    try:
        if job.stage == "analyze":
            job.log("[analyze] 开始（单会话提纯，会调模型 judge，耗时取决于步数）")
        elif job.stage == "collect":
            job.log(f"[collect] 开始（analyze={bool(job.extra.get('analyze'))}）")
        elif job.stage == "delete":
            job.log("[delete] 开始（删除当前数据集及下游产物，保留源会话与冻结交付包）")
        else:
            job.log(f"[{job.stage}] 开始（force={job.force}）")
        if job.stage == "delete":
            from pipeline.session_delete import delete_session

            sid = str((job.extra.get("sessions") or [""])[0])
            job.result = delete_session(sid, cfg=cfg, log=job.log)
            _ROW_CACHE.clear()
        elif job.stage == "stage1":
            job.result = stage1_extract.run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "stage2":
            job.result = stage2_gold.run(force=job.force, log=job.log, cfg=cfg)
        elif job.stage == "analyze":
            sids = [s for s in (job.extra.get("sessions") or []) if s]
            if not sids:
                raise RuntimeError("没有指定要分析的会话")
            job.result = stage2_gold.run(force=False, log=job.log, cfg=cfg, only=sids)
        elif job.stage == "collect":
            from pipeline.collect import ingest_selected

            sids = ingest_selected(
                str(job.extra.get("source") or ""),
                job.extra.get("ids") or [],
                cfg,
                job.log,
            )
            result: Dict[str, Any] = {"sessions": sids}
            if job.extra.get("analyze"):
                result["stage2"] = stage2_gold.run(
                    force=False, log=job.log, cfg=cfg, only=sids
                )
            job.result = result
        elif job.stage == "stage3":
            # 与全流程一致：构建前先评估，用户不必先跳到训练页单独点「DPO 评估」。
            job.log("[stage3] 先评估待复核 DPO，再合成训练集")
            evaluation = _evaluate_dpo_or_raise(job.log, cfg)
            r3 = stage3_synth.run(force=job.force, log=job.log, cfg=cfg)
            job.result = {"dpo_evaluation": evaluation, "stage3": r3}
        elif job.stage == "dataset_evaluate":
            job.result = _evaluate_dpo_or_raise(job.log, cfg)
        elif job.stage == "all":
            r1 = stage1_extract.run(force=job.force, log=job.log, cfg=cfg)
            r2 = stage2_gold.run(force=job.force, log=job.log, cfg=cfg)
            evaluation = _evaluate_dpo_or_raise(job.log, cfg)
            r3 = stage3_synth.run(force=job.force, log=job.log, cfg=cfg)
            job.result = {"stage1": r1, "stage2": r2, "dpo_evaluation": evaluation, "stage3": r3}
        elif job.stage == "review":
            from pipeline.review import run_review
            res = run_review(
                session_ids=job.extra.get("sessions"), force=job.force, log=job.log, cfg=cfg
            )
            # 评审完成后先刷新 DPO 评估，再构建；门禁失败必须让任务失败可见。
            _evaluate_dpo_or_raise(job.log, cfg)
            stage3_synth.run(force=False, log=job.log, cfg=cfg)
            job.result = res
        elif job.stage == "refine":
            from pipeline.review import run_refine
            res = run_refine((job.extra.get("sessions") or [None])[0], log=job.log, cfg=cfg)
            _evaluate_dpo_or_raise(job.log, cfg)
            stage3_synth.run(force=False, log=job.log, cfg=cfg)
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
        elif job.stage == "package":
            from pipeline.package import build_delivery_zip  # noqa: PLC0415

            build_dir, manifest = dataset_build.resolve_build(cfg, job.extra.get("build_id"))
            build_cfg = dataset_build.load_build_config(build_dir, cfg)
            result = build_delivery_zip(
                build_dir,
                build_cfg,
                log=job.log,
                profile=job.extra.get("profile"),
                delivery_dir=stage_dirs(cfg)["root"] / "04_release" / "delivery",
            )
            job.result = {**result, "build_id": manifest["build_id"]}
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
        try:
            task_registry.sync_job(job, cfg)
        except Exception:  # noqa: BLE001
            pass
        try:
            _pump_queue()
        except Exception:  # noqa: BLE001
            pass


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
    job = Job(stage, force, extra)
    start_now = False
    with JOBS_LOCK:
        blocker = _active_blocker(stage)
        if blocker:
            job.status = "queued"
            job.log(f"[{stage}] 已加入排队（等待 {blocker.stage} {blocker.id}）")
        else:
            job.status = "running"
            start_now = True
        JOBS[job.id] = job
    try:
        task_registry.register_job(job, load_config())
    except Exception:  # noqa: BLE001
        pass
    if start_now:
        _launch_job_thread(job)
    return job


def _assert_no_delete_running() -> None:
    """同步写接口与后台级联删除互斥，避免旧索引覆盖删除结果。"""
    with JOBS_LOCK:
        blocking = next(
            (
                j
                for j in JOBS.values()
                if j.stage == "delete" and j.status in ("running", "queued")
            ),
            None,
        )
    if blocking:
        state = "正在删除会话" if blocking.status == "running" else "删除任务排队中"
        raise HTTPException(409, f"{state}，请稍后再试: {blocking.id}")


def _assert_no_gold_write_running() -> None:
    """同步触发 Stage 2 / Stage 3 重建前，确保没有冲突的后台写作业正在运行。"""
    _assert_no_delete_running()
    with JOBS_LOCK:
        blocking = next(
            (
                j
                for j in JOBS.values()
                if j.stage in _GOLD_WRITE_STAGES and j.status == "running"
            ),
            None,
        )
    if blocking:
        raise HTTPException(409, f"后台作业 {blocking.stage} ({blocking.id}) 正在运行，请等待其完成后再提交")


def _guard_sensitive_config_change(new_cfg: Dict[str, Any]) -> None:
    """无鉴权本地模式下，禁止通过 HTTP 接口篡改 pi_models_json 指向任意外部文件（防 !cmd 命令注入）。"""
    if os.environ.get("GOLDTHREAD_API_TOKEN", "").strip():
        return
    new_paths = new_cfg.get("paths")
    if not isinstance(new_paths, dict) or "pi_models_json" not in new_paths:
        return
    cur_cfg = load_config()
    cur_val = str((cur_cfg.get("paths") or {}).get("pi_models_json") or "")
    next_val = str(new_paths.get("pi_models_json") or "")
    if next_val and next_val != cur_val:
        raise HTTPException(
            403,
            "安全限制：未配置 GOLDTHREAD_API_TOKEN 时，禁止通过 HTTP API 修改 paths.pi_models_json；请直接在本机编辑 config/pipeline.yaml",
        )


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

class ConfigBody(BaseModel):
    yaml: str


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    cfg = load_config()
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    return JSONResponse(
        {
            "path": str(CONFIG_PATH),
            "yaml": text,
            "parsed": {k: v for k, v in cfg.items() if not k.startswith("_")},
        },
        headers={"Cache-Control": "no-store"},
    )


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
    _guard_sensitive_config_change(parsed)
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
    if not os.environ.get("GOLDTHREAD_API_TOKEN", "").strip():
        cur_pm = str((load_config().get("paths") or {}).get("pi_models_json") or "")
        for v in body.values:
            if isinstance(v, dict) and v.get("path") == "paths.pi_models_json":
                if str(v.get("value") or "") not in ("", cur_pm):
                    raise HTTPException(
                        403,
                        "安全限制：未配置 GOLDTHREAD_API_TOKEN 时，禁止通过 HTTP API 修改 paths.pi_models_json",
                    )
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
        seams_failed = 0
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
            seams_failed += int(aud.get("seams_failed", 0) or 0)
        put("s2.normalize", "done" if gold_steps else "ready", f"静默 {silent} 步", "机械规则，无需模型")
        put("s2.judge_steps", "warn" if failed_batches else ("done" if decided else "ready"),
            f"{decided} 条决定", f"失败批 {failed_batches}（点续跑补齐）" if failed_batches else "模型逐条判断")
        put("s2.apply", "done" if gold_steps else "ready", f"黄金 {gold_steps} 步",
            f"剪除 {pruned} / 折叠 {folded}")
        put("s2.seams_extract", "done" if seams_total else "ready", f"{seams_total} 处接缝", "程序机械提取")
        put("s2.judge_seams", "warn" if seams_failed else ("done" if seams_total else "ready"),
            f"修复 {seams_fixed} 处" + (f"（失败 {seams_failed}）" if seams_failed else ""),
            f"接缝 {seams_total} 处" + (" · 点「运行 Stage 2 补齐」重试失败项" if seams_failed else ""))
        audited_n = sum(verdicts.values())
        missing_q = len(gold_sessions) - audited_n
        put("s2.judge_quality", "warn" if missing_q > 0 else ("done" if verdicts else "ready"),
            f"{audited_n} / {len(gold_sessions)} 条已审",
            (" · ".join(f"{k} {n}" for k, n in sorted(verdicts.items()))
             + (f" · 缺 {missing_q} 条未审（点续跑补齐）" if missing_q > 0 else "")) or "尚未审计")
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
        delivery_dir = train_root.parent / "04_release" / "delivery"
        zips = _delivery_zips(delivery_dir)
        if zips:
            z = zips[0]
            mb = z.stat().st_size / (1024 * 1024)
            put("s3.package", "done", z.name, f"{mb:.1f} MB · {_package_layout_note(z.name)}")
        elif ho.exists():
            put("s3.package", "ready", "待打包", "到「交付」页选格式并生成交付包")
        else:
            put("s3.package", "todo", "等待构建", "先在训练数据页构建训练集")
    else:
        for sid in ("s3.gate", "s3.windows", "s3.traj", "s3.dpo_review",
                    "s3.split", "s3.manifest", "s3.handoff", "s3.package"):
            put(sid, "todo", "等待 stage2", "")

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
    remaining = [{"id": x["id"], "label": x["label"], "phase": x["phase"],
                  "owner": x["owner"], "status": x["status"]}
                 for x in steps_out if x["status"] not in ("done",)]
    completed = [{"id": x["id"], "label": x["label"], "phase": x["phase"],
                  "value": x.get("value", "")}
                 for x in steps_out if x["status"] == "done"]
    current = next((x for x in steps_out if x["status"] == "running"), None)
    train_root = stage_dirs(cfg)["train"]
    delivery_dir = train_root.parent / "04_release" / "delivery"
    zips = _delivery_zips(delivery_dir)
    _zip0 = zips[0] if zips else None
    _man = _zip_manifest(_zip0) if _zip0 is not None else {}
    _pkg_h = _man.get("package_hash")
    if _zip0 is not None and not _pkg_h:
        _pkg_h = _package_hash_from_name(_zip0.stem)
    _fmt = _man.get("format_contract") if isinstance(_man.get("format_contract"), dict) else None
    deliverable = {
        "ready": bool(zips) and done >= total - 1,
        "handoff_exists": (train_root / "handoff.md").exists(),
        "zip_path": str(_zip0) if _zip0 else None,
        "zip_name": _zip0.name if _zip0 else None,
        "zip_size_bytes": _zip0.stat().st_size if _zip0 else 0,
        "zip_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(_zip0.stat().st_mtime)) if _zip0 else None,
        "package_hash": _pkg_h,
        "format": _fmt,
    }
    # 建议的下一步操作（界面直接渲染按钮）
    actions: List[Dict[str, str]] = []
    if nxt:
        if nxt["id"].startswith("s1.") or nxt["id"].startswith("s2."):
            actions.append({"kind": "run", "target": "all" if nxt["phase"] == "stage1" else "stage2", "label": "运行管线"})
        elif nxt["phase"] == "stage3":
            actions.append({"kind": "run", "target": "stage3", "label": "运行 stage3"})
        if nxt["status"] == "warn":
            actions.append({"kind": "tab", "target": "runs", "label": "查看详情"})
    if deliverable["handoff_exists"] and not deliverable["zip_path"]:
        actions.append({"kind": "export", "target": "zip", "label": "生成交付包"})
    if deliverable["zip_path"]:
        actions.append({"kind": "download", "target": "zip", "label": "下载交付 zip"})
        actions.append({"kind": "handoff", "target": "handoff", "label": "查看 handoff.md"})
    return {
        "phases": phases,
        "total": total,
        "done": done,
        "remaining_count": len(remaining),
        "remaining": remaining,
        "completed": completed,
        "current": current,
        "job": running.as_dict(with_logs=False) if running else None,
        "next": ({"id": nxt["id"], "label": nxt["label"], "phase": nxt["phase"],
                  "owner": nxt["owner"], "status": nxt["status"],
                  "detail": nxt.get("detail", ""), "output": nxt.get("output", "")} if nxt else None),
        "actions": actions,
        "deliverable": deliverable,
        "export_profiles": export_profiles_public(),
        "config_hash": config_hash(cfg),
        "data_dir": str(cfg["paths"]["data_dir"]),
        "handoff": str(train_root / "handoff.md"),
        "flow": [
            {"id": "source", "label": "pi 会话", "path": str(cfg["paths"].get("sessions_dir", "~/.pi/agent/sessions")),
             "desc": "Agent 执行轨迹源文件"},
            {"id": "stage1", "label": "01_raw", "path": str(stage_dirs(cfg)["raw"]),
             "desc": "准入 + 回合分类 + 不可变副本"},
            {"id": "stage2", "label": "02_gold", "path": str(stage_dirs(cfg)["gold"]),
             "desc": "模型判断 + 程序执行"},
            {"id": "stage3", "label": "03_train", "path": str(train_root),
             "desc": "SFT/DPO 样本 + handoff"},
            {"id": "delivery", "label": "交付 zip", "path": str(delivery_dir),
             "desc": "按 goldthread-train 模板打包，交给 autotrain"},
        ],
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
    if body.stage not in ("stage1", "stage2", "stage3", "dataset_evaluate", "all", "review", "refine", "classify"):
        raise HTTPException(400, "stage 必须是 stage1|stage2|stage3|dataset_evaluate|all|review|refine|classify")
    job = _start(body.stage, body.force, {"sessions": body.sessions})
    return {"job_id": job.id, "status": job.status}


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


@app.get("/api/tasks")
def list_tasks_api(limit: int = Query(50, ge=1, le=100)) -> Dict[str, Any]:
    cfg = load_config()
    with JOBS_LOCK:
        active = set(JOBS.keys())
    task_registry.reconcile_stale(active, cfg)
    data = task_registry.list_tasks(cfg, limit=limit)
    enriched: List[Dict[str, Any]] = []
    with JOBS_LOCK:
        for t in data["tasks"]:
            rec = dict(t)
            job = JOBS.get(t["id"])
            if job:
                rec = task_registry.enrich_with_live_job(rec, job)
            enriched.append(rec)
    running_live = sum(1 for t in enriched if t.get("status") == "running")
    queued_live = sum(1 for t in enriched if t.get("status") == "queued")
    return {"running": running_live, "queued": queued_live, "tasks": enriched}


@app.get("/api/tasks/{task_id}")
def get_task_api(task_id: str, offset: int = 0) -> Dict[str, Any]:
    cfg = load_config()
    with JOBS_LOCK:
        job = JOBS.get(task_id)
        active = set(JOBS.keys())
    task_registry.reconcile_stale(active, cfg)
    rec = task_registry.get_task(task_id, cfg)
    if not rec and not job:
        raise HTTPException(404, "任务不存在")
    if job:
        base = rec or {}
        merged = task_registry.enrich_with_live_job({**base, **task_registry.task_record_from_job(job)}, job)
        d = dict(merged)
        d["logs"] = job.logs[offset:]
        d["offset"] = offset
        d["log_count"] = len(job.logs)
        d["live"] = True
        return d
    if rec:
        stored = rec.get("logs") if isinstance(rec.get("logs"), list) else []
        d = {k: v for k, v in rec.items() if k != "logs"}
        d["logs"] = stored[offset:]
        d["offset"] = offset
        d["log_count"] = len(stored)
        d["live"] = False
        return d
    raise HTTPException(404, "任务不存在")


@app.post("/api/tasks/{task_id}/dismiss")
def dismiss_task_api(task_id: str) -> Dict[str, Any]:
    if not task_registry.dismiss_task(task_id, load_config()):
        raise HTTPException(404, "任务不存在")
    return {"ok": True}


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
    source_models = Counter()
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
        source_models[_dominant_model(e) or "unknown"] += 1
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
        "source_models": dict(source_models),
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

def _dpo_source_bucket(source: str) -> str:
    src = str(source or "")
    if src == "stage2_pruned":
        return "pruned"
    if src == "review_rewrite":
        return "review"
    if src == "negative_step":
        return "negative"
    return ""


def _dpo_written_by_session(train_root: Path) -> Dict[str, Dict[str, Any]]:
    """按 session_id 统计训练集里实际写入的 DPO 偏好对（门禁后）。"""
    path = Path(train_root) / "dpo_preference_pairs.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for row in read_jsonl(path):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        sid = str(meta.get("session_id") or row.get("session_id") or "")
        if not sid:
            continue
        slot = out.setdefault(sid, {"total": 0, "parts": {"pruned": 0, "review": 0, "negative": 0}})
        slot["total"] += 1
        bucket = _dpo_source_bucket(str(meta.get("source") or row.get("source") or ""))
        if bucket:
            slot["parts"][bucket] += 1
    return out


def _session_lineage(
    sid: str,
    entry: Dict[str, Any],
    funnels: Dict[str, Any],
    dpo_written: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """一条黄金会话的去向：有用步 → SFT，弯路 → 这条会话的 DPO。

    ``dpo_train`` 必须是训练集里实际写入的偏好对数，不能用门禁前的候选合计，
    否则列表「→ N 对」会和详情页 DPO 标签页对不上。
    """
    sf = funnels.get(sid) if isinstance(funnels, dict) else None
    sf = sf if isinstance(sf, dict) else {}
    funnel = sf.get("funnel") if isinstance(sf.get("funnel"), dict) else {}
    stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
    written = sid in (funnels or {})
    candidates = {
        "pruned": int(sf.get("dpo_pruned") or 0),
        "review": int(sf.get("dpo_review") or 0),
        "negative": int(sf.get("dpo_negative") or 0),
    }
    written_info = None
    if isinstance(dpo_written, dict):
        written_info = dpo_written.get(sid)
    if written_info is None and isinstance(sf.get("dpo_written"), int):
        parts = sf.get("dpo_written_parts") if isinstance(sf.get("dpo_written_parts"), dict) else {}
        written_info = {
            "total": int(sf.get("dpo_written") or 0),
            "parts": {
                "pruned": int(parts.get("pruned") or 0),
                "review": int(parts.get("review") or 0),
                "negative": int(parts.get("negative") or 0),
            },
        }
    if written and written_info is not None:
        parts = {
            "pruned": int((written_info.get("parts") or {}).get("pruned") or 0),
            "review": int((written_info.get("parts") or {}).get("review") or 0),
            "negative": int((written_info.get("parts") or {}).get("negative") or 0),
        }
        dpo_train = int(written_info.get("total") or (parts["pruned"] + parts["review"] + parts["negative"]))
    elif written:
        parts = candidates
        dpo_train = parts["pruned"] + parts["review"] + parts["negative"]
    else:
        parts = candidates
        dpo_train = None
    return {
        "gold_steps": stats.get("gold_steps"),
        "raw_steps": stats.get("raw_steps"),
        "pruned": stats.get("pruned_detours"),
        "dpo_made": int(entry.get("dpo_count") or 0),
        "split": sf.get("split") or "",
        "sft_windows": funnel.get("admitted_steps") if written else None,
        "sft_full": bool(sf.get("full_sft_allowed")) if written else False,
        "train_written": written,
        "dpo_train": dpo_train,
        "dpo_parts": parts if written else None,
        "dpo_candidates": candidates if written else None,
    }


def _dominant_model(entry: Dict[str, Any]) -> str:
    if entry.get("dominant_model"):
        return str(entry["dominant_model"])
    models = entry.get("models") or {}
    if not isinstance(models, dict) or not models:
        return ""
    return max(models.items(), key=lambda item: int(item[1] or 0))[0]


@app.get("/api/raw/sessions")
def raw_sessions(
    label: str = "",
    project: str = "",
    q: str = "",
    adopted: str = "",
    model: str = "",
    limit: int = 500,
) -> Dict[str, Any]:
    cfg = load_config()
    dirs = stage_dirs(cfg)
    index = load_index(dirs["raw"] / "index.json")
    gold_idx = load_index(dirs["gold"] / "index.json")
    gold_sessions_map = gold_idx.get("sessions") or {}
    rows = []
    adopted_n = 0
    pool = 0
    for sid, e in index["sessions"].items():
        if model and _dominant_model(e) != model:
            continue
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
        g = gold_sessions_map.get(sid) or {}
        is_adopted = bool(g)
        pool += 1
        if is_adopted:
            adopted_n += 1
        if adopted == "yes" and not is_adopted:
            continue
        if adopted == "no" and is_adopted:
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
                "origin": e.get("origin") or "",
                "adopted": is_adopted,
                "gold_status": g.get("status") or "",
                "analyzed_at": g.get("last_run") or "",
                "gold_steps": (g.get("stats") or {}).get("gold_steps") if g else None,
                "dpo_count": g.get("dpo_count") if g else None,
            }
        )
    rows.sort(key=lambda r: (0 if r["adopted"] else 1, r["label"] or "", r["project_group"] or "", r["session_id"]))
    return {
        "total": len(rows),
        "pool": pool,
        "adopted": adopted_n,
        "source_models": sorted({_dominant_model(e) for e in index["sessions"].values() if _dominant_model(e)}),
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
    auto_sync: bool = False


@app.post("/api/raw/session/{sid}/relabel")
def relabel(sid: str, body: RelabelBody) -> Dict[str, Any]:
    _assert_no_delete_running()
    if body.label not in ("positive", "negative", "pending"):
        raise HTTPException(400, "label 必须是 positive|negative|pending")
    cfg = load_config()
    try:
        res = stage1_extract.relabel_session(sid, body.label, cfg=cfg)
        return {**res, "note": "已更新标签并同步 raw / gold 索引"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


class TurnOverridesBody(BaseModel):
    turns: Optional[Dict[str, Any]] = None
    reset: bool = False
    recompute_downstream: bool = True


@app.put("/api/raw/session/{sid}/turns")
def save_turn_overrides(sid: str, body: TurnOverridesBody) -> Dict[str, Any]:
    if body.recompute_downstream:
        _assert_no_gold_write_running()
    else:
        _assert_no_delete_running()
    cfg = load_config()
    try:
        res = stage1_extract.update_turn_overrides(
            sid=sid,
            turns_patch=body.turns,
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


@app.post("/api/raw/session/{sid}/analyze")
def analyze_one(sid: str) -> Dict[str, Any]:
    """分析或重新分析一条原始会话：有用步写入黄金会话，弯路写入该会话的 DPO。"""
    cfg = load_config()
    index = load_index(stage_dirs(cfg)["raw"] / "index.json")
    if sid not in (index.get("sessions") or {}):
        raise HTTPException(404, "会话不存在")
    job = _start("analyze", True, {"sessions": [sid]})
    return {"job_id": job.id, "status": job.status}


@app.delete("/api/session/{sid}")
def delete_session_api(sid: str) -> Dict[str, Any]:
    """从原始、黄金与当前训练工作区联动删除一条会话。"""
    from pipeline.session_delete import validate_sid

    try:
        sid = validate_sid(sid)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    cfg = load_config()
    dirs = stage_dirs(cfg)
    raw_index = load_index(dirs["raw"] / "index.json")
    gold_index = load_index(dirs["gold"] / "index.json")
    if sid not in (raw_index.get("sessions") or {}) and sid not in (gold_index.get("sessions") or {}):
        raise HTTPException(404, "会话不存在")
    job = _start("delete", False, {"sessions": [sid]})
    return {"job_id": job.id, "status": job.status}


class CollectBody(BaseModel):
    source: str
    ids: List[str] = []
    analyze: bool = False


@app.get("/api/collect/sessions")
def collect_sessions(source: str = "pi", q: str = "") -> Dict[str, Any]:
    from pipeline.collect import list_sessions

    try:
        return list_sessions(source, q)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/collect/preview")
def collect_preview(source: str, id: str) -> Dict[str, Any]:
    from pipeline.collect import preview_session

    try:
        return preview_session(source, id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/collect")
def collect_start(body: CollectBody) -> Dict[str, Any]:
    if body.source not in ("pi", "cursor"):
        raise HTTPException(400, "source 必须是 pi 或 cursor")
    if not body.ids:
        raise HTTPException(400, "先选择会话")
    if len(body.ids) > 12:
        raise HTTPException(400, "一次最多采集 12 条")
    job = _start(
        "collect",
        False,
        {"source": body.source, "ids": body.ids, "analyze": body.analyze},
    )
    return {"job_id": job.id, "status": job.status}


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
def gold_sessions(label: str = "", project: str = "", status: str = "", q: str = "", model: str = "", output: str = "") -> Dict[str, Any]:
    cfg = load_config()
    dirs = stage_dirs(cfg)
    idx = load_index(dirs["gold"] / "index.json")
    train_stats = read_json(dirs["train"] / "stats.json", default={}) or {}
    session_funnels = train_stats.get("session_funnels") or {}
    dpo_written = _dpo_written_by_session(dirs["train"])
    draft = dataset_draft.load_draft(cfg)
    split_plan = draft.get("split_plan") or {}
    rows = []
    for sid, e in idx["sessions"].items():
        if model and _dominant_model(e) != model:
            continue
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
        lineage = _session_lineage(sid, e, session_funnels, dpo_written)
        project_group = e.get("project_group", "unknown")
        layer = dataset_draft.layer_exclusion(draft, "session", sid, project_group)
        usage = dataset_draft.session_kind_exclusions(draft, sid, project_group)
        lineage["split"] = dataset_draft.effective_split(project_group, sid, split_plan)
        if output == "sft" and not (lineage["train_written"] and (lineage["sft_windows"] or 0) > 0):
            continue
        if output == "dpo" and not (lineage["train_written"] and (lineage["dpo_train"] or 0) > 0):
            continue
        if output == "unwritten" and lineage["train_written"]:
            continue
        rows.append(
            {
                **{k: v for k, v in e.items() if k != "annotation_file"},
                "session_id": sid,
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
                "lineage": lineage,
                "dataset_excluded": layer["action"] == "exclude",
                "exclusion_scope": layer.get("scope", ""),
                "exclusion_reason": layer.get("reason", ""),
                "sft_excluded": usage["sft_excluded"],
                "sft_exclusion_scope": usage["sft_exclusion_scope"],
                "dpo_excluded": usage["dpo_excluded"],
                "dpo_exclusion_scope": usage["dpo_exclusion_scope"],
                "split": lineage["split"],
                "split_source": dataset_draft.split_source(project_group, sid, split_plan),
            }
        )
    rows.sort(key=lambda r: (r.get("label", ""), r.get("project_group", ""), r["session_id"]))
    return {
        "total": len(rows),
        "source_models": sorted({_dominant_model(e) for e in idx["sessions"].values() if _dominant_model(e)}),
        "sessions": rows,
    }


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
    train_root = stage_dirs(cfg)["train"]
    train_stats = read_json(train_root / "stats.json", default={}) or {}
    funnels = train_stats.get("session_funnels") or {}
    dpo_written = _dpo_written_by_session(train_root)
    draft = dataset_draft.load_draft(cfg)
    project_group = str(e.get("project_group") or rows[0].get("project_group") or "unknown")
    layer = dataset_draft.layer_exclusion(draft, "session", sid, project_group)
    usage = dataset_draft.session_kind_exclusions(draft, sid, project_group)
    lineage = _session_lineage(sid, e, funnels, dpo_written)
    return {
        "record": rows[0],
        "annotation": ann,
        "effective": curation.resolve(ann, cfg)["state"],
        "progress": curation.progress(ann),
        "pruned": read_jsonl(Path(e["reject_file"])) if e.get("reject_file") else [],
        "dpo": read_jsonl(Path(e["dpo_file"])) if e.get("dpo_file") else [],
        "diagnostics": read_jsonl(Path(e["diag_file"])) if e.get("diag_file") else [],
        "synthesis_preview": synth_preview,
        "lineage": lineage,
        "dataset_excluded": layer["action"] == "exclude",
        "exclusion_scope": layer.get("scope", ""),
        "exclusion_reason": layer.get("reason", ""),
        "sft_excluded": usage["sft_excluded"],
        "sft_exclusion_scope": usage["sft_exclusion_scope"],
        "dpo_excluded": usage["dpo_excluded"],
        "dpo_exclusion_scope": usage["dpo_exclusion_scope"],
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
    _assert_no_delete_running()
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
    _assert_no_delete_running()
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
    if body.auto_sync_stage3:
        _assert_no_gold_write_running()
    else:
        _assert_no_delete_running()
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
    _assert_no_delete_running()
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
    _assert_no_gold_write_running()
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
    try:
        build_dir, _manifest = dataset_build.resolve_build(cfg)
        p = build_dir / "handoff.md"
    except FileNotFoundError:
        p = stage_dirs(cfg)["train"] / "handoff.md"
    if not p.exists():
        p = stage_dirs(cfg)["train"] / "handoff.md"
    if not p.exists():
        raise HTTPException(404, "handoff.md 还没生成，跑一次 stage3 就会产出")
    return {"path": str(p), "markdown": p.read_text(encoding="utf-8")}


class ExportPackageBody(BaseModel):
    profile: Optional[str] = None
    build_id: Optional[str] = None


_ZIP_NAME_PREFIXES = ("goldthread-train-", "goldthread-delivery-")


def _delivery_zips(delivery_dir: Path) -> List[Path]:
    if not delivery_dir.exists():
        return []
    found: List[Path] = []
    for prefix in _ZIP_NAME_PREFIXES:
        found.extend(delivery_dir.glob(f"{prefix}*.zip"))
    return sorted(found, key=lambda item: item.stat().st_mtime, reverse=True)


def _package_layout_note(name: str) -> str:
    if name.startswith("goldthread-train-"):
        return "goldthread-train 模板"
    return "交付包"


def _package_hash_from_name(name: str) -> str:
    for prefix in _ZIP_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return ""


def _zip_manifest(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    import zipfile  # noqa: PLC0415

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for arc in ("MANIFEST.json", "goldthread-train/data/MANIFEST.json"):
                if arc in names:
                    data = json.loads(zf.read(arc).decode("utf-8"))
                    return data if isinstance(data, dict) else {}
            return {}
    except Exception:  # noqa: BLE001
        return {}


def _zip_length_stats(path: Path) -> Dict[str, Any]:
    """优先读包内 ``data/length_stats.json``；旧包才回退计算并写 sidecar。"""
    from pipeline.common import length_stats as ls  # noqa: PLC0415

    packed = ls.read_length_stats_from_zip(path)
    if packed:
        return packed
    side = ls.read_length_stats_sidecar(path)
    if side:
        return side
    try:
        files = ls.rows_from_zip_lf(path)
        man = _zip_manifest(path)
        stats = ls.build_length_stats(
            files,
            length_gate=man.get("length_gate") if isinstance(man.get("length_gate"), dict) else {},
            source="legacy",
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "datasets": {}}
    if stats.get("ok"):
        try:
            ls.write_length_stats_sidecar(path, stats)
        except OSError:
            pass
    return stats


def _zip_doc_markdown(path: Path, build_id: str = "") -> Dict[str, Any]:
    """包说明：优先该 Build 的 handoff.md，其次 zip 内 README，最后才回退工作区 handoff。"""
    import zipfile  # noqa: PLC0415

    cfg = load_config()
    source = ""
    markdown = ""
    if build_id:
        try:
            build_dir, _ = dataset_build.resolve_build(cfg, build_id)
            handoff = build_dir / "handoff.md"
            if handoff.is_file():
                markdown = handoff.read_text(encoding="utf-8")
                source = str(handoff)
        except (FileNotFoundError, ValueError):
            pass
    docs: List[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            docs = sorted(
                n for n in names
                if n.lower().endswith(".md") and not n.endswith("/")
            )
            if not markdown:
                for arc in (
                    "goldthread-train/README.md",
                    "README.md",
                    "goldthread-train/docs/DEPLOY.md",
                ):
                    if arc in names:
                        markdown = zf.read(arc).decode("utf-8", errors="replace")
                        source = arc
                        break
    except Exception:  # noqa: BLE001
        docs = []
    if not markdown:
        train_ho = stage_dirs(cfg)["train"] / "handoff.md"
        if train_ho.is_file():
            markdown = train_ho.read_text(encoding="utf-8")
            source = str(train_ho)
    return {"markdown": markdown, "source": source, "docs": docs}


@app.post("/api/export/package")
def export_package(body: Optional[ExportPackageBody] = None) -> Dict[str, Any]:
    """异步生成交付 zip（复用 Job 轮询；约 15–30 秒）。

    ``profile`` 是导出格式（Qwen XML / Hermes JSON）。省略时沿用上次的选择，
    没有上次选择则用 Qwen XML。返回 ``{job_id}``。
    """
    cfg = load_config()
    requested_build = body.build_id if body else None
    try:
        _build_dir, manifest = dataset_build.resolve_build(cfg, requested_build)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    chosen = body.profile if body else None
    if chosen:
        try:
            resolve_export_profile(chosen)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    job = _start("package", True, extra={"profile": chosen, "build_id": manifest["build_id"]})
    return {"job_id": job.id}


@app.get("/api/export/package")
def export_package_info() -> Dict[str, Any]:
    """返回当前交付包信息（若存在）。

    zip 文件名用 package_hash（内容指纹）；config_hash 仍是 pipeline.yaml 配置版本，
    二者可以不同。下载始终取 delivery/ 下 mtime 最新的训练 zip。
    """
    cfg = load_config()
    train_root = stage_dirs(cfg)["train"]
    delivery_dir = train_root.parent / "04_release" / "delivery"
    zips = _delivery_zips(delivery_dir)
    if not zips:
        return {"exists": False, "handoff_exists": (train_root / "handoff.md").exists(), "format": None}
    z = zips[0]
    man = _zip_manifest(z)
    package_hash = man.get("package_hash")
    stem = z.stem
    if not package_hash:
        package_hash = _package_hash_from_name(stem)
    fmt = man.get("format_contract") if isinstance(man.get("format_contract"), dict) else None
    return {
        "exists": True,
        "zip_path": str(z),
        "zip_name": z.name,
        "size_bytes": z.stat().st_size,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(z.stat().st_mtime)),
        "package_hash": package_hash,
        "build_id": man.get("build_id"),
        "config_hash": man.get("config_hash"),
        "format": fmt,
    }


def _delivery_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    return stage_dirs(cfg)["train"].parent / "04_release" / "delivery"


def _resolve_delivery_zip(delivery_dir: Path, name: str = "") -> Path:
    if name:
        base = Path(name).name
        if base != name or not base.endswith(".zip"):
            raise HTTPException(400, "非法文件名")
        path = (delivery_dir / base).resolve()
        try:
            path.relative_to(delivery_dir.resolve())
        except ValueError as exc:
            raise HTTPException(400, "非法文件名") from exc
        if not path.is_file():
            raise HTTPException(404, "找不到这个包")
        return path
    zips = sorted(delivery_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True) if delivery_dir.exists() else []
    if not zips:
        raise HTTPException(404, "交付包尚未生成，请运行 stage3 或 POST /api/export/package")
    return zips[0]


@app.get("/api/export/history")
def export_history() -> Dict[str, Any]:
    """交付目录里的全部 zip，新的在前。

    ``length`` 来自包内/旁路静态统计，列表阶段不分词。
    """
    from pipeline.common import length_stats as ls  # noqa: PLC0415

    delivery_dir = _delivery_dir()
    rows = []
    if delivery_dir.exists():
        zips = sorted(delivery_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        for z in zips:
            man = _zip_manifest(z)
            fmt = man.get("format_contract") if isinstance(man.get("format_contract"), dict) else {}
            counts = man.get("counts") if isinstance(man.get("counts"), dict) else {}
            length = ls.pack_length_preview(ls.read_length_stats_static(z))
            rows.append({
                "name": z.name,
                "size_bytes": z.stat().st_size,
                "generated_at": man.get("generated_at") or time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(z.stat().st_mtime)
                ),
                "package_hash": man.get("package_hash") or "",
                "build_id": man.get("build_id") or "",
                "format_label": (fmt or {}).get("label") or "",
                "profile": (fmt or {}).get("profile") or "",
                "sft_train": counts.get("sft.train.lf.jsonl"),
                "sft_eval": counts.get("sft.eval.lf.jsonl"),
                "dpo_train": counts.get("dpo.train.lf.jsonl"),
                "dpo_eval": counts.get("dpo.eval.lf.jsonl"),
                "length": length,
            })
    return {"total": len(rows), "packs": rows}


@app.get("/api/export/pack/docs")
def export_pack_docs(name: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """某个历史包的说明文档（handoff / README）。"""
    z = _resolve_delivery_zip(_delivery_dir(), name)
    man = _zip_manifest(z)
    build_id = str(man.get("build_id") or "")
    docs = _zip_doc_markdown(z, build_id=build_id)
    return {
        "name": z.name,
        "build_id": build_id,
        "package_hash": man.get("package_hash") or "",
        **docs,
    }


@app.get("/api/export/pack/length-stats")
def export_pack_length_stats(name: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """某个历史包的训练内容长度分布。"""
    z = _resolve_delivery_zip(_delivery_dir(), name)
    man = _zip_manifest(z)
    stats = _zip_length_stats(z)
    return {
        "name": z.name,
        "build_id": man.get("build_id") or "",
        "package_hash": man.get("package_hash") or "",
        **stats,
    }


@app.get("/api/export/download")
def export_download(name: str = ""):
    """下载交付 zip。给出 name 时下载该次打包，否则下载最新一份。"""
    z = _resolve_delivery_zip(_delivery_dir(), name)
    return FileResponse(
        path=str(z),
        filename=z.name,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.delete("/api/export/pack")
def export_pack_delete(name: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """删除某个历史交付 zip（及旁路 length_stats sidecar）。"""
    from pipeline.common import length_stats as ls  # noqa: PLC0415

    z = _resolve_delivery_zip(_delivery_dir(), name)
    side = ls.sidecar_path(z)
    deleted = [z.name]
    try:
        z.unlink()
    except OSError as exc:
        raise HTTPException(500, f"删除失败：{exc}") from exc
    if side.is_file():
        try:
            side.unlink()
            deleted.append(side.name)
        except OSError:
            pass
    return {"ok": True, "name": z.name, "deleted": deleted}


@app.get("/api/dataset/builds")
def list_dataset_builds() -> Dict[str, Any]:
    cfg = load_config()
    rows = dataset_build.list_builds(cfg)
    try:
        _path, latest = dataset_build.resolve_build(cfg)
        latest_id = latest["build_id"]
    except FileNotFoundError:
        latest_id = ""
    summaries = [{
        "build_id": row["build_id"],
        "plan_hash": row.get("plan_hash"),
        "created_at": row.get("created_at"),
        "config_hash": row.get("config_hash"),
        "counts": row.get("counts") or {},
        "split_plan": row.get("split_plan") or {},
        "file_count": len(row.get("files") or []),
    } for row in rows]
    return {"total": len(summaries), "latest_build_id": latest_id, "builds": summaries}


class PublishReleaseBody(BaseModel):
    build_id: Optional[str] = None


@app.get("/api/releases")
def get_releases() -> Dict[str, Any]:
    return release.list_releases(load_config())


@app.post("/api/releases")
def publish_release(body: Optional[PublishReleaseBody] = None) -> Dict[str, Any]:
    """冻结一个 Build 为发布版本。不修改当前训练版本。"""
    try:
        return release.publish(load_config(), body.build_id if body else None)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/releases/rollback")
def rollback_release() -> Dict[str, Any]:
    try:
        return release.rollback(load_config())
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/releases/{version}/activate")
def activate_release(version: str) -> Dict[str, Any]:
    try:
        return release.activate(load_config(), version)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/releases/{version}/import-build")
def import_release_build(version: str) -> Dict[str, Any]:
    """把历史发布目录固化成 Build。不切换当前训练版本，也不改最新 Build。"""
    try:
        return release.import_legacy(load_config(), version)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/dataset/builds/{build_id}")
def get_dataset_build(build_id: str) -> Dict[str, Any]:
    try:
        _path, manifest = dataset_build.resolve_build(load_config(), build_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return manifest


class DraftDecisionItem(BaseModel):
    sample_id: str
    action: str
    reason: str = ""


class DraftDecisionsBody(BaseModel):
    decisions: List[DraftDecisionItem]


class DraftSplitBody(BaseModel):
    assignments: Dict[str, str] = {}
    session_assignments: Dict[str, str] = {}


class ScopeDecisionItem(BaseModel):
    scope: str
    id: str
    action: str
    reason: str = ""


class ScopeDecisionsBody(BaseModel):
    updates: List[ScopeDecisionItem]


class DraftReplanBody(BaseModel):
    eval_ratio: float
    seed: int = 42
    dry_run: bool = False
    expected_hash: Optional[str] = None


@app.get("/api/dataset/draft/sample")
def dataset_draft_sample(sample_id: str) -> Dict[str, Any]:
    try:
        return stage3_synth.draft_sample(sample_id, load_config())
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc


@app.get("/api/dataset/draft")
def dataset_draft_preview(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    q: str = "",
    project: str = "",
    split: str = "",
    kind: str = "",
    decision: str = "",
) -> Dict[str, Any]:
    preview = stage3_synth.draft_preview(load_config())
    rows = preview["rows"]
    if project:
        rows = [r for r in rows if r["project_group"] == project]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    if decision:
        rows = [r for r in rows if r["decision"] == decision]
    if q:
        needle = q.lower()
        rows = [r for r in rows if needle in json.dumps(r, ensure_ascii=False).lower()]
    total = len(rows)
    if total == 0:
        offset = 0
    elif offset >= total:
        offset = ((total - 1) // limit) * limit
    project_stats: Dict[str, Dict[str, int]] = {}
    for session in preview["sessions"]:
        project = session["project_group"]
        stat = project_stats.setdefault(project, {
            "candidates": 0, "included": 0, "excluded": 0, "gate_excluded": 0,
            "sft": 0, "trajectory": 0, "dpo": 0, "sessions": 0,
        })
        stat["sessions"] += 1
    for row in preview["rows"]:
        stat = project_stats.setdefault(row["project_group"], {
            "candidates": 0, "included": 0, "excluded": 0, "gate_excluded": 0,
            "sft": 0, "trajectory": 0, "dpo": 0, "sessions": 0,
        })
        stat["candidates"] += 1
        stat["included"] += int(row["included"])
        stat["excluded"] += int(not row["included"] and row["decision"] == "exclude")
        stat["gate_excluded"] += int(not row["included"] and row["decision"] != "exclude")
        stat[row["kind"]] += 1
    return {
        **preview,
        "rows": rows[offset : offset + limit],
        "total": total,
        "offset": offset,
        "projects": sorted({s["project_group"] for s in preview["sessions"]}),
        "project_stats": project_stats,
    }


def _draft_tree_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "candidates": len(samples),
        "included": sum(1 for r in samples if r["included"]),
        "excluded": sum(1 for r in samples if not r["included"] and r["decision"] == "exclude"),
        "gate_excluded": sum(1 for r in samples if not r["included"] and r["decision"] != "exclude"),
        "sft": sum(1 for r in samples if r["kind"] == "sft"),
        "trajectory": sum(1 for r in samples if r["kind"] == "trajectory"),
        "dpo": sum(1 for r in samples if r["kind"] == "dpo"),
    }


def _draft_sample_actionable(row: Dict[str, Any]) -> bool:
    """与控制台树行「恢复 / 停用」按钮条件一致：能点操作的排前面。"""
    if row.get("included"):
        return True
    scope = str(row.get("decision_scope") or "")
    return row.get("decision") == "exclude" and scope in ("", "sample")


def _nest_kind_buckets(child_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """轨迹下固定挂 SFT、DPO 两个可展开节点。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {"sft": [], "dpo": []}
    for row in child_rows:
        kind = str(row.get("kind") or "")
        if kind in buckets:
            buckets[kind].append(row)
    for kind in ("sft", "dpo"):
        # 能点「恢复/停用」的在前；其中已纳入优先，再按步序；门禁丢弃等无法点的沉底。
        buckets[kind].sort(key=lambda r: (
            not _draft_sample_actionable(r),
            not bool(r.get("included")),
            r.get("ref_step") is None,
            int(r.get("ref_step") or 0),
            str(r.get("sample_id") or ""),
        ))
    return [
        {
            "kind": kind,
            "counts": _draft_tree_counts(buckets[kind]),
            "samples": buckets[kind],
        }
        for kind in ("sft", "dpo")
    ]


def _nest_session_trajectories(sid: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """会话下挂轨迹；轨迹下再挂 SFT / DPO 节点，其下才是切片样本。"""
    traj_rows = [r for r in samples if r.get("kind") == "trajectory"]
    child_rows = [r for r in samples if r.get("kind") != "trajectory"]
    if not samples:
        return []
    if not traj_rows:
        return [{
            "trajectory_id": f"traj::{sid}" if sid else "traj::unknown",
            "sample": None,
            "counts": _draft_tree_counts(child_rows),
            "kinds": _nest_kind_buckets(child_rows),
        }]
    trajectories: List[Dict[str, Any]] = []
    for index, traj in enumerate(traj_rows):
        kids = child_rows if index == 0 else []
        trajectories.append({
            "trajectory_id": str(traj.get("sample_id") or f"traj::{sid}"),
            "sample": traj,
            "counts": _draft_tree_counts([traj, *kids]),
            "kinds": _nest_kind_buckets(kids),
        })
    return trajectories


def _flatten_session_samples(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for traj in node.get("trajectories") or []:
        if traj.get("sample"):
            flat.append(traj["sample"])
        for bucket in traj.get("kinds") or []:
            flat.extend(bucket.get("samples") or [])
        flat.extend(traj.get("samples") or [])
    return flat


@app.get("/api/dataset/draft/tree")
def dataset_draft_tree(group_by: str = "project") -> Dict[str, Any]:
    """层级树：分组 → 黄金会话 → 轨迹 → SFT/DPO 节点 → 切片。停用与切分都沿这棵树向下生效。"""
    if group_by not in ("project", "split"):
        raise HTTPException(400, "group_by 必须是 project|split")
    preview = stage3_synth.draft_preview(load_config())
    plan = preview.get("split_plan") or {}
    scopes = preview.get("scope_decisions") or {}

    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for row in preview["rows"]:
        by_session.setdefault(str(row.get("session_id") or ""), []).append(row)

    session_nodes: Dict[str, Dict[str, Any]] = {}
    for info in preview["sessions"]:
        sid = str(info["session_id"])
        samples = by_session.pop(sid, [])
        usage = dataset_draft.session_kind_exclusions(
            {"scope_decisions": scopes}, sid, info["project_group"]
        )
        session_nodes[sid] = {
            "session_id": sid,
            "project_group": info["project_group"],
            "split": info["split"],
            "split_source": info.get("split_source", "project"),
            "excluded": bool(info.get("excluded")),
            "exclusion_scope": info.get("exclusion_scope", ""),
            "exclusion_reason": info.get("exclusion_reason", ""),
            "own_excluded": (scopes.get(f"session:{sid}") or {}).get("action") == "exclude",
            "sft_excluded": usage["sft_excluded"],
            "dpo_excluded": usage["dpo_excluded"],
            "sft_exclusion_scope": usage["sft_exclusion_scope"],
            "dpo_exclusion_scope": usage["dpo_exclusion_scope"],
            "effective_state": info.get("effective_state", ""),
            "counts": _draft_tree_counts(samples),
            "trajectories": _nest_session_trajectories(sid, samples),
        }
    for sid, samples in by_session.items():
        first = samples[0]
        project_group = first.get("project_group", "unknown")
        usage = dataset_draft.session_kind_exclusions(
            {"scope_decisions": scopes}, sid, project_group
        )
        session_nodes[sid] = {
            "session_id": sid,
            "project_group": project_group,
            "split": first.get("split", "train"),
            "split_source": first.get("split_source", "project"),
            "excluded": False,
            "exclusion_scope": "",
            "exclusion_reason": "",
            "own_excluded": False,
            "sft_excluded": usage["sft_excluded"],
            "dpo_excluded": usage["dpo_excluded"],
            "sft_exclusion_scope": usage["sft_exclusion_scope"],
            "dpo_exclusion_scope": usage["dpo_exclusion_scope"],
            "effective_state": "",
            "counts": _draft_tree_counts(samples),
            "trajectories": _nest_session_trajectories(sid, samples),
        }

    groups: Dict[str, Dict[str, Any]] = {}
    for node in session_nodes.values():
        if group_by == "split":
            key = node["split"]
            title = "验证集 eval" if key == "eval" else "训练集 train"
        else:
            key = node["project_group"]
            title = key
        group = groups.setdefault(key, {
            "key": key,
            "title": title,
            "project_group": node["project_group"] if group_by == "project" else "",
            "split": (plan.get("assignments") or {}).get(key, "train") if group_by == "project" else key,
            "own_excluded": (scopes.get(f"project:{key}") or {}).get("action") == "exclude" if group_by == "project" else False,
            "sessions": [],
            "samples": [],
        })
        group["sessions"].append(node)
        group["samples"].extend(_flatten_session_samples(node))
    out_groups = []
    for group in groups.values():
        group["counts"] = _draft_tree_counts(group.pop("samples"))
        group["sessions"].sort(key=lambda n: n["session_id"])
        out_groups.append(group)
    out_groups.sort(key=lambda g: g["key"])
    return {
        "group_by": group_by,
        "groups": out_groups,
        "summary": preview["summary"],
        "split_plan": plan,
        "scope_decisions": scopes,
        "dirty": preview.get("dirty", False),
        "plan_hash": preview.get("plan_hash"),
        "last_build": preview.get("last_build"),
    }


@app.put("/api/dataset/draft/decisions")
def update_dataset_draft_decisions(body: DraftDecisionsBody) -> Dict[str, Any]:
    _assert_no_delete_running()
    try:
        result = dataset_draft.update_decisions(
            [item.model_dump() for item in body.decisions], load_config()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.put("/api/dataset/draft/scopes")
def update_dataset_draft_scopes(body: ScopeDecisionsBody) -> Dict[str, Any]:
    """项目层或会话层停用/恢复，下面的切片一并生效。"""
    _assert_no_delete_running()
    try:
        result = dataset_draft.update_scope_decisions(
            [item.model_dump() for item in body.updates], load_config()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@app.put("/api/dataset/draft/split")
def update_dataset_draft_split(body: DraftSplitBody) -> Dict[str, Any]:
    _assert_no_delete_running()
    cfg = load_config()
    stage3_synth.draft_preview(cfg)  # 解析当前项目集合并初始化 Split Plan。
    try:
        if body.assignments:
            dataset_draft.set_project_splits(body.assignments, cfg)
        if body.session_assignments:
            dataset_draft.set_session_splits(body.session_assignments, cfg)
        plan = dataset_draft.load_draft(cfg)["split_plan"]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "split_plan": plan}


@app.post("/api/dataset/draft/split/replan")
def replan_dataset_draft_split(body: DraftReplanBody) -> Dict[str, Any]:
    _assert_no_delete_running()
    cfg = load_config()
    preview = stage3_synth.draft_preview(cfg, persist_split=False)
    try:
        result = dataset_draft.replan_split(
            cfg,
            eval_ratio=body.eval_ratio,
            seed=body.seed,
            dry_run=body.dry_run,
            base_plan=preview["split_plan"] if body.dry_run else None,
            expected_hash=body.expected_hash,
        )
    except dataset_draft.SplitPlanConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        **result,
        "base_hash": dataset_draft.split_plan_hash(preview["split_plan"]),
        "base_plan": preview["split_plan"],
    }


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
    draft = dataset_draft.load_draft(cfg)
    is_dpo = name.startswith("dpo")
    legacy = stage3_synth._load_dpo_exclusions(stage_dirs(cfg)["state"]) if is_dpo else {}
    out_rows = []
    for orig_idx, r in page:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {
            "session_id": r.get("session_id"),
            "project_group": r.get("project_group"),
            "split": r.get("split"),
            "model": r.get("model"),
            "quality_score": r.get("quality_score"),
            "pathologies": r.get("pathologies", []),
            "tool_calls": r.get("tool_calls", []),
        }
        sample_id = str(r.get("sample_id") or r.get("diagnostic_id") or "")
        decision = dataset_draft.exclusion_for(
            sample_id,
            str(meta.get("project_group") or "unknown"),
            str(meta.get("session_id") or ""),
            draft,
            legacy,
            is_dpo=is_dpo,
        )
        out_rows.append({
            "index": orig_idx,
            "sample_id": sample_id,
            "metadata": meta,
            "preview": _preview(r, name),
            "decision": decision.get("action") or "auto",
            "decision_scope": decision.get("scope") or "",
            "decision_reason": decision.get("reason") or "",
            "included": decision.get("action") != "exclude",
        })
    return {
        "name": name,
        "total": len(filtered),
        "unfiltered_total": len(all_rows),
        "offset": offset,
        "rows": out_rows,
    }


@app.get("/api/train/dataset/{name}/{index}")
def dataset_row(name: str, index: int) -> Dict[str, Any]:
    if "/" in name or "\\" in name or not name.endswith(".jsonl"):
        raise HTTPException(400, "非法文件名")
    cfg = load_config()
    root = stage_dirs(cfg)["train"]
    rows = read_jsonl(root / name)
    if index < 0 or index >= len(rows):
        raise HTTPException(404, "行不存在")
    raw = rows[index]
    out = dict(raw)
    meta = out.get("metadata") or {}
    sample_id = out.get("sample_id") or meta.get("sample_id") or ""
    out["_lineage"] = {"same_sample": _files_with_sample(root, sample_id, name)}
    return out


def _files_with_sample(root: Path, sample_id: str, skip: str) -> List[str]:
    """同一 sample_id 还出现在哪些训练文件里。编号被重写过的发布稿对不上，就返回空。"""
    if not sample_id:
        return []
    found: List[str] = []
    needle = '"' + sample_id + '"'
    for path in sorted(root.glob("*.jsonl")):
        if path.name == skip:
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if needle in line:
                        found.append(path.name)
                        break
        except OSError:
            continue
    return found


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


def _pref_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        inner = value.get("value", "")
        return inner if isinstance(inner, str) else ""
    return ""


def _preview(row: Dict[str, Any], dataset_name: str = "") -> Dict[str, Any]:
    # 1) DPO 偏好对
    if "chosen" in row and "rejected" in row:
        c_val = _pref_text(row.get("chosen"))
        r_val = _pref_text(row.get("rejected"))
        c_think, c_body, c_tools = _extract_think_and_body(c_val)
        r_think, r_body, r_tools = _extract_think_and_body(r_val)
        convs = row.get("conversations") or []
        first_user = next(
            (m.get("value") for m in convs if m.get("from") in ("human", "user")),
            "",
        )
        # body 已剥掉 tool；无思考时仅回退到非 tool 正文，tool 只出现在 *_tools
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
