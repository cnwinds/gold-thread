"""流水线步骤目录：运行界面与实际执行的一一对应关系（唯一事实源）。

每个步骤都对应代码里真实做的一件事，顺序即执行顺序。
- phase  ：所属阶段（stage1 / stage2 / stage3）
- owner  ：谁在执行（程序 / 模型 / 人工）
- label  ：界面上显示的名字
- detail ：这一步到底做什么（可对到代码位置）
- output ：产出物（用于界面显示"做完了有什么"）

pipeline 各 stage 通过：
- `begin()` / `finish()`：写入可见执行日志（开始说明工作内容，结束带耗时），并同步 `@@STEP`
- `emit()`：仅上报实时进度（`@@STEP {json}`），不刷可见日志（用于循环内进度）

server 解析 `@@STEP` 后挂到 job.progress，运行界面据此显示"正在做什么"。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

PHASES: List[Dict[str, Any]] = [
    {"id": "stage1", "name": "① 采集与清洗", "desc": "把 pi 会话变成 01_raw 里带回合标注的不可变副本"},
    {"id": "stage2", "name": "② 黄金提纯", "desc": "模型决定删/折/改/排，程序执行并留血缘，再接缝修复 + 质量报告"},
    {"id": "stage3", "name": "③ 训练集合成", "desc": "门控、窗口拼装、DPO 复核、切分、生成 handoff"},
]

STEPS: List[Dict[str, Any]] = [
    # ---------------- stage1 ----------------
    {"id": "s1.scan", "phase": "stage1", "owner": "程序", "label": "扫描会话目录",
     "detail": "递归扫描 ~/.pi/agent/sessions 下的 .jsonl，统计文件数与所属文件夹",
     "output": "扫描计数"},
    {"id": "s1.admit", "phase": "stage1", "owner": "程序", "label": "目录准入过滤",
     "detail": "只保留 admission.workspace_prefixes（如 --d--crack-），排除 exclude_folders",
     "output": "准入会话清单"},
    {"id": "s1.dedupe", "phase": "stage1", "owner": "程序", "label": "同源副本去重",
     "detail": "同 cwd + 同首条用户消息的迁移/复制副本只留最长一份",
     "output": "重复清单"},
    {"id": "s1.classify", "phase": "stage1", "owner": "模型", "label": "回合语义分类",
     "detail": "逐回合判定 root/task/setup/noise/continuation/correction（turns.use_model）",
     "output": "turn_annotations"},
    {"id": "s1.flow", "phase": "stage1", "owner": "程序", "label": "回合成流并落盘",
     "detail": "按分类剥噪音、折叠催促、纠偏转 system、根任务标准化；写不可变副本 + meta.json",
     "output": "01_raw/{positive,negative,pending} + index.json"},

    # ---------------- stage2 ----------------
    {"id": "s2.normalize", "phase": "stage2", "owner": "程序", "label": "机械规整",
     "detail": "中间步静默执行（手册强制）、nohup+sleep+tail 轮询命令改写",
     "output": "规范化后的消息流"},
    {"id": "s2.judge_steps", "phase": "stage2", "owner": "模型", "label": "模型步级判断",
     "detail": "按批送判，逐条给 keep/prune/fold/rewrite/exclude 与理由（judge.decide_model）",
     "output": "步级决定 + 理由"},
    {"id": "s2.apply", "phase": "stage2", "owner": "程序", "label": "执行决定 + 留血缘",
     "detail": "确定性执行删除/折叠/改写/排除；守不变量（首步不可删、剪除≤60%、补丁不新增事实）",
     "output": "gold 消息流 + pruned 记录"},
    {"id": "s2.seams_extract", "phase": "stage2", "owner": "程序", "label": "提取接缝",
     "detail": "机械找出被删改切断的位置：被移除内容摘要 + 前后相邻步",
     "output": "接缝清单"},
    {"id": "s2.judge_seams", "phase": "stage2", "owner": "模型", "label": "接缝审核与修复",
     "detail": "判断接缝是否平滑；不平滑给最小补丁（只改接缝后那步，禁新增事实）",
     "output": "seam 审计 + 补丁"},
    {"id": "s2.judge_quality", "phase": "stage2", "owner": "模型", "label": "质量报告 + 高光",
     "detail": "分块审计整条会话：可用/需人工/不建议训练 + 是否需整条重写 + 高光步",
     "output": "model_report（含 highlights）"},
    {"id": "s2.dpo_build", "phase": "stage2", "owner": "程序", "label": "构造 DPO 对",
     "detail": "把剪掉的弯路配成 chosen/rejected；反例会话的病态步另起来源",
     "output": "dpo_pairs.jsonl"},

    # ---------------- stage3 ----------------
    {"id": "s3.gate", "phase": "stage3", "owner": "程序", "label": "入库门控",
     "detail": "标签/复核态/模型否决（discard、needs_wholesale_rewrite）三态过滤，决定每条会话是否产样本",
     "output": "门控结论 + 漏斗"},
    {"id": "s3.windows", "phase": "stage3", "owner": "程序", "label": "单步窗口合成",
     "detail": "根任务锚定 + 历史 thinking 剥离 + 阶段权重（高光步由模型给权重）",
     "output": "sft_step_windows"},
    {"id": "s3.traj", "phase": "stage3", "owner": "程序", "label": "完整轨迹合成",
     "detail": "整条会话按留出规则产出长轨迹样本",
     "output": "sft_gold_trajectories"},
    {"id": "s3.dpo_review", "phase": "stage3", "owner": "模型", "label": "DPO 逐对复核",
     "detail": "判断 chosen 是否真更优、rejected 是否真错（风格差异不算），输出 good/swap/drop",
     "output": "DPO 复核结果（含缓存）"},
    {"id": "s3.split", "phase": "stage3", "owner": "程序", "label": "打乱与切分",
     "detail": "跨项目全局打乱；按项目整包切 train/eval，防相邻步泄漏",
     "output": "split 映射"},
    {"id": "s3.manifest", "phase": "stage3", "owner": "程序", "label": "写清单与切分文件",
     "detail": "dataset_info.json（LLaMA-Factory 可直接读）、stats.json、*.train/*.eval 文件",
     "output": "03_train/*.jsonl + dataset_info"},
    {"id": "s3.handoff", "phase": "stage3", "owner": "程序", "label": "生成 handoff 文档",
     "detail": "输出 handoff.md + train_sft.yaml / train_dpo.yaml：从环境搭建到训练的全流程",
     "output": "03_train/handoff.md"},
    {"id": "s3.package", "phase": "stage3", "owner": "程序", "label": "打包交付 zip",
     "detail": "规整成 sft/dpo 的 *.lf.jsonl，dataset_info 与 YAML 的 gt_*_lf 对齐，打成可直接训练的 zip",
     "output": "04_release/delivery/*.zip"},
]

STEP_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in STEPS}
PHASE_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PHASES}
STEPS_BY_PHASE: Dict[str, List[Dict[str, Any]]] = {}
for _s in STEPS:
    STEPS_BY_PHASE.setdefault(_s["phase"], []).append(_s)

# log 回调稳定键 → {step_id: monotonic 起点}
# 注意：不能用 id(bound_method)，CPython 每次取 bound method 可能是新对象。
_ACTIVE: Dict[Any, Dict[str, float]] = {}


def _log_key(log) -> Any:
    """为 log 回调生成稳定键，使 begin/finish 能对上同一计时器。

    不能用 id(bound_method)：CPython 每次取绑定方法可能是新对象。
    list.append 等 builtin method 没有 __func__，改用 (id(self), __name__)。
    """
    self = getattr(log, "__self__", None)
    if self is not None:
        name = getattr(log, "__name__", None)
        if not name:
            func = getattr(log, "__func__", None)
            name = getattr(func, "__name__", None)
        if name:
            return (id(self), name)
        func = getattr(log, "__func__", None)
        if func is not None:
            return (id(self), func)
    return id(log)


def format_elapsed(seconds: float) -> str:
    """把秒数格式成一眼能懂的耗时文案。"""
    if seconds < 0:
        seconds = 0.0
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    total = int(round(seconds))
    m, s = divmod(total, 60)
    if m < 60:
        return f"{m}分{s:02d}秒"
    h, m = divmod(m, 60)
    return f"{h}小时{m:02d}分{s:02d}秒"


def _meta(step_id: str) -> Dict[str, Any]:
    return STEP_BY_ID.get(step_id) or {"id": step_id, "label": step_id, "detail": "", "owner": "", "phase": ""}


def _phase_name(step_id: str) -> str:
    phase = _meta(step_id).get("phase") or ""
    return (PHASE_BY_ID.get(phase) or {}).get("name") or phase or "步骤"


def _safe_log(log, msg: str) -> None:
    try:
        log(msg)
    except Exception:  # noqa: BLE001
        pass


def emit(log, step_id: str, done: Optional[int] = None, total: Optional[int] = None,
         msg: str = "") -> None:
    """上报实时进度。server 会解析 `@@STEP` 行并挂到 job.progress（不进可见日志）。"""
    payload: Dict[str, Any] = {"id": step_id}
    if done is not None:
        payload["done"] = done
    if total is not None:
        payload["total"] = total
    if msg:
        payload["msg"] = msg
    try:
        log("@@STEP " + json.dumps(payload, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass


def begin(log, step_id: str, msg: str = "") -> None:
    """标记步骤开始：写可见日志（工作内容一眼能懂）+ 上报进度。"""
    meta = _meta(step_id)
    label = meta.get("label") or step_id
    detail = (meta.get("detail") or "").strip()
    owner = (meta.get("owner") or "").strip()
    phase = _phase_name(step_id)

    _ACTIVE.setdefault(_log_key(log), {})[step_id] = time.monotonic()

    extra = f"（{msg}）" if msg else ""
    _safe_log(log, f"▶ 开始 · [{phase}] {label}{extra}")
    if detail:
        who = f"；执行方：{owner}" if owner else ""
        _safe_log(log, f"   要做：{detail}{who}")
    elif msg:
        _safe_log(log, f"   要做：{msg}")

    emit(log, step_id, msg=msg or label)


def finish(log, step_id: str, done: Optional[int] = None, total: Optional[int] = None,
           msg: str = "") -> None:
    """标记步骤完成：写可见日志（含耗时）+ 上报完成进度。"""
    started = _ACTIVE.get(_log_key(log), {}).pop(step_id, None)
    elapsed = (time.monotonic() - started) if started is not None else None
    meta = _meta(step_id)
    label = meta.get("label") or step_id
    phase = _phase_name(step_id)

    parts = [f"✓ 完成 · [{phase}] {label}"]
    if msg:
        parts.append(msg)
    if elapsed is not None:
        parts.append(f"耗时 {format_elapsed(elapsed)}")
    _safe_log(log, " · ".join(parts))

    # 未显式给 done/total 时，按整步完成上报，便于 runbook 勾选
    if done is None and total is None:
        done, total = 1, 1
    emit(log, step_id, done=done, total=total, msg=msg or label)
