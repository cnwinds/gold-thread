"""流水线步骤目录：运行界面与实际执行的一一对应关系（唯一事实源）。

每个步骤都对应代码里真实做的一件事，顺序即执行顺序。
- phase  ：所属阶段（stage1 / stage2 / human / stage3）
- owner  ：谁在执行（程序 / 模型 / 人工）
- label  ：界面上显示的名字
- detail ：这一步到底做什么（可对到代码位置）
- output ：产出物（用于界面显示"做完了有什么"）

pipeline 各 stage 通过 `emit()` 上报实时进度（`@@STEP {json}` 行），
server 解析后挂到 job.progress，运行界面据此显示"正在做什么"。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

PHASES: List[Dict[str, Any]] = [
    {"id": "stage1", "name": "① 采集与清洗", "desc": "把 pi 会话变成 01_raw 里带回合标注的不可变副本"},
    {"id": "stage2", "name": "② 黄金提纯", "desc": "模型决定删/折/改/排，程序执行并留血缘，再接缝修复 + 质量报告"},
    {"id": "human", "name": "③ 人工复核", "desc": "逐条浏览模型判断，采纳/不采纳/手改，最后定稿"},
    {"id": "stage3", "name": "④ 训练集合成", "desc": "门控、窗口拼装、DPO 复核、切分、生成 handoff"},
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

    # ---------------- human ----------------
    {"id": "h.review", "phase": "human", "owner": "人工", "label": "逐条复核",
     "detail": "黄金数据页浏览改写/问题步，采纳/不采纳/手改/排除；可写意见让模型重做",
     "output": "annotations.human_review.items"},
    {"id": "h.finalize", "phase": "human", "owner": "人工", "label": "定稿",
     "detail": "确认无误后标记定稿，未处置项按模型结果默认生效；定稿态才允许稳定入库",
     "output": "human_review.finalized"},

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
     "detail": "把 handoff + 数据集 + 训练配置打成 goldthread-delivery-*.zip，可直接交给下游 AI",
     "output": "03_train/delivery/*.zip"},
]

STEP_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in STEPS}
STEPS_BY_PHASE: Dict[str, List[Dict[str, Any]]] = {}
for _s in STEPS:
    STEPS_BY_PHASE.setdefault(_s["phase"], []).append(_s)


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