"""模型判断层（judge）：唯一有权做"删除 / 折叠 / 改写 / 排除 / 评分 / 复核"的部件。

职责边界（对应 DATASET_BEST_PRACTICES.md）：

  程序（本模块之外）只做模式化工作
      - 解析 .jsonl、按 user 切块、组窗口、拼 payload
      - 确定性执行模型给出的决定（删/折/改/清/排）
      - 记录血缘（by / reason / patch / 原文快照）、校验不变量、计数、落盘
      - 从已执行的变更中机械提取"接缝"

  模型（本模块）负责全部判断
      - decide_steps   ：哪些步保留 / 剪除 / 折叠 / 改写 / 清空播报 / 排除出训练
      - audit_seams    ：被删改后的接缝是否平滑，需要时给最小补丁
      - audit_quality  ：整条会话质量报告 + 高光步骤（决定权重）
      - review_dpo     ：偏好对是真好还是仅风格差异
      - （回合级分类见 turnclassify.py，同一原则）

所有提示词都把手册里的硬规则写进去，避免模型自由发挥。
模型调用失败时返回空决定 -> 程序保持原样（不删不改），保证"失败只退化为保守"。
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common.config import load_config
from .common.llm import chat_json

PROMPT_VERSION = "judge.v1"

MAX_STEPS_KEPT = 0.6  # 单会话最多剪除比例（安全阀，由程序强制）

# ---------------------------------------------------------------------------
# 提示词：把 DATASET_BEST_PRACTICES.md 的规则原文固化进去
# ---------------------------------------------------------------------------

_RULES = """你必须严格遵守以下手册规则，不得自由发挥：

【弯路与剪枝（阶段 1）】
A. 状态折叠：write 写完同一文件后紧跟的 edit（修 bug）应合并进 write，不单独保留。
B. 手滑与环境报错：命令写错、传错目录、cp 手滑覆盖、调用不存在的命令 -> 剪除，并把下一步思考开头的懊恼词（Oops / It didn't recognize / Hold on）删掉。
C. 假阴性死胡同：基于错误前提反复试探且没有产出（例如用一个未暴露的 API 反复撞墙）-> 剪除试错段，但保留"查证源码确认真相"的那一步。
D. 真实业务探针排错【必须保留，严禁剪除】：遇到未知格式/异常 -> 写探针打印字节 -> 定位结构 -> 修脚本重跑 的科学闭环，是黄金数据，只能压缩发散措辞，不能删步。

【噪音（第 2 层网关）】必须剪除：模型配置/API 连通性测试、粘贴 Error 400/502/503、硬件与磁盘闲聊（设备能不能 root、C 盘清理）、与本任务无关的提问、纯"继续/可以/hi"。

【思维链（阶段 2）】
1. 严禁空 thinking：空思考且报错的步 -> 剪除；空思考但成功的步 -> 用它的对外播报文本内化为 thinking，或依据其工具调用合成一句"意图"。
2. 超长发散思考（>2400 字或出现多次 Wait/Actually/Hold on 自我推翻）-> 压缩到 300~1600 字，保留"观测事实 -> 结构假设 -> 下一步验证动作"三段式，删除反复自我否定与默写的大段代码。
3. 中间步骤静默执行：最终总结之前的工具步，对外文本应清空，只留 thinking + tool_call。【此项已由程序机械完成，你不需要再返回 clear_text】
4. 心算十六进制/RJ-754 并自行打勾确认（如 "0x1e1e ✓"）-> 必须改写为"需用脚本实测验证"的表述，不得保留心算结论。

【纪律红线】
- 严禁引入原文没有的偏移量、密钥、路径、版本号、结论；改写只能是"删减 + 措辞收敛"，不能新增事实。
- 跨目录偷看（读/拷贝同级其他实验目录，如 /d/crack/xxx-glm/ 里读本不该读的东西）-> 该步标记 exclude（保留在流程中但排除出训练集）。
- 语言必须与原文一致（英文原文输出英文）。
- 段落结构沿用原文空行分段。"""

STEPS_SYSTEM = f"""你是工程分析 Agent 训练数据的判断员。给你一条已按手册清洗过的会话里的若干步骤，你逐条决定它们该怎么处理。

{_RULES}

处理动作（只能选一个）：
- keep    ：保留原样。
- prune   ：整步删除（仅限上述 B/C 类弯路、噪音；D 类科学排错严禁 prune）。
- fold    ：合并进上一步（仅当上一步是 write，本步是同一路径的 edit）。
- rewrite ：替换 thinking（用于：空思考补齐、超长压缩、心算打勾修正、剔除措辞噪音）。必须给出 rewritten。
- exclude ：保留在流程中但排除出训练集（跨目录违规等）。
注意：对外播报的静默清空已由程序机械执行，不要返回 clear_text。

判断要点：
1. 默认 keep。只有存在明确问题的步才动，宁可少动不可错删。
2. D 类科学排错闭环、关键密钥/结构发现、并发侦察 -> 一律 keep。
3. rewrite/压缩时必须保留全部事实（偏移量、字节数、密钥、路径、版本、结论），只删自我推翻与冗余。
4. 只输出 JSON。"""

STEPS_SCHEMA = """输出 JSON：
{"decisions":[{"msg_id":"原 id","action":"keep|prune|fold|rewrite|exclude",
  "reason":"不超过 20 字","rewritten":"action=rewrite 时必填，否则省略"}]}
约束：decisions 只需包含"非 keep"的步；没有需要处理的步就输出 {"decisions":[]}。"""

SEAM_SYSTEM = f"""你是训练数据接缝审查员。程序的机械清洗会删除或折叠一些内容，可能留下断裂的上下文。
给你若干"接缝"：每处包含被移除的内容摘要、删除前后的相邻步骤。

{_RULES}

你的任务：判断接缝是否平滑。
1. 看后续步骤是否残留对被删内容的指代（例如 "the previous command failed"、"刚才那个包"、"上一步报错"）而该内容已被删除 -> 不平滑。
2. 看上下文是否出现逻辑跳空（前一步还在 A，下一步直接跳到 C 且没有过渡）-> 不平滑。
3. 不平滑时给出最小补丁：只允许重写"接缝后那一步"的 thinking，删掉悬空指代或补一句承接，**严禁新增事实、严禁补做被删的事**。
4. 平滑则 smooth=true，不给补丁。
只输出 JSON。"""

SEAM_SCHEMA = """输出 JSON：
{"seams":[{"seam_id":"原 id","smooth":true 或 false,
  "issue":"不平滑时简述","patch_msg_id":"要重写的目标步 id（通常是 after）",
  "patched_thinking":"重写后的完整 thinking（仅 smooth=false 时给）"}]}
约束：seams 只需包含 smooth=false 的项；全部平滑就输出 {"seams":[]}。"""

QUALITY_SYSTEM = f"""你是训练数据质量审计员。给你一条会话的概览与抽样步骤，输出质量报告和高光步骤。

{_RULES}

质量报告要求：
1. verdict：
   - trainable   ：整体可用，接缝已修、无系统性病态。
   - needs_human ：局部可疑，需人工定夺。
   - discard     ：不值得训练（大量空思考、全程盲调工具、反复自我推翻、假阴性死胡同占主体、明显未完成就收尾）。
2. 若"需要整条重写才能用"，必须置 needs_wholesale_rewrite=true，并在 reason 说明；
   这种情况程序会直接报告并放弃该会话，**你不允许通过重写把它救回来**。
3. highlights：挑出真正的关键决策步（突破性发现、并发侦察、密钥/结构确认、正确止损切换方案），
   每步给 why。其余步骤是常规步。高光步用于训练过采样权重。
只输出 JSON。"""

QUALITY_SCHEMA = """输出 JSON：
{"verdict":"trainable|needs_human|discard",
 "confidence":0.0-1.0,
 "reason":"一句话结论",
 "problems":["具体问题"],
 "needs_wholesale_rewrite":true 或 false,
 "highlights":[{"msg_id":"步 id","why":"不超过 24 字","weight":2 或 3}],
 "opening_msg_id":"开场侦察的代表步 id（可空）",
 "closing_msg_id":"收尾核验/写报告的代表步 id（可空）"}"""

DPO_SYSTEM = f"""你是偏好数据复核员。给你若干偏好对（同一上下文下的 chosen 与 rejected），判断这对是否成立。

{_RULES}

判定标准：
- good：rejected 确实是错误行为（手滑命令、假阴性死胡同、心算打勾、反复自我推翻、脑补源码），chosen 确实是修正后的正确行为。
- swap：两者对应关系反了（chosen 才是错的）。
- drop：这对不成立——例如差别只是风格/长短偏好、rejected 其实没错、或两者都不对。
宁可 drop 也不要放进错误的偏好信号。
只输出 JSON。"""

DPO_SCHEMA = """输出 JSON：
{"pairs":[{"sample_id":"原 id","verdict":"good|swap|drop","reason":"不超过 20 字"}]}
约束：只需返回非 good 的项；全部成立就输出 {"pairs":[]}。"""


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------

def _jcfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or load_config()
    j = dict(cfg.get("judge", {}) or {})
    j.setdefault("enabled", True)
    j.setdefault("model", "")
    j.setdefault("step_batch", 24)
    j.setdefault("seam_batch", 12)
    j.setdefault("dpo_batch", 8)
    j.setdefault("max_steps_per_audit", 40)
    j.setdefault("labels", ["positive", "pending"])
    j.setdefault("concurrency", 6)
    j.setdefault("session_workers", 1)
    j.setdefault("fallback_to_rules", True)
    acts = j.get("decisions") or {}
    j["decisions"] = {
        "prune": acts.get("prune", True),
        "fold": acts.get("fold", True),
        "rewrite": acts.get("rewrite", True),
        "clear_text": acts.get("clear_text", True),
        "exclude": acts.get("exclude", True),
    }
    return j


def enabled(cfg: Optional[Dict[str, Any]] = None, label: Optional[str] = None) -> bool:
    j = _jcfg(cfg)
    if not j["enabled"]:
        return False
    if label is None:
        return True
    return label in (j.get("labels") or ["positive", "pending"])


def _model(cfg: Dict[str, Any], j: Dict[str, Any]) -> Optional[str]:
    return j.get("model") or cfg["llm"].get("model")


_SEM_LOCK = threading.Lock()
_SEM = None
_SEM_SIZE = -1


def _sem(concurrency: int):
    """全局在飞请求闸门：批内并发与会话并发共享同一上限，避免打爆网关。"""
    global _SEM, _SEM_SIZE
    size = max(1, int(concurrency))
    with _SEM_LOCK:
        if _SEM is None or _SEM_SIZE != size:
            _SEM = threading.Semaphore(size)
            _SEM_SIZE = size
        return _SEM


def _call(system: str, schema: str, payload: Any, cfg: Dict[str, Any], j: Dict[str, Any],
          tag: str, log=print, temperature: float = 0.0) -> Dict[str, Any]:
    """调用模型；失败返回 {}（程序侧据此保持原样，绝不误删）。"""
    user = schema + "\n\n待判断数据：\n" + json.dumps(payload, ensure_ascii=False)
    sem = _sem(j.get("concurrency", 6))
    try:
        with sem:
            out = chat_json(
                system,
                user,
                model=_model(cfg, j),
                temperature=temperature,
                prompt_version=f"{PROMPT_VERSION}.{tag}",
            )
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log(f"[judge] {tag} 调用失败，保持原样: {exc}")
        return {}


def _pmap(fn, items: List[Any], workers: int) -> List[Any]:
    """批量并发执行；workers<=1 时退化为顺序（便于对比与限流）。"""
    workers = max(1, int(workers))
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    out: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception:  # noqa: BLE001
                out[i] = None
    return out


def _batched(items: List[Any], n: int) -> List[List[Any]]:
    n = max(1, int(n))
    return [items[i: i + n] for i in range(0, len(items), n)]


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n] + "…"


def _tool_brief(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for tc in m.get("tool_calls", []) or []:
        args = tc.get("arguments") or {}
        out.append({
            "name": tc.get("name"),
            "args": _clip(json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args, 220),
        })
    return out


def _result_brief(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": tr.get("name"), "err": bool(tr.get("isError")), "out": _clip(tr.get("content"), 260)}
        for tr in (m.get("tool_results") or [])
    ]


def _step_payload(m: Dict[str, Any], no: int) -> Dict[str, Any]:
    return {
        "msg_id": m.get("id"),
        "no": no,
        "turn": m.get("turn_index"),
        "thinking": _clip(m.get("raw_thinking") or m.get("thinking"), 1200),
        "think_len": len((m.get("raw_thinking") or m.get("thinking") or "")),
        "text": _clip(m.get("raw_text") or m.get("text"), 200),
        "tools": _tool_brief(m),
        "results": _result_brief(m),
        "flags": [f for f in (m.get("flags") or []) if f in (
            "silent_executed", "thinking_compressed", "thinking_backfilled", "thinking_synthesized")],
    }


# ---------------------------------------------------------------------------
# 1) 步级决定
# ---------------------------------------------------------------------------

def decide_steps(gold_messages: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None,
                 log=print) -> Dict[str, Dict[str, Any]]:
    """返回 {msg_id: {action, reason, rewritten}}；模型未提及的步即 keep。"""
    cfg = cfg or load_config()
    j = _jcfg(cfg)
    if not j["enabled"]:
        return {}
    asst = [m for m in gold_messages if m.get("role") == "assistant"]
    if not asst:
        return {}
    root = next((m for m in gold_messages if m.get("role") == "user"), None)
    decisions: Dict[str, Dict[str, Any]] = {}

    def _one(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "root_task": _clip((root or {}).get("content"), 300),
            "steps": [_step_payload(m, i + 1) for i, m in enumerate(batch)],
        }
        return _call(STEPS_SYSTEM, STEPS_SCHEMA, payload, cfg, j, "steps", log)

    batches = _batched(asst, j["step_batch"])
    outs = _pmap(_one, batches, j["concurrency"])
    for batch, out in zip(batches, outs):
        out = out or {}
        for d in out.get("decisions", []) or []:
            if not isinstance(d, dict):
                continue
            mid, act = d.get("msg_id"), (d.get("action") or "").strip()
            if mid not in {m.get("id") for m in batch}:
                continue
            if act not in ("keep", "prune", "fold", "rewrite", "exclude"):
                continue
            if act != "keep" and not j["decisions"].get(act, True):
                continue
            if act == "rewrite" and not (d.get("rewritten") or "").strip():
                continue
            decisions[mid] = {"action": act, "reason": _clip(d.get("reason"), 60),
                              "rewritten": (d.get("rewritten") or "").strip()}
    return decisions


def prewarm(sessions: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None,
            log=print) -> Dict[str, Any]:
    """会话级并行预热：多条会话同时发起步级判断。

    结果丢弃（主循环会重算），但 llm.chat 以 (prompt_version+model+messages+参数) 的 sha256
    落盘缓存，主循环二次调用即缓存命中 —— 真实模型延迟被会话并行摊掉，判断内容完全一致。

    sessions: [{"sid","raw_file","annotations","manual_overrides","max_tool_output_chars"}]
    """
    from .common import pilog  # noqa: PLC0415

    cfg = cfg or load_config()
    j = _jcfg(cfg)
    if not j["enabled"] or not sessions:
        return {"sessions": 0, "steps": 0}
    workers = max(1, int(j.get("session_workers", 1)))

    def _one(s: Dict[str, Any]) -> int:
        try:
            sess = pilog.parse_session(Path(s["raw_file"]),
                                       max_tool_output_chars=int(s.get("max_tool_output_chars", 12000)))
            flow, _marks, _st = pilog.build_flow(sess, cfg, s.get("annotations"), s.get("manual_overrides"))
            decide_steps(flow, cfg=cfg, log=lambda *_: None)  # 目的：写入磁盘缓存
            return sum(1 for m in flow if m.get("role") == "assistant")
        except Exception as exc:  # noqa: BLE001
            log(f"[judge] 预热失败 {str(s.get('sid'))[:8]}: {exc}")
            return 0

    t0 = time.time()
    results = _pmap(_one, sessions, workers)
    steps = sum(r or 0 for r in results)
    log(f"[judge] 会话并行预热 {len(sessions)} 条（{workers} 会话 × {j.get('concurrency')} 批并发）"
        f"共 {steps} 步，耗时 {time.time() - t0:.0f}s")
    return {"sessions": len(sessions), "steps": steps}


# ---------------------------------------------------------------------------
# 2) 接缝审核
# ---------------------------------------------------------------------------

def audit_seams(seams: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None,
                log=print) -> Dict[str, Dict[str, Any]]:
    """返回 {seam_id: {smooth, issue, patch_msg_id, patched_thinking}}。"""
    cfg = cfg or load_config()
    j = _jcfg(cfg)
    if not j["enabled"] or not seams:
        return {}
    out_map: Dict[str, Dict[str, Any]] = {}
    batches = _batched(seams, j["seam_batch"])
    outs = _pmap(lambda b: _call(SEAM_SYSTEM, SEAM_SCHEMA, {"seams": b}, cfg, j, "seams", log),
                 batches, j["concurrency"])
    for batch, out in zip(batches, outs):
        out = out or {}
        for s in out.get("seams", []) or []:
            if not isinstance(s, dict) or not s.get("seam_id"):
                continue
            out_map[str(s["seam_id"])] = {
                "smooth": bool(s.get("smooth", True)),
                "issue": _clip(s.get("issue"), 120),
                "patch_msg_id": s.get("patch_msg_id"),
                "patched_thinking": (s.get("patched_thinking") or "").strip(),
            }
    return out_map


# ---------------------------------------------------------------------------
# 3) 质量报告 + 高光
# ---------------------------------------------------------------------------

def audit_quality(gold_messages: List[Dict[str, Any]], stats: Dict[str, Any],
                  cfg: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    cfg = cfg or load_config()
    j = _jcfg(cfg)
    if not j["enabled"]:
        return {}
    asst = [m for m in gold_messages if m.get("role") == "assistant"]
    if not asst:
        return {}
    cap = int(j["max_steps_per_audit"])
    if len(asst) <= cap:
        sample = asst
    else:  # 均匀抽样
        step = len(asst) / cap
        sample = [asst[min(len(asst) - 1, int(i * step))] for i in range(cap)]
    root = next((m for m in gold_messages if m.get("role") == "user"), None)
    payload = {
        "root_task": _clip((root or {}).get("content"), 300),
        "session_stats": {k: stats.get(k) for k in (
            "raw_steps", "gold_steps", "folded", "pruned_detours",
            "thinking_compressed", "thinking_backfilled", "thinking_synthesized",
            "silent_cleared") if k in stats},
        "sampled": len(sample),
        "total_steps": len(asst),
        "steps": [_step_payload(m, i + 1) for i, m in enumerate(sample)],
    }
    out = _call(QUALITY_SYSTEM, QUALITY_SCHEMA, payload, cfg, j, "quality", log)
    if not out:
        return {}
    valid = {m.get("id") for m in asst}
    verdict = str(out.get("verdict") or "").strip()
    if verdict not in ("trainable", "needs_human", "discard"):
        verdict = "needs_human"
    try:
        conf = float(out.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    highlights = []
    for hl in out.get("highlights", []) or []:
        if not isinstance(hl, dict) or hl.get("msg_id") not in valid:
            continue
        try:
            w = int(hl.get("weight") or 2)
        except (TypeError, ValueError):
            w = 2
        highlights.append({"msg_id": hl["msg_id"], "why": _clip(hl.get("why"), 60),
                           "weight": max(1, min(5, w))})
    return {
        "verdict": verdict,
        "confidence": round(conf, 3),
        "reason": _clip(out.get("reason"), 200),
        "problems": [_clip(p, 160) for p in (out.get("problems") or []) if isinstance(p, str)][:12],
        "needs_wholesale_rewrite": bool(out.get("needs_wholesale_rewrite")),
        "highlights": highlights,
        "opening_msg_id": out.get("opening_msg_id") if out.get("opening_msg_id") in valid else None,
        "closing_msg_id": out.get("closing_msg_id") if out.get("closing_msg_id") in valid else None,
        "audited_steps": len(sample),
        "total_steps": len(asst),
        "model": _model(cfg, j),
        "prompt_version": PROMPT_VERSION,
    }


# ---------------------------------------------------------------------------
# 4) DPO 复核
# ---------------------------------------------------------------------------

def review_dpo(pairs: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None,
               log=print) -> Dict[str, Dict[str, Any]]:
    """返回 {sample_id: {verdict, reason}}（非 good 项）。"""
    cfg = cfg or load_config()
    j = _jcfg(cfg)
    if not j["enabled"] or not pairs:
        return {}
    out_map: Dict[str, Dict[str, Any]] = {}

    def _one(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"pairs": [
            {
                "sample_id": p.get("sample_id"),
                "reason": (p.get("metadata") or {}).get("reason"),
                "context_tail": [
                    {"from": c.get("from"), "value": _clip(c.get("value"), 400)}
                    for c in (p.get("conversations") or [])[-4:]
                ],
                "chosen": _clip((p.get("chosen") or {}).get("value"), 1400),
                "rejected": _clip((p.get("rejected") or {}).get("value"), 1400),
            } for p in batch]}
        return _call(DPO_SYSTEM, DPO_SCHEMA, payload, cfg, j, "dpo", log)

    batches = _batched(pairs, j["dpo_batch"])
    outs = _pmap(_one, batches, j["concurrency"])
    for batch, out in zip(batches, outs):
        out = out or {}
        for it in out.get("pairs", []) or []:
            if not isinstance(it, dict):
                continue
            sid, v = it.get("sample_id"), (it.get("verdict") or "").strip()
            if v not in ("swap", "drop"):
                continue
            out_map[str(sid)] = {"verdict": v, "reason": _clip(it.get("reason"), 60)}
    return out_map


# ---------------------------------------------------------------------------
# 5) 接缝提取（程序侧机械生成，供 audit_seams 使用）
# ---------------------------------------------------------------------------

def extract_seams(gold_messages: List[Dict[str, Any]], pruned: List[Dict[str, Any]],
                  turn_marks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """从已执行的删/折操作机械推导接缝：被移除内容的摘要 + 前后相邻步。

    不生成任何判断，只是把"哪里被切断了"结构化出来，交给模型审。
    """
    asst = [m for m in gold_messages if m.get("role") == "assistant"]
    by_id = {m.get("id"): i for i, m in enumerate(asst)}
    seams: List[Dict[str, Any]] = []

    def _ctx(i: int) -> Dict[str, Any]:
        m = asst[i] if 0 <= i < len(asst) else None
        if not m:
            return {}
        return {
            "msg_id": m.get("id"),
            "thinking": _clip(m.get("thinking"), 500),
            "text": _clip(m.get("text"), 160),
            "tools": [t["name"] for t in _tool_brief(m)],
        }

    # 1) 步级剪枝：被删步的原位置 -> 现在相邻的前后两步
    for k, pd in enumerate(pruned):
        rid = pd.get("dropped_msg_id")
        if not rid:
            continue
        # 找到被删步在原序列中的后继（在 gold 里被删步已不存在，用 pruned 记录的 rejected 内容定位）
        after = None
        for i, m in enumerate(asst):
            if m.get("pruned_detour") == rid:
                after = i
                break
        seams.append({
            "seam_id": f"prune::{rid}",
            "type": "detour_pruned",
            "removed": {
                "msg_id": rid,
                "reason": pd.get("reason"),
                "thinking": _clip((pd.get("rejected_msg") or {}).get("thinking"), 500),
                "tools": [tc.get("name") for tc in ((pd.get("rejected_msg") or {}).get("tool_calls") or [])],
                "error_output": _clip(pd.get("error_output"), 220),
            },
            "before": _ctx(after - 1) if after is not None else {},
            "after": _ctx(after) if after is not None else {},
        })

    # 2) 状态折叠：合并步本身
    for m in asst:
        if not m.get("folded_from"):
            continue
        i = by_id.get(m.get("id"))
        seams.append({
            "seam_id": f"fold::{m.get('id')}",
            "type": "state_folded",
            "removed": {"msg_id": m.get("folded_from"), "reason": "write+edit 合并",
                        "thinking": _clip((m.get("fold_details") or {}).get("edit_thinking"), 400)},
            "before": _ctx((i or 0) - 1),
            "after": _ctx(i or 0),
        })

    # 3) 回合级：催促折叠 / 噪音删除（上下文跳接）
    for mk in (turn_marks or []):
        if mk.get("action") not in ("folded", "internalized", "dropped"):
            continue
        if not (mk.get("folded") or mk.get("dropped")):
            continue
        seams.append({
            "seam_id": f"turn::{mk.get('turn_index')}",
            "type": f"turn_{mk.get('action')}",
            "removed": {
                "turn_index": mk.get("turn_index"),
                "kind": mk.get("kind"),
                "reason": mk.get("reason"),
                "user_text": _clip(mk.get("text"), 200),
                "steps": mk.get("assistant_steps"),
                "popped_await_msg": mk.get("popped_await_msg"),
            },
            "before": {}, "after": {},
        })

    # 4) 静默执行 / 空思考合成：单步内的"内容消失"
    for m in asst:
        flags = m.get("flags") or []
        if "silent_executed" in flags:
            seams.append({
                "seam_id": f"silent::{m.get('id')}",
                "type": "silent_cleared",
                "removed": {"msg_id": m.get("id"), "reason": "中间步播报清空",
                            "text": _clip(m.get("raw_text"), 300)},
                "before": {}, "after": _ctx(by_id.get(m.get("id"), 0)),
            })
        if "thinking_synthesized" in flags:
            seams.append({
                "seam_id": f"synth::{m.get('id')}",
                "type": "thinking_synthesized",
                "removed": {"msg_id": m.get("id"), "reason": "空思考由程序合成",
                            "thinking": _clip(m.get("thinking"), 300)},
                "before": _ctx(by_id.get(m.get("id"), 0) - 1),
                "after": _ctx(by_id.get(m.get("id"), 0)),
            })
    return seams