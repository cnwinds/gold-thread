# -*- coding: utf-8 -*-
"""一次性补丁：review.py 增加 全面优化 + 独立审核循环"""
src = open('pipeline/review.py', encoding='utf-8').read()

# 1. 新提示词（插在 REFINE_REWRITE_SCHEMA 之后）
anchor = '''REFINE_REWRITE_SCHEMA = """输出 JSON：
{"rewrites": [{"msg_id": "原消息 id", "rewritten": "修订后的 thinking", "reply": "一句话说明按意见改了什么"}]}"""'''
assert src.count(anchor) == 1
prompts = anchor + '''

OPTIMIZE_SYSTEM = """你是工程分析 Agent 训练数据的全面优化器。对给出的每一步 thinking 做生产级优化，目标是改完即可直接进入训练集：
- 保留：观测到的事实/字节/工具输出 -> 结构假设 -> 下一步验证动作，三段式结构完整；
- 修复：心算十六进制并打勾（改写为"用脚本实测验证"的表述）、脑补源码、反复自我推翻（Wait/Actually/Hold on）、与任务无关的复述；
- 语言不变：与原文语言一致，严禁翻译；
- 段落不变：沿用原文的空行分段与列表格式，一个主题一段；
- 事实红线：偏移量、密钥、字节数、路径、版本号、结论必须与原文一致，严禁引入原文没有的信息，严禁丢失关键数据；
- 若该步已足够好，输出轻修版本即可；但输入的每一条都必须返回。
严格只输出 JSON。"""

OPTIMIZE_SCHEMA = """输出 JSON：
{"rewrites": [{"msg_id": "原消息 id", "rewritten": "优化后的完整 thinking"}]}
要求：输入的每一条 msg_id 都必须在 rewrites 中出现。"""

AUDIT_SYSTEM = """你是独立的工程分析训练数据审计员，与优化器不是同一个模型。逐条审计优化后的 thinking 是否达到"可直接进入训练集"的标准：
- 事实保真：偏移/字节/密钥/路径/版本/结论是否与原版一致；是否引入原文没有的信息；是否丢失关键数据；
- 语言与结构：语言是否与原版一致；段落/列表结构是否保留；
- 质量达标：自我推翻循环是否清除；是否仍有心算打勾；是否保持第一人称；是否短到丢失实质内容。
逐条给出问题；全部达标时 passed=true 且 issues 为空。
严格只输出 JSON。"""

AUDIT_SCHEMA = """输出 JSON：
{"passed": true 或 false,
 "overall": "一句话总评",
 "issues": [{"msg_id": "原消息 id", "problem": "问题简述", "instruction": "给优化器的修改指令"}]}"""

OPTIMIZE_REVISE_SYSTEM = """你是工程分析 Agent 训练数据的全面优化器（修订轮）。独立审计员否决了部分条目并给出逐条修改指令：
- 以 original_thinking 为唯一事实来源，逐条落实 audit_instruction；
- 语言与段落结构与原文一致；严禁引入原文没有的信息；
- 每一条都必须返回修订后的完整 thinking。
严格只输出 JSON。"""'''
src = src.replace(anchor, prompts)

# 2. optimize_session + run_optimize（追加到 run_refine 结束后）
anchor2 = '''    log(
        f"[refine] {session_id[:8]} -> 第 {ann['review_rounds'][-1]['round']} 轮 {rnd['verdict']} "
        f"(conf {rnd['confidence']:.2f}，重写 {len(rnd['changed_msg_ids'])} 步)"
    )
    return {"round": ann["review_rounds"][-1]['round'], "verdict": rnd["verdict"], "changed": rnd["changed_msg_ids"]}'''
assert src.count(anchor2) == 1
funcs = anchor2 + '''


def optimize_session(
    entry: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    log=print,
) -> Dict[str, Any]:
    """全面优化：优化器模型全量改写 -> 独立审核模型逐条审计 -> 不合格退回修订，循环至过审。"""
    cfg = cfg or load_config()
    rcfg = cfg.get("review", {})
    ocfg = rcfg.get("optimize", {}) or {}
    opt_model = ocfg.get("optimizer_model") or cfg["llm"]["model"]
    rev_model = ocfg.get("reviewer_model") or cfg["llm"]["model"]
    max_loops = max(1, int(ocfg.get("max_loops", 3)))
    batch_size = max(1, int(ocfg.get("batch_size", 6)))
    rows = read_jsonl(Path(entry["gold_file"]))
    if not rows:
        raise RuntimeError(f"黄金文件为空: {entry.get('gold_file')}")
    rec = rows[0]
    asst = [m for m in rec["messages"] if m["role"] == "assistant"]
    by_id = {m["id"]: m for m in asst}
    root_prompt = rec.get("stats", {}).get("root_prompt", "")

    def _orig(mid: str) -> str:
        return _orig_thinking(by_id[mid]) if mid in by_id else ""

    def _collect(out: Any) -> Dict[str, str]:
        got: Dict[str, str] = {}
        for rw in (out or {}).get("rewrites", []):
            if isinstance(rw, dict) and rw.get("msg_id") in by_id:
                t = (rw.get("rewritten") or "").strip()
                if t:
                    got[rw["msg_id"]] = t
        return got

    log(f"[optimize] {entry['session_id'][:8]} 全量优化 {len(asst)} 步"
        f"（优化器 {opt_model} / 独立审核 {rev_model}，至多 {max_loops} 轮）")
    current: Dict[str, str] = {}
    for bi in range(0, len(asst), batch_size):
        batch = asst[bi: bi + batch_size]
        body = {"root_prompt": root_prompt, "steps": [
            {"msg_id": m["id"], "thinking": _orig_thinking(m)[:12000], "text": (m.get("text") or "")[:300],
             "tools": [tc.get("name") for tc in (m.get("tool_calls") or [])][:6], "flags": m.get("flags", [])}
            for m in batch]}
        out = chat_json(OPTIMIZE_SYSTEM, OPTIMIZE_SCHEMA + "\n\n待优化：\n" + json.dumps(body, ensure_ascii=False),
                        model=opt_model, temperature=float(rcfg.get("temperature", 0.0)),
                        prompt_version=PROMPT_VERSION + ".optimize")
        current.update(_collect(out))
        log(f"[optimize] 优化进度 {min(bi + batch_size, len(asst))}/{len(asst)}")
    current = {mid: t for mid, t in current.items() if mid in by_id and t != _orig(mid)}

    loops = []
    passed = False
    overall = ""
    for loop in range(1, max_loops + 1):
        issues = []
        passed = True
        for bi in range(0, len(asst), batch_size * 2):
            batch = [m for m in asst[bi: bi + batch_size * 2] if m["id"] in current]
            if not batch:
                continue
            body = {"root_prompt": root_prompt, "steps": [
                {"msg_id": m["id"], "original": _orig_thinking(m)[:8000], "optimized": current[m["id"]][:8000]}
                for m in batch]}
            out = chat_json(AUDIT_SYSTEM, AUDIT_SCHEMA + "\n\n待审计：\n" + json.dumps(body, ensure_ascii=False),
                            model=rev_model, temperature=0.0, prompt_version=PROMPT_VERSION + ".audit")
            overall = (out or {}).get("overall") or overall
            if not (out or {}).get("passed"):
                passed = False
            for i in (out or {}).get("issues", []):
                if isinstance(i, dict) and i.get("msg_id") in current:
                    issues.append(i)
        loops.append({"loop": loop, "phase": "audit", "passed": passed, "issues": len(issues)})
        if passed:
            break
        problem_ids = list(dict.fromkeys(i["msg_id"] for i in issues))
        instr = {i["msg_id"]: i for i in issues}
        log(f"[optimize] 第 {loop} 轮审核未过（{len(problem_ids)} 步），退回优化器修订")
        loops.append({"loop": loop, "phase": "revise", "steps": len(problem_ids)})
        if loop >= max_loops:
            break
        for bi in range(0, len(problem_ids), batch_size):
            pb = [mid for mid in problem_ids[bi: bi + batch_size] if mid in by_id]
            if not pb:
                continue
            body = {"root_prompt": root_prompt, "steps": [
                {"msg_id": mid, "original_thinking": _orig(mid)[:10000], "previous_rewrite": current.get(mid, ""),
                 "audit_problem": (instr.get(mid) or {}).get("problem", ""),
                 "audit_instruction": (instr.get(mid) or {}).get("instruction", "")} for mid in pb]}
            out = chat_json(OPTIMIZE_REVISE_SYSTEM, OPTIMIZE_SCHEMA + "\n\n待修订：\n" + json.dumps(body, ensure_ascii=False),
                            model=opt_model, temperature=0.0, prompt_version=PROMPT_VERSION + ".optimize_revise")
            current.update(_collect(out))

    changed = {mid: t for mid, t in current.items() if mid in by_id and t != _orig(mid)}
    verdict = "gold" if passed else "needs_review"
    n_audit = len([l for l in loops if l["phase"] == "audit"])
    log(f"[optimize] {entry['session_id'][:8]} -> {verdict}，改动 {len(changed)}/{len(asst)} 步，审核 {n_audit} 轮")
    return {
        "kind": "optimize",
        "verdict": verdict,
        "confidence": 0.92 if passed else 0.55,
        "reasons": [
            f"全面优化 {len(changed)}/{len(asst)} 步",
            f"独立审核{'通过' if passed else '未完全通过'}（共 {n_audit} 轮）",
            overall,
        ],
        "bad_step_ids": [],
        "thinking_rewrites": [{"msg_id": mid, "rewritten": t} for mid, t in changed.items()],
        "optimizer_model": opt_model,
        "reviewer_model": rev_model,
        "loops": loops,
        "audit_overall": overall,
        "reviewed_steps": len(asst),
        "total_steps": len(asst),
        "model": opt_model,
        "prompt_version": PROMPT_VERSION,
    }


def run_optimize(session_id: str, log=print, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    index = load_index(stage_dirs(cfg)["gold"] / "index.json")
    entry = index["sessions"].get(session_id)
    if not entry:
        raise RuntimeError(f"会话不存在: {session_id}")
    ann_path = Path(entry["annotation_file"])
    rnd = optimize_session(entry, cfg=cfg, log=log)
    ann = read_json(ann_path, default={}) or {}
    curation.ensure_rounds(ann)
    hr = curation.human(ann)
    reset_ids = [mid for mid, it in hr["items"].items() if it.get("rewrite")]
    curation.append_round(ann, rnd, reset_ids=reset_ids)
    write_json_atomic(ann_path, ann)
    return {
        "round": ann["review_rounds"][-1]["round"],
        "verdict": rnd["verdict"],
        "changed": len(rnd["thinking_rewrites"]),
        "loops": rnd["loops"],
        "optimizer_model": rnd["optimizer_model"],
        "reviewer_model": rnd["reviewer_model"],
    }'''
src = src.replace(anchor2, funcs)

open('pipeline/review.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('review.py 修改完成，语法 OK')
