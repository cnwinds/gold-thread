"""GoldThread Agent-First 统一命令行入口（gt）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pipeline.common.config import load_config

from . import (
    assist_tools,
    build_tools,
    curate_tools,
    eval_tools,
    inspect_tools,
    ledger,
    lineage_graph,
    report_tools,
)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and getattr(stream, "buffer", None) is not None:
            enc = (getattr(stream, "encoding", "") or "").lower()
            if enc != "utf-8":
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _format_inspect_source(res: Dict[str, Any]) -> None:
    print(
        f"[source] {res['source_dir']} | 文件 {res['total_files']} | "
        f"已解析 {res['parsed_sessions']} (已入库 {res['adopted_count']} / 未入库 {res['unadopted_count']})"
    )
    if res["new_projects"]:
        print(f"  未覆盖新项目 ({len(res['new_projects'])}): {', '.join(res['new_projects'])}")
    for proj, info in res["by_project"].items():
        tag = "已有" if info["in_corpus"] else "新项目"
        print(
            f"  - {proj:<24} [{tag}] total={info['total']} "
            f"adopted={info['adopted']} unadopted={info['unadopted']}"
        )
    for s in res["sessions"][:30]:
        mark = "已入库" if s["adopted"] else "未入库"
        print(
            f"    {s['session_id'][:8]} [{mark}] {s['project_group']:<22} "
            f"{s['label']:<8} {s['dominant_model']:<22} steps={s['assistant_steps']}"
        )


def _format_inspect_corpus(res: Dict[str, Any]) -> None:
    tc = res["train_counts"]
    print(
        f"[corpus] raw={res['raw_sessions']} gold={res['gold_sessions']} "
        f"projects={res['project_count']} (规范 {res['canonical_projects']}) "
        f"latest_build={res['latest_build_id'] or '-'}"
    )
    print(
        f"  train_projects({len(res['split_projects']['train'])}): "
        f"{', '.join(res['split_projects']['train']) or '-'}"
    )
    print(
        f"  eval_projects({len(res['split_projects']['eval'])}): "
        f"{', '.join(res['split_projects']['eval']) or '-'}"
    )
    print(
        f"  SFT={tc['sft_windows']} (train={tc['sft_windows_train']}, eval={tc['sft_windows_eval']}, "
        f"traj={tc['sft_trajectories']}) | DPO={tc['dpo_total']} "
        f"(train={tc['dpo_train']}, eval={tc['dpo_eval']})"
    )
    for proj, pb in res["by_project"].items():
        print(
            f"  - {proj:<24} [{pb['split']:<5}] sessions={pb['sessions']} "
            f"gold_steps={pb['gold_steps']} sft={pb['sft_windows']} dpo={pb['dpo_written']}"
        )


def _format_inspect_session(res: Dict[str, Any]) -> None:
    eff = res.get("effective") or {}
    mrep = res.get("model_report") or {}
    print(
        f"[session] {res['session_id']} | {res['project_group']} | "
        f"label={res['label']} status={res['status']} effective={eff.get('state')} "
        f"verdict={mrep.get('verdict') or '-'}"
    )
    print(f"  root_prompt: {res.get('root_prompt', '')[:160]}")
    print(f"  turns={len(res.get('turn_marks') or [])} | model={res.get('dominant_model')}")
    if mrep.get("reason"):
        print(f"  model_report: {mrep['reason']}")
    for st in res.get("steps") or []:
        tools = ",".join(st["tool_calls"]) or "STOP"
        flags = ",".join(st["flags"]) or "-"
        ex = " [EXCLUDED]" if st["excluded"] else ""
        print(
            f"    #{st['step_index']:<3} {st['msg_id']} T{st['turn_index']} "
            f"think={st['think_chars']}/{st['raw_think_chars']} tools={tools} "
            f"flags={flags}{ex}"
        )


def _format_inspect_step(res: Dict[str, Any]) -> None:
    print(
        f"[step] {res['session_id'][:8]} #{res['step_index']}/{res['total_steps']} "
        f"msg={res['msg_id']} turn={res['turn_index']} excluded={res['excluded']}"
    )
    print(f"  flags={res['flags']} judge_reason={res['judge_reason'] or '-'}")
    print(f"  raw_thinking({len(res['raw_thinking'])}): {res['raw_thinking'][:200]!r}")
    print(f"  thinking({len(res['thinking'])}): {res['thinking'][:200]!r}")
    print(f"  tool_calls({len(res['tool_calls'])}): {[t['name'] for t in res['tool_calls']]}")
    if res["dpo_pairs"]:
        print(f"  dpo_pairs({len(res['dpo_pairs'])}): {[p['sample_id'] for p in res['dpo_pairs']]}")


def _format_eval_pack(res: Dict[str, Any]) -> None:
    status = "HEALTHY" if res["healthy"] else "UNHEALTHY"
    print(
        f"[eval.pack] {res['resolved_path']} -> {status} "
        f"(style={res['expected_style']}, build={res.get('build_id') or '-'})"
    )
    print(
        f"  SFT: train={res['sft_train_rows']} (unique={res['sft_train_unique']}), "
        f"eval={res['sft_eval_rows']} (unique={res['sft_eval_unique']})"
    )
    print(
        f"  DPO: train={res['dpo_train_rows']}, eval={res['dpo_eval_rows']} | "
        f"sources={res['dpo_by_source']}"
    )
    print(
        f"  Projects={res['project_count']} (non_canonical={res['non_canonical_project_count']}) | "
        f"shape={res['thinking_shape']} ({res['hist_think_ratio']:.1%}) | "
        f"empty_target_think={res['empty_target_thinking']}"
    )
    print(
        f"  stop_targets={res['stop_targets']} (premature={res['premature_stop_targets']}) | "
        f"tag_leaks={res['prompt_tag_leaks']} ({res['prompt_tag_leak_rate']:.2%}) | "
        f"parallel(train={res['sft_train_parallel_rate']:.1%}, eval={res['sft_eval_parallel_rate']:.1%})"
    )
    print(
        f"  toolcall: ok={res['ok_blocks']} wrong_style={res['wrong_style']} "
        f"stray_xml={res['stray_xml']} bad={res['bad_blocks']}"
    )
    for v in res["violations"]:
        print(f"  ! [{v['severity']}] {v['code']}: {v['message']}")


def _format_eval_diff(res: Dict[str, Any]) -> None:
    print(res["markdown_table"])


def _format_eval_student(res: Dict[str, Any]) -> None:
    s = res["summary"]
    print(
        f"[eval.student] {res['student_dir']} | runs={s['total_runs']} "
        f"steps={s['total_steps']} tools={s['total_tool_calls']} "
        f"parallel={s['parallel_call_rate']:.1%} empty_think={s['empty_think_steps']} "
        f"turn0_stop={s['turn0_premature_stops']} overflow={s['max_context_overflows']} "
        f"emu={s['offline_emu_runs']}"
    )
    for r in res["runs"]:
        tb = r.get("teacher_benchmark")
        t_info = f" teacher_steps={tb['assistant_steps']} ({tb['step_ratio']}x)" if tb else ""
        print(
            f"  - {r['run_name']:<26} {r['session_id'][:8]} {r['project_group']:<20} "
            f"turns={r['turns']} steps={r['assistant_steps']} par={r['parallel_call_rate']:.1%} "
            f"t0_stop={r['turn0_premature_stop']} emu={r['offline_emu_used']}{t_info}"
        )


def _format_report_list(reports: List[Dict[str, Any]]) -> None:
    print(f"[reports] 共 {len(reports)} 份评估报告")
    for r in reports:
        print(
            f"  - {r['report_id']} | {r['created_at']} | {r['title']} "
            f"(build={r.get('build_id') or '-'}, diff={r.get('has_diff')})"
        )


def _format_ledger_list(events: List[Dict[str, Any]]) -> None:
    print(f"[ledger] 共 {len(events)} 条操作记录")
    for ev in events:
        print(
            f"  - {ev['op_id']} | {ev['timestamp']} | {ev['actor']} | "
            f"{ev['tool']} [{ev['status']}] {ev['reason']}"
        )


def _format_gate_check(res: Dict[str, Any]) -> None:
    status = "PASSED" if res["passed"] else "FAILED"
    print(
        f"[gate.check] {res['target']} -> {status} "
        f"(kind={res['resolved_kind']}, style={res['expected_style']}, "
        f"build={res.get('build_id') or '-'}, pack={res.get('package_hash') or '-'})"
    )
    for chk in res.get("checks") or []:
        mark = "OK" if chk["passed"] else "FAIL"
        print(f"  - [{mark}] {chk['name']}: {chk['detail']}")
    for v in res.get("violations") or []:
        print(f"  ! [{v.get('severity', 'error')}] {v.get('code')}: {v.get('message')}")


def _format_build_res(res: Dict[str, Any]) -> None:
    if res.get("dry_run"):
        gp = res.get("gate_preview") or {}
        print(
            f"[build.dry-run] plan_hash={res.get('plan_hash')} dirty={res.get('dirty')} | "
            f"SFT(train={gp.get('sft_train_rows', 0)}, eval={gp.get('sft_eval_rows', 0)}) | "
            f"DPO(train={gp.get('dpo_train_rows', 0)}, eval={gp.get('dpo_eval_rows', 0)}) | "
            f"pending_dpo={gp.get('pending_evaluations', 0)}"
        )
        return
    s3 = res.get("stage3") or {}
    print(
        f"[build] build_id={res.get('build_id')} ok={res.get('ok')} op_id={res.get('op_id')} | "
        f"SFT={s3.get('sft_windows', 0)} (train={s3.get('sft_train', 0)}, eval={s3.get('sft_eval', 0)}) | "
        f"DPO={s3.get('dpo_total', 0)} (train={s3.get('dpo_train', 0)}, eval={s3.get('dpo_eval', 0)})"
    )
    if res.get("gate"):
        _format_gate_check(res["gate"])


def _format_pack_res(res: Dict[str, Any]) -> None:
    print(
        f"[pack] {res.get('zip_name')} hash={res.get('package_hash')} "
        f"build={res.get('build_id') or '-'} ok={res.get('ok')} op_id={res.get('op_id')}"
    )
    if res.get("copied_to"):
        print(f"  copied_to: {res['copied_to']}")
    if res.get("gate"):
        _format_gate_check(res["gate"])


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="指定 pipeline.yaml 路径")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="以 JSON 格式输出")

    parser = argparse.ArgumentParser(
        prog="gt",
        description="GoldThread Agent-First CLI",
    )
    parser.add_argument("--config", default=None, help="指定 pipeline.yaml 路径")
    parser.add_argument("--json", action="store_true", default=False, help="以 JSON 格式输出")
    sub = parser.add_subparsers(dest="group", required=True)

    # inspect
    p_ins = sub.add_parser("inspect", help="只读检查器")
    ins_sub = p_ins.add_subparsers(dest="action", required=True)

    p_ins_src = ins_sub.add_parser("source", parents=[common], help="检查源会话目录")
    p_ins_src.add_argument("--dir", required=True, help="源会话目录")
    p_ins_src.add_argument("--unadopted", action="store_true", help="仅列出未入库会话")

    p_ins_cor = ins_sub.add_parser("corpus", parents=[common], help="检查当前语料库全景")
    p_ins_cor.add_argument("--project", default=None, help="按项目名或规范项目组过滤")

    p_ins_ses = ins_sub.add_parser("session", parents=[common], help="检查单条会话详情")
    p_ins_ses.add_argument("sid", help="会话 ID 或唯一前缀")
    p_ins_ses.add_argument("--steps", action="store_true", help="附带紧凑步骤摘要列表")

    p_ins_stp = ins_sub.add_parser("step", parents=[common], help="检查单步三态血缘")
    p_ins_stp.add_argument("sid", help="会话 ID 或唯一前缀")
    p_ins_stp.add_argument("step_ref", help="msg_id（支持前缀）或 0-based 步骤序号")

    ins_sub.add_parser("graph", parents=[common], help="查看虚拟数据集演进关系图（DAG）与工作空间状态")

    # eval
    p_ev = sub.add_parser("eval", help="训练包体检与实测评估")
    ev_sub = p_ev.add_subparsers(dest="action", required=True)

    p_ev_pack = ev_sub.add_parser("pack", parents=[common], help="全维度体检训练包或 Build")
    p_ev_pack.add_argument("target", help="zip 路径 / 8位包哈希 / build_id / latest / 目录")
    p_ev_pack.add_argument("--style", choices=("coder-xml", "hermes-json"), default=None)

    p_ev_diff = ev_sub.add_parser("diff", parents=[common], help="对比两个训练包/Build")
    p_ev_diff.add_argument("before", help="旧包/Build")
    p_ev_diff.add_argument("after", help="新包/Build")

    p_ev_stu = ev_sub.add_parser("student", parents=[common], help="分析学生实测目录与会话日志")
    p_ev_stu.add_argument("--student-dir", required=True, help="学生实测工作目录")
    p_ev_stu.add_argument("--sessions-dir", default=None, help="pi 会话日志根目录")
    p_ev_stu.add_argument("--teacher-dir", default=None, help="教师基准目录")

    # report
    p_rep = sub.add_parser("report", help="评估报告归档与查询")
    rep_sub = p_rep.add_subparsers(dest="action", required=True)

    p_rep_pub = rep_sub.add_parser("publish", parents=[common], help="发布评估报告")
    p_rep_pub.add_argument("--title", required=True, help="报告标题")
    src_grp = p_rep_pub.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--md", help="Markdown 文件路径")
    src_grp.add_argument("--text", help="Markdown 正文文本")
    p_rep_pub.add_argument("--id", dest="report_id", default=None, help="指定 report_id")
    p_rep_pub.add_argument("--before", dest="before_pack", default=None, help="对比前包/Build")
    p_rep_pub.add_argument("--after", dest="after_pack", default=None, help="对比后包/Build")
    p_rep_pub.add_argument("--build", dest="build_id", default=None, help="关联 Build ID")
    p_rep_pub.add_argument("--summary", default="", help="一句话摘要")
    p_rep_pub.add_argument("--reason", default="", help="操作原因说明")
    p_rep_pub.add_argument("--actor", default="agent", help="操作者")

    rep_sub.add_parser("list", parents=[common], help="列出已归档评估报告")

    p_rep_show = rep_sub.add_parser("show", parents=[common], help="查看指定评估报告")
    p_rep_show.add_argument("report_id", help="报告 ID")

    # ledger
    p_led = sub.add_parser("ledger", help="统一操作账本查询")
    led_sub = p_led.add_subparsers(dest="action", required=True)
    p_led_list = led_sub.add_parser("list", parents=[common], help="查询操作账本")
    p_led_list.add_argument("--limit", type=int, default=100)
    p_led_list.add_argument("--sid", default=None)
    p_led_list.add_argument("--project", default=None)
    p_led_list.add_argument("--tool", default=None)
    p_led_list.add_argument("--build", dest="build_id", default=None)
    p_led_list.add_argument("--actor", default=None)

    # curate
    p_cur = sub.add_parser("curate", help="纯确定性语料与策展写操作")
    cur_sub = p_cur.add_subparsers(dest="action", required=True)

    p_cur_adopt = cur_sub.add_parser("adopt", parents=[common], help="批量采纳源会话")
    p_cur_adopt.add_argument("--from", dest="from_dir", required=True, help="源会话目录")
    p_cur_adopt.add_argument("--select", action="append", default=[], help="指定 sid / 前缀 / 文件名（逗号分隔或多次传入）")
    p_cur_adopt.add_argument("--all-unadopted", action="store_true", help="采纳全部未入库会话")
    p_cur_adopt.add_argument("--origin", choices=("pi", "cursor"), default="pi")
    p_cur_adopt.add_argument("--reason", default="", help="操作原因说明")
    p_cur_adopt.add_argument("--actor", default="agent", help="操作者")

    p_cur_drop = cur_sub.add_parser("drop-session", parents=[common], help="安全级联删除指定会话")
    p_cur_drop.add_argument("sid", help="会话 ID 或唯一前缀")
    p_cur_drop.add_argument("--reason", default="", help="操作原因说明")
    p_cur_drop.add_argument("--actor", default="agent", help="操作者")

    p_cur_turns = cur_sub.add_parser("turns", parents=[common], help="更新会话回合覆盖")
    p_cur_turns.add_argument("sid", help="会话 ID 或唯一前缀")
    turns_grp = p_cur_turns.add_mutually_exclusive_group(required=True)
    turns_grp.add_argument("--patch", help="JSON 字符串或 @file.json")
    turns_grp.add_argument("--file", dest="patch_file", help="回合补丁 JSON 文件路径")
    turns_grp.add_argument("--reset", action="store_true", help="清空该会话回合覆盖")
    p_cur_turns.add_argument("--reason", default="", help="操作原因说明")
    p_cur_turns.add_argument("--actor", default="agent", help="操作者")

    p_cur_steps = cur_sub.add_parser("steps", parents=[common], help="写入步级结构化决策并重算黄金层")
    p_cur_steps.add_argument("--file", dest="decisions_file", required=True, help="决策 JSON 文件路径")
    p_cur_steps.add_argument("--no-apply", action="store_true", help="仅写入 step_decisions.json，不立即重算 stage2")
    p_cur_steps.add_argument("--reason", default="", help="操作原因说明")
    p_cur_steps.add_argument("--actor", default="agent", help="操作者")

    p_cur_dpo = cur_sub.add_parser("dpo", parents=[common], help="写入 DPO 偏好对裁决")
    p_cur_dpo.add_argument("--verdicts", required=True, help="裁决 JSON 文件、@file.json 或内联 JSON")
    p_cur_dpo.add_argument("--reason", default="", help="操作原因说明")
    p_cur_dpo.add_argument("--actor", default="agent", help="操作者")

    p_cur_split = cur_sub.add_parser("split", parents=[common], help="管理项目/会话切分计划")
    p_cur_split.add_argument("--pin", action="append", default=[], help="锁定项目切分 game::xxx=eval|train|auto")
    p_cur_split.add_argument("--session-pin", action="append", default=[], help="锁定会话切分 sid=eval|train|inherit")
    p_cur_split.add_argument("--eval-ratio", type=float, default=None, help="eval 占比 (0~1)")
    p_cur_split.add_argument("--seed", type=int, default=None, help="切分随机种子")
    p_cur_split.add_argument("--replan", action="store_true", help="按当前 ratio/seed 重新切分未锁定项目")
    p_cur_split.add_argument("--reason", default="", help="操作原因说明")
    p_cur_split.add_argument("--actor", default="agent", help="操作者")

    p_cur_excl = cur_sub.add_parser("exclude", parents=[common], help="停用或恢复样本/作用域")
    p_cur_excl.add_argument("--sample", action="append", default=[], help="sample_id（可多次传入或逗号分隔）")
    p_cur_excl.add_argument("--scope", action="append", default=[], help="project:xxx / session:sid / session_sft:sid / session_dpo:sid")
    p_cur_excl.add_argument("--restore", action="store_true", help="恢复为 auto（取消排除）")
    p_cur_excl.add_argument("--reason", default="", help="操作原因说明")
    p_cur_excl.add_argument("--actor", default="agent", help="操作者")

    # build
    p_build = sub.add_parser("build", parents=[common], help="纯确定性构建训练集与不可变 Build")
    p_build.add_argument("--force", action="store_true", help="忽略增量缓存强制重建")
    p_build.add_argument("--scan-raw", action="store_true", help="构建前先扫描 raw 源目录 (stage1)")
    p_build.add_argument("--dry-run", action="store_true", help="仅预览 Stage 3 规模与门禁预期，不落盘")
    p_build.add_argument("--no-gate", action="store_true", help="构建后跳过 gate check")
    p_build.add_argument("--require-reviewed-dpo", action="store_true", help="严格要求全部 DPO 候选均已显式裁决（拒绝未审 bypass）")
    p_build.add_argument("--reason", default="", help="操作原因说明")
    p_build.add_argument("--actor", default="agent", help="操作者")

    # gate
    p_gate = sub.add_parser("gate", help="硬性质量门禁")
    gate_sub = p_gate.add_subparsers(dest="action", required=True)
    p_gate_chk = gate_sub.add_parser("check", parents=[common], help="校验 Build 或交付包硬性门禁")
    p_gate_chk.add_argument("target", nargs="?", default="latest", help="latest / build_id / pack_hash / zip 路径")
    p_gate_chk.add_argument("--style", choices=("coder-xml", "hermes-json"), default=None)

    # pack
    p_pack = sub.add_parser("pack", parents=[common], help="从不可变 Build 打包交付 zip")
    p_pack.add_argument("--build", dest="build_id", default=None, help="指定 Build ID（默认 latest）")
    p_pack.add_argument("--profile", default=None, help="导出格式 profile（如 qwen-xml / hermes-json）")
    p_pack.add_argument("--parent", dest="parent_pack", default=None, help="显式指定父交付包哈希（默认自动继承最新包）")
    p_pack.add_argument("--copy-to", default=None, help="同步拷贝一份到指定目录或文件")
    p_pack.add_argument("--no-gate", action="store_true", help="打包后跳过 gate check")
    p_pack.add_argument("--reason", default="", help="操作原因说明")
    p_pack.add_argument("--actor", default="agent", help="操作者")

    # assist
    p_ast = sub.add_parser("assist", help="显式可选的小模型初筛子命令")
    ast_sub = p_ast.add_subparsers(dest="action", required=True)

    p_ast_turns = ast_sub.add_parser("turns", parents=[common], help="小模型回合分类初筛")
    p_ast_turns.add_argument("sid", help="会话 ID 或唯一前缀")
    p_ast_turns.add_argument("--apply", action="store_true", help="写入 meta.json 的 turn_annotations")
    p_ast_turns.add_argument("--reason", default="", help="操作原因说明")
    p_ast_turns.add_argument("--actor", default="agent", help="操作者")

    p_ast_judge = ast_sub.add_parser("judge", parents=[common], help="小模型步级判断与质量审计初筛")
    p_ast_judge.add_argument("sid", help="会话 ID 或唯一前缀")
    p_ast_judge.add_argument("--apply", action="store_true", help="写入 step_decisions.json 并重算黄金层")
    p_ast_judge.add_argument("--reason", default="", help="操作原因说明")
    p_ast_judge.add_argument("--actor", default="agent", help="操作者")

    p_ast_dpo = ast_sub.add_parser("dpo", parents=[common], help="小模型对未审 DPO 跑初筛缓存")
    p_ast_dpo.add_argument("--reason", default="", help="操作原因说明")
    p_ast_dpo.add_argument("--actor", default="agent", help="操作者")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8_stdio()
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    json_mode = "--json" in raw_argv

    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    json_mode = bool(getattr(args, "json", False))

    try:
        cfg = load_config(args.config) if getattr(args, "config", None) else load_config()

        if args.group == "inspect":
            if args.action == "source":
                res = inspect_tools.inspect_source(
                    args.dir, cfg=cfg, unadopted_only=bool(args.unadopted)
                )
                _print_json(res) if json_mode else _format_inspect_source(res)
            elif args.action == "corpus":
                res = inspect_tools.inspect_corpus(cfg=cfg, project=args.project)
                _print_json(res) if json_mode else _format_inspect_corpus(res)
            elif args.action == "session":
                res = inspect_tools.inspect_session(
                    args.sid, cfg=cfg, include_steps=bool(args.steps)
                )
                _print_json(res) if json_mode else _format_inspect_session(res)
            elif args.action == "step":
                res = inspect_tools.inspect_step(args.sid, args.step_ref, cfg=cfg)
                _print_json(res) if json_mode else _format_inspect_step(res)
            elif args.action == "graph":
                res = lineage_graph.build_dataset_graph(cfg=cfg)
                if json_mode:
                    _print_json(res)
                else:
                    ws = res["workspace"]
                    print(
                        f"[inspect.graph] packs={len(res['nodes'])} edges={len(res['edges'])} "
                        f"head={res['head_pack'] or '-'} workspace_dirty={ws['dirty']} "
                        f"cas_objects={res['cas_total_objects']}"
                    )
                    for n in res["nodes"]:
                        print(
                            f"  pack {n['id']} (parent={n.get('resolved_parent') or '-'}) "
                            f"projects={n['projects_count']} sft={n['sft_train']}/{n['sft_eval']} "
                            f"dpo={n['dpo_train']}/{n['dpo_eval']}"
                        )

        elif args.group == "eval":
            if args.action == "pack":
                res = eval_tools.eval_pack(
                    args.target, cfg=cfg, expected_style=args.style
                )
                _print_json(res) if json_mode else _format_eval_pack(res)
            elif args.action == "diff":
                res = eval_tools.eval_diff(args.before, args.after, cfg=cfg)
                _print_json(res) if json_mode else _format_eval_diff(res)
            elif args.action == "student":
                res = eval_tools.eval_student(
                    args.student_dir,
                    sessions_dir=args.sessions_dir,
                    teacher_dir=args.teacher_dir,
                    cfg=cfg,
                )
                _print_json(res) if json_mode else _format_eval_student(res)

        elif args.group == "report":
            if args.action == "publish":
                md_content = (
                    Path(args.md).read_text(encoding="utf-8")
                    if args.md
                    else str(args.text or "")
                )
                res = report_tools.publish_report(
                    title=args.title,
                    md_content=md_content,
                    report_id=args.report_id,
                    before_pack=args.before_pack,
                    after_pack=args.after_pack,
                    build_id=args.build_id,
                    summary=args.summary,
                    actor=args.actor,
                    reason=args.reason,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[report.publish] {res['report_id']} 已归档 -> {res['md_file']} "
                        f"(op_id={res['op_id']})"
                    )
            elif args.action == "list":
                rows = report_tools.list_reports(cfg=cfg)
                _print_json({"reports": rows}) if json_mode else _format_report_list(rows)
            elif args.action == "show":
                res = report_tools.get_report(args.report_id, cfg=cfg)
                if json_mode:
                    _print_json(res)
                else:
                    print(f"# {res.get('title')}\n(report_id={res.get('report_id')}, created_at={res.get('created_at')})\n")
                    print(res.get("markdown", ""))

        elif args.group == "ledger":
            if args.action == "list":
                events = ledger.read_ledger(
                    cfg=cfg,
                    limit=args.limit,
                    sid=args.sid,
                    project=args.project,
                    tool=args.tool,
                    build_id=args.build_id,
                    actor=args.actor,
                )
                _print_json({"events": events, "count": len(events)}) if json_mode else _format_ledger_list(events)

        elif args.group == "curate":
            if args.action == "adopt":
                res = curate_tools.adopt_sessions(
                    args.from_dir,
                    select=args.select,
                    all_unadopted=bool(args.all_unadopted),
                    origin=args.origin,
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[curate.adopt] 已采纳 {res['adopted_count']} 条会话 "
                        f"(projects={', '.join(res['projects']) or '-'}, op_id={res['op_id']})"
                    )
            elif args.action == "drop-session":
                res = curate_tools.drop_session(
                    args.sid,
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[curate.drop_session] 已删除 {res['session_id']} "
                        f"(deleted_files={res['deleted_files']}, removed_rows={res['removed_rows']}, op_id={res['op_id']})"
                    )
            elif args.action == "turns":
                patch_src = Path(args.patch_file) if args.patch_file else args.patch
                res = curate_tools.curate_turns(
                    args.sid,
                    patch=patch_src,
                    reset=bool(args.reset),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(f"[curate.turns] 已更新 {res['session_id']} 回合覆盖 (op_id={res['op_id']})")
            elif args.action == "steps":
                res = curate_tools.curate_steps(
                    Path(args.decisions_file),
                    apply=not bool(args.no_apply),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[curate.steps] 已保存 {len(res['sessions'])} 条会话步级决策 "
                        f"(applied={res['applied']}, op_id={res['op_id']})"
                    )
            elif args.action == "dpo":
                res = curate_tools.curate_dpo(
                    args.verdicts,
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(f"[curate.dpo] 已写入 {res['updated']} 条 DPO 裁决 (total={res['total']}, op_id={res['op_id']})")
            elif args.action == "split":
                res = curate_tools.curate_split(
                    pin=args.pin,
                    session_pin=args.session_pin,
                    eval_ratio=args.eval_ratio,
                    seed=args.seed,
                    replan=bool(args.replan),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(f"[curate.split] 切分计划已更新 (changes={len(res['changes'])}, op_id={res['op_id']})")
            elif args.action == "exclude":
                res = curate_tools.curate_exclude(
                    samples=args.sample,
                    scopes=args.scope,
                    restore=bool(args.restore),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[curate.exclude] action={res['action']} "
                        f"samples_changed={res['samples_changed']} scopes_changed={res['scopes_changed']} "
                        f"(op_id={res['op_id']})"
                    )

        elif args.group == "build":
            res = build_tools.build_pipeline(
                force=bool(args.force),
                scan_raw=bool(args.scan_raw),
                dry_run=bool(args.dry_run),
                run_gate=not bool(args.no_gate),
                require_reviewed_dpo=bool(getattr(args, "require_reviewed_dpo", False)),
                reason=args.reason,
                actor=args.actor,
                cfg=cfg,
            )
            _print_json(res) if json_mode else _format_build_res(res)
            if not res.get("ok", True):
                return 2

        elif args.group == "gate":
            if args.action == "check":
                res = build_tools.gate_check(args.target, style=args.style, cfg=cfg)
                _print_json(res) if json_mode else _format_gate_check(res)
                return 0 if res["passed"] else 2

        elif args.group == "pack":
            res = build_tools.pack_delivery(
                build_id=args.build_id,
                profile=args.profile,
                parent_pack=args.parent_pack,
                copy_to=args.copy_to,
                run_gate=not bool(args.no_gate),
                reason=args.reason,
                actor=args.actor,
                cfg=cfg,
            )
            _print_json(res) if json_mode else _format_pack_res(res)
            if not res.get("ok", True):
                return 2

        elif args.group == "assist":
            if args.action == "turns":
                res = assist_tools.assist_turns(
                    args.sid,
                    apply=bool(args.apply),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(f"[assist.turns] {res['session_id']} applied={res['applied']} op_id={res.get('op_id') or '-'}")
            elif args.action == "judge":
                res = assist_tools.assist_judge(
                    args.sid,
                    apply=bool(args.apply),
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[assist.judge] {res['session_id']} decisions={len(res.get('decisions') or {})} "
                        f"applied={res['applied']} op_id={res.get('op_id') or '-'}"
                    )
            elif args.action == "dpo":
                res = assist_tools.assist_dpo(
                    reason=args.reason,
                    actor=args.actor,
                    cfg=cfg,
                )
                if json_mode:
                    _print_json(res)
                else:
                    print(
                        f"[assist.dpo] candidates={res.get('candidates', 0)} evaluated={res.get('evaluated', 0)} "
                        f"good={res.get('good', 0)} swap={res.get('swap', 0)} drop={res.get('drop', 0)} "
                        f"op_id={res.get('op_id')}"
                    )

        return 0
    except Exception as exc:  # noqa: BLE001
        if json_mode:
            _print_json({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        else:
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
