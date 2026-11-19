"""GoldThread Agent-First Phase 3 自动化测试：留痕看板 API、三态血缘、步级应急开关与前端可观测视图。"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from fastapi.testclient import TestClient

import server.app as app_mod
from goldthread import build_tools, curate_tools, ledger, report_tools
from pipeline.common.config import DEFAULTS
from pipeline.common.store import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _make_sandbox_cfg(tmp_path: Path) -> tuple[Dict[str, Any], Path]:
    cfg = copy.deepcopy(DEFAULTS)
    data_dir = tmp_path / "data"
    sessions_dir = tmp_path / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["sessions_dir"] = str(sessions_dir)
    cfg["gold"]["min_gold_steps"] = 2
    cfg["synth"]["eval_ratio"] = 0.5
    cfg["synth"]["shuffle"] = False
    cfg["export_profiles"] = [
        {
            "id": "qwen-xml",
            "label": "Qwen3.8 Coder XML",
            "default": True,
            "tool_call_style": "coder-xml",
            "template": "qwen3_8",
            "tool_format": "qwen3_coder",
        }
    ]
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    cfg["_config_path"] = str(cfg_path)
    return cfg, cfg_path


def _write_pi_session(
    path: Path,
    *,
    session_id: str,
    cwd: str,
    model: str = "glm-5.3-flash",
    turns: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[Dict[str, Any]] = [
        {"type": "session", "id": session_id, "cwd": cwd, "timestamp": "2026-11-18T10:00:00+08:00"},
        {"type": "model_change", "modelId": model, "provider": "zai"},
    ]
    for t_idx, turn in enumerate(turns):
        lines.append(
            {
                "type": "message",
                "id": f"u_{session_id[:6]}_{t_idx}",
                "timestamp": "2026-11-18T10:01:00+08:00",
                "message": {"role": "user", "content": [{"type": "text", "text": turn["user"]}]},
            }
        )
        for s_idx, step in enumerate(turn.get("steps") or []):
            content: List[Dict[str, Any]] = []
            if step.get("thinking"):
                content.append({"type": "thinking", "thinking": step["thinking"]})
            if step.get("text"):
                content.append({"type": "text", "text": step["text"]})
            for tc_idx, tc in enumerate(step.get("tool_calls") or []):
                content.append(
                    {
                        "type": "toolCall",
                        "id": tc.get("id") or f"call_{session_id[:4]}_{t_idx}_{s_idx}_{tc_idx}",
                        "name": tc["name"],
                        "arguments": tc.get("arguments") or {},
                    }
                )
            lines.append(
                {
                    "type": "message",
                    "id": step.get("id") or f"asst_{t_idx}_{s_idx}",
                    "timestamp": "2026-11-18T10:02:00+08:00",
                    "message": {
                        "role": "assistant",
                        "model": model,
                        "provider": "zai",
                        "content": content,
                        "stopReason": step.get("stop_reason", "toolUse" if step.get("tool_calls") else "stop"),
                    },
                }
            )
            for tr in step.get("tool_results") or []:
                lines.append(
                    {
                        "type": "message",
                        "id": f"tr_{session_id[:4]}_{t_idx}_{s_idx}_{tr.get('tool_call_id', '0')}",
                        "timestamp": "2026-11-18T10:03:00+08:00",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": tr.get("tool_call_id") or f"call_{session_id[:4]}_{t_idx}_{s_idx}_0",
                            "toolName": tr.get("name", "bash"),
                            "isError": bool(tr.get("isError", False)),
                            "content": [{"type": "text", "text": tr.get("content", "ok")}],
                        },
                    }
                )
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")


@pytest.fixture()
def phase3_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    monkeypatch.setattr(app_mod, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(app_mod, "CONFIG_PATH", cfg_path)
    app_mod._EVAL_PACK_CACHE.clear()

    src_dir = tmp_path / "ext_teacher"
    sid1 = "01a1aa11-1111-7000-8000-000000000001"
    sid2 = "01a1bb22-2222-7000-8000-000000000002"
    sid3 = "01a1cc33-3333-7000-8000-000000000003"

    _write_pi_session(
        src_dir / f"{sid1}.jsonl",
        session_id=sid1,
        cwd=r"D:\crack\com.demo.alpha",
        turns=[
            {
                "user": "请完整逆向 com.demo.alpha.apk 并还原全部加密配置。",
                "steps": [
                    {
                        "id": "m_a1",
                        "thinking": "先解包 APK 结构并检查 lib 与 assets 目录，定位核心文件。",
                        "tool_calls": [{"id": "c_a1", "name": "bash", "arguments": {"command": "apktool d alpha.apk"}}],
                        "tool_results": [{"tool_call_id": "c_a1", "name": "bash", "content": "decoded ok"}],
                    },
                    {
                        "id": "m_a1b",
                        "thinking": "读取配置文件头十六字节，核对自定义魔数与加密方式。",
                        "tool_calls": [{"id": "c_a1b", "name": "bash", "arguments": {"command": "head -c 16 assets/cfg.bin"}}],
                        "tool_results": [{"tool_call_id": "c_a1b", "name": "bash", "content": "4d44424b"}],
                    },
                    {
                        "id": "m_a_err",
                        "thinking": "尝试用错误参数运行脚本导致报错，随后需要剪枝。",
                        "tool_calls": [{"id": "c_a_err", "name": "bash", "arguments": {"command": "python bad.py"}}],
                        "tool_results": [{"tool_call_id": "c_a_err", "name": "bash", "isError": True, "content": "Traceback: FileNotFoundError"}],
                    },
                    {
                        "id": "m_a2",
                        "thinking": "修正路径后用脚本提取全部配置 JSON 并校验格式完整性。",
                        "tool_calls": [{"id": "c_a2", "name": "bash", "arguments": {"command": "python unpack_cfg.py"}}],
                        "tool_results": [{"tool_call_id": "c_a2", "name": "bash", "content": "extracted 12 files"}],
                    },
                ],
            },
            {
                "user": "继续",
                "steps": [
                    {
                        "id": "m_a3",
                        "thinking": "核验所有还原产物完整性无误，准备输出最终逆向总结报告。",
                        "text": "# 还原完成\n已成功解出全部 12 个配置文件。",
                        "stop_reason": "stop",
                    }
                ],
            },
        ],
    )

    def _clean_4_steps(prefix: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"m_{prefix}1",
                "thinking": f"先检查 {prefix} 安装包目录结构并确认加密资产列表。",
                "tool_calls": [{"id": f"c_{prefix}1", "name": "bash", "arguments": {"command": f"unzip -l {prefix}.apk"}}],
                "tool_results": [{"tool_call_id": f"c_{prefix}1", "name": "bash", "content": "assets/data.bin"}],
            },
            {
                "id": f"m_{prefix}2",
                "thinking": f"探查 {prefix} 的 data.bin 文件头与密钥派生逻辑。",
                "tool_calls": [{"id": f"c_{prefix}2", "name": "bash", "arguments": {"command": "python probe.py"}}],
                "tool_results": [{"tool_call_id": f"c_{prefix}2", "name": "bash", "content": "magic ok"}],
            },
            {
                "id": f"m_{prefix}3",
                "thinking": f"执行批量解密脚本还原 {prefix} 全部数据文件。",
                "tool_calls": [{"id": f"c_{prefix}3", "name": "bash", "arguments": {"command": f"python dec_{prefix}.py"}}],
                "tool_results": [{"tool_call_id": f"c_{prefix}3", "name": "bash", "content": "decrypted ok"}],
            },
            {
                "id": f"m_{prefix}4",
                "thinking": f"所有 {prefix} 资产已经过脚本校验无误，输出完整收尾结论。",
                "text": f"{prefix} 全部资源已还原。",
                "stop_reason": "stop",
            },
        ]

    _write_pi_session(
        src_dir / f"{sid2}.jsonl",
        session_id=sid2,
        cwd=r"D:\crack\com.demo.beta",
        turns=[{"user": "请逆向 com.demo.beta.apk。", "steps": _clean_4_steps("b")}],
    )

    _write_pi_session(
        src_dir / f"{sid3}.jsonl",
        session_id=sid3,
        cwd=r"D:\crack\com.demo.gamma",
        turns=[{"user": "请逆向 com.demo.gamma.apk。", "steps": _clean_4_steps("g")}],
    )

    curate_tools.adopt_sessions(
        src_dir,
        select=[sid1, sid2],
        reason="先采纳 alpha 与 beta 构造 v1 基线包",
        actor="agent",
        cfg=cfg,
    )
    curate_tools.curate_split(
        pin=["game::alpha=train", "game::beta=eval"],
        reason="固定 train/eval 切分确保门禁非空",
        actor="agent",
        cfg=cfg,
    )
    b1 = build_tools.build_pipeline(reason="构建 v1", actor="agent", cfg=cfg)
    assert b1["ok"] is True, f"b1 failed: {b1}"
    p1 = build_tools.pack_delivery(
        build_id=b1["build_id"],
        profile="qwen-xml",
        reason="打包 v1",
        actor="agent",
        cfg=cfg,
    )
    assert p1["ok"] is True, f"p1 failed: {p1}"

    # 再采纳 gamma 并对 alpha 的 m_a2 写入改写决策，构建 v2
    curate_tools.adopt_sessions(
        src_dir,
        select=[sid3],
        reason="补齐 gamma 教师轨迹",
        actor="agent",
        cfg=cfg,
    )
    curate_tools.curate_split(
        pin=["game::gamma=train"],
        reason="将 gamma 归入训练集",
        actor="agent",
        cfg=cfg,
    )
    curate_tools.curate_steps(
        {
            "session_id": sid1,
            "steps": {
                "m_a_err": {
                    "action": "prune",
                    "reason": "错误参数报错弯路",
                },
                "m_a2": {
                    "action": "rewrite",
                    "rewritten": "定位正确的配置目录后执行脚本提取全部十二份配置文件。",
                    "reason": "精简思考表述",
                },
            },
            "model_report": {
                "verdict": "trainable",
                "confidence": 0.92,
                "reason": "完整逆向闭环",
                "highlights": [{"msg_id": "m_a2", "weight": 1.5, "why": "关键解密步"}],
            },
        },
        apply=True,
        reason="精炼 alpha 关键解密步思考",
        actor="agent",
        cfg=cfg,
    )
    b2 = build_tools.build_pipeline(reason="构建 v2", actor="agent", cfg=cfg)
    assert b2["ok"] is True, f"b2 failed: {b2}"
    p2 = build_tools.pack_delivery(
        build_id=b2["build_id"],
        profile="qwen-xml",
        reason="打包 v2",
        actor="agent",
        cfg=cfg,
    )
    assert p2["ok"] is True

    rep = report_tools.publish_report(
        title="v1 到 v2 评估演进报告",
        md_content="# 评估结论\n补齐 beta 并优化 alpha 思维链后，覆盖项目翻倍。",
        report_id="rep_v1_to_v2",
        before_pack=p1["package_hash"],
        after_pack=p2["package_hash"],
        build_id=b2["build_id"],
        summary="项目数从 1 提升至 2，门禁 100% 通过",
        actor="agent",
        reason="归档迭代报告",
        cfg=cfg,
    )

    client = TestClient(app_mod.app)
    return {
        "cfg": cfg,
        "client": client,
        "sid1": sid1,
        "sid2": sid2,
        "b1": b1,
        "b2": b2,
        "p1": p1,
        "p2": p2,
        "report": rep,
    }


def test_overview_includes_build_pack_reports_and_ledger(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    body = resp.json()

    assert body["latest_build"]["build_id"] == phase3_env["b2"]["build_id"]
    assert body["latest_pack"]["package_hash"] == phase3_env["p2"]["package_hash"]
    assert len(body["reports"]) == 1
    assert body["reports"][0]["report_id"] == "rep_v1_to_v2"
    assert len(body["recent_ledger"]) >= 5


def test_ledger_and_reports_endpoints_support_filters(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    sid1 = phase3_env["sid1"]

    r_all = client.get("/api/ledger?limit=50")
    assert r_all.status_code == 200
    all_data = r_all.json()
    assert all_data["count"] >= 5
    assert "curate.adopt" in all_data["tools"]
    assert "agent" in all_data["actors"]

    r_sid = client.get(f"/api/ledger?sid={sid1[:8]}&tool=curate.steps")
    assert r_sid.status_code == 200
    sid_events = r_sid.json()["events"]
    assert len(sid_events) == 1
    assert sid_events[0]["reason"] == "精炼 alpha 关键解密步思考"

    r_rep_list = client.get("/api/reports")
    assert r_rep_list.status_code == 200
    assert r_rep_list.json()["count"] == 1

    r_rep = client.get("/api/reports/rep_v1_to_v2")
    assert r_rep.status_code == 200
    rep_detail = r_rep.json()
    assert rep_detail["title"] == "v1 到 v2 评估演进报告"
    assert "评估结论" in rep_detail["markdown"]
    assert rep_detail["diff_summary"]["deltas"]["project_count"]["delta"] == 1

    r_404 = client.get("/api/reports/non_existent_report")
    assert r_404.status_code == 404


def test_eval_pack_and_diff_endpoints(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    h1 = phase3_env["p1"]["package_hash"]
    h2 = phase3_env["p2"]["package_hash"]

    r_pack = client.get(f"/api/eval/pack?target={h2}")
    assert r_pack.status_code == 200
    pdata = r_pack.json()
    assert pdata["gate"]["passed"] is True
    assert pdata["eval"]["healthy"] is True
    assert pdata["eval"]["premature_stop_targets"] == 0
    assert pdata["eval"]["prompt_tag_leaks"] == 0

    r_diff = client.get(f"/api/eval/diff?before={h1}&after={h2}")
    assert r_diff.status_code == 200
    ddata = r_diff.json()
    assert ddata["deltas"]["project_count"]["before"] == 2
    assert ddata["deltas"]["project_count"]["after"] == 3
    assert "| 指标 |" in ddata["markdown_table"]


def test_sessions_and_three_state_lineage_and_step_gate(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    cfg = phase3_env["cfg"]
    sid1 = phase3_env["sid1"]

    r_sessions = client.get("/api/sessions")
    assert r_sessions.status_code == 200
    sdata = r_sessions.json()
    assert sdata["total"] == 3
    assert set(sdata["projects"]) == {"game::alpha", "game::beta", "game::gamma"}

    # 用唯一前缀请求三态血缘
    r_lin = client.get(f"/api/sessions/{sid1[:8]}/lineage")
    assert r_lin.status_code == 200
    lin = r_lin.json()
    assert lin["session_id"] == sid1
    assert any(t["folded"] for t in lin["turns"]), "第二轮“继续”应呈现为折叠状态"

    states_by_mid = {s["msg_id"]: s for s in lin["steps"]}
    assert states_by_mid["m_a1"]["state"] == "kept"
    assert states_by_mid["m_a_err"]["state"] == "pruned"
    assert states_by_mid["m_a2"]["state"] == "rewritten"
    assert states_by_mid["m_a2"]["changed"] is True
    assert states_by_mid["m_a2"]["highlight_weight"] == 1.5
    assert len(lin["ledger_events"]) >= 2

    # Web 应急步级开关：排除 m_a1 并确认自动写入 ledger (actor="human")
    r_gate_off = client.put(
        f"/api/gold/session/{sid1[:8]}/step-gate",
        json={"msg_id": "m_a1", "action": "exclude", "reason": "Web 人工测试排除"},
    )
    assert r_gate_off.status_code == 200
    assert r_gate_off.json()["ok"] is True

    r_lin_after = client.get(f"/api/sessions/{sid1}/lineage")
    after_steps = {s["msg_id"]: s for s in r_lin_after.json()["steps"]}
    assert after_steps["m_a1"]["excluded"] is True
    assert after_steps["m_a1"]["state"] == "excluded"

    human_events = ledger.read_ledger(cfg=cfg, actor="human", sid=sid1)
    assert any(ev["tool"] == "curate.steps" and "Web 人工测试排除" in ev["reason"] for ev in human_events)

    # 恢复 m_a1
    r_gate_on = client.put(
        f"/api/gold/session/{sid1}/step-gate",
        json={"msg_id": "m_a1", "action": "keep"},
    )
    assert r_gate_on.status_code == 200
    r_lin_restored = client.get(f"/api/sessions/{sid1}/lineage")
    restored_steps = {s["msg_id"]: s for s in r_lin_restored.json()["steps"]}
    assert restored_steps["m_a1"]["excluded"] is False


def test_draft_exclusion_switches_record_human_ledger_and_legacy_review_removed(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    cfg = phase3_env["cfg"]
    sid2 = phase3_env["sid2"]

    # Web 作用域停用开关自动记账本
    r_scope = client.put(
        "/api/dataset/draft/scopes",
        json={"updates": [{"scope": "session", "id": sid2, "action": "exclude", "reason": "应急停用 beta"}]},
    )
    assert r_scope.status_code == 200
    scope_events = ledger.read_ledger(cfg=cfg, tool="web.draft_scopes", actor="human")
    assert len(scope_events) == 1
    assert scope_events[0]["reason"] == "应急停用 beta"

    # 已移除的 6 个旧人工审核路由应返回 404/405，且 /api/run 拒绝 review/refine
    for method, path in (
        ("POST", f"/api/gold/session/{sid2}/review"),
        ("PUT", f"/api/gold/session/{sid2}/review/items"),
        ("POST", f"/api/gold/session/{sid2}/review/batch"),
        ("POST", f"/api/gold/session/{sid2}/review/decision"),
        ("POST", f"/api/gold/session/{sid2}/review/refine"),
        ("POST", f"/api/gold/session/{sid2}/finalize"),
    ):
        resp = client.request(method, path, json={})
        assert resp.status_code in (404, 405), f"旧路由 {method} {path} 未清理干净"

    for dead_stage in ("review", "refine"):
        r_run = client.post("/api/run", json={"stage": dead_stage})
        assert r_run.status_code == 400


def test_web_observability_console_wires_phase3_views(tmp_path: Path) -> None:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "openReportDrawer(",
        "loadLedger(",
        "paintLedgerView(",
        "openLedgerDetail(",
        "openSessionLineageDrawer(",
        "renderLineageHtml(",
        "openPackEval(",
        "openPackDiffDrawer(",
        "/api/gold/session/${encodeURIComponent(sid)}/step-gate",
        'data-sm="lineage"',
        "TAB_ALIASES",
    ):
        assert required in html, f"web/index.html 缺少 Phase 3 关键实现: {required}"

    for removed in (
        "/review/items",
        "/review/decision",
        "/review/refine",
        "/review/batch",
        "function revFinalize",
        "function revDecide",
    ):
        assert removed not in html, f"web/index.html 仍残留废弃审核代码: {removed}"

    if shutil.which("node"):
        script = "\n".join(re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html))
        js_file = tmp_path / "phase3_inline_check.js"
        js_file.write_text(script, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(js_file)], capture_output=True, text=True, encoding="utf-8")
        assert proc.returncode == 0, proc.stderr
    assert "const jsArg =" in html


def test_audit_fixes_step_reset_dpo_shorthand_and_split_ledger(phase3_env: Dict[str, Any]) -> None:
    client: TestClient = phase3_env["client"]
    cfg = phase3_env["cfg"]
    sid1 = phase3_env["sid1"]

    # 1) m_a2 原本在 fixture 中被 rewrite；传 None 重置后不应从旧 .gold.jsonl 反向复活
    curate_tools.curate_steps(
        {"session_id": sid1, "steps": {"m_a2": None}},
        apply=True,
        reason="撤销 m_a2 人工改写恢复原始思考",
        actor="agent",
        cfg=cfg,
    )
    r_lin = client.get(f"/api/sessions/{sid1}/lineage")
    assert r_lin.status_code == 200
    steps_map = {s["msg_id"]: s for s in r_lin.json()["steps"]}
    assert steps_map["m_a2"]["state"] == "kept"
    assert steps_map["m_a2"]["changed"] is False

    # 2) curate_dpo 支持 all=good 快捷串
    dpo_res = curate_tools.curate_dpo("all=good", reason="快捷放行全部候选偏好对", actor="agent", cfg=cfg)
    assert dpo_res["ok"] is True
    assert dpo_res["updated"] >= 1

    # 3) Web 切分调整自动写入 ledger (web.draft_split)
    r_sp = client.put(
        "/api/dataset/draft/split",
        json={"assignments": {"game::gamma": "eval"}},
    )
    assert r_sp.status_code == 200
    sp_events = ledger.read_ledger(cfg=cfg, tool="web.draft_split", actor="human")
    assert len(sp_events) == 1
    assert "game::gamma" in sp_events[0]["affected"]["projects"]

