"""虚拟数据集条目、CAS 内容寻址时间旅行快照与演进 DAG 关系图的自动化测试。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from goldthread import build_tools, curate_tools
from pipeline.common.config import DEFAULTS
from pipeline.common.store import read_json


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
        {"type": "session", "id": session_id, "cwd": cwd, "timestamp": "2026-11-19T09:00:00+08:00"},
        {"type": "model_change", "modelId": model, "provider": "zai"},
    ]
    for t_idx, turn in enumerate(turns):
        lines.append(
            {
                "type": "message",
                "id": f"u_{session_id[:6]}_{t_idx}",
                "timestamp": "2026-11-19T09:01:00+08:00",
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
                    "timestamp": "2026-11-19T09:02:00+08:00",
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
                        "timestamp": "2026-11-19T09:03:00+08:00",
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


def _seed_two_sessions(tmp_path: Path) -> tuple[Path, str, str]:
    src_dir = tmp_path / "ext_src"
    sid1 = "01b2aa11-1111-7000-8000-000000000001"
    sid2 = "01b2bb22-2222-7000-8000-000000000002"
    _write_pi_session(
        src_dir / f"{sid1}.jsonl",
        session_id=sid1,
        cwd=r"D:\crack\com.demo.alpha",
        turns=[
            {
                "user": "完整逆向 com.demo.alpha.apk 并导出加密配置。",
                "steps": [
                    {
                        "id": "a_s1",
                        "thinking": "初始版本思考：先解包 APK 检查 assets 目录结构与加密资产清单。",
                        "tool_calls": [{"id": "ca1", "name": "bash", "arguments": {"command": "apktool d a.apk"}}],
                        "tool_results": [{"tool_call_id": "ca1", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "a_s1b",
                        "thinking": "读取头部十六字节确认自定义魔数与算法标识。",
                        "tool_calls": [{"id": "ca1b", "name": "bash", "arguments": {"command": "head -c 16 assets/cfg.bin"}}],
                        "tool_results": [{"tool_call_id": "ca1b", "name": "bash", "content": "4d44424b"}],
                    },
                    {
                        "id": "a_err",
                        "thinking": "尝试错误脚本参数导致报错，随后将被剪枝沉淀为 DPO。",
                        "tool_calls": [{"id": "cae", "name": "bash", "arguments": {"command": "python wrong.py"}}],
                        "tool_results": [{"tool_call_id": "cae", "name": "bash", "isError": True, "content": "Traceback: FileNotFoundError"}],
                    },
                    {
                        "id": "a_s2",
                        "thinking": "用脚本提取配置并输出验收总结。",
                        "tool_calls": [{"id": "ca2", "name": "bash", "arguments": {"command": "python dec.py"}}],
                        "tool_results": [{"tool_call_id": "ca2", "name": "bash", "content": "10 files"}],
                    },
                    {
                        "id": "a_s3",
                        "thinking": "核对全部输出文件无误后完成交付。",
                        "text": "已完成 alpha 全部解密。",
                    },
                ],
            }
        ],
    )
    _write_pi_session(
        src_dir / f"{sid2}.jsonl",
        session_id=sid2,
        cwd=r"D:\crack\com.demo.beta",
        turns=[
            {
                "user": "完整逆向 com.demo.beta.apk 资源。",
                "steps": [
                    {
                        "id": "b_s1",
                        "thinking": "检查 beta 资源包头部魔数与文件目录结构。",
                        "tool_calls": [{"id": "cb1", "name": "bash", "arguments": {"command": "head -c 16 res.pak"}}],
                        "tool_results": [{"tool_call_id": "cb1", "name": "bash", "content": "50414b31"}],
                    },
                    {
                        "id": "b_s2",
                        "thinking": "编写解包脚本导出全部纹理与数据表。",
                        "tool_calls": [{"id": "cb2", "name": "bash", "arguments": {"command": "python unpack_pak.py"}}],
                        "tool_results": [{"tool_call_id": "cb2", "name": "bash", "content": "done"}],
                    },
                    {
                        "id": "b_s2b",
                        "thinking": "用校验脚本核对导出的全部 PNG 与配置条目数。",
                        "tool_calls": [{"id": "cb2b", "name": "bash", "arguments": {"command": "python check.py"}}],
                        "tool_results": [{"tool_call_id": "cb2b", "name": "bash", "content": "verified"}],
                    },
                    {
                        "id": "b_s3",
                        "thinking": "验证导出纹理尺寸与完整性后输出完整结论收尾。",
                        "text": "已完成 beta 资源导出。",
                    },
                ],
            }
        ],
    )
    return src_dir, sid1, sid2


def test_cas_snapshot_survives_workspace_mutation_and_deletion(tmp_path: Path) -> None:
    from pipeline import cas_store, stage2_gold

    cfg, _ = _make_sandbox_cfg(tmp_path)
    src_dir, sid1, sid2 = _seed_two_sessions(tmp_path)

    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2], reason="采纳初始两条会话", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行初始 DPO", cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::beta=eval"], reason="固定切分", cfg=cfg)
    b1 = build_tools.build_pipeline(reason="构建 v1", cfg=cfg)
    assert b1["ok"] is True, f"b1 gate: {b1.get('gate')}"
    b1_id = b1["build_id"]

    # 断言 v1 构建已归档 CAS 快照清单
    b1_cas = cas_store.load_build_cas_manifest(b1_id, cfg=cfg)
    assert set(b1_cas["sessions"].keys()) == {sid1, sid2}
    sid1_v1_gold_sha = b1_cas["sessions"][sid1]["files"]["gold_file"]
    sid2_v1_raw_sha = b1_cas["sessions"][sid2]["files"]["raw_file"]
    assert len(sid1_v1_gold_sha) == 64
    assert len(sid2_v1_raw_sha) == 64

    # 在工作区修改 sid1 的黄金思考，并彻底删除 sid2
    curate_tools.curate_steps(
        {
            "session_id": sid1,
            "steps": {
                "a_s1": {
                    "action": "rewrite",
                    "rewritten": "演进后改写思考：直接用并行命令探查 APK 目录与 lib 架构。",
                    "reason": "精炼开局思考",
                }
            },
        },
        apply=True,
        reason="v2 修补 sid1 开局思考",
        cfg=cfg,
    )
    curate_tools.drop_session(sid2, reason="v2 移除旧会话 sid2", cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行新改写产生的 DPO", cfg=cfg)

    # 即使工作区已改写 sid1、删除了 sid2，读取 v1 的 CAS 快照仍 100% 还原当时原貌
    v1_sid1_gold_rows = cas_store.read_cas_jsonl(sid1_v1_gold_sha, cfg=cfg)
    assert len(v1_sid1_gold_rows) == 1
    v1_asst = [m for m in v1_sid1_gold_rows[0]["messages"] if m.get("id") == "a_s1"][0]
    assert "初始版本思考" in v1_asst["thinking"]

    v1_sid2_raw_path = cas_store.get_cas_path(sid2_v1_raw_sha, cfg=cfg)
    assert v1_sid2_raw_path is not None and v1_sid2_raw_path.is_file()
    assert sid2 in v1_sid2_raw_path.read_text(encoding="utf-8")

    # v1 中 sid1 的 raw_file 未变，构建 v2 时相同 sha256 对象应自动去重复用
    sid1_raw_sha = b1_cas["sessions"][sid1]["files"]["raw_file"]
    b2 = build_tools.build_pipeline(run_gate=False, reason="构建单项目 v2 验证去重", cfg=cfg)
    b2_cas = cas_store.load_build_cas_manifest(b2["build_id"], cfg=cfg)
    assert b2_cas["sessions"][sid1]["files"]["raw_file"] == sid1_raw_sha
    assert b2_cas["sessions"][sid1]["files"]["gold_file"] != sid1_v1_gold_sha
    assert sid2 not in b2_cas["sessions"]


def test_lineage_dag_workspace_dirty_transition_and_four_layer_time_travel(tmp_path: Path) -> None:
    from goldthread import lineage_graph, report_tools
    from pipeline import stage2_gold

    cfg, _ = _make_sandbox_cfg(tmp_path)
    src_dir, sid1, sid2 = _seed_two_sessions(tmp_path)
    sid3 = "01b2cc33-3333-7000-8000-000000000003"
    _write_pi_session(
        src_dir / f"{sid3}.jsonl",
        session_id=sid3,
        cwd=r"D:\crack\com.demo.gamma",
        turns=[
            {
                "user": "完整逆向 com.demo.gamma.apk 全部脚本与配置。",
                "steps": [
                    {
                        "id": "g_s1",
                        "thinking": "解包 gamma.apk 检查目录。",
                        "tool_calls": [{"id": "cg1", "name": "bash", "arguments": {"command": "apktool d g.apk"}}],
                        "tool_results": [{"tool_call_id": "cg1", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "g_s2",
                        "thinking": "探查加密脚本头部。",
                        "tool_calls": [{"id": "cg2", "name": "bash", "arguments": {"command": "head -c 16 main.enc"}}],
                        "tool_results": [{"tool_call_id": "cg2", "name": "bash", "content": "47414d45"}],
                    },
                    {
                        "id": "g_s3",
                        "thinking": "运行解密器还原全部脚本。",
                        "tool_calls": [{"id": "cg3", "name": "bash", "arguments": {"command": "python dec_g.py"}}],
                        "tool_results": [{"tool_call_id": "cg3", "name": "bash", "content": "done"}],
                    },
                    {
                        "id": "g_s4",
                        "thinking": "校验输出文件数量无误后收尾。",
                        "text": "gamma 全部还原完成。",
                    },
                ],
            }
        ],
    )

    # 1) 构建并打包 v1 -> Clean 状态，工作空间应隐藏
    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2], reason="采纳 alpha/beta", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行 v1 DPO", cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::beta=eval"], reason="固定 v1 切分", cfg=cfg)
    b1 = build_tools.build_pipeline(reason="构建 v1", cfg=cfg)
    p1 = build_tools.pack_delivery(build_id=b1["build_id"], profile="qwen-xml", reason="打包 v1", cfg=cfg)
    assert p1["ok"] is True
    h1 = p1["package_hash"]

    g_clean = lineage_graph.build_dataset_graph(cfg=cfg)
    assert g_clean["workspace"]["dirty"] is False
    assert [n["id"] for n in g_clean["nodes"]] == [h1]
    assert g_clean["nodes"][0]["is_head"] is True

    # 2) 在工作区新增 sid3、改写 sid1、删除 sid2（未打包） -> Dirty 状态，出现虚线草稿边
    curate_tools.adopt_sessions(src_dir, select=[sid3], reason="v2 新增 gamma", cfg=cfg)
    curate_tools.curate_steps(
        {
            "session_id": sid1,
            "steps": {
                "a_s1": {
                    "action": "rewrite",
                    "rewritten": "演进后精炼思考：先用并行命令检查安装包目录与底层库架构分布。",
                    "reason": "压缩开局思考",
                }
            },
        },
        apply=True,
        reason="v2 修补 alpha 思考",
        cfg=cfg,
    )
    curate_tools.drop_session(sid2, reason="v2 删除旧 beta", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行 v2 DPO", cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::gamma=eval"], reason="固定 v2 切分", cfg=cfg)

    g_dirty = lineage_graph.build_dataset_graph(cfg=cfg)
    assert g_dirty["workspace"]["dirty"] is True
    draft_edges = [e for e in g_dirty["edges"] if e["to"] == "workspace"]
    assert len(draft_edges) == 1
    assert draft_edges[0]["from"] == h1
    assert len(draft_edges[0]["ledger_ops"]) >= 3

    # 3) 构建并打包 v2（自动继承 parent_pack=h1），发布评估报告 -> 重新回到 Clean 态
    b2 = build_tools.build_pipeline(reason="构建 v2", cfg=cfg)
    assert b2["ok"] is True
    p2 = build_tools.pack_delivery(build_id=b2["build_id"], profile="qwen-xml", reason="打包 v2", cfg=cfg)
    assert p2["ok"] is True
    h2 = p2["package_hash"]
    assert p2.get("parent_pack") == h1

    report_tools.publish_report(
        title="v1 到 v2 演进评估报告",
        md_content="# 评估结论\n\n成功替换 beta 为 gamma 并精炼 alpha 开局思考。\n",
        before_pack=h1,
        after_pack=h2,
        build_id=b2["build_id"],
        reason="归档 v1->v2 报告",
        cfg=cfg,
    )

    g_v2 = lineage_graph.build_dataset_graph(cfg=cfg)
    assert g_v2["workspace"]["dirty"] is False
    assert {n["id"] for n in g_v2["nodes"]} == {h1, h2}
    e_1_2 = [e for e in g_v2["edges"] if e["from"] == h1 and e["to"] == h2]
    assert len(e_1_2) == 1
    assert len(e_1_2[0]["reports"]) == 1
    assert e_1_2[0]["reports"][0]["title"] == "v1 到 v2 演进评估报告"
    assert any(op["tool"] == "curate.steps" for op in e_1_2[0]["ledger_ops"])

    # 4) 四层时间旅行验证：点开旧包 h1，范围精准隔离 sid3，且已删 sid2 / 已改 sid1 均 100% 还原当时原貌
    h1_gold = lineage_graph.list_virtual_dataset_gold(h1, cfg=cfg)
    h1_gold_map = {s["session_id"]: s for s in h1_gold["sessions"]}
    assert set(h1_gold_map.keys()) == {sid1, sid2}
    assert h1_gold_map[sid1]["snapshot_status"] == "workspace_modified"
    assert h1_gold_map[sid2]["snapshot_status"] == "workspace_deleted"

    h1_raw = lineage_graph.list_virtual_dataset_raw(h1, cfg=cfg)
    assert {s["session_id"] for s in h1_raw["sessions"]} == {sid1, sid2}

    h1_train = lineage_graph.list_virtual_dataset_train(h1, cfg=cfg)
    assert h1_train["total"] > 0
    assert all("gamma" not in (row.get("project_group") or "") for row in h1_train["rows"])

    lin_sid2_v1 = lineage_graph.get_virtual_dataset_session_lineage(h1, sid2, cfg=cfg)
    assert lin_sid2_v1["readonly"] is True
    assert lin_sid2_v1["snapshot_status"] == "workspace_deleted"
    assert any(s["msg_id"] == "b_s1" for s in lin_sid2_v1["steps"])
    assert any("已完成 beta 资源导出" in str(m.get("text") or "") for m in lin_sid2_v1["gold_messages"])
    assert len(lin_sid2_v1["raw_messages"]) > 0

    lin_sid1_v1 = lineage_graph.get_virtual_dataset_session_lineage(h1, sid1, cfg=cfg)
    s1_v1 = [s for s in lin_sid1_v1["steps"] if s["msg_id"] == "a_s1"][0]
    assert "初始版本思考" in s1_v1["gold_thinking"]

    lin_sid1_v2 = lineage_graph.get_virtual_dataset_session_lineage(h2, sid1, cfg=cfg)
    s1_v2 = [s for s in lin_sid1_v2["steps"] if s["msg_id"] == "a_s1"][0]
    assert "演进后精炼思考" in s1_v2["gold_thinking"]


def test_datasets_http_endpoints_and_cli_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    import server.app as app_mod
    from goldthread import cli
    from pipeline import stage2_gold

    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    monkeypatch.setattr(app_mod, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(app_mod, "CONFIG_PATH", cfg_path)
    app_mod._EVAL_PACK_CACHE.clear()

    src_dir, sid1, sid2 = _seed_two_sessions(tmp_path)
    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2], reason="采纳 alpha/beta", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行 DPO", cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::beta=eval"], reason="固定切分", cfg=cfg)
    b1 = build_tools.build_pipeline(reason="构建 v1", cfg=cfg)
    assert cli.main(["--config", str(cfg_path), "--json", "pack", "--profile", "qwen-xml", "--reason", "CLI 打包 v1"]) == 0

    client = TestClient(app_mod.app)
    r_graph = client.get("/api/datasets/graph")
    assert r_graph.status_code == 200
    g_body = r_graph.json()
    assert g_body["workspace"]["dirty"] is False
    assert len(g_body["nodes"]) == 1
    h1 = g_body["nodes"][0]["id"]

    # CLI inspect graph
    assert cli.main(["--config", str(cfg_path), "--json", "inspect", "graph"]) == 0

    # 删除 sid2 使工作空间变为 Dirty，验证 HTTP 看到虚线草稿及 h1 的 CAS 只读还原
    curate_tools.drop_session(sid2, reason="工作区删除 beta", cfg=cfg)
    r_graph_dirty = client.get("/api/datasets/graph")
    assert r_graph_dirty.status_code == 200
    assert r_graph_dirty.json()["workspace"]["dirty"] is True

    r_ov = client.get(f"/api/datasets/{h1}")
    assert r_ov.status_code == 200
    assert r_ov.json()["id"] == h1
    assert r_ov.json()["readonly"] is True
    bd = r_ov.json()["breakdown"]
    assert len(bd["projects"]) == 2
    assert {p["project"] for p in bd["projects"]} == {"game::alpha", "game::beta"}

    r_train = client.get(f"/api/datasets/{h1}/train?split=eval")
    assert r_train.status_code == 200
    assert r_train.json()["total"] > 0

    r_gold = client.get(f"/api/datasets/{h1}/gold")
    assert r_gold.status_code == 200
    assert {s["session_id"] for s in r_gold.json()["sessions"]} == {sid1, sid2}

    r_raw = client.get(f"/api/datasets/{h1}/raw")
    assert r_raw.status_code == 200
    assert {s["session_id"] for s in r_raw.json()["sessions"]} == {sid1, sid2}

    r_lin = client.get(f"/api/datasets/{h1}/sessions/{sid2[:8]}")
    assert r_lin.status_code == 200
    lin_body = r_lin.json()
    assert lin_body["readonly"] is True
    assert lin_body["snapshot_status"] == "workspace_deleted"
    assert lin_body["session_id"] == sid2


def test_web_console_renders_variant_a_lineage_dag_and_cas_views(tmp_path: Path) -> None:
    import re
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "renderDatasetDagHtml(",
        "selectVirtualDataset(",
        "selectLineageEdge(",
        "loadVirtualDatasetSubView(",
        "openDatasetSessionDrawer(",
        "switchVdsSessionMode(",
        "renderVdsBreakdownHtml(",
        "openVirtualTrainSampleDrawer(",
        "/api/datasets/",
        "vds-node",
        "vds-edge-card",
        "vds-subtabs",
        "助手对外回复内容",
        "workspace_deleted",
        "workspace_modified",
    ):
        assert required in html, f"web/index.html 缺少 Variant A 关键组件: {required}"
    assert "工作区底座全景漏斗与报告/账本速览" not in html

    # 抛弃型探索原型不应残留在生产静态目录中
    assert not (root / "web" / "prototype-datasets.html").exists()

    if shutil.which("node"):
        script = "\n".join(re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html))
        js_file = tmp_path / "variant_a_check.js"
        js_file.write_text(script, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(js_file)], capture_output=True, text=True, encoding="utf-8")
        assert proc.returncode == 0, proc.stderr


def test_session_sft_exclusion_shows_zero_sft_and_preserves_dpo(tmp_path: Path) -> None:
    from goldthread import lineage_graph
    from pipeline import stage2_gold

    cfg, _ = _make_sandbox_cfg(tmp_path)
    src_dir, sid1, sid2 = _seed_two_sessions(tmp_path)
    sid3 = "01b2cc33-3333-7000-8000-000000000003"
    _write_pi_session(
        src_dir / f"{sid3}.jsonl",
        session_id=sid3,
        cwd=r"D:\crack\com.demo.gamma",
        turns=[
            {
                "user": "完整逆向 com.demo.gamma.apk 全部资源。",
                "steps": [
                    {
                        "id": "g_1",
                        "thinking": "解包 gamma 检查目录。",
                        "tool_calls": [{"id": "cg1", "name": "bash", "arguments": {"command": "apktool d g.apk"}}],
                        "tool_results": [{"tool_call_id": "cg1", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "g_2",
                        "thinking": "读取资源头部魔数。",
                        "tool_calls": [{"id": "cg2", "name": "bash", "arguments": {"command": "head -c 16 g.bin"}}],
                        "tool_results": [{"tool_call_id": "cg2", "name": "bash", "content": "47414d45"}],
                    },
                    {
                        "id": "g_3",
                        "thinking": "导出全部明文资源。",
                        "tool_calls": [{"id": "cg3", "name": "bash", "arguments": {"command": "python dec.py"}}],
                        "tool_results": [{"tool_call_id": "cg3", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "g_4",
                        "thinking": "核对无误后收尾。",
                        "text": "gamma 完成。",
                    },
                ],
            }
        ],
    )

    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2, sid3], reason="采纳三条会话", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_dpo("all=good", reason="放行 DPO", cfg=cfg)
    curate_tools.curate_split(
        pin=["game::alpha=train", "game::gamma=train", "game::beta=eval"],
        reason="固定切分保证 train/eval 非空",
        cfg=cfg,
    )
    # 停用 sid1 的全部 SFT，仅保留其剪枝产生的 DPO 负例
    curate_tools.curate_exclude(
        scopes=[f"session_sft:{sid1}"],
        reason="sid1 仅用于提供 DPO 偏好对，不学 SFT",
        cfg=cfg,
    )

    b1 = build_tools.build_pipeline(reason="构建含仅DPO会话的数据集", cfg=cfg)
    assert b1["ok"] is True
    p1 = build_tools.pack_delivery(build_id=b1["build_id"], profile="qwen-xml", reason="打包", cfg=cfg)
    assert p1["ok"] is True
    h1 = p1["package_hash"]

    # 正式包与工作区的黄金列表都必须精确显示 sid1 的 sft_windows == 0 且 dpo_count >= 1
    for target in (h1, "workspace"):
        gold_res = lineage_graph.list_virtual_dataset_gold(target, cfg=cfg)
        by_sid = {s["session_id"]: s for s in gold_res["sessions"]}
        assert by_sid[sid1]["sft_windows"] == 0
        assert by_sid[sid1]["dpo_count"] >= 1
        assert by_sid[sid1]["sft_excluded"] is True
        # 加权展开副本去重校验：gamma 的唯一逻辑窗口数必须严格小于展开后的 LF 训练行数
        assert by_sid[sid3]["sft_windows"] == 4
    pack_gold = lineage_graph.list_virtual_dataset_gold(h1, cfg=cfg)
    gamma_pack_row = {s["session_id"]: s for s in pack_gold["sessions"]}[sid3]
    assert gamma_pack_row["sft_windows"] < gamma_pack_row["sft_lf_rows"]

    # 冻结历史包后整项目停用 gamma：历史计数/状态/包标记必须完全冻结，实时停用仅走 workspace_usage 叠加
    frozen_fields = (
        "project_group",
        "split",
        "raw_steps",
        "gold_steps",
        "sft_windows",
        "sft_lf_rows",
        "dpo_count",
        "sft_excluded",
        "dpo_excluded",
        "snapshot_status",
        "readonly",
    )
    before = {s["session_id"]: {k: s.get(k) for k in frozen_fields} for s in pack_gold["sessions"]}
    before_gamma_usage = copy.deepcopy(gamma_pack_row["workspace_usage"])
    assert before_gamma_usage["in_workspace"] is True
    assert before_gamma_usage["session_excluded"] is False

    curate_tools.curate_exclude(
        scopes=["project:game::gamma"],
        reason="整项 gamma 停用，验证历史包冻结与工作区实时叠加分离",
        cfg=cfg,
    )

    pack_after = lineage_graph.list_virtual_dataset_gold(h1, cfg=cfg)
    after_rows = {s["session_id"]: s for s in pack_after["sessions"]}
    # 历史包所有行的冻结字段与包内 SFT/DPO 标记、CAS 状态保持不变
    for sid, snap in before.items():
        assert {k: after_rows[sid].get(k) for k in frozen_fields} == snap
    gamma_after = after_rows[sid3]
    assert gamma_after["workspace_usage"]["in_workspace"] is True
    assert gamma_after["workspace_usage"]["session_excluded"] is True
    assert gamma_after["workspace_usage"]["session_exclusion_scope"] == "project"
    assert gamma_after["workspace_usage"]["sft_excluded"] is True
    assert gamma_after["workspace_usage"]["dpo_excluded"] is True

    # 工作区黄金列表：实时停用后 gamma 的行计数归零，并携带同形状的停用状态
    ws_gold = lineage_graph.list_virtual_dataset_gold("workspace", cfg=cfg)
    gamma_ws_row = {s["session_id"]: s for s in ws_gold["sessions"]}[sid3]
    assert gamma_ws_row["sft_windows"] == 0
    assert gamma_ws_row["dpo_count"] == 0
    assert gamma_ws_row["sft_excluded"] is True
    assert gamma_ws_row["dpo_excluded"] is True
    assert gamma_ws_row["workspace_usage"] == gamma_after["workspace_usage"]

    # 抽屉与列表共用同一 workspace_usage 形状/取值
    drawer = lineage_graph.get_virtual_dataset_session_lineage(h1, sid3, cfg=cfg)
    assert drawer["workspace_usage"] == gamma_after["workspace_usage"]
    drawer_ws = lineage_graph.get_virtual_dataset_session_lineage("workspace", sid3, cfg=cfg)
    assert drawer_ws["workspace_usage"] == gamma_ws_row["workspace_usage"]

    # 恢复后实时叠加归零，历史包仍应保持冻结结果
    curate_tools.curate_exclude(
        scopes=["project:game::gamma"],
        restore=True,
        reason="恢复 gamma 以清理测试沙箱状态",
        cfg=cfg,
    )
    restored = {s["session_id"]: s for s in lineage_graph.list_virtual_dataset_gold(h1, cfg=cfg)["sessions"]}
    assert restored[sid3]["workspace_usage"]["session_excluded"] is False
    assert restored[sid3]["workspace_usage"]["sft_excluded"] is False
    assert {k: restored[sid3].get(k) for k in frozen_fields} == before[sid3]
    restored_ws = {s["session_id"]: s for s in lineage_graph.list_virtual_dataset_gold("workspace", cfg=cfg)["sessions"]}
    assert restored_ws[sid3]["workspace_usage"]["session_excluded"] is False
    assert restored_ws[sid3]["sft_windows"] == 4


def test_audit_hardening_gate_staging_dpo_validation_and_pyc_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """覆盖审计 P1/P2：失败包不发布不复制、无 assistant/缺项目门禁拦截、未知 DPO ID 拒绝、模板排除 .pyc。"""
    import shutil
    import zipfile

    import server.app as app_mod
    from pipeline import length_gate, package, stage2_gold

    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    cfg["export_profiles"].append(
        {
            "id": "hermes-json",
            "label": "Hermes JSON",
            "default": False,
            "tool_call_style": "hermes-json",
            "template": "qwen3",
            "tool_format": "hermes",
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(app_mod, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(app_mod, "CONFIG_PATH", cfg_path)

    src_dir, sid1, sid2 = _seed_two_sessions(tmp_path)
    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2], reason="采纳会话", cfg=cfg)
    stage2_gold.run(cfg=cfg)

    # 1) DPO 严格审核：未裁决时 --require-reviewed-dpo 应识别 bypassed_unreviewed >= 1；臆造 ID 必须报错
    dry_unreviewed = build_tools.build_pipeline(dry_run=True, require_reviewed_dpo=True, cfg=cfg)
    assert dry_unreviewed["ok"] is False
    assert dry_unreviewed["gate_preview"]["bypassed_unreviewed"] >= 1
    with pytest.raises(ValueError, match="未知 DPO sample_id"):
        curate_tools.curate_dpo({"dpo::fake_sid::999": "good"}, reason="测试假ID", cfg=cfg)

    curate_tools.curate_dpo("all=good", reason="合法放行全部候选", cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::beta=eval"], reason="固定切分", cfg=cfg)
    b1 = build_tools.build_pipeline(require_reviewed_dpo=True, reason="构建 v1", cfg=cfg)
    assert b1["ok"] is True

    # 2) 打包暂存门禁：模拟分词器不可用导致 length_gate 跳过 -> CLI 与 Web 均不得发布或复制不合格 ZIP
    monkeypatch.setattr(length_gate, "load_gate_tokenizer", lambda *_a, **_kw: (None, "unavailable_in_test"))
    copy_dest = tmp_path / "student_drop"
    bad_pack = build_tools.pack_delivery(
        build_id=b1["build_id"],
        profile="qwen-xml",
        copy_to=copy_dest,
        reason="模拟门禁失败打包",
        cfg=cfg,
    )
    assert bad_pack["ok"] is False
    delivery_dir = Path(cfg["paths"]["data_dir"]) / "04_release" / "delivery"
    assert list(delivery_dir.glob("*.zip")) == []
    assert not copy_dest.exists() or list(copy_dest.glob("*.zip")) == []

    # Web POST /api/export/package 公开路由同样应失败且不产出正式包
    import time as _time
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    r_pkg = client.post("/api/export/package", json={"build_id": b1["build_id"], "profile": "qwen-xml"})
    assert r_pkg.status_code == 200
    jid = r_pkg.json()["job_id"]
    for _ in range(40):
        j_info = client.get(f"/api/jobs/{jid}").json()
        if j_info["status"] not in ("running", "queued"):
            break
        _time.sleep(0.05)
    assert j_info["status"] == "failed"
    assert list(delivery_dir.glob("*.zip")) == []

    # 3) 逐行监督目标与项目归组门禁负例：仅含 user 无 assistant 目标 / 缺失 project_group 均不得通过门禁
    def _write_custom_zip(zpath: Path, train_row: Dict[str, Any], eval_row: Dict[str, Any]) -> None:
        manifest = {
            "package_hash": "deadbeef",
            "format_contract": {"tool_style": "coder-xml"},
            "length_gate": {"sft.train": {"kept": 1}, "sft.eval": {"kept": 1}},
        }
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("goldthread-train/data/MANIFEST.json", json.dumps(manifest))
            zf.writestr("goldthread-train/data/sft.train.lf.jsonl", json.dumps(train_row, ensure_ascii=False) + "\n")
            zf.writestr("goldthread-train/data/sft.eval.lf.jsonl", json.dumps(eval_row, ensure_ascii=False) + "\n")

    user_only_zip = tmp_path / "user_only.zip"
    _write_custom_zip(
        user_only_zip,
        {"sample_id": "u1", "conversations": [{"from": "human", "value": "只有提问没有助手回复"}], "metadata": {"project_group": "game::alpha"}},
        {"sample_id": "u2", "conversations": [{"from": "human", "value": "只有提问"}], "metadata": {"project_group": "game::beta"}},
    )
    g_user_only = build_tools.gate_check(str(user_only_zip), cfg=cfg)
    assert g_user_only["passed"] is False
    assert any(v["code"] in ("sft_non_empty", "missing_assistant_target") for v in g_user_only["violations"])

    no_proj_zip = tmp_path / "no_proj.zip"
    valid_conv = [
        {"from": "human", "value": "逆向目标"},
        {"from": "gpt", "value": "<think>\n足够长度的合法目标思考内容用于测试门禁校验。\n</think>\n已完成总结。"},
    ]
    _write_custom_zip(
        no_proj_zip,
        {"sample_id": "p1", "conversations": valid_conv, "metadata": {}},
        {"sample_id": "p2", "conversations": valid_conv, "metadata": {"project_group": "game::beta"}},
    )
    g_no_proj = build_tools.gate_check(str(no_proj_zip), cfg=cfg)
    assert g_no_proj["passed"] is False
    assert any(v["code"] == "canonical_project_groups" for v in g_no_proj["violations"])

    # 4) 模板收集器排除 __pycache__ / *.pyc，保持哈希不受本地字节码缓存污染
    tmpl_copy = tmp_path / "tmpl_copy"
    shutil.copytree(package._TEMPLATE_ROOT, tmpl_copy)
    for pyc in tmpl_copy.rglob("*.pyc"):
        pyc.unlink()
    monkeypatch.setattr(package, "_TEMPLATE_ROOT", tmpl_copy)
    clean_files = package._template_static_files()
    assert not any(k.endswith(".pyc") or "__pycache__" in k for k in clean_files)
    pycache_dir = tmpl_copy / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)
    (pycache_dir / "autotrain.cpython-312.pyc").write_bytes(b"\x00\x01\x02\x03")
    after_pyc_files = package._template_static_files()
    assert clean_files == after_pyc_files


def test_repack_idempotency_acyclic_dag_frozen_config_and_web_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """覆盖审计 P1/P2：重打旧包不产生环路、配置变更触发 Dirty 但旧包回放恒定、Web 构建路由与步级开关 sid 精确绑定。"""
    import re
    import shutil
    import subprocess
    from fastapi.testclient import TestClient

    import server.app as app_mod
    from goldthread import lineage_graph
    from pipeline import stage2_gold

    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    cfg["export_profiles"].append(
        {
            "id": "hermes-json",
            "label": "Hermes JSON",
            "default": False,
            "tool_call_style": "hermes-json",
            "template": "qwen3",
            "tool_format": "hermes",
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(app_mod, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(app_mod, "CONFIG_PATH", cfg_path)

    src_dir = tmp_path / "src_cfg"
    sid1 = "01c3aa11-1111-7000-8000-000000000001"
    sid2 = "01c3bb22-2222-7000-8000-000000000002"
    long_out = "X" * 300
    _write_pi_session(
        src_dir / f"{sid1}.jsonl",
        session_id=sid1,
        cwd=r"D:\crack\com.demo.alpha",
        turns=[
            {
                "user": "完整逆向 alpha.apk。",
                "steps": [
                    {
                        "id": "shared_m1",
                        "thinking": "检查安装包结构与超长输出截断一致性。",
                        "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
                        "tool_results": [{"tool_call_id": "c1", "name": "bash", "content": long_out}],
                    },
                    {
                        "id": "m2",
                        "thinking": "读取配置头验证魔数结构。",
                        "tool_calls": [{"id": "c2", "name": "bash", "arguments": {"command": "head cfg"}}],
                        "tool_results": [{"tool_call_id": "c2", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "m3",
                        "thinking": "批量还原配置文件并核对数量。",
                        "tool_calls": [{"id": "c3", "name": "bash", "arguments": {"command": "python dec.py"}}],
                        "tool_results": [{"tool_call_id": "c3", "name": "bash", "content": "done"}],
                    },
                    {
                        "id": "m4",
                        "thinking": "全部文件核验通过后输出最终总结。",
                        "text": "alpha 完成。",
                    },
                ],
            }
        ],
    )
    _write_pi_session(
        src_dir / f"{sid2}.jsonl",
        session_id=sid2,
        cwd=r"D:\crack\com.demo.beta",
        turns=[
            {
                "user": "完整逆向 beta.apk。",
                "steps": [
                    {
                        "id": "shared_m1",
                        "thinking": "beta 第一步具有相同 msg_id 用于测试前端不串会话。",
                        "tool_calls": [{"id": "b1", "name": "bash", "arguments": {"command": "ls"}}],
                        "tool_results": [{"tool_call_id": "b1", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "bm2",
                        "thinking": "beta 第二步探查字节。",
                        "tool_calls": [{"id": "b2", "name": "bash", "arguments": {"command": "head b"}}],
                        "tool_results": [{"tool_call_id": "b2", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "bm3",
                        "thinking": "beta 第三步执行还原。",
                        "tool_calls": [{"id": "b3", "name": "bash", "arguments": {"command": "python b.py"}}],
                        "tool_results": [{"tool_call_id": "b3", "name": "bash", "content": "ok"}],
                    },
                    {
                        "id": "bm4",
                        "thinking": "beta 收尾核验完成。",
                        "text": "beta 完成。",
                    },
                ],
            }
        ],
    )

    curate_tools.adopt_sessions(src_dir, select=[sid1, sid2], reason="采纳会话", cfg=cfg)
    stage2_gold.run(cfg=cfg)
    curate_tools.curate_split(pin=["game::alpha=train", "game::beta=eval"], reason="固定切分", cfg=cfg)
    b1 = build_tools.build_pipeline(reason="构建 b1", cfg=cfg)
    assert b1["ok"] is True

    # 1) 打包 A (qwen-xml, 根) -> 打包 B (hermes-json, 子) -> 重打 A：DAG 必须保持 A 为根、无环！
    pa1 = build_tools.pack_delivery(build_id=b1["build_id"], profile="qwen-xml", reason="打包 A", cfg=cfg)
    pb = build_tools.pack_delivery(build_id=b1["build_id"], profile="hermes-json", reason="打包 B", cfg=cfg)
    pa2 = build_tools.pack_delivery(build_id=b1["build_id"], profile="qwen-xml", reason="重打 A 验证幂等", cfg=cfg)
    ha, hb = pa1["package_hash"], pb["package_hash"]
    assert pa2["generated_at"] == pa1["generated_at"]
    assert pa2["parent_pack"] is None

    graph = lineage_graph.build_dataset_graph(cfg=cfg)
    roots = [n["id"] for n in graph["nodes"] if n.get("resolved_parent") is None]
    assert ha in roots
    edge_pairs = {(e["from"], e["to"]) for e in graph["edges"]}
    assert not ((ha, hb) in edge_pairs and (hb, ha) in edge_pairs), f"检测到环路: {edge_pairs}"

    # 2) 冻结配置回放与配置变更 Dirty 检测：修改当前 max_tool_output_chars=50 后，旧包回放不变，工作区报 Dirty
    lin_before = lineage_graph.get_virtual_dataset_session_lineage(ha, sid1, cfg=cfg)
    cfg_modified = copy.deepcopy(cfg)
    cfg_modified["synth"]["max_tool_output_chars"] = 50
    lin_after = lineage_graph.get_virtual_dataset_session_lineage(ha, sid1, cfg=cfg_modified)
    assert lin_before["raw_messages"] == lin_after["raw_messages"]
    g_mod = lineage_graph.build_dataset_graph(cfg=cfg_modified)
    assert g_mod["workspace"]["dirty"] is True
    assert "synth_config_changed" in g_mod["workspace"]["reasons"]

    # 3) Web “1. 构建 Build” 按钮真实请求路径通过 TestClient 取得合法 job_id
    html = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    m_build_fn = re.search(r"async function buildWorkspaceFromGraph\(\) \{([\s\S]*?)\n\}", html)
    assert m_build_fn is not None
    m_url = re.search(r'api\("([^"]+)"', m_build_fn.group(1))
    assert m_url is not None and m_url.group(1) == "/api/run"
    assert re.search(r'JSON\.stringify\(\{\s*stage:\s*"build",\s*force:\s*false\s*\}\)', m_build_fn.group(1))
    client = TestClient(app_mod.app)
    r_build_btn = client.post(m_url.group(1), json={"stage": "build", "force": False})
    assert r_build_btn.status_code == 200
    assert r_build_btn.json().get("job_id")

    # 4) Node 直接执行前端 goldSetStepGate：首次打开虚拟数据集抽屉及 A->B 切换时均精确发往目标 sid
    if shutil.which("node"):
        script = "\n".join(re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html))
        m_gate = re.search(r"(async function goldSetStepGate\([\s\S]*?\n\})", script)
        assert m_gate is not None
        node_code = f"""
let GOLD_ACTIVE_SID = "";
let STATE = {{ tab: "overview", drawerData: null }};
let GOLD_VIEW = {{ sid: "", cache: {{}} }};
let _VDS_SESSION_CACHE = null;
const calls = [];
async function api(url, opts) {{ calls.push({{ url, body: JSON.parse(opts.body) }}); return {{ ok: true }}; }}
function toast() {{}}
async function openDatasetSessionDrawer() {{}}
async function openSessionLineageDrawer() {{}}
async function loadOverview() {{}}
{m_gate.group(1)}
(async () => {{
  // 首次打开会话 A（GOLD_ACTIVE_SID 尚未被旧页面设置）
  _VDS_SESSION_CACHE = {{ dataset_id: "workspace", session_id: "{sid1}", steps: [{{ msg_id: "shared_m1", excluded: false }}] }};
  await goldSetStepGate("shared_m1", "exclude", "{sid1}");
  // 切换到会话 B（即使 GOLD_ACTIVE_SID 残留为 sid1，显式传参和 _VDS_SESSION_CACHE 也必须发往 sid2）
  GOLD_ACTIVE_SID = "{sid1}";
  _VDS_SESSION_CACHE = {{ dataset_id: "workspace", session_id: "{sid2}", steps: [{{ msg_id: "shared_m1", excluded: true }}] }};
  await goldSetStepGate("shared_m1", "exclude", "{sid2}");
  console.log(JSON.stringify(calls));
}})().catch(e => {{ console.error(e); process.exit(1); }});
"""
        js_path = tmp_path / "step_gate_sid_test.js"
        js_path.write_text(node_code, encoding="utf-8")
        proc = subprocess.run(["node", str(js_path)], check=True, capture_output=True, text=True, encoding="utf-8")
        recorded = json.loads(proc.stdout.strip().splitlines()[-1])
        assert len(recorded) == 2
        assert f"/api/gold/session/{sid1}/step-gate" in recorded[0]["url"]
        assert recorded[0]["body"] == {"msg_id": "shared_m1", "action": "exclude", "reason": ""}
        assert f"/api/gold/session/{sid2}/step-gate" in recorded[1]["url"]
        assert recorded[1]["body"] == {"msg_id": "shared_m1", "action": "keep", "reason": ""}





