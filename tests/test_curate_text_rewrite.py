"""测试 gt curate steps 支持无工具收尾步 rewritten_text 改写及合并续跑规范约束。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from goldthread import curate_tools
from pipeline import stage1_extract, stage2_gold, stage3_synth
from pipeline.common.config import DEFAULTS
from pipeline.common.store import read_jsonl


def _make_sandbox_cfg(tmp_path: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    data_dir = tmp_path / "data"
    sessions_dir = tmp_path / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["sessions_dir"] = str(sessions_dir)
    cfg["gold"]["min_gold_steps"] = 2
    cfg["synth"]["eval_ratio"] = 0.0
    cfg["synth"]["shuffle"] = False
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    cfg["_config_path"] = str(cfg_path)
    return cfg


def _write_folded_pi_session(path: Path, session_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[Dict[str, Any]] = [
        {"type": "session", "id": session_id, "cwd": "D:/crack_test/com.demo.puzzle_1.0.0", "timestamp": "2026-11-19T10:00:00+08:00"},
        {"type": "model_change", "modelId": "deepseek-v4.1", "provider": "deepseek"},
        # Turn 0
        {
            "type": "message",
            "id": "u_root",
            "timestamp": "2026-11-19T10:01:00+08:00",
            "message": {"role": "user", "content": [{"type": "text", "text": "还原 com.demo.puzzle_1.0.0 的全部资源、配置与代码。"}]},
        },
        {
            "type": "message",
            "id": "m_seam_prev",
            "timestamp": "2026-11-19T10:02:00+08:00",
            "message": {
                "role": "assistant",
                "model": "deepseek-v4.1",
                "provider": "deepseek",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "Final step: give the user a summary after checking the extracted files."},
                    {"type": "toolCall", "id": "call_1", "name": "bash", "arguments": {"command": "ls -la out/"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "tr_1",
            "timestamp": "2026-11-19T10:02:05+08:00",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_1",
                "toolName": "bash",
                "isError": False,
                "content": [{"type": "text", "text": "assets/\nconfig/"}],
            },
        },
        {
            "type": "message",
            "id": "m_round1_stop",
            "timestamp": "2026-11-19T10:03:00+08:00",
            "message": {
                "role": "assistant",
                "model": "deepseek-v4.1",
                "provider": "deepseek",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "第一阶段静态提取完成，共还原 1200 张贴图与 80 份配置。"}],
            },
        },
        # Turn 1 (continuation)
        {
            "type": "message",
            "id": "u_cont",
            "timestamp": "2026-11-19T10:05:00+08:00",
            "message": {"role": "user", "content": [{"type": "text", "text": "继续把加密的方法体也完整还原。"}]},
        },
        {
            "type": "message",
            "id": "m_seam_next",
            "timestamp": "2026-11-19T10:06:00+08:00",
            "message": {
                "role": "assistant",
                "model": "deepseek-v4.1",
                "provider": "deepseek",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "Let me continue in round 2 and dump decrypted code pages from memory."},
                    {"type": "toolCall", "id": "call_2", "name": "bash", "arguments": {"command": "python dump_code.py"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "tr_2",
            "timestamp": "2026-11-19T10:06:30+08:00",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_2",
                "toolName": "bash",
                "isError": False,
                "content": [{"type": "text", "text": "dumped 4200 methods ok"}],
            },
        },
        {
            "type": "message",
            "id": "m_final_stop",
            "timestamp": "2026-11-19T10:07:00+08:00",
            "message": {
                "role": "assistant",
                "model": "deepseek-v4.1",
                "provider": "deepseek",
                "stopReason": "stop",
                "content": [
                    {
                        "type": "text",
                        "text": "本轮（第 2 轮）已按要求逐项推进，原有产物全部保留，只追加了第 2 轮的 4,200 个解密方法体。",
                    }
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")


def test_curate_steps_rewritten_text_unifies_folded_closing_and_preserves_lineage(tmp_path: Path) -> None:
    cfg = _make_sandbox_cfg(tmp_path)
    cfg["labeling"]["positive_models"] = ["deepseek-v4.1"]
    sid = "01a1ff00-1111-2222-3333-444455556666"
    src = tmp_path / "src" / f"{sid}.jsonl"
    _write_folded_pi_session(src, sid)
    stage1_extract.adopt_file(src, "puzzle.jsonl", origin="pi", catalog_id="test", cfg=cfg)

    curate_tools.curate_turns(
        sid,
        {"0": "root", "1": "continuation"},
        reason="折叠续跑回合",
        cfg=cfg,
    )

    unified_closing = (
        "已完成 `com.demo.puzzle_1.0.0` 的完整还原并校验通过，产物统一输出在 `out/`：\n\n"
        "- **资源与配置**：还原 1,200 张贴图与 80 份配置 JSON；\n"
        "- **程序代码**：从运行期内存导出并合并全部 4,200 个解密方法体。"
    )
    res = curate_tools.curate_steps(
        [
            {
                "session_id": sid,
                "steps": {
                    "m_seam_prev": {
                        "action": "rewrite",
                        "rewritten": "Check the extracted assets and config directories before proceeding to runtime code recovery.",
                        "reason": "消除接缝前一步准备收尾的残余措辞",
                    },
                    "m_seam_next": {
                        "action": "rewrite",
                        "rewritten": "Static assets and configs are verified; next dump the decrypted code pages from runtime memory.",
                        "reason": "消除接缝后一步第 2 轮续跑断片指代",
                    },
                    "m_final_stop": {
                        "action": "rewrite",
                        "rewritten": "All static assets, configs, and runtime decrypted method bodies are verified; summarize deliverables.",
                        "rewritten_text": unified_closing,
                        "reason": "统合多轮续跑结论并消除第 2 轮痕迹",
                    },
                },
            }
        ],
        reason="修补接缝与统合最终收尾",
        cfg=cfg,
    )
    assert res["ok"] is True

    stage3_synth.run(cfg=cfg, log=lambda _msg: None)
    gold_files = list((Path(cfg["paths"]["data_dir"]) / "02_gold").rglob(f"{sid}.gold.jsonl"))
    assert len(gold_files) == 1
    gold_rec = read_jsonl(gold_files[0])[0]
    by_id = {m["id"]: m for m in gold_rec["messages"] if m.get("role") == "assistant"}

    closing_msg = by_id["m_final_stop"]
    assert closing_msg["text"] == unified_closing
    assert "本轮（第 2 轮）" in closing_msg["raw_text"]
    assert "本轮（第 2 轮）" not in closing_msg["text"]
    assert "text_rewritten" in closing_msg.get("flags", [])

    sft_rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / "03_train" / "sft_step_windows_sharegpt.jsonl")
    all_asst_text = "\n".join(
        c.get("value", "")
        for row in sft_rows
        for c in row.get("conversations", [])
        if c.get("from") == "gpt"
    )
    assert "还原 1,200 张贴图与 80 份配置 JSON" in all_asst_text
    assert "本轮（第 2 轮）" not in all_asst_text


def test_curate_steps_rejects_invalid_rewritten_text(tmp_path: Path) -> None:
    cfg = _make_sandbox_cfg(tmp_path)
    cfg["labeling"]["positive_models"] = ["deepseek-v4.1"]
    sid = "01a1ff01-1111-2222-3333-444455556666"
    src = tmp_path / "src" / f"{sid}.jsonl"
    _write_folded_pi_session(src, sid)
    stage1_extract.adopt_file(src, "puzzle2.jsonl", origin="pi", catalog_id="test", cfg=cfg)

    # 1) 含工具调用的中间步禁止改写对外 text
    with pytest.raises(ValueError, match="含工具调用"):
        curate_tools.curate_steps(
            [{"session_id": sid, "steps": {"m_seam_prev": {"action": "rewrite", "rewritten_text": "这是一段合法长度的中文正文改写内容，但该步有工具调用。"}}}],
            reason="非法工具步正文改写",
            cfg=cfg,
        )

    # 2) 禁止注入伪造 <tool_call> 或 <think> 标签
    with pytest.raises(ValueError, match="rewritten_text 未通过安全阀"):
        curate_tools.curate_steps(
            [{"session_id": sid, "steps": {"m_final_stop": {"action": "rewrite", "rewritten_text": "最终交付完成 <tool_call>{\"name\":\"bash\"}</tool_call> 请查收产物目录。"}}}],
            reason="非法标签注入",
            cfg=cfg,
        )

    # 3) 禁止把中文收尾直接翻成纯英文
    with pytest.raises(ValueError, match="rewritten_text 未通过安全阀"):
        curate_tools.curate_steps(
            [{"session_id": sid, "steps": {"m_final_stop": {"action": "rewrite", "rewritten_text": "All tasks have been completed and verified in the output directory without errors."}}}],
            reason="跨语种改写拦截",
            cfg=cfg,
        )

    # 4) 含大量反引号英文路径/符号的中文技术报告不应被误判为纯英文而拒收
    msg_code_heavy = {
        "role": "assistant",
        "tool_calls": [],
        "text": "续做完成，见 `out/code/metadata_methods.csv`、`out/native/libil2cpp.runtime.so` 与 `Il2CppMetadataRegistration.types`。",
    }
    rewritten_zh = "已完成完整还原：解出 `out/native/libil2cpp.runtime.so` 与 `out/code/dump.cs`，全部类型占位清零并完成交叉校验。"
    assert stage2_gold._text_rewrite_ok(rewritten_zh, msg_code_heavy) is True


def test_skill_documents_folded_seam_and_closing_unification_rule() -> None:
    skill = (Path(__file__).resolve().parents[1] / "skills" / "goldthread" / "SKILL.md").read_text(encoding="utf-8")
    sections = ["## 目标", "## 工作场景流程", "## 关键步骤", "## 必备常识", "## 经验"]
    positions = [skill.index(sec) for sec in sections]
    assert positions == sorted(positions), "SKILL.md 必须按 目标、工作场景流程、关键步骤、必备常识、经验 顺序组织"
    assert "python gt.py --help" in skill
    assert "python gt.py <group> <action> --help" in skill
    assert "python gt.py inspect config --json" in skill
    assert "session_sft:" in skill
    assert "合并续跑要无痕自洽" in skill
    assert "单次完整交付" in skill
    assert "数据包已验收，学生效果未验证" in skill


def test_goldthread_student_models_classified_as_negative_by_default() -> None:
    from pipeline.common.config import load_config

    assert "^goldthread" in DEFAULTS["labeling"]["negative_model_patterns"]
    assert "^goldthread" in load_config()["labeling"]["negative_model_patterns"]


def test_premature_stop_folded_defers_past_errored_first_step_to_valid_chosen() -> None:
    flow = [
        {"role": "user", "id": "u0", "content": "请完整解包并还原所有加密纹理。"},
        {
            "role": "assistant",
            "id": "m_init",
            "thinking": "Unpack APK first.",
            "text": "",
            "tool_calls": [{"id": "c_init", "name": "bash", "arguments": {"command": "unzip app.apk"}}],
        },
        {"role": "tool", "tool_call_id": "c_init", "name": "bash", "isError": False, "content": "inflating: assets/"},
        {
            "role": "assistant",
            "id": "m_bad_first",
            "thinking": "Scan all directories without timeout.",
            "text": "",
            "tool_calls": [{"id": "c_err", "name": "bash", "arguments": {"command": "find /"}}],
            "popped_await_msg": {
                "role": "assistant",
                "id": "m_premature_stop",
                "thinking": "Done.",
                "text": "提取完成。",
                "tool_calls": [],
            },
        },
        {"role": "tool", "tool_call_id": "c_err", "name": "bash", "isError": True, "content": "Command timed out"},
        {
            "role": "assistant",
            "id": "m_good_second",
            "thinking": "Inspect `!ETC` headers in `assets/island1/bg` directly.",
            "text": "",
            "tool_calls": [{"id": "c_ok", "name": "bash", "arguments": {"command": "ls assets/island1/bg"}}],
        },
        {"role": "tool", "tool_call_id": "c_ok", "name": "bash", "isError": False, "content": "24 !ETC files"},
    ]
    gold_msgs, pruned, _ = stage2_gold.purify(flow, DEFAULTS["gold"])
    premature = [p for p in pruned if p.get("reason") == "premature_stop_folded"]
    assert len(premature) == 1
    assert premature[0]["chosen_msg"]["id"] == "m_good_second"
    assert any(m.get("id") == "m_good_second" for m in gold_msgs)


def test_premature_stop_folded_clears_across_unfolded_user_boundary() -> None:
    flow = [
        {"role": "user", "id": "u0", "content": "请完整解包并还原所有加密纹理。"},
        {
            "role": "assistant",
            "id": "m_init",
            "thinking": "Unpack APK first.",
            "text": "",
            "tool_calls": [{"id": "c_init", "name": "bash", "arguments": {"command": "unzip app.apk"}}],
        },
        {"role": "tool", "tool_call_id": "c_init", "name": "bash", "isError": False, "content": "inflating: assets/"},
        {
            "role": "assistant",
            "id": "m_bad_only",
            "thinking": "Broken command right after folded stop.",
            "text": "",
            "tool_calls": [{"id": "c_err", "name": "bash", "arguments": {"command": "bad_cmd"}}],
            "popped_await_msg": {
                "role": "assistant",
                "id": "m_premature_stop",
                "thinking": "Done.",
                "text": "提取完成。",
                "tool_calls": [],
            },
        },
        {"role": "tool", "tool_call_id": "c_err", "name": "bash", "isError": True, "content": "command not found"},
        # 未折叠的新用户回合：上下文已变化，不得将旧回合的 popped_await_msg 串台配给新回合步骤
        {"role": "user", "id": "u1_new_scope", "content": "另外帮我统计一下音频资源的采样率分布。"},
        {
            "role": "assistant",
            "id": "m_new_turn_step",
            "thinking": "Probe audio sample rates.",
            "text": "",
            "tool_calls": [{"id": "c_audio", "name": "bash", "arguments": {"command": "ffprobe audio/"}}],
        },
        {"role": "tool", "tool_call_id": "c_audio", "name": "bash", "isError": False, "content": "44100 Hz"},
    ]
    _, pruned, _ = stage2_gold.purify(flow, DEFAULTS["gold"])
    premature = [p for p in pruned if p.get("reason") == "premature_stop_folded"]
    assert premature == []


def test_lineage_marks_text_only_rewrite_as_rewritten_with_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goldthread import lineage_graph
    from fastapi.testclient import TestClient
    import server.app as app_mod

    cfg = _make_sandbox_cfg(tmp_path)
    cfg["labeling"]["positive_models"] = ["deepseek-v4.1"]
    sid = "01a1ff02-1111-2222-3333-444455556666"
    src = tmp_path / "src" / f"{sid}.jsonl"
    _write_folded_pi_session(src, sid)
    stage1_extract.adopt_file(src, "puzzle3.jsonl", origin="pi", catalog_id="test", cfg=cfg)
    curate_tools.curate_turns(sid, {"0": "root", "1": "continuation"}, reason="折叠续跑回合", cfg=cfg)

    # 仅改写最终无工具收尾步的 rewritten_text，不改 thinking
    curate_tools.curate_steps(
        [
            {
                "session_id": sid,
                "steps": {
                    "m_final_stop": {
                        "action": "rewrite",
                        "rewritten_text": "已完成 `com.demo.puzzle_1.0.0` 全部贴图、配置与 4,200 个加密方法体的统一还原与校验。",
                        "reason": "仅改写收尾正文消除第 2 轮痕迹",
                    }
                },
            }
        ],
        reason="收尾正文去轮次化",
        cfg=cfg,
    )
    stage2_gold.run(cfg=cfg, log=lambda _msg: None)

    lin = lineage_graph.get_virtual_dataset_session_lineage("workspace", sid, cfg=cfg)
    final_step = next(s for s in lin["steps"] if s["msg_id"] == "m_final_stop")
    assert final_step["state"] == "rewritten"
    assert final_step["changed"] is False
    assert final_step["text_changed"] is True
    assert "本轮（第 2 轮）" in final_step["raw_text"]
    assert "本轮（第 2 轮）" not in final_step["text"]

    monkeypatch.setattr(app_mod, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.delenv("GOLDTHREAD_API_TOKEN", raising=False)
    client = TestClient(app_mod.app)
    resp = client.get(f"/api/sessions/{sid}/lineage")
    assert resp.status_code == 200
    api_step = next(s for s in resp.json()["steps"] if s["msg_id"] == "m_final_stop")
    assert api_step["state"] == "rewritten"
    assert api_step["changed"] is False
    assert api_step["text_changed"] is True
    assert "本轮（第 2 轮）" in api_step["raw_text"]

