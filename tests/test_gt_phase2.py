"""GoldThread Agent-First Phase 2 自动化测试（完全隔离沙箱，绝不触碰真实 data/）。"""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

from goldthread import assist_tools, build_tools, cli, curate_tools, ledger
from pipeline import stage1_extract, stage2_gold, stage3_synth
from pipeline.common.config import DEFAULTS
from pipeline.common.format import default_tools_json
from pipeline.common.store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic


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


def _sample_turns_zh(game_hint: str) -> List[Dict[str, Any]]:
    return [
        {
            "user": f'<file name="prompt.txt">\n帮我破解 {game_hint}.xapk，还原全部资源和代码。\n</file>',
            "steps": [
                {
                    "id": f"m_{game_hint}_0",
                    "thinking": "先用脚本查看目录结构与 XAPK 内容，确认主包和配置文件位置。",
                    "tool_calls": [{"id": f"c_{game_hint}_0", "name": "bash", "arguments": {"command": "ls -la"}}],
                    "tool_results": [{"tool_call_id": f"c_{game_hint}_0", "content": "base.apk\nassets/"}],
                },
                {
                    "id": f"m_{game_hint}_1",
                    "thinking": ("等等，是不是哪里不对？重新考虑一下参数偏移。" * 15) + "再检查一遍文件头确认。",
                    "tool_calls": [{"id": f"c_{game_hint}_1", "name": "bash", "arguments": {"command": "head -c 16 base.apk"}}],
                    "tool_results": [{"tool_call_id": f"c_{game_hint}_1", "content": "504b0304"}],
                },
                {
                    "id": f"m_{game_hint}_2",
                    "thinking": "手滑执行了错误命令，等会需要剪除。",
                    "tool_calls": [{"id": f"c_{game_hint}_2", "name": "bash", "arguments": {"command": "badcmd_xyz"}}],
                    "tool_results": [{"tool_call_id": f"c_{game_hint}_2", "isError": True, "content": "command not found"}],
                },
                {
                    "id": f"m_{game_hint}_3",
                    "thinking": "使用 Python 脚本解析资产文件头并核对解密偏移量。",
                    "tool_calls": [
                        {"id": f"c_{game_hint}_3a", "name": "read", "arguments": {"path": "assets/cfg.json"}},
                        {"id": f"c_{game_hint}_3b", "name": "bash", "arguments": {"command": "python parse.py"}},
                    ],
                    "tool_results": [
                        {"tool_call_id": f"c_{game_hint}_3a", "content": "{}"},
                        {"tool_call_id": f"c_{game_hint}_3b", "content": "decoded ok"},
                    ],
                },
                {
                    "id": f"m_{game_hint}_4",
                    "thinking": "所有资产和代码逻辑已通过脚本实测验证完成，准备输出最终完整报告。",
                    "text": f"# {game_hint} 逆向完成\n已导出全部配置与脚本。",
                    "tool_calls": [],
                },
            ],
        }
    ]


# ---------------------------------------------------------------------------
# 1. 默认关闭隐式 LLM 与兼容性保护
# ---------------------------------------------------------------------------

def test_defaults_disable_implicit_llm_calls() -> None:
    assert DEFAULTS["turns"]["use_model"] is False
    assert DEFAULTS["judge"]["enabled"] is False
    from pipeline.common.config import load_config
    real_cfg = load_config()
    assert real_cfg["turns"]["use_model"] is False
    assert real_cfg["judge"]["enabled"] is False
    assert "review" not in yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    )


def test_stage2_preserves_existing_model_report_and_reuses_when_judge_disabled(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    src = tmp_path / "src" / "01a11111-1111-7111-8111-111111111111.jsonl"
    _write_pi_session(
        src,
        session_id="01a11111-1111-7111-8111-111111111111",
        cwd=r"D:\crack_test\teacher\com.oakever.jigsawcard_1.0",
        turns=_sample_turns_zh("jigsawcard"),
    )
    sid = stage1_extract.adopt_file(src, "jigsawcard.jsonl", origin="pi", catalog_id="test", cfg=cfg)

    # 先在 judge.enabled=True 下构造出历史黄金与 model_report
    cfg_on = copy.deepcopy(cfg)
    cfg_on["judge"]["enabled"] = True
    fake_report = {
        "verdict": "trainable",
        "confidence": 0.91,
        "reason": "完整逆向轨迹",
        "problems": [],
        "needs_wholesale_rewrite": False,
        "highlights": [{"msg_id": "m_jigsawcard_3", "why": "双工具并行解析", "weight": 3}],
    }
    with patch("pipeline.judge.decide_steps", return_value={
        "m_jigsawcard_1": {"action": "rewrite", "rewritten": "用脚本检查 ZIP 魔数头部字段。", "reason": "压缩反刍"},
    }), patch("pipeline.judge.audit_seams", return_value={}), patch(
        "pipeline.judge.audit_quality", return_value=fake_report
    ):
        stage2_gold.run(force=True, cfg=cfg_on, log=lambda *_: None)

    # 切回 judge.enabled=False 跑增量 stage2：必须复用而不推倒重算
    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM")):
        res_reuse = stage2_gold.run(force=False, cfg=cfg, log=lambda *_: None)
    assert res_reuse["reused"] == 1
    assert res_reuse["written"] == 0

    # 即使单条 force=True 重算（未提供新 model_report），必须保留 prev_ann 的 model_report 和历史改写
    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM")):
        res_recompute = stage2_gold.run(only=[sid], force=True, cfg=cfg, log=lambda *_: None)
    assert res_recompute["written"] == 1

    ann_file = Path(cfg["paths"]["data_dir"]) / "02_gold" / "annotations" / f"{sid}.json"
    ann = read_json(ann_file)
    assert ann["model_report"] == fake_report
    gold_file = Path(cfg["paths"]["data_dir"]) / "02_gold" / "sessions" / "game_jigsawcard" / f"{sid}.gold.jsonl"
    gold_rec = read_jsonl(gold_file)[0]
    assert gold_rec["model_report"] == fake_report
    m1 = next(m for m in gold_rec["messages"] if m.get("id") == "m_jigsawcard_1")
    assert m1["thinking"] == "用脚本检查 ZIP 魔数头部字段。"


# ---------------------------------------------------------------------------
# 2. curate adopt / drop-session / turns / steps / dpo / split / exclude
# ---------------------------------------------------------------------------

def test_curate_adopt_and_drop_session_with_ledger_and_cross_layer_cleanup(tmp_path: Path) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    teacher = tmp_path / "teacher"
    sid1 = "01a1d2ec-2fa8-729b-a5f3-ad8fbfb12caa"
    sid2 = "01a1bf57-e645-77d2-a686-a35402e131b5"
    _write_pi_session(
        teacher / "puzzle" / f"2026_{sid1}.jsonl",
        session_id=sid1,
        cwd=r"D:\crack_test\teacher\com.fugo.words_5.6.0\from_crack_puzzle",
        turns=_sample_turns_zh("puzzle"),
    )
    _write_pi_session(
        teacher / "riddle" / f"2026_{sid2}.jsonl",
        session_id=sid2,
        cwd=r"D:\crack_test\teacher\com.dopuz.klotski.riddle_1.62\from_crack_minimax",
        turns=_sample_turns_zh("riddle"),
    )

    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM")):
        rc = cli.main([
            "--config", str(cfg_path), "--json",
            "curate", "adopt",
            "--from", str(teacher),
            "--select", f"{sid1[:8]},{sid2[:8]}",
            "--reason", "采纳缺失游戏教师轨迹",
        ])
    assert rc == 0

    raw_idx = read_json(Path(cfg["paths"]["data_dir"]) / "01_raw" / "index.json")
    assert set(raw_idx["sessions"]) == {sid1, sid2}
    assert raw_idx["sessions"][sid1]["project_group"] == "game::puzzle"
    assert raw_idx["sessions"][sid2]["project_group"] == "game::riddle"
    # <file> 标签已被剥离
    assert "<file" not in raw_idx["sessions"][sid1]["stats"]["root_prompt"]

    # 写入 step_decisions & dpo_verdicts，验证 drop-session 会联动清理
    write_json_atomic(Path(cfg["paths"]["data_dir"]) / "00_state" / "step_decisions.json", {
        sid1: {"steps": {}},
        sid2: {"steps": {}},
    })
    write_json_atomic(Path(cfg["paths"]["data_dir"]) / "00_state" / "dpo_verdicts.json", {
        f"dpo::{sid1}::pruned0::x": {"verdict": "good"},
        f"dpo::{sid2}::pruned0::y": {"verdict": "good"},
    })

    rc_drop = cli.main([
        "--config", str(cfg_path), "--json",
        "curate", "drop-session", sid1[:8],
        "--reason", "清理测试桩会话",
    ])
    assert rc_drop == 0

    raw_after = read_json(Path(cfg["paths"]["data_dir"]) / "01_raw" / "index.json")
    assert set(raw_after["sessions"]) == {sid2}
    sd_after = read_json(Path(cfg["paths"]["data_dir"]) / "00_state" / "step_decisions.json")
    assert set(sd_after) == {sid2}
    dv_after = read_json(Path(cfg["paths"]["data_dir"]) / "00_state" / "dpo_verdicts.json")
    assert set(dv_after) == {f"dpo::{sid2}::pruned0::y"}

    events = ledger.read_ledger(cfg=cfg)
    assert [e["tool"] for e in events] == ["curate.drop_session", "curate.adopt"]


def test_curate_turns_updates_meta_and_rejects_invalid_kind(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    sid = "01a20001-0000-7000-8000-000000000001"
    src = tmp_path / "src" / f"{sid}.jsonl"
    turns = _sample_turns_zh("mahjong") + [
        {
            "user": "请继续深度还原脚本逻辑",
            "steps": [
                {
                    "id": "m_mahjong_extra",
                    "thinking": "继续深入分析核心脚本结构并输出完整解密代码。",
                    "tool_calls": [],
                    "text": "完整代码已导出。",
                }
            ],
        }
    ]
    _write_pi_session(src, session_id=sid, cwd=r"D:\crack\mahjong", turns=turns)
    curate_tools.adopt_sessions(tmp_path / "src", all_unadopted=True, reason="adopt", cfg=cfg)

    with pytest.raises(ValueError, match="非法回合类型"):
        curate_tools.curate_turns(sid[:8], {"1": {"kind": "bad_kind"}}, reason="bad", cfg=cfg)

    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM")):
        res = curate_tools.curate_turns(
            sid[:8],
            [{"index": 1, "kind": "continuation"}],
            reason="将第二回合折叠进主线",
            cfg=cfg,
        )
    assert res["ok"] is True
    assert res["manual_turn_overrides"]["turns"]["1"]["kind"] == "continuation"
    meta = read_json(Path(cfg["paths"]["data_dir"]) / "01_raw" / "index.json")["sessions"][sid]
    assert meta["turns"][1]["kind"] == "continuation"
    assert meta["turns"][1]["folded"] is True
    assert ledger.read_ledger(cfg=cfg, tool="curate.turns")[0]["affected"]["sids"] == [sid]


def test_curate_steps_safety_valves_and_deterministic_dpo_output(tmp_path: Path) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    sid = "01a30001-0000-7000-8000-000000000001"
    src = tmp_path / "src" / f"{sid}.jsonl"
    _write_pi_session(src, session_id=sid, cwd=r"D:\crack\jigsawcard", turns=_sample_turns_zh("jigsawcard"))
    curate_tools.adopt_sessions(tmp_path / "src", all_unadopted=True, reason="adopt", cfg=cfg)

    # 1) 首步 prune 拦截
    with pytest.raises(ValueError, match="不允许 prune"):
        curate_tools.curate_steps({
            "session_id": sid[:8],
            "steps": {"m_jigsawcard_0": {"action": "prune", "reason": "bad"}},
        }, reason="test", cfg=cfg)

    # 2) 超 35% prune 拦截 (5 步助手，max_prune = max(1, int(5*0.35)) = 1)
    with pytest.raises(ValueError, match="35%"):
        curate_tools.curate_steps({
            "session_id": sid[:8],
            "steps": {
                "m_jigsawcard_1": {"action": "prune", "reason": "p1"},
                "m_jigsawcard_2": {"action": "prune", "reason": "p2"},
            },
        }, reason="test", cfg=cfg)

    # 3) 中翻英 rewrite 拦截
    with pytest.raises(ValueError, match="安全阀校验"):
        curate_tools.curate_steps({
            "session_id": sid[:8],
            "steps": {
                "m_jigsawcard_1": {
                    "action": "rewrite",
                    "rewritten": "Check the ZIP magic bytes at offset zero using python script.",
                    "reason": "translated",
                },
            },
        }, reason="test", cfg=cfg)

    # 确认拦截后绝未写入脏状态
    assert not stage2_gold.step_decisions_path(cfg).exists()

    # 4) 合法 rewrite + prune + exclude + model_report
    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({
        "session_id": sid[:8],
        "model_report": {
            "verdict": "trainable",
            "confidence": 0.95,
            "reason": "主模型审核通过",
            "highlights": [{"msg_id": "m_jigsawcard_3", "why": "并行工具调用", "weight": 3}],
        },
        "steps": {
            "m_jigsawcard_1": {
                "action": "rewrite",
                "rewritten": "通过脚本读取安装包前十六字节，核对压缩包头部魔数并确认结构位置。",
                "reason": "压缩冗长反刍",
            },
            "m_jigsawcard_2": {"action": "prune", "reason": "错误命令弯路"},
            "m_jigsawcard_3": {"action": "exclude", "reason": "测试排除标记"},
        },
    }, ensure_ascii=False), encoding="utf-8")

    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM")):
        rc = cli.main([
            "--config", str(cfg_path), "--json",
            "curate", "steps",
            "--file", str(decisions_file),
            "--reason", "主模型步级策展",
        ])
    assert rc == 0

    gold_idx = read_json(Path(cfg["paths"]["data_dir"]) / "02_gold" / "index.json")
    gold_rec = read_jsonl(Path(gold_idx["sessions"][sid]["gold_file"]))[0]
    by_id = {m["id"]: m for m in gold_rec["messages"] if m.get("role") == "assistant"}
    assert "m_jigsawcard_2" not in by_id  # pruned
    assert by_id["m_jigsawcard_1"]["thinking"] == "通过脚本读取安装包前十六字节，核对压缩包头部魔数并确认结构位置。"
    assert by_id["m_jigsawcard_3"]["excluded"] is True

    # 还原 m_jigsawcard_3 为 keep 以允许进入 SFT 并生成完整 DPO
    curate_tools.curate_steps({
        "session_id": sid,
        "steps": {"m_jigsawcard_3": {"action": "keep", "reason": "恢复纳入训练"}},
    }, reason="恢复 step 3", cfg=cfg)

    records = read_json(Path(cfg["paths"]["data_dir"]) / "02_gold" / "index.json")["sessions"]
    dpo_pairs, _ = stage3_synth.synthesize_dpo_candidates(records, cfg, {"game::jigsawcard": "train"})
    sources = {p["metadata"]["source"] for p in dpo_pairs}
    assert "stage2_pruned" in sources
    assert "review_rewrite" in sources


def test_curate_dpo_split_exclude_and_zero_llm_e2e_build_gate_pack(tmp_path: Path) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    teacher = tmp_path / "teacher"
    sid_tr = "01a40001-0000-7000-8000-000000000001"
    sid_ev = "01a40002-0000-7000-8000-000000000002"
    _write_pi_session(
        teacher / "jigsawcard.jsonl",
        session_id=sid_tr,
        cwd=r"D:\crack_test\teacher\com.oakever.jigsawcard_1.0",
        turns=_sample_turns_zh("jigsawcard"),
    )
    _write_pi_session(
        teacher / "klondike.jsonl",
        session_id=sid_ev,
        cwd=r"D:\crack_test\teacher\com.oakever.klondike_1.0",
        turns=_sample_turns_zh("klondike"),
    )

    with patch("pipeline.common.llm.chat_json", side_effect=AssertionError("禁止隐式 LLM 请求")):
        # 1) adopt
        curate_tools.adopt_sessions(teacher, all_unadopted=True, reason="采纳两条游戏轨迹", cfg=cfg)

        # 2) steps 策展（为 train 生成 review_rewrite + stage2_pruned DPO）
        curate_tools.curate_steps({
            "sessions": {
                sid_tr: {
                    "steps": {
                        "m_jigsawcard_1": {
                            "action": "rewrite",
                            "rewritten": "通过脚本读取安装包前十六字节，核对压缩包魔数偏移。",
                            "reason": "压缩反刍",
                        },
                        "m_jigsawcard_2": {"action": "prune", "reason": "错误命令弯路"},
                    }
                },
                sid_ev: {
                    "steps": {
                        "m_klondike_1": {
                            "action": "rewrite",
                            "rewritten": "通过脚本读取安装包前十六字节，核对压缩包魔数偏移。",
                            "reason": "压缩反刍",
                        },
                    }
                },
            }
        }, reason="清洗反刍与弯路", cfg=cfg)

        # 3) split 锁定
        split_res = curate_tools.curate_split(
            pin=["game::jigsawcard=train", "game::klondike=eval"],
            reason="固定 train/eval 项目切分",
            cfg=cfg,
        )
        assert split_res["split_plan"]["assignments"]["game::jigsawcard"] == "train"
        assert split_res["split_plan"]["assignments"]["game::klondike"] == "eval"

        # 4) dpo 裁决：一条 good，一条 swap，一条 drop
        records = read_json(Path(cfg["paths"]["data_dir"]) / "02_gold" / "index.json")["sessions"]
        cands, _ = stage3_synth.synthesize_dpo_candidates(
            records, cfg, split_res["split_plan"]["assignments"]
        )
        assert len(cands) >= 3
        c0, c1, c2 = cands[0], cands[1], cands[2]
        orig_c1_rejected = c1["rejected"]["value"]
        curate_tools.curate_dpo({
            c0["sample_id"]: {"verdict": "good", "reason": "有效负例"},
            c1["sample_id"]: {"verdict": "swap", "reason": "偏好对调"},
            c2["sample_id"]: {"verdict": "drop", "reason": "无效丢弃"},
        }, reason="主模型 DPO 复核裁决", cfg=cfg)

        # 5) build --dry-run 不落盘 03_train / 04_release / ledger
        ledger_before = len(ledger.read_ledger(cfg=cfg))
        dry_res = build_tools.build_pipeline(dry_run=True, cfg=cfg)
        assert dry_res["dry_run"] is True
        assert not (Path(cfg["paths"]["data_dir"]) / "04_release" / "builds" / "LATEST_BUILD.json").exists()
        assert len(ledger.read_ledger(cfg=cfg)) == ledger_before

        # 6) 正式 gt build（零 LLM）
        assert cli.main([
            "--config", str(cfg_path), "--json",
            "build", "--reason", "纯确定性构建 v1",
        ]) == 0

        dpo_built = read_jsonl(Path(cfg["paths"]["data_dir"]) / "03_train" / "dpo_preference_pairs.jsonl")
        by_sid = {r["sample_id"]: r for r in dpo_built}
        assert c0["sample_id"] in by_sid
        assert by_sid[c0["sample_id"]]["metadata"]["model_review"] == "good"
        assert c1["sample_id"] in by_sid
        assert by_sid[c1["sample_id"]]["metadata"]["model_review"] == "swap"
        assert by_sid[c1["sample_id"]]["chosen"]["value"] == orig_c1_rejected
        assert c2["sample_id"] not in by_sid

        # 7) gate check latest -> 0
        assert cli.main(["--config", str(cfg_path), "--json", "gate", "check", "latest"]) == 0

        # 8) pack + copy_to + gate
        copy_dir = tmp_path / "exported"
        assert cli.main([
            "--config", str(cfg_path), "--json",
            "pack", "--copy-to", str(copy_dir), "--reason", "打包交付包",
        ]) == 0
        zips = list(copy_dir.glob("goldthread-train-*.zip"))
        assert len(zips) == 1
        assert cli.main(["--config", str(cfg_path), "--json", "gate", "check", str(zips[0])]) == 0

        # 9) exclude & restore 校验
        excl_sample = f"step::{sid_tr}::m_jigsawcard_0"
        curate_tools.curate_exclude(samples=[excl_sample], reason="临时排除测试", cfg=cfg)
        prev_excl = stage3_synth.draft_preview(cfg)
        row_excl = next(r for r in prev_excl["rows"] if r["sample_id"] == excl_sample)
        assert row_excl["included"] is False

        curate_tools.curate_exclude(samples=[excl_sample], restore=True, reason="恢复样本", cfg=cfg)
        prev_rest = stage3_synth.draft_preview(cfg)
        row_rest = next(r for r in prev_rest["rows"] if r["sample_id"] == excl_sample)
        assert row_rest["included"] is True


# ---------------------------------------------------------------------------
# 3. gate check 拦截异常包（早停 / <file> 标签泄漏 / length_gate 跳过）
# ---------------------------------------------------------------------------

def test_gate_check_fails_with_exit_code_2_on_polluted_pack(tmp_path: Path) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    bad_zip = tmp_path / "bad-pack.zip"
    sft_train_rows = [
        {
            "sample_id": "step::s1::m1",
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": '<file name="a.txt">破解 a.apk</file>'},
                {
                    "from": "gpt",
                    "value": "<think>\n阶段总结一下，请问是否继续？\n</think>\n已完成初步拆包，请指示。",
                },
            ],
            "tools": default_tools_json(),
            "metadata": {
                "sample_id": "step::s1::m1",
                "session_id": "s1",
                "project_group": "bad_proj_name",
                "step_rank": 1,
                "total_steps": 3,
                "stage": "routine",
            },
        }
    ]
    manifest = {
        "package_hash": "deadbeef",
        "format": {"tool_call": "coder-xml", "train_template": "qwen3_8"},
        "length_gate": {"_skip": "no_tokenizer"},
    }
    with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("goldthread-train/data/MANIFEST.json", json.dumps(manifest, ensure_ascii=False))
        zf.writestr("goldthread-train/data/sft.train.lf.jsonl", json.dumps(sft_train_rows[0], ensure_ascii=False) + "\n")
        zf.writestr("goldthread-train/data/sft.eval.lf.jsonl", "")

    gate = build_tools.gate_check(str(bad_zip), cfg=cfg)
    assert gate["passed"] is False
    codes = {v["code"] for v in gate["violations"]}
    assert "sft_non_empty" in codes
    assert "zero_premature_stop" in codes
    assert "zero_prompt_tag_leaks" in codes
    assert "canonical_project_groups" in codes
    assert "length_gate_effective" in codes

    assert cli.main(["--config", str(cfg_path), "--json", "gate", "check", str(bad_zip)]) == 2


@pytest.mark.parametrize(("target", "target_role", "error_code"), [
    ("<think>Confirm the measured total before reporting it.</think>Done.", "gpt", None),
    ("<think>Confirm the total.</think><think>Double-check it.</think>Done.", "gpt", "invalid_target_think_tags"),
    ("</think>Confirm the measured total.<think>Done.", "gpt", "invalid_target_think_tags"),
    ("<think>Confirm the measured total. Done.", "gpt", "invalid_target_think_tags"),
    ("<think>Confirm the total.</think><think>Double-check it.</think>Done.", "assistant", "missing_assistant_target"),
])
def test_gate_checks_exactly_one_ordered_think_pair_in_sft_target(
    tmp_path: Path, target: str, target_role: str, error_code: str | None,
) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    path = tmp_path / "think-pairs.zip"
    manifest = {
        "format": {"tool_call": "coder-xml"},
        "length_gate": {"sft.train": {}, "sft.eval": {}},
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("goldthread-train/data/MANIFEST.json", json.dumps(manifest))
        for split in ("train", "eval"):
            row = {
                "sample_id": f"step::{split}::m1",
                "metadata": {"project_group": f"game::{split}"},
                "conversations": [
                    {"from": "system", "value": "Report measurements."},
                    {"from": "human", "value": "What is the baseline?"},
                    {"from": "gpt", "value": "<think>Check the baseline.</think>Baseline is 5."},
                    {"from": "human", "value": "Confirm the total."},
                    {"from": target_role, "value": target} if split == "train" else
                    {"from": "gpt", "value": "<think>Confirm the measured total.</think>Total is 7."},
                ],
            }
            zf.writestr(f"goldthread-train/data/sft.{split}.lf.jsonl", json.dumps(row) + "\n")

    gate = build_tools.gate_check(path, cfg=cfg)
    assert gate["passed"] is (error_code is None), gate["violations"]
    assert cli.main(["--config", str(cfg_path), "gate", "check", str(path), "--json"]) == (0 if error_code is None else 2)
    if error_code:
        violations = {item["code"]: item for item in gate["violations"]}
        assert error_code in violations
        if error_code == "invalid_target_think_tags":
            example = violations[error_code]["examples"][0]
            assert example["dataset"] == "sft.train"
            assert example["row_number"] == 1
            assert example["sample_id"] == "step::train::m1"


# ---------------------------------------------------------------------------
# 4. gt assist turns / judge / dpo 显式调用与 --apply 留痕
# ---------------------------------------------------------------------------

def test_assist_turns_judge_dpo_with_mock_llm(tmp_path: Path) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    sid = "01a50001-0000-7000-8000-000000000001"
    src = tmp_path / "src" / f"{sid}.jsonl"
    _write_pi_session(src, session_id=sid, cwd=r"D:\crack\jigsawcard", turns=_sample_turns_zh("jigsawcard"))
    curate_tools.adopt_sessions(tmp_path / "src", all_unadopted=True, reason="adopt", cfg=cfg)

    fake_turns = {
        "root_turn": 0,
        "turns": [{"index": 0, "kind": "root", "drop": False, "why": "破解主任务"}],
        "notes": "ok",
    }
    fake_judge_steps = {
        "decisions": [
            {
                "msg_id": "m_jigsawcard_1",
                "action": "rewrite",
                "reason": "压缩反刍",
                "rewritten": "读取 base.apk 前 16 字节验证文件头。",
            }
        ]
    }
    fake_quality = {
        "verdict": "trainable",
        "confidence": 0.92,
        "reason": "闭环完整",
        "problems": [],
        "needs_wholesale_rewrite": False,
        "highlights": [{"msg_id": "m_jigsawcard_3", "why": "并行核验", "weight": 3}],
        "opening_msg_id": "m_jigsawcard_0",
        "closing_msg_id": "m_jigsawcard_4",
    }
    fake_dpo_out = {
        "pairs": [{"sample_id": f"dpo_review::{sid}::m_jigsawcard_1", "verdict": "good", "reason": "压缩更优"}]
    }

    def _fake_chat(system: str, user: str, **_kwargs) -> Dict[str, Any]:
        if "回合" in system:
            return fake_turns
        if "判断员" in system:
            return fake_judge_steps
        if "审计员" in system:
            return fake_quality
        if "偏好数据复核员" in system:
            return fake_dpo_out
        return {}

    with patch("pipeline.turnclassify.chat_json", side_effect=_fake_chat), patch(
        "pipeline.judge.chat_json", side_effect=_fake_chat
    ):
        # assist turns --apply
        assert cli.main([
            "--config", str(cfg_path), "--json",
            "assist", "turns", sid[:8], "--apply", "--reason", "初筛回合",
        ]) == 0
        meta = read_json(Path(cfg["paths"]["data_dir"]) / "01_raw" / "index.json")["sessions"][sid]
        assert meta["turn_annotations"]["root_turn"] == 0

        # assist judge --apply
        assert cli.main([
            "--config", str(cfg_path), "--json",
            "assist", "judge", sid[:8], "--apply", "--reason", "初筛步级",
        ]) == 0
        sd = stage2_gold.load_step_decisions(cfg)
        assert sd[sid]["steps"]["m_jigsawcard_1"]["action"] == "rewrite"
        assert sd[sid]["model_report"]["verdict"] == "trainable"

        # assist dpo
        assert cli.main([
            "--config", str(cfg_path), "--json",
            "assist", "dpo", "--reason", "初筛 DPO",
        ]) == 0

    tools_logged = [e["tool"] for e in ledger.read_ledger(cfg=cfg)]
    assert "assist.turns" in tools_logged
    assert "assist.judge" in tools_logged
    assert "assist.dpo" in tools_logged
