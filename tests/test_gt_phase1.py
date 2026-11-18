"""GoldThread Agent-First Phase 1 自动化测试（完全隔离临时目录，不触碰真实 data/）。"""
from __future__ import annotations

import copy
import json
import re
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from goldthread import cli, eval_tools, inspect_tools, ledger, report_tools
from pipeline.common.config import DEFAULTS
from pipeline.common.format import default_tools_json
from pipeline.common.store import write_json_atomic, write_jsonl_atomic


def _make_sandbox_cfg(tmp_path: Path) -> tuple[Dict[str, Any], Path]:
    cfg = copy.deepcopy(DEFAULTS)
    data_dir = tmp_path / "data"
    sessions_dir = tmp_path / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["sessions_dir"] = str(sessions_dir)
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
    extra_errors: List[Dict[str, Any]] | None = None,
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
                        "id": tc.get("id") or f"call_{t_idx}_{s_idx}_{tc_idx}",
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
                        "id": f"tr_{t_idx}_{s_idx}_{tr.get('tool_call_id', '0')}",
                        "timestamp": "2026-11-18T10:02:05+08:00",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": tr.get("tool_call_id") or "",
                            "toolName": tr.get("name") or "bash",
                            "isError": bool(tr.get("isError", False)),
                            "content": [{"type": "text", "text": tr.get("content", "ok")}],
                        },
                    }
                )
    for err in extra_errors or []:
        lines.append({"type": "message", "id": err.get("id", "err_1"), "message": err})
    path.write_text(
        "\n".join(json.dumps(obj, ensure_ascii=False) for obj in lines) + "\n",
        encoding="utf-8",
    )


def _seed_corpus(tmp_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = Path(cfg["paths"]["data_dir"])
    sid1 = "01a10001-1111-4000-8000-000000000001"
    sid2 = "01a10002-2222-4000-8000-000000000002"

    raw1 = data_dir / "01_raw" / "positive" / "game_jigsawcard" / f"{sid1}.jsonl"
    meta1 = data_dir / "01_raw" / "positive" / "game_jigsawcard" / f"{sid1}.meta.json"
    gold1 = data_dir / "02_gold" / "sessions" / "game_jigsawcard" / f"{sid1}.gold.jsonl"
    ann1 = data_dir / "02_gold" / "annotations" / f"{sid1}.json"
    dpo1 = data_dir / "02_gold" / "rejects" / f"{sid1}.dpo.jsonl"

    raw1.parent.mkdir(parents=True, exist_ok=True)
    raw1.write_text('{"type":"session","id":"' + sid1 + '"}\n', encoding="utf-8")
    meta1_obj = {
        "session_id": sid1,
        "label": "positive",
        "project_group": "game::jigsawcard",
        "cwd": "D:\\crack\\jigsawcard",
        "timestamp": "2026-11-18T10:00:00+08:00",
        "models": {"glm-5.3-flash": 3},
        "source": {"rel_path": "jigsaw/s1.jsonl", "abs_path": str(raw1)},
        "stats": {
            "assistant_steps": 3,
            "turns_total": 2,
            "turns_kept": 2,
            "turns_folded": 1,
            "root_prompt": "破解 jigsawcard.xapk",
            "raw_root_prompt": "<file name=\"p.txt\">破解 jigsawcard.xapk</file>",
        },
        "turns": [
            {
                "turn_index": 0,
                "kind": "root",
                "action": "kept",
                "dropped": False,
                "folded": False,
                "reason": "root_task",
                "assistant_steps": 2,
                "kept_steps": 2,
                "text_preview": "破解 jigsawcard.xapk",
            },
            {
                "turn_index": 1,
                "kind": "continuation",
                "action": "folded",
                "dropped": False,
                "folded": True,
                "reason": "folded_into_mainline",
                "assistant_steps": 1,
                "kept_steps": 1,
                "text_preview": "继续",
            },
        ],
        "pathologies": {"over_rumination_length": 1},
        "raw_file": str(raw1),
        "meta_file": str(meta1),
    }
    write_json_atomic(meta1, meta1_obj)

    long_cmd = "echo " + ("x" * 750)
    long_out = "OUTPUT:" + ("y" * 950)
    gold_msgs = [
        {"id": f"sys_{sid1}", "role": "system", "content": "sys"},
        {"id": f"usr_{sid1}", "role": "user", "content": "破解 jigsawcard.xapk"},
        {
            "id": "aa11bb22",
            "role": "assistant",
            "turn_index": 0,
            "thinking": "压缩后的思考：检查目录结构。",
            "raw_thinking": "原始超长思考：Wait, actually let me check " + ("abc " * 30),
            "text": "",
            "raw_text": "我先看看目录",
            "flags": ["silent_executed", "thinking_compressed"],
            "judge_reason": "压缩冗长反刍",
            "folded_from": "aa11bb21",
            "fold_details": {"path": "run.py", "edits_count": 1},
            "excluded": False,
            "exclude_reason": "",
            "raw_quality": {"is_clean": False, "score": 70.0, "pathologies": ["over_rumination_length"]},
            "quality": {"is_clean": True, "score": 100.0, "pathologies": []},
            "tool_calls": [
                {"id": "tc_1", "name": "bash", "arguments": {"command": long_cmd}},
            ],
            "tool_results": [
                {"tool_call_id": "tc_1", "name": "bash", "isError": False, "raw_len": len(long_out), "content": long_out},
            ],
        },
        {
            "id": "cc33dd44",
            "role": "assistant",
            "turn_index": 1,
            "thinking": "已完成全部资产与配置解密，准备输出最终总结。",
            "raw_thinking": "已完成全部资产与配置解密，准备输出最终总结。",
            "text": "破解完成，已全部还原。",
            "raw_text": "破解完成，已全部还原。",
            "flags": [],
            "judge_reason": "",
            "excluded": False,
            "exclude_reason": "",
            "raw_quality": {"is_clean": True, "score": 100.0, "pathologies": []},
            "quality": {"is_clean": True, "score": 100.0, "pathologies": []},
            "tool_calls": [],
            "tool_results": [],
        },
    ]
    write_jsonl_atomic(
        gold1,
        [
            {
                "gold_id": f"gold::{sid1}",
                "session_id": sid1,
                "label": "positive",
                "project_group": "game::jigsawcard",
                "cwd": "D:\\crack\\jigsawcard",
                "dominant_model": "glm-5.3-flash",
                "stats": {
                    "raw_steps": 3,
                    "gold_steps": 2,
                    "turns_folded": 1,
                    "root_prompt": "破解 jigsawcard.xapk",
                    "raw_root_prompt": "<file name=\"p.txt\">破解 jigsawcard.xapk</file>",
                },
                "messages": gold_msgs,
                "model_report": {
                    "verdict": "trainable",
                    "reason": "主线清晰",
                    "problems": [],
                    "highlights": [{"msg_id": "aa11bb22", "weight": 3}],
                    "opening_msg_id": "aa11bb22",
                    "closing_msg_id": "cc33dd44",
                },
            }
        ],
    )
    write_json_atomic(
        ann1,
        {
            "session_id": sid1,
            "label": "positive",
            "status": "gold",
            "model_report": {
                "verdict": "trainable",
                "confidence": 0.9,
                "reason": "主线清晰",
                "problems": [],
                "highlights": [{"msg_id": "aa11bb22", "weight": 3}],
                "opening_msg_id": "aa11bb22",
                "closing_msg_id": "cc33dd44",
            },
        },
    )
    dpo_pair = {
        "sample_id": f"dpo_review::{sid1}::aa11bb22",
        "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "破解 jigsawcard.xapk"}],
        "chosen": {"from": "gpt", "value": "<think>\n压缩后的思考：检查目录结构。\n</think>"},
        "rejected": {"from": "gpt", "value": "<think>\n原始超长思考\n</think>"},
        "metadata": {
            "source": "review_rewrite",
            "reason": "over_rumination_length",
            "session_id": sid1,
            "project_group": "game::jigsawcard",
            "msg_id": "aa11bb22",
        },
    }
    write_jsonl_atomic(dpo1, [dpo_pair])
    write_jsonl_atomic(data_dir / "02_gold" / "dpo_pairs.jsonl", [dpo_pair])

    # 第二个项目用于过滤测试
    meta2_obj = {
        "session_id": sid2,
        "label": "positive",
        "project_group": "game::meowdoku",
        "cwd": "D:\\crack\\meowdoku",
        "models": {"glm-5.3-flash": 2},
        "stats": {"assistant_steps": 2, "turns_folded": 0, "root_prompt": "破解 meowdoku"},
        "turns": [],
    }
    write_json_atomic(
        data_dir / "01_raw" / "index.json",
        {"generated_at": "2026-11-18T10:00:00+08:00", "sessions": {sid1: meta1_obj, sid2: meta2_obj}},
    )
    write_json_atomic(
        data_dir / "02_gold" / "index.json",
        {
            "generated_at": "2026-11-18T10:00:00+08:00",
            "sessions": {
                sid1: {
                    "session_id": sid1,
                    "label": "positive",
                    "status": "gold",
                    "project_group": "game::jigsawcard",
                    "dominant_model": "glm-5.3-flash",
                    "stats": {"raw_steps": 3, "gold_steps": 2, "turns_folded": 1},
                    "dpo_count": 1,
                    "pathologies": {"over_rumination_length": 1},
                    "gold_file": str(gold1),
                    "dpo_file": str(dpo1),
                    "annotation_file": str(ann1),
                    "model_report": {"verdict": "trainable", "reason": "主线清晰"},
                },
                sid2: {
                    "session_id": sid2,
                    "label": "positive",
                    "status": "gold",
                    "project_group": "game::meowdoku",
                    "dominant_model": "glm-5.3-flash",
                    "stats": {"raw_steps": 2, "gold_steps": 2, "turns_folded": 0},
                    "dpo_count": 0,
                    "pathologies": {},
                },
            },
        },
    )
    write_json_atomic(
        data_dir / "03_train" / "stats.json",
        {
            "counts": {
                "sft_gold_trajectories_sharegpt.jsonl": 2,
                "sft_step_windows_sharegpt.jsonl": 4,
                "sft_step_windows_sharegpt.train.jsonl": 2,
                "sft_step_windows_sharegpt.eval.jsonl": 2,
                "dpo_preference_pairs.jsonl": 1,
                "dpo_preference_pairs.train.jsonl": 1,
                "dpo_preference_pairs.eval.jsonl": 0,
            },
            "split": {"game::jigsawcard": "train", "game::meowdoku": "eval"},
            "weighted_steps": 6,
            "session_funnels": {
                sid1: {
                    "session_id": sid1,
                    "project_group": "game::jigsawcard",
                    "split": "train",
                    "effective_state": "auto_gold",
                    "funnel": {"admitted_steps": 2, "gold_steps": 2, "raw_steps": 3},
                    "dpo_written": 1,
                },
                sid2: {
                    "session_id": sid2,
                    "project_group": "game::meowdoku",
                    "split": "eval",
                    "effective_state": "auto_gold",
                    "funnel": {"admitted_steps": 2, "gold_steps": 2, "raw_steps": 2},
                    "dpo_written": 0,
                },
            },
        },
    )
    write_json_atomic(
        data_dir / "04_release" / "builds" / "LATEST_BUILD.json",
        {"build_id": "abcdef0123456789"},
    )
    return {"sid1": sid1, "sid2": sid2}


def _build_test_zip(
    zip_path: Path,
    *,
    sft_train: List[Dict[str, Any]],
    sft_eval: List[Dict[str, Any]],
    dpo_train: List[Dict[str, Any]],
    dpo_eval: List[Dict[str, Any]],
    template: str = "qwen3_8",
    tool_style: str = "coder-xml",
    package_hash: str = "11223344",
    build_id: str = "abcdef0123456789",
    length_gate: Dict[str, Any] | None = None,
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package_hash": package_hash,
        "build_id": build_id,
        "format": {"tool_call": tool_style, "train_template": template},
        "length_gate": length_gate or {},
    }
    settings_yaml = f"project:\n  template: {template}\n"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("goldthread-train/settings.yaml", settings_yaml)
        zf.writestr("goldthread-train/data/MANIFEST.json", json.dumps(manifest, ensure_ascii=False))
        for fname, rows in (
            ("sft.train.lf.jsonl", sft_train),
            ("sft.eval.lf.jsonl", sft_eval),
            ("dpo.train.lf.jsonl", dpo_train),
            ("dpo.eval.lf.jsonl", dpo_eval),
        ):
            payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            zf.writestr(f"goldthread-train/data/{fname}", payload)
    return zip_path


# ---------------------------------------------------------------------------
# 1. ledger tests
# ---------------------------------------------------------------------------

def test_ledger_record_and_filter(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    _seed_corpus(tmp_path, cfg)

    snap = ledger.corpus_snapshot_summary(cfg)
    assert snap["raw_sessions"] == 2
    assert snap["gold_sessions"] == 2
    assert snap["projects"] == ["game::jigsawcard", "game::meowdoku"]
    assert snap["latest_build_id"] == "abcdef0123456789"

    ev1 = ledger.record_operation(
        "curation.decide",
        reason="人工复核通过",
        actor="human",
        affected={"sids": ["01a10001-1111-4000-8000-000000000001"], "projects": ["game::jigsawcard"]},
        cfg=cfg,
    )
    ev2 = ledger.record_operation(
        "report.publish",
        reason="发布评估报告",
        actor="agent",
        affected={"build_id": "abcdef0123456789", "projects": ["game::meowdoku"]},
        cfg=cfg,
    )
    assert re.fullmatch(r"op_\d{8}_\d{6}_[0-9a-f]{4}", ev1["op_id"])

    all_events = ledger.read_ledger(cfg=cfg)
    assert [e["op_id"] for e in all_events] == [ev2["op_id"], ev1["op_id"]]
    assert len(ledger.read_ledger(cfg=cfg, sid="01a10001")) == 1
    assert len(ledger.read_ledger(cfg=cfg, project="jigsawcard")) == 1
    assert len(ledger.read_ledger(cfg=cfg, tool="report.publish")) == 1
    assert len(ledger.read_ledger(cfg=cfg, build_id="abcdef01")) == 1
    assert len(ledger.read_ledger(cfg=cfg, actor="human")) == 1
    assert len(ledger.read_ledger(cfg=cfg, limit=1)) == 1

    # 并发写入安全性
    errs: List[Exception] = []

    def _worker(i: int) -> None:
        try:
            ledger.record_operation("test.concurrent", reason=f"w{i}", cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs
    assert len(ledger.read_ledger(cfg=cfg, tool="test.concurrent", limit=50)) == 10


# ---------------------------------------------------------------------------
# 2. inspect tests
# ---------------------------------------------------------------------------

def test_inspect_source_corpus_session_and_step(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    seeded = _seed_corpus(tmp_path, cfg)

    # inspect_source: 1 adopted (by filename tail prefix), 1 unadopted (new project), 1 malformed
    src_dir = tmp_path / "source_logs"
    _write_pi_session(
        src_dir / "--d--crack-jigsawcard--" / "2026-11-18T10-00-00_01a10001.jsonl",
        session_id=seeded["sid1"],
        cwd="D:\\crack\\com.oakever.jigsawcard_1.5.22",
        turns=[{"user": "破解 jigsawcard.xapk", "steps": [{"thinking": "ok", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}]}],
    )
    _write_pi_session(
        src_dir / "--d--crack-puzzle--" / "2026-11-18T11-00-00_01a19999.jsonl",
        session_id="01a19999-9999-4000-8000-000000000099",
        cwd="D:\\crack_test\\teacher\\com.easybrain.block.puzzle.games_3.32.0",
        model="MiniMax-M3.1-Flash",
        turns=[{"user": "破解 puzzle", "steps": [{"thinking": "check", "tool_calls": [{"name": "read", "arguments": {"path": "a"}}]}]}],
    )
    (src_dir / "broken.jsonl").write_text("{bad json\n", encoding="utf-8")

    src_all = inspect_tools.inspect_source(src_dir, cfg=cfg)
    assert src_all["adopted_count"] == 1
    assert src_all["unadopted_count"] == 1
    assert src_all["parse_error_count"] == 0
    assert "game::puzzle" in src_all["new_projects"]
    assert src_all["by_project"]["game::puzzle"]["in_corpus"] is False

    src_unadopted = inspect_tools.inspect_source(src_dir, cfg=cfg, unadopted_only=True)
    assert len(src_unadopted["sessions"]) == 1
    assert src_unadopted["sessions"][0]["project_group"] == "game::puzzle"
    assert src_unadopted["sessions"][0]["adopted"] is False

    # inspect_corpus
    corp = inspect_tools.inspect_corpus(cfg=cfg)
    assert corp["raw_sessions"] == 2
    assert corp["gold_sessions"] == 2
    assert corp["project_count"] == 2
    assert corp["train_counts"]["sft_windows"] == 4
    corp_filtered = inspect_tools.inspect_corpus(cfg=cfg, project="jigsawcard")
    assert len(corp_filtered["sessions"]) == 1
    assert corp_filtered["sessions"][0]["project_group"] == "game::jigsawcard"

    # inspect_session (prefix + steps)
    ses = inspect_tools.inspect_session("01a10001", cfg=cfg, include_steps=True)
    assert ses["session_id"] == seeded["sid1"]
    assert ses["effective"]["state"] == "auto_gold"
    assert len(ses["turn_marks"]) == 2
    assert len(ses["steps"]) == 2
    assert ses["steps"][0]["tool_calls"] == ["bash"]
    assert "tool_results" not in ses["steps"][0]

    with pytest.raises(ValueError):
        inspect_tools.inspect_session("01a1", cfg=cfg)
    with pytest.raises(KeyError):
        inspect_tools.inspect_session("ffffffff", cfg=cfg)

    # inspect_step (by 0-based index, #N, and msg_id prefix)
    st0 = inspect_tools.inspect_step("01a10001", "0", cfg=cfg)
    st0_hash = inspect_tools.inspect_step("01a10001", "#0", cfg=cfg)
    st0_mid = inspect_tools.inspect_step("01a10001", "aa11", cfg=cfg)
    assert st0["msg_id"] == st0_hash["msg_id"] == st0_mid["msg_id"] == "aa11bb22"
    assert "原始超长思考" in st0["raw_thinking"]
    assert st0["thinking"] == "压缩后的思考：检查目录结构。"
    assert st0["raw_text"] == "我先看看目录"
    assert st0["text"] == ""
    assert st0["folded_from"] == "aa11bb21"
    assert st0["fold_details"]["path"] == "run.py"
    assert "[truncated" in st0["tool_calls"][0]["arguments"]["command"]
    assert "[truncated" in st0["tool_results"][0]["content_preview"]
    assert len(st0["dpo_pairs"]) == 1


# ---------------------------------------------------------------------------
# 3. eval tests (pack, diff, student)
# ---------------------------------------------------------------------------

def test_eval_pack_and_diff(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    tools_str = default_tools_json(cfg)

    # 不健康包：含中途假收尾 stop、<file> 污染、hermes-json 错风格、非规范项目组
    bad_zip = _build_test_zip(
        tmp_path / "bad.zip",
        package_hash="bad00001",
        sft_train=[
            {
                "sample_id": "step::s1::m1",
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": '<file name="prompt.txt">\n破解游戏\n</file>'},
                    {
                        "from": "gpt",
                        "value": '<think>\n这里要准备继续分析，但是我先停下来问用户是否继续。\n</think>\n等待指令。',
                    },
                ],
                "tools": tools_str,
                "metadata": {
                    "project_group": "workspace::weird_dir",
                    "step_rank": 5,
                    "total_steps": 20,
                    "stage": "routine",
                },
            },
            {
                "sample_id": "step::s1::m2",
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "破解游戏"},
                    {
                        "from": "gpt",
                        "value": (
                            '<think>\n调用 bash 检查文件。\n</think>\n'
                            '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
                        ),
                    },
                ],
                "tools": tools_str,
                "metadata": {
                    "project_group": "workspace::weird_dir",
                    "step_rank": 6,
                    "total_steps": 20,
                    "stage": "routine",
                },
            },
        ],
        sft_eval=[
            {
                "sample_id": "step::s2::m1",
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "破解 eval 游戏"},
                    {
                        "from": "gpt",
                        "value": (
                            "<think>\n检查目录和文件列表。\n</think>\n"
                            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
                        ),
                    },
                ],
                "tools": tools_str,
                "metadata": {"project_group": "game::altosodyssey", "step_rank": 1, "total_steps": 10, "stage": "opening"},
            }
        ],
        dpo_train=[],
        dpo_eval=[],
        length_gate={"_skip": {"skipped": True}},
    )

    ev_bad = eval_tools.eval_pack(bad_zip, cfg=cfg)
    assert ev_bad["healthy"] is False
    codes = {v["code"] for v in ev_bad["violations"]}
    assert "premature_stop_targets" in codes
    assert "prompt_tag_leaks" in codes
    assert "toolcall_contract" in codes
    assert "non_canonical_project_group" in codes
    assert "length_gate_skipped" in codes
    assert ev_bad["premature_stop_targets"] == 1
    assert ev_bad["prompt_tag_leaks"] == 1
    assert ev_bad["wrong_style"] == 1

    # 合规 coder-xml 包
    good_zip = _build_test_zip(
        tmp_path / "good.zip",
        package_hash="good0002",
        sft_train=[
            {
                "sample_id": "step::s3::m1",
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "破解 jigsawcard"},
                    {
                        "from": "gpt",
                        "value": (
                            "<think>\n先并行查看目录与文件。\n</think>\n"
                            "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>\n"
                            "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"
                        ),
                    },
                    {"from": "observation", "value": "ok"},
                    {
                        "from": "gpt",
                        "value": "<think>\n确认所有文件已完成解密与明文核验，输出最终报告。\n</think>\n已完成全部解密。",
                    },
                ],
                "tools": tools_str,
                "metadata": {
                    "project_group": "game::jigsawcard",
                    "step_rank": 2,
                    "total_steps": 2,
                    "stage": "closing",
                },
            }
        ],
        sft_eval=[
            {
                "sample_id": "step::s4::m1",
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "破解 altosodyssey"},
                    {
                        "from": "gpt",
                        "value": (
                            "<think>\n查看解包目录结构并检查清单，确认资源文件数量与格式。\n</think>\n"
                            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
                        ),
                    },
                ],
                "tools": tools_str,
                "metadata": {
                    "project_group": "game::altosodyssey",
                    "step_rank": 1,
                    "total_steps": 10,
                    "stage": "opening",
                },
            }
        ],
        dpo_train=[
            {
                "sample_id": "dpo::s3::m1",
                "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "破解 jigsawcard"}],
                "chosen": {
                    "from": "gpt",
                    "value": (
                        "<think>\n直接执行工具调用，不要半途停工。\n</think>\n"
                        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
                    ),
                },
                "rejected": {"from": "gpt", "value": "<think>\n我先停一下等待用户。\n</think>\n是否继续？"},
                "tools": tools_str,
                "metadata": {
                    "source": "stage2_pruned",
                    "reason": "premature_stop_folded",
                    "project_group": "game::jigsawcard",
                },
            }
        ],
        dpo_eval=[],
    )

    ev_good = eval_tools.eval_pack(good_zip, cfg=cfg)
    assert ev_good["healthy"] is True
    assert ev_good["violations"] == []
    assert ev_good["premature_stop_targets"] == 0
    assert ev_good["prompt_tag_leaks"] == 0
    assert ev_good["wrong_style"] == 0
    assert ev_good["sft_train_parallel_rate"] == 0.5
    assert ev_good["dpo_by_source"] == {"stage2_pruned": 1}

    diff = eval_tools.eval_diff(bad_zip, good_zip, cfg=cfg)
    assert diff["deltas"]["premature_stop_targets"]["delta"] == -1
    assert diff["deltas"]["prompt_tag_leaks"]["delta"] == -1
    assert diff["deltas"]["non_canonical_project_count"]["delta"] == -1
    assert diff["deltas"]["dpo_total_rows"]["delta"] == 1
    assert "| 指标 |" in diff["markdown_table"]


def test_eval_student_with_teacher_benchmark(tmp_path: Path) -> None:
    cfg, _ = _make_sandbox_cfg(tmp_path)
    student_dir = tmp_path / "student" / "goldthread-test"
    (student_dir / "meowdoku-27b").mkdir(parents=True)
    (student_dir / "puzzle-27b").mkdir(parents=True)
    sessions_dir = tmp_path / "sessions"
    teacher_dir = tmp_path / "teacher"

    _write_pi_session(
        sessions_dir / "--D--crack_test-student-goldthread-test-meowdoku-27b--" / "s_meow.jsonl",
        session_id="stu-meow-001",
        cwd="D:\\crack_test\\student\\goldthread-test\\meowdoku-27b",
        model="qwen3.8-27b",
        turns=[
            {
                "user": "破解 meowdoku",
                "steps": [
                    {
                        "thinking": "使用 unicorn 离线模拟解码。",
                        "tool_calls": [
                            {"name": "bash", "arguments": {"command": "python emu_unicorn.py"}},
                            {"name": "read", "arguments": {"path": "out.txt"}},
                        ],
                    },
                    {"thinking": "先总结一下", "text": "阶段完成"},
                ],
            },
            {
                "user": "继续深挖 .bin",
                "steps": [
                    {"thinking": "继续解析", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                ],
            },
        ],
    )
    _write_pi_session(
        sessions_dir / "--D--crack_test-student-goldthread-test-puzzle-27b--" / "s_puz.jsonl",
        session_id="stu-puz-001",
        cwd="D:\\crack_test\\student\\goldthread-test\\puzzle-27b",
        model="qwen3.8-27b",
        turns=[
            {
                "user": "破解 puzzle",
                "steps": [{"thinking": "", "text": "abort", "stop_reason": "aborted"}],
            }
        ],
        extra_errors=[{"role": "assistant", "errorMessage": "maximum context length 131072 exceeded", "content": []}],
    )
    _write_pi_session(
        teacher_dir / "com.oakever.meowdoku_1.21.0" / "t_meow.jsonl",
        session_id="tea-meow-001",
        cwd="D:\\train-project\\teacher\\com.oakever.meowdoku_1.21.0",
        model="glm-5.3-flash",
        turns=[
            {
                "user": "破解 meowdoku",
                "steps": [
                    {"thinking": "s1", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                    {"thinking": "s2", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                    {"thinking": "s3", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                    {"thinking": "s4", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
                ],
            }
        ],
    )

    res = eval_tools.eval_student(
        student_dir, sessions_dir=sessions_dir, teacher_dir=teacher_dir, cfg=cfg
    )
    assert res["summary"]["total_runs"] == 2
    assert res["summary"]["offline_emu_runs"] == 1
    assert res["summary"]["max_context_overflows"] == 1
    assert res["summary"]["empty_think_steps"] == 1
    assert res["summary"]["turn0_premature_stops"] == 1

    meow_run = next(r for r in res["runs"] if r["project_group"] == "game::meowdoku")
    assert meow_run["offline_emu_used"] is True
    assert meow_run["turn0_premature_stop"] is True
    assert meow_run["teacher_benchmark"] is not None
    assert meow_run["teacher_benchmark"]["assistant_steps"] == 4


# ---------------------------------------------------------------------------
# 4. report & cli tests
# ---------------------------------------------------------------------------

def test_report_publish_and_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    _seed_corpus(tmp_path, cfg)
    tools_str = default_tools_json(cfg)

    z_before = _build_test_zip(
        tmp_path / "b.zip",
        package_hash="aaaa1111",
        sft_train=[{"sample_id": "s1", "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "<file name=\"a\">hi</file>"}, {"from": "gpt", "value": "<think>\nabcdefghijklmnopqrstuv\n</think>\nhi"}], "tools": tools_str, "metadata": {"project_group": "workspace::x", "step_rank": 1, "total_steps": 5, "stage": "routine"}}],
        sft_eval=[{"sample_id": "s2", "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "hi"}, {"from": "gpt", "value": "<think>\nabcdefghijklmnopqrstuv\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"}], "tools": tools_str, "metadata": {"project_group": "game::altosodyssey", "step_rank": 1, "total_steps": 1, "stage": "closing"}}],
        dpo_train=[],
        dpo_eval=[],
    )
    z_after = _build_test_zip(
        tmp_path / "a.zip",
        package_hash="bbbb2222",
        sft_train=[{"sample_id": "s3", "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "hi"}, {"from": "gpt", "value": "<think>\nabcdefghijklmnopqrstuv\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"}], "tools": tools_str, "metadata": {"project_group": "game::jigsawcard", "step_rank": 1, "total_steps": 1, "stage": "closing"}}],
        sft_eval=[{"sample_id": "s4", "conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": "hi"}, {"from": "gpt", "value": "<think>\nabcdefghijklmnopqrstuv\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"}], "tools": tools_str, "metadata": {"project_group": "game::altosodyssey", "step_rank": 1, "total_steps": 1, "stage": "closing"}}],
        dpo_train=[],
        dpo_eval=[],
    )

    md_path = tmp_path / "my_report.md"
    md_path.write_text("# 测试评估报告\n\n本次修复了早停与标签污染问题。\n", encoding="utf-8")

    rc = cli.main(
        [
            "--config", str(cfg_path),
            "report", "publish",
            "--title", "Phase 1 验证报告",
            "--md", str(md_path),
            "--id", "eval_test_001",
            "--before", str(z_before),
            "--after", str(z_after),
            "--build", "abcdef0123456789",
            "--reason", "单元测试发布",
            "--json",
        ]
    )
    assert rc == 0
    pub_out = json.loads(capsys.readouterr().out)
    assert pub_out["report_id"] == "eval_test_001"
    assert pub_out["diff_summary"] is not None
    assert pub_out["diff_summary"]["deltas"]["prompt_tag_leaks"]["delta"] == -1
    assert Path(pub_out["md_file"]).is_file()
    assert Path(pub_out["meta_file"]).is_file()

    # report list & show
    assert cli.main(["--config", str(cfg_path), "report", "list", "--json"]) == 0
    lst = json.loads(capsys.readouterr().out)
    assert len(lst["reports"]) == 1
    assert lst["reports"][0]["report_id"] == "eval_test_001"

    assert cli.main(["--config", str(cfg_path), "report", "show", "eval_test_001", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert "测试评估报告" in shown["markdown"]

    # ledger list 自动包含 report.publish 记录
    assert cli.main(["--config", str(cfg_path), "ledger", "list", "--tool", "report.publish", "--json"]) == 0
    led_out = json.loads(capsys.readouterr().out)
    assert led_out["count"] == 1
    assert led_out["events"][0]["op_id"] == pub_out["op_id"]

    # inspect corpus via CLI --json
    assert cli.main(["--config", str(cfg_path), "inspect", "corpus", "--json"]) == 0
    corp_out = json.loads(capsys.readouterr().out)
    assert corp_out["raw_sessions"] == 2

    # 错误处理 (--json 模式返回非 0 且输出 ok=False)
    err_rc = cli.main(["--config", str(cfg_path), "inspect", "session", "nonexistent", "--json"])
    assert err_rc != 0
    err_out = json.loads(capsys.readouterr().out)
    assert err_out["ok"] is False
    assert "error" in err_out
