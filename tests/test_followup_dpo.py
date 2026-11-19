"""Strict/offline DPO regressions using public operations on a temporary workspace."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from goldthread import build_tools, cli, curate_tools
from pipeline import stage2_gold, stage3_synth
from pipeline.common.config import stage_dirs
from pipeline.common.store import load_index, read_json, write_json_atomic, write_jsonl_atomic
from test_virtual_datasets import _make_sandbox_cfg, _write_pi_session


@pytest.fixture
def workspace(tmp_path):
    cfg, cfg_path = _make_sandbox_cfg(tmp_path)
    source = tmp_path / "teacher"
    sid = "01b2aa11-1111-7000-8000-000000000001"
    path = "table.csv"

    def read_step(step_id, file, result, *, failed=False):
        return {
            "id": step_id,
            "thinking": "读取表格内容并核对字段名称和数据行。",
            "tool_calls": [{"id": step_id, "name": "read", "arguments": {"path": file}}],
            "tool_results": [{"tool_call_id": step_id, "name": "read", "content": result, "isError": failed}],
        }

    _write_pi_session(
        source / f"{sid}.jsonl", session_id=sid, cwd=r"D:\demo\table-summary",
        turns=[{"user": "读取表格并总结包含的字段和记录数量。", "steps": [
            read_step("first", "README.md", "Data is in table.csv"),
            read_step("typo", "tabel.csv", "ENOENT: file not found", failed=True),
            read_step("correct", path, "name,count\nrow,1"),
            {"id": "finish", "thinking": "表格只有一条记录，现在总结字段名称。", "text": "字段为 name 和 count，共一条记录。"},
        ]}],
    )
    curate_tools.adopt_sessions(source, select=[sid], reason="sandbox fixture", cfg=cfg)
    return cfg, cfg_path, sid


def _rows(cfg):
    return {row["sample_id"]: row for row in stage3_synth.draft_preview(cfg, persist_split=False)["rows"]
            if row["kind"] == "dpo"}


def _build(cfg, **kwargs):
    return build_tools.build_pipeline(cfg=cfg, run_gate=False, **kwargs)


def _strict(cfg, **kwargs):
    return _build(cfg, require_reviewed_dpo=True, **kwargs)


def _build_files(cfg):
    root = stage_dirs(cfg)["root"] / "04_release" / "builds"
    return root / "LATEST_BUILD.json", sorted(p.name for p in root.iterdir()) if root.exists() else []


def _pairs(sid):
    def pair(name):
        return {
            "sample_id": f"dpo::{sid}::{name}",
            "conversations": [{"from": "human", "value": "读取并核对表格。"}],
            "chosen": {"from": "gpt", "value": "先读取表头，再统计数据行。"},
            "rejected": {"from": "gpt", "value": "无需读取，直接猜测条数。"},
            "metadata": {"session_id": sid, "project_group": "game::table-summary", "source": "stage2_pruned"},
        }

    return pair("first"), pair("second")


def _seed_gold_pairs(cfg, sid, *pairs):
    # Synthetic Stage 2 output allows controlled same-ID mutations and malformed candidates.
    gold = stage_dirs(cfg)["gold"]
    path = load_index(gold / "index.json")["sessions"][sid]["dpo_file"]
    write_jsonl_atomic(Path(path), pairs)
    write_jsonl_atomic(gold / "dpo_pairs.jsonl", pairs)


def test_strict_raw_only_build_refreshes_pool_without_publishing(workspace, capsys):
    cfg, cfg_path, _ = workspace
    assert not (stage_dirs(cfg)["gold"] / "index.json").exists()
    rc = cli.main(["--config", str(cfg_path), "--json", "build", "--require-reviewed-dpo", "--no-gate"])
    failure = json.loads(capsys.readouterr().out)
    assert rc != 0 and failure["error_type"] == "DPOReviewRequired"
    assert failure["dpo_audit"]["bypassed_unreviewed"] > 0
    assert "gt curate dpo" in failure["error"]
    latest, builds = _build_files(cfg)
    assert not latest.exists() and builds == []
    pending = _rows(cfg)
    assert pending and all(r["gate_status"] == "bypassed" for r in pending.values())
    curate_tools.curate_dpo({sample_id: "good" for sample_id in pending}, reason="review actual pool", cfg=cfg)
    result = _strict(cfg)
    assert result["ok"] and result["dpo_audit"]["explicit_verdicts"] == len(pending)
    assert read_json(latest)["build_id"] == result["build_id"]


@pytest.mark.parametrize("field", ["chosen", "conversations"])
def test_same_id_content_change_requires_re_review_without_advancing_latest(workspace, field):
    cfg, _, sid = workspace
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    changed, stable = _pairs(sid)
    ids = [changed["sample_id"], stable["sample_id"]]
    _seed_gold_pairs(cfg, sid, changed, stable)
    curate_tools.curate_dpo({ids[0]: "swap", ids[1]: "good"}, reason="review initial versions", cfg=cfg)
    assert _strict(cfg)["ok"]
    latest, before_builds = _build_files(cfg)
    before_pointer = latest.read_bytes()
    earlier = _rows(cfg)
    (changed[field][0] if field == "conversations" else changed[field])["value"] += " 已更新。"
    _seed_gold_pairs(cfg, sid, changed, stable)
    after = _rows(cfg)
    assert earlier[ids[0]]["gate_status"] == "swap"
    assert after[ids[0]]["content_fingerprint"] != earlier[ids[0]]["content_fingerprint"]
    assert after[ids[0]]["gate_status"] == "pending" and not after[ids[0]]["included"]
    assert after[ids[1]]["content_fingerprint"] == earlier[ids[1]]["content_fingerprint"]
    assert after[ids[1]]["review_source"] == "explicit" and after[ids[1]]["included"]
    with pytest.raises(stage3_synth.DPOReviewRequired) as exc:
        _strict(cfg)
    assert exc.value.audit["explicit_verdicts"] == exc.value.audit["pending_evaluations"] == 1
    with pytest.raises(RuntimeError, match="gt curate dpo"):
        _build(cfg)
    assert latest.read_bytes() == before_pointer and _build_files(cfg)[1] == before_builds
    review = curate_tools.curate_dpo("pending=good", reason="review changed content", cfg=cfg)
    assert review["sample_ids"] == [ids[0]]
    assert _strict(cfg)["dpo_audit"]["explicit_verdicts"] == 2


def test_excluded_and_invalid_pairs_do_not_block_strict_build(workspace):
    cfg, _, sid = workspace
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    excluded, invalid = _pairs(sid)
    invalid["rejected"] = deepcopy(invalid["chosen"])
    _seed_gold_pairs(cfg, sid, excluded, invalid)
    curate_tools.curate_exclude(samples=[excluded["sample_id"]], reason="intentionally excluded", cfg=cfg)
    rows = _rows(cfg)
    assert rows[excluded["sample_id"]]["decision"] == "exclude"
    assert rows[invalid["sample_id"]]["gate_status"] == "invalid"
    assert not any(r["included"] for r in rows.values())
    dry = _strict(cfg, dry_run=True)
    audit = _strict(cfg)["dpo_audit"]
    assert dry["ok"] and dry["dpo_audit"] == audit
    assert audit["total_candidates"] == 2 and audit["eligible_candidates"] == 0
    assert audit["readiness"] == "ready"


@pytest.mark.parametrize("legacy", ["drop", "swap", "good", "anonymous-cache"])
def test_legacy_unbound_review_does_not_authorize_inclusion(workspace, legacy):
    cfg, _, sid = workspace
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    pair, _ = _pairs(sid)
    _seed_gold_pairs(cfg, sid, pair)
    sample_id = pair["sample_id"]
    state = stage_dirs(cfg)["state"]
    if legacy == "anonymous-cache":
        write_json_atomic(state / "dpo_review_cache.json", {"legacy-key": {"verdict": "drop"}})
    else:
        write_json_atomic(state / "dpo_verdicts.json", {sample_id: {"verdict": legacy}})
    row = _rows(cfg)[sample_id]
    assert row["gate_status"] == "pending" and not row["included"]
    with pytest.raises(stage3_synth.DPOReviewRequired):
        _strict(cfg)
    with pytest.raises(RuntimeError, match="gt curate dpo"):
        _build(cfg)
    assert not _build_files(cfg)[0].exists()
    assert curate_tools.curate_dpo("pending=drop", reason="review to omit", cfg=cfg)["sample_ids"] == [sample_id]
    result = _strict(cfg)
    assert result["dpo_audit"]["explicit_verdicts"] == 1
    assert result["stage3"]["dpo_total"] == 0


def test_cached_reviews_are_bound_per_content_even_with_duplicate_ids(workspace, monkeypatch):
    cfg, _, sid = workspace
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    drop, keep = _pairs(sid)
    keep["sample_id"] = drop["sample_id"]
    drop["chosen"]["value"] = "删除版本：先读取表头，再统计数据行。"
    drop["conversations"].extend([
        {"from": "gpt", "value": "已找到表格。"},
        {"from": "human", "value": "先核对字段。"},
        {"from": "gpt", "value": "正在检查表头。"},
    ])
    _seed_gold_pairs(cfg, sid, drop, keep)

    def judge_response(_system, _schema, payload, *_args):
        pair, = payload["pairs"]
        return {"pairs": [{"sample_id": pair["sample_id"], "verdict": (
            "drop" if "删除版本" in str(pair["chosen"]) else "good"
        )}]}

    judge_cfg = deepcopy(cfg)
    judge_cfg["judge"]["enabled"] = True
    monkeypatch.setattr("pipeline.judge._call", judge_response)
    evaluation = stage3_synth.evaluate_dpo_pending(cfg=judge_cfg, log=lambda *_: None)
    assert evaluation["good"] == evaluation["drop"] == 1
    cached = _strict(cfg)
    assert cached["dpo_audit"]["cached_reviews"] == 2 and cached["stage3"]["dpo_total"] == 1

    # The old key only hashed the last three messages; the review now binds the entire context.
    drop["conversations"][0]["value"] = "请读取另一份表格。"
    _seed_gold_pairs(cfg, sid, drop, keep)
    audit = _strict(cfg, dry_run=True)["dpo_audit"]
    assert audit["cached_reviews"] == audit["pending_evaluations"] == 1
    with pytest.raises(RuntimeError, match="gt curate dpo"):
        _build(cfg)


def test_strict_dry_run_is_read_only_and_detects_old_purification(workspace, capsys):
    cfg, cfg_path, _ = workspace
    dirs = stage_dirs(cfg)

    def snapshot():
        return {p.relative_to(dirs["root"]): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
                for p in dirs["root"].rglob("*")}

    def dry(readiness, **kwargs):
        before = snapshot()
        result = _strict(cfg, dry_run=True, **kwargs)
        assert snapshot() == before
        assert result["dpo_audit"]["readiness"] == readiness
        assert result["ok"] == (readiness == "ready")
        return result

    assert dry("undetermined")["gate_preview"]["dpo_train_rows"] is None
    before = snapshot()
    assert cli.main(["--config", str(cfg_path), "build", "--dry-run", "--require-reviewed-dpo"]) == 2
    assert "readiness=undetermined" in capsys.readouterr().out
    assert snapshot() == before
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    assert dry("review_required")["dpo_audit"]["bypassed_unreviewed"] > 0
    curate_tools.curate_dpo("pending=good", reason="review fixture", cfg=cfg)
    dry("ready")
    assert dry("undetermined", force=True)["dpo_audit"]["reason"] == "forced_rebuild"
    assert dry("undetermined", scan_raw=True)["dpo_audit"]["reason"] == "raw_scan_requested"
    state_path = dirs["state"] / "stage2.json"
    state = read_json(state_path)
    for item in state["files"].values():
        del item["purify_version"]
    write_json_atomic(state_path, state)
    assert dry("undetermined")["dpo_audit"]["reason"] == "stage2_inputs_or_outputs_out_of_date"
    stage2_gold.run(cfg=cfg, log=lambda *_: None)
    dry("ready")
