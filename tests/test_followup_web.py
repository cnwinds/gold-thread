"""Regression tests for the graph's deterministic workspace build job."""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import server.app as web
from goldthread import curate_tools, eval_tools, inspect_tools, ledger
from pipeline import judge, stage1_extract, stage3_synth, turnclassify
from pipeline.common.config import DEFAULTS, stage_dirs
from pipeline.common.store import load_index, read_jsonl


class SyntheticTokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(char) for char in text])


def _write_session(path: Path, sid: str, item: str) -> None:
    rows = [{"type": "session", "id": sid, "cwd": f"D:/crack/com.example.{item}"}]

    def message(msg_id, role, content, **fields):
        rows.append({"type": "message", "id": msg_id, "message": {"role": role, "content": content, **fields}})

    model = {"model": "glm-fixture", "provider": "fixture"}
    message("prompt", "user", [{"type": "text", "text": f"Summarize the inventory in {item}.txt."}])
    # Three tool steps put the final answer beyond Stage 3's opening window.
    for i, name in enumerate((f"{item}.txt", "summary.txt", "totals.txt")):
        call_id = f"read-{i}"
        message(f"step-{i}", "assistant", [
            {"type": "thinking", "thinking": f"Read {name} to verify the inventory total before reporting."},
            {"type": "toolCall", "id": call_id, "name": "read", "arguments": {"path": name}},
        ], stopReason="toolUse", **model)
        message(f"result-{i}", "toolResult", [{"type": "text", "text": f"{item}: 2"}],
                toolCallId=call_id, toolName="read", isError=False)
    message("final", "assistant", [
        {"type": "thinking", "thinking": "All records agree, so report the verified inventory total."},
        {"type": "text", "text": f"The inventory contains two {item}."},
    ], stopReason="stop", **model)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-source"
    source.mkdir()
    cfg = copy.deepcopy(DEFAULTS)
    cfg["paths"].update(data_dir=str(tmp_path / "sandbox"), sessions_dir=str(source),
                        pi_models_json=str(tmp_path / "absent-models.json"))
    cfg["gold"]["min_gold_steps"] = 2
    cfg["synth"].update(eval_ratio=0.5, shuffle=False)
    cfg["domain"]["system_prompt"] = "Summarize the inventory in local text fixtures."
    # The new job must remain offline even if model assistance is enabled in the server configuration.
    cfg["turns"]["use_model"] = True
    cfg["judge"]["enabled"] = True
    monkeypatch.setattr(web, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(web, "JOBS", {})
    monkeypatch.setattr(eval_tools, "_find_eval_tokenizer", lambda: (SyntheticTokenizer(), "fixture"))

    forbidden_calls = []

    def forbid(name):
        def unexpected(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"Unexpected {name} during workspace build")
        return unexpected

    monkeypatch.setattr(judge, "chat_json", forbid("judge request"))
    monkeypatch.setattr(turnclassify, "chat_json", forbid("classification request"))
    monkeypatch.setattr(stage1_extract, "run", forbid("Stage 1 external scan"))

    threads = []

    def launch(job):
        thread = threading.Thread(target=web._run_job, args=(job,), daemon=True)
        threads.append(thread)
        thread.start()

    monkeypatch.setattr(web, "_launch_job_thread", launch)
    client = TestClient(web.app)
    try:
        yield cfg, source, client, forbid
    finally:
        client.close()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
        assert forbidden_calls == []


def _adopt(source, cfg, count):
    sids = []
    for i, item in enumerate(("apples", "pears", "plums"), start=1):
        sid = f"a0b0c0d{i}-0000-4000-8000-000000000001"
        _write_session(source / f"{sid}.jsonl", sid, item)
        sids.append(sid)
    adopted = sids[:count]
    curate_tools.adopt_sessions(source, select=adopted, reason="Adopt synthetic inventory fixtures", cfg=cfg)
    assert set(load_index(stage_dirs(cfg)["raw"] / "index.json")["sessions"]) == set(adopted)
    assert not (stage_dirs(cfg)["gold"] / "index.json").exists()
    return adopted


def _submit(client, stage="build"):
    response = client.post("/api/run", json={"stage": stage, "force": False})
    assert response.status_code == 200, response.text
    return response.json()


def _wait(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("done", "failed") and job["finished_at"]:
            return job
        time.sleep(0.02)
    pytest.fail(f"Job did not finish: {job}")


def test_raw_only_graph_build_uses_adopted_sessions_and_checks_gate(workspace, monkeypatch):
    cfg, source, client, forbid = workspace
    adopted = _adopt(source, cfg, count=2)
    monkeypatch.setattr(inspect_tools, "inspect_source", forbid("external source inspection"))
    monkeypatch.setattr(stage3_synth, "evaluate_dpo_pending", forbid("legacy DPO evaluation"))

    job = _wait(client, _submit(client)["job_id"])
    assert job["status"] == "done", job
    task = client.get(f"/api/tasks/{job['id']}")
    assert task.status_code == 200, task.text
    assert task.json()["title"] == job["title"] == "工作区构建"
    assert task.json()["affects"] == ["overview", "gold", "train"]
    assert task.json()["links"]["open_tab"] == "train"
    result = job["result"]
    assert result["ok"] and result["gate"]["passed"]
    assert result["stage1"] is None
    build_id = result["build_id"]
    response = client.get(f"/api/dataset/builds/{build_id}")
    assert response.status_code == 200, response.text
    assert set(response.json()["cas_manifest"]["sessions"]) == set(adopted)
    samples = [
        row for split in ("train", "eval")
        for row in read_jsonl(Path(result["build_dir"]) / f"sft_step_windows_sharegpt.{split}.jsonl")
    ]
    assert {row["metadata"]["session_id"] for row in samples} == set(adopted)
    assert set(load_index(stage_dirs(cfg)["raw"] / "index.json")["sessions"]) == set(adopted)
    assert client.get("/api/dataset/builds").json()["latest_build_id"] == build_id
    assert not list((stage_dirs(cfg)["root"] / "04_release" / "delivery").glob("*.zip"))
    event = ledger.read_ledger(cfg=cfg, tool="build.run", limit=1)[0]
    assert event["op_id"] == result["op_id"]
    assert event["actor"] == "human" and event["params"]["scan_raw"] is False


def test_stage3_stays_stage3_only_and_workspace_gate_failure_is_visible(workspace):
    cfg, source, client, _ = workspace
    _adopt(source, cfg, count=1)

    legacy = _wait(client, _submit(client, "stage3")["job_id"])
    assert legacy["status"] == "failed" and "stage2 未产出数据" in legacy["error"]
    assert not (stage_dirs(cfg)["gold"] / "index.json").exists()

    # One project cannot populate both SFT splits. Stage 2/3 can run, but the normal gate must fail the job.
    job = _wait(client, _submit(client)["job_id"])
    assert job["status"] == "failed", job
    assert job["result"]["gate"]["passed"] is False
    assert "sft_non_empty" in job["error"]


def test_step_gate_rejects_mutation_while_all_is_running(workspace, monkeypatch):
    _, _, client, _ = workspace
    blocker = web.Job("all", False)
    with web.JOBS_LOCK:
        web.JOBS[blocker.id] = blocker
    curate_steps = Mock(return_value={})
    monkeypatch.setattr(curate_tools, "curate_steps", curate_steps)

    response = client.put(
        "/api/gold/session/a0b0c0d1-0000-4000-8000-000000000001/step-gate",
        json={"msg_id": "step-0", "action": "exclude"},
    )

    assert response.status_code == 409, response.text
    assert f"all ({blocker.id})" in response.json()["detail"]
    curate_steps.assert_not_called()


@pytest.mark.parametrize("blocking_stage", ["stage1", "collect", "stage3", "package", "all", "delete"])
def test_conflicting_workspace_build_queues(workspace, blocking_stage):
    cfg, source, client, _ = workspace
    _adopt(source, cfg, count=2)
    blocker = web.Job(blocking_stage, False)
    with web.JOBS_LOCK:
        web.JOBS[blocker.id] = blocker
    reply = _submit(client)
    assert reply["status"] == "queued"
    assert client.get(f"/api/jobs/{reply['job_id']}").json()["status"] == "queued"
    assert not (stage_dirs(cfg)["gold"] / "index.json").exists()
    with web.JOBS_LOCK:
        blocker.status = "done"
    web._pump_queue()
    job = _wait(client, reply["job_id"])
    assert job["status"] == "done", job
