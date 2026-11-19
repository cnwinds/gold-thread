"""Follow-up finding 4: replay confirmed write/edit pairs without losing file state."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from pipeline import stage2_gold as s2
from pipeline.common.config import DEFAULTS, stage_dirs
from pipeline.common.store import dump_index, load_index, read_json, read_jsonl, write_json_atomic, write_jsonl_atomic


PATH = "notes.txt"
ROOT = {"id": "root", "role": "user", "content": "Update the garden notes."}
INTRO = {"id": "intro", "role": "assistant", "thinking": "Plan the garden notes.", "text": ""}


def step(mid, name, args, receipt, *, error=False, thinking="Update the notes."):
    call = {"id": f"call-{mid}", "name": name, "arguments": args}
    result = {"id": f"result-{mid}", "role": "tool", "tool_call_id": call["id"],
              "name": name, "content": receipt, "isError": error}
    msg = {"id": mid, "role": "assistant", "thinking": thinking, "text": "",
           "tool_calls": [call], "tool_results": [result]}
    return [msg, result]


def blocks(*pairs):
    return [{"oldText": old, "newText": new} for old, new in pairs]


def pair(original="red\nblue\n", replacements=None):
    replacements = blocks(("red", "green")) if replacements is None else replacements
    write = step("write", "write", {"path": PATH, "content": original}, f"Successfully wrote to {PATH}")
    edit = step("edit", "edit", {"path": PATH, "edits": replacements},
                f"Successfully replaced {len(replacements)} block(s) in {PATH}.",
                thinking="Oops! Use the updated garden notes.")
    return write + edit


def purify(mode, flow, monkeypatch, *, decisions=None, max_edits=8):
    decisions = decisions or {}
    cfg = {"gold": {"max_fold_edits": max_edits}}
    if mode == "rules":
        return s2.purify(flow, cfg["gold"])
    if mode == "mixed":
        decisions = {"intro": {"action": "keep"}, **decisions}
    else:
        decisions = {"edit": {"action": "fold"}, **decisions}
    if mode == "judge":
        monkeypatch.setattr(s2.judge, "decide_steps", lambda *a, **k: decisions)
        monkeypatch.setattr(s2.judge, "audit_seams", lambda *a, **k: {})
        return s2.purify_by_judge(flow, cfg, log=lambda *_: None)[:3]
    return s2.apply_step_decisions(flow, cfg, step_decision={"steps": decisions})


def ids(messages):
    return [m["id"] for m in messages if m["role"] == "assistant"]


def content(msg):
    return msg["tool_calls"][0]["arguments"]["content"]


@pytest.fixture(params=["rules", "mixed", "explicit", "judge"])
def mode(request):
    return request.param


def test_successful_atomic_fold_uses_original_spans(mode, monkeypatch):
    # The first replacement introduces the second oldText. Sequential replace
    # would edit the newly inserted text instead of the original second block.
    original = "\ufeffred\r\nblue\r\n"
    write, wr, edit, er = pair(original, blocks(("red\n", "blue\n"), ("blue\n", "green\n")))
    flow = [ROOT, INTRO, write, wr, edit, er]
    snapshot = deepcopy(flow)
    gold, pruned, stats = purify(mode, flow, monkeypatch)
    assert flow == snapshot
    assert ids(gold) == ["intro", "write"]
    assert [m["id"] for m in gold if m["role"] == "tool"] == ["result-write"]
    assert stats["folded"] == 1
    folded = next(m for m in gold if m["id"] == "write")
    assert content(folded) == "\ufeffblue\r\ngreen\r\n"
    assert folded["folded_from"] == "edit"
    assert folded["fold_details"]["edits_count"] == 2
    assert len(pruned) == 1 and pruned[0]["dropped_msg_id"] == "edit"
    if mode in ("rules", "mixed"):
        assert pruned[0]["has_dpo"]
        assert content(pruned[0]["chosen_msg"]) == content(folded)
        assert content(pruned[0]["rejected_msg"]) == original
    else:
        assert not pruned[0]["has_dpo"]


@pytest.mark.parametrize("shape", ["legacy", "object", "json", "combined"])
def test_supported_pi_edit_argument_shapes(shape, mode, monkeypatch):
    write, wr, edit, er = pair()
    wr["content"] = f"Successfully wrote 9 bytes to {PATH}"
    args = {"path": PATH}
    single = {"oldText": "red\n", "newText": ""}
    if shape == "legacy":
        args.update(single)
        er["content"] = f"Successfully replaced text in {PATH}."
    elif shape == "combined":
        args.update(single, edits=blocks(("blue", "yellow")))
        er["content"] = f"Successfully replaced 2 block(s) in {PATH}."
    else:
        args["edits"] = single if shape == "object" else json.dumps([single])
    edit["tool_calls"][0]["arguments"] = args
    gold, _, stats = purify(mode, [ROOT, INTRO, write, wr, edit, er], monkeypatch)
    assert stats["folded"] == 1
    assert content(gold[-2]) == ("yellow\n" if shape == "combined" else "blue\n")


@pytest.mark.parametrize("bad", ["write_error", "edit_error", "missing_write", "missing_edit", "bad_id", "bad_name",
                                 "duplicate_result", "partial_receipt", "unconfirmed", "truncated", "contradictory"])
def test_requires_identified_complete_success_on_both_calls(bad, mode, monkeypatch):
    write, wr, edit, er = pair(replacements=blocks(("red", "green"), ("blue", "yellow")))
    edit["thinking"] = "" if bad == "edit_error" else edit["thinking"]
    write["thinking"] = "" if bad == "write_error" else write["thinking"]
    flow = [ROOT, INTRO, write, wr, edit, er]
    if bad == "write_error":
        wr["isError"] = True
    elif bad == "edit_error":
        er["isError"] = True
        # Error-pruning must not erase this edit in favor of the following read.
        flow.extend(step("read", "read", {"path": PATH}, "red\nblue\n"))
    elif bad == "missing_write":
        flow.remove(wr)  # an inline mirror alone must not replace the observation
    elif bad == "missing_edit":
        flow.remove(er)
    elif bad == "bad_id":
        er["tool_call_id"] = "different-call"
    elif bad == "bad_name":
        er["name"] = "write"
    elif bad == "duplicate_result":
        flow.append({**er, "id": "duplicate"})
    elif bad == "partial_receipt":
        er["content"] = f"Successfully replaced 1 block(s) in {PATH}."
    elif bad == "unconfirmed":
        er.pop("isError")
    elif bad == "truncated":
        er["raw_len"] = len(er["content"]) + 25
    elif bad == "contradictory":
        edit["tool_results"] = [{**er, "isError": True}]
    gold, pruned, stats = purify(mode, flow, monkeypatch)
    assert [m["id"] for m in gold] == [m["id"] for m in flow]
    assert content(next(m for m in gold if m["id"] == "write")) == "red\nblue\n"
    assert stats["folded"] == 0
    assert not pruned


@pytest.mark.parametrize("original,replacements,limit", [
    ("red red", blocks(("red", "green")), 8),
    ('"red" and “red”', blocks(('"red"', "green")), 8),
    ("Look at the ‘red’ label.", blocks(("'red'", "green")), 8),  # fuzzy-only match
    ("aaa", blocks(("aa", "b")), 8),  # overlapping occurrences also count as ambiguous
    ("red blue", blocks(("red blue", "green"), ("blue", "yellow")), 8),
    ("red blue", blocks(("red", "green"), ("green", "yellow")), 8),
    ("red blue", blocks(("red", "green"), ("missing", "yellow")), 8),
    ("red blue", [{"oldText": "red", "newText": "green"}, {"oldText": "blue"}], 8),
    ("red blue", blocks(("red", "green"), ("blue", "yellow")), 1),
    ("012345678", blocks(*[(str(i), "x") for i in range(9)]), 8),
])
def test_replay_failure_is_atomic(mode, monkeypatch, original, replacements, limit):
    gold, pruned, stats = purify(mode, [ROOT, INTRO, *pair(original, replacements)], monkeypatch, max_edits=limit)
    assert ids(gold) == ["intro", "write", "edit"]
    assert content(next(m for m in gold if m["id"] == "write")) == original
    assert stats["folded"] == 0
    assert not pruned


@pytest.mark.parametrize("mode", ["mixed", "explicit", "judge"])
def test_never_fold_across_pruned_source_step(mode, monkeypatch):
    write, wr, edit, er = pair()
    between = step("between", "read", {"path": PATH}, "red\nblue\n")
    decisions = {"between": {"action": "prune"}}
    gold, pruned, stats = purify(mode, [ROOT, INTRO, write, wr, *between, edit, er], monkeypatch,
                                  decisions=decisions)
    assert ids(gold) == ["intro", "write", "edit"]
    assert [p["dropped_msg_id"] for p in pruned] == ["between"]
    assert stats["folded"] == 0
    assert content(next(m for m in gold if m["id"] == "write")) == "red\nblue\n"


def test_no_fold_over_a_turn_removed_upstream(mode, monkeypatch):
    write, wr, edit, er = pair()
    write["turn_index"] = 0
    edit["turn_index"] = 1
    gold, pruned, stats = purify(mode, [ROOT, INTRO, write, wr, edit, er], monkeypatch)
    assert not pruned
    assert ids(gold) == ["intro", "write", "edit"]
    assert stats["folded"] == 0


@pytest.mark.parametrize("mode", ["rules", "mixed"])
def test_never_fold_across_automatically_removed_step(mode, monkeypatch):
    write, wr, edit, er = pair()
    between = step("between", "read", {"path": PATH}, "Missing garden notes.", error=True)
    gold, pruned, stats = purify(mode, [ROOT, INTRO, write, wr, *between, edit, er], monkeypatch)
    assert ids(gold) == ["intro", "write", "edit"]
    assert [p["dropped_msg_id"] for p in pruned] == ["between"]
    assert stats["folded"] == 0
    assert content(next(m for m in gold if m["id"] == "write")) == "red\nblue\n"


def test_failed_edit_cannot_be_a_detour_correction(mode, monkeypatch):
    _, _, edit, er = pair()
    er["isError"] = True
    before = step("before", "read", {"path": PATH}, "The notes were not found.", error=True)
    decisions = {"before": {"action": "prune"}} if mode in ("explicit", "judge") else None
    gold, pruned, stats = purify(mode, [ROOT, INTRO, *before, edit, er], monkeypatch, decisions=decisions)
    assert "edit" in ids(gold)
    assert stats["folded"] == 0
    assert not any(p["has_dpo"] for p in pruned)
    if decisions is None:
        assert "before" in ids(gold)
    else:
        assert [p["dropped_msg_id"] for p in pruned] == ["before"]


def pi_rows(sid, original, replacements, *, error):
    rows = [{"type": "session", "id": sid, "cwd": "garden"}]
    for m in [ROOT, *pair(original, replacements)]:
        if m["role"] == "user":
            message = {"role": "user", "content": [{"type": "text", "text": m["content"]}]}
        elif m["role"] == "assistant":
            message = {"role": "assistant", "model": "synthetic", "content": [
                {"type": "thinking", "thinking": m["thinking"]},
                *[{"type": "toolCall", **call} for call in m["tool_calls"]],
            ]}
        else:
            message = {"role": "toolResult", "toolCallId": m["tool_call_id"], "toolName": m["name"],
                       "isError": error and m["name"] == "edit",
                       "content": [{"type": "text", "text": m["content"]}]}
        rows.append({"type": "message", "id": m["id"], "message": message})
    return rows


def test_stage2_recomputes_legacy_cached_gold_without_force(tmp_path):
    cfg = deepcopy(DEFAULTS)
    cfg["paths"]["data_dir"] = str(tmp_path)
    cfg["gold"]["min_gold_steps"] = 1
    dirs = stage_dirs(cfg)
    raw_index = {}
    for sid, error in [("good", False), ("bad", True)]:
        raw = dirs["raw"] / f"{sid}.jsonl"
        meta = raw.with_suffix(".meta.json")
        write_jsonl_atomic(raw, pi_rows(sid, "red\n", blocks(("red", "green")), error=error))
        write_json_atomic(meta, {})
        raw_index[sid] = {"raw_file": str(raw), "meta_file": str(meta),
                          "project_group": "game::garden", "label": "positive"}
    dump_index(dirs["raw"] / "index.json", raw_index, {})
    run = lambda **kw: s2.run(cfg=cfg, log=lambda *_: None, **kw)
    assert run()["written"] == 2
    gold_index = load_index(dirs["gold"] / "index.json")["sessions"]
    good = read_jsonl(Path(gold_index["good"]["gold_file"]))[0]
    assert ids(good["messages"]) == ["write"]
    assert content(next(m for m in good["messages"] if m["id"] == "write")) == "green\n"
    # Simulate pre-fix cached output for the failed edit, including its DPO pair.
    bad_file = Path(gold_index["bad"]["gold_file"])
    bad_dpo = Path(gold_index["bad"]["dpo_file"])
    write_jsonl_atomic(bad_file, [{**good, "session_id": "bad"}])
    write_jsonl_atomic(bad_dpo, read_jsonl(Path(gold_index["good"]["dpo_file"])))
    state_file = dirs["state"] / "stage2.json"
    state = read_json(state_file)
    state["files"]["bad"].pop("purify_version")
    write_json_atomic(state_file, state)
    assert run(only=["good"])["written"] == 1
    assert ids(read_jsonl(bad_file)[0]["messages"]) == ["write"]
    summary = run()
    assert (summary["written"], summary["reused"]) == (1, 1)
    assert ids(read_jsonl(bad_file)[0]["messages"]) == ["write", "edit"]
    assert read_jsonl(bad_dpo) == []
    assert summary["dpo_pairs"] == 1
    assert run()["reused"] == 2
