"""虚拟数据集训练切片的会话级只读查询（HTTP 公开路由）。

测试只使用临时目录中的中性合成 ZIP 与工作区训练文件，不触碰真实数据，
也不触发任何构建 / 训练流程。
"""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from goldthread import lineage_graph
from pipeline.common.config import DEFAULTS
import server.app as web


PACK_HASH = "neutral0001"
LONG_CHOSEN = "CHOSEN-" + ("a" * 600)
LONG_REJECTED = "REJECTED-" + ("b" * 600)
LONG_CONTEXT = "CONTEXT-" + ("c" * 500)


def _cfg(tmp_path: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["_config_path"] = str(tmp_path / "pipeline.yaml")
    return cfg


def _sft_row(
    sample_id: str,
    *,
    sid: str | None,
    text: str,
    top_level_sid: str | None = None,
    project: str = "neutral::alpha",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "conversations": [
            {"from": "human", "value": f"question {sample_id}"},
            {"from": "gpt", "value": text},
        ],
    }
    meta: Dict[str, Any] = {"project_group": project}
    if sid is not None:
        meta["session_id"] = sid
    row["metadata"] = meta
    if top_level_sid is not None:
        row["session_id"] = top_level_sid
    return row


def _dpo_row(
    sample_id: str,
    *,
    sid: str | None,
    chosen: str,
    rejected: str,
    context: str = LONG_CONTEXT,
    top_level_sid: str | None = None,
    project: str = "neutral::alpha",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "conversations": [{"from": "human", "value": context}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
    }
    meta: Dict[str, Any] = {"project_group": project}
    if sid is not None:
        meta["session_id"] = sid
    row["metadata"] = meta
    if top_level_sid is not None:
        row["session_id"] = top_level_sid
    return row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _write_zip_pack(data_dir: Path, files: Dict[str, List[Dict[str, Any]]]) -> Path:
    delivery = data_dir / "04_release" / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    zpath = delivery / f"goldthread-train-{PACK_HASH}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "goldthread-train/data/MANIFEST.json",
            json.dumps({"package_hash": PACK_HASH, "package": "neutral-pack"}),
        )
        for name, rows in files.items():
            zf.writestr(
                f"goldthread-train/data/{name}",
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            )
    return zpath


def _workspace_train(data_dir: Path) -> Path:
    train = data_dir / "03_train"
    train.mkdir(parents=True, exist_ok=True)
    return train


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    cfg = _cfg(tmp_path)
    data_dir = Path(cfg["paths"]["data_dir"])
    monkeypatch.setattr(web, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.delenv("GOLDTHREAD_API_TOKEN", raising=False)
    return {"cfg": cfg, "data_dir": data_dir, "client": TestClient(web.app)}


def test_session_exact_filter_empty_positive_and_pagination(env: Dict[str, Any]) -> None:
    data_dir: Path = env["data_dir"]
    client: TestClient = env["client"]

    # 冻结包：SFT / DPO 均包含 alpha 会话，且存在“前缀相同”的碰撞会话
    dpo_rows = [
        _dpo_row("dpo-other-0", sid="other-session", chosen="O", rejected="o"),
        _dpo_row("dpo-alpha-0", sid="sess-alpha", chosen=LONG_CHOSEN, rejected=LONG_REJECTED),
        _dpo_row("dpo-long-0", sid="sess-alpha-long", chosen="L", rejected="l"),
        _dpo_row("dpo-alpha-1", sid="sess-alpha", chosen="C1", rejected="R1"),
        # metadata 缺失时回退顶层 session_id
        _dpo_row("dpo-alpha-2", sid=None, top_level_sid="sess-alpha", chosen="C2", rejected="R2"),
    ]
    sft_rows = [
        _sft_row("sft-other-0", sid="other-session", text="other"),
        _sft_row("sft-alpha-0", sid="sess-alpha", text="A" * 400),
        _sft_row("sft-long-0", sid="sess-alpha-long", text="long"),
    ]
    _write_zip_pack(
        data_dir,
        {
            "dpo.train.lf.jsonl": dpo_rows,
            "dpo.eval.lf.jsonl": [],
            "sft.train.lf.jsonl": sft_rows,
            "sft.eval.lf.jsonl": [],
        },
    )

    empty = client.get(f"/api/datasets/{PACK_HASH}/train", params={"session_id": "sess-missing"}).json()
    assert empty["total"] == 0
    assert empty["rows"] == []

    full = client.get(f"/api/datasets/{PACK_HASH}/train", params={"session_id": "sess-alpha"}).json()
    # 精确匹配：不能命中 sess-alpha-long
    assert {r["session_id"] for r in full["rows"]} == {"sess-alpha"}
    assert full["total"] == 4  # 3 DPO + 1 SFT
    assert {r["sample_id"] for r in full["rows"]} == {
        "dpo-alpha-0",
        "dpo-alpha-1",
        "dpo-alpha-2",
        "sft-alpha-0",
    }

    # 分页在会话过滤之后生效：offset 跳过的必须是匹配行
    page = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "sess-alpha", "offset": 2, "limit": 2},
    ).json()
    assert page["total"] == 4
    assert page["offset"] == 2
    assert len(page["rows"]) == 2
    assert {r["session_id"] for r in page["rows"]} == {"sess-alpha"}
    paged_ids = {r["sample_id"] for r in full["rows"][2:4]}
    assert {r["sample_id"] for r in page["rows"]} == paged_ids

    # 只看 DPO 时仍是精确会话过滤
    dpo_only = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "sess-alpha", "kind": "dpo"},
    ).json()
    assert dpo_only["total"] == 3
    assert all(r["kind"] == "dpo" for r in dpo_only["rows"])


def test_raw_row_complete_untruncated_and_stored_direction(env: Dict[str, Any]) -> None:
    data_dir: Path = env["data_dir"]
    client: TestClient = env["client"]
    dpo_row = _dpo_row(
        "dpo-full",
        sid="sess-full",
        chosen=LONG_CHOSEN,
        rejected=LONG_REJECTED,
        context=LONG_CONTEXT,
    )
    _write_zip_pack(
        data_dir,
        {
            "dpo.train.lf.jsonl": [dpo_row],
            "dpo.eval.lf.jsonl": [],
            "sft.train.lf.jsonl": [],
            "sft.eval.lf.jsonl": [],
        },
    )

    body = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "sess-full", "kind": "dpo"},
    ).json()
    assert body["total"] == 1
    row = body["rows"][0]
    raw = row["raw_row"]

    # chosen / rejected 保持原始方向与完整长度，context 不被截断
    assert raw["chosen"] == {"from": "gpt", "value": LONG_CHOSEN}
    assert raw["rejected"] == {"from": "gpt", "value": LONG_REJECTED}
    assert raw["conversations"][0]["value"] == LONG_CONTEXT
    assert len(raw["chosen"]["value"]) == len(LONG_CHOSEN)
    assert len(raw["rejected"]["value"]) == len(LONG_REJECTED)
    assert len(raw["conversations"][0]["value"]) == len(LONG_CONTEXT)
    assert raw == dpo_row


def test_workspace_vs_frozen_and_rows_removed_from_next_build(env: Dict[str, Any]) -> None:
    data_dir: Path = env["data_dir"]
    client: TestClient = env["client"]

    # 冻结包保留旧会话 old-session；工作区最后一次生成的 03_train 已不再包含它
    frozen_old = _dpo_row("old-frozen", sid="old-session", chosen="frozen-old", rejected="rej-old")
    frozen_same = _dpo_row("same-frozen", sid="sess-same", chosen="frozen-value", rejected="frozen-rej")
    _write_zip_pack(
        data_dir,
        {
            "dpo.train.lf.jsonl": [frozen_old, frozen_same],
            "dpo.eval.lf.jsonl": [],
            "sft.train.lf.jsonl": [_sft_row("frozen-sft", sid="sess-same", text="frozen-sft")],
            "sft.eval.lf.jsonl": [],
        },
    )

    train = _workspace_train(data_dir)
    ws_rows = [
        _dpo_row("same-ws", sid="sess-same", chosen="workspace-value", rejected="workspace-rej"),
        _dpo_row("old-ws", sid="old-session", chosen="workspace-old", rejected="workspace-rej-old"),
    ]
    _write_jsonl(train / "dpo_preference_pairs.train.jsonl", ws_rows)
    _write_jsonl(train / "dpo_preference_pairs.eval.jsonl", [])
    _write_jsonl(
        train / "sft_step_windows_sharegpt.train.jsonl",
        [_sft_row("ws-sft", sid="sess-same", text="workspace-sft")],
    )
    _write_jsonl(train / "sft_step_windows_sharegpt.eval.jsonl", [])

    # 冻结包：历史会话仍可只读查看，内容来自 ZIP 而非工作区
    frozen_old_body = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "old-session", "kind": "dpo"},
    ).json()
    assert frozen_old_body["total"] == 1
    assert frozen_old_body["rows"][0]["raw_row"]["chosen"]["value"] == "frozen-old"

    frozen_same = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "sess-same"},
    ).json()
    assert frozen_same["total"] == 2
    frozen_values = {r["sample_id"]: r["raw_row"] for r in frozen_same["rows"]}
    assert frozen_values["same-frozen"]["chosen"]["value"] == "frozen-value"
    assert frozen_values["frozen-sft"]["conversations"][-1]["value"] == "frozen-sft"

    # 工作区：同一会话返回最后一次生成的 03_train 内容（与冻结包不同）
    ws_same = client.get(
        "/api/datasets/workspace/train",
        params={"session_id": "sess-same"},
    ).json()
    assert ws_same["total"] == 2
    ws_values = {r["sample_id"]: r["raw_row"] for r in ws_same["rows"]}
    assert ws_values["same-ws"]["chosen"]["value"] == "workspace-value"
    assert ws_values["ws-sft"]["conversations"][-1]["value"] == "workspace-sft"

    # 工作区已不再包含 old-session 的下一次构建内容；但冻结包仍可查看它
    ws_old = client.get(
        "/api/datasets/workspace/train",
        params={"session_id": "removed-from-workspace", "kind": "dpo"},
    ).json()
    assert ws_old["total"] == 0


def test_dpo_session_query_skips_sft_files_and_unrelated_compaction(
    env: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir: Path = env["data_dir"]
    client: TestClient = env["client"]

    _write_zip_pack(
        data_dir,
        {
            "dpo.train.lf.jsonl": [
                _dpo_row("freeze-hit", sid="sess-hit", chosen="c", rejected="r"),
                _dpo_row("freeze-miss", sid="sess-other", chosen="c2", rejected="r2"),
            ],
            "dpo.eval.lf.jsonl": [],
            "sft.train.lf.jsonl": [_sft_row("freeze-sft-must-not-load", sid="sess-hit", text="x")],
            "sft.eval.lf.jsonl": [_sft_row("freeze-sft-eval-must-not-load", sid="sess-hit", text="y")],
        },
    )
    train = _workspace_train(data_dir)
    _write_jsonl(
        train / "dpo_preference_pairs.train.jsonl",
        [
            _dpo_row("ws-hit", sid="sess-hit", chosen="wc", rejected="wr"),
            _dpo_row("ws-miss", sid="sess-other", chosen="wc2", rejected="wr2"),
        ],
    )
    _write_jsonl(train / "dpo_preference_pairs.eval.jsonl", [])
    _write_jsonl(
        train / "sft_step_windows_sharegpt.train.jsonl",
        [_sft_row("ws-sft-must-not-load", sid="sess-hit", text="x")],
    )
    _write_jsonl(
        train / "sft_step_windows_sharegpt.eval.jsonl",
        [_sft_row("ws-sft-eval-must-not-load", sid="sess-hit", text="y")],
    )

    opened: List[str] = []
    compacted: List[str] = []
    real_read_jsonl = lineage_graph.read_jsonl
    real_read_zip_jsonl = lineage_graph._read_zip_jsonl
    real_compact = lineage_graph._compact_train_row

    def spy_read_jsonl(path: Path) -> List[Dict[str, Any]]:
        opened.append(Path(path).name)
        return real_read_jsonl(path)

    def spy_read_zip_jsonl(zp: Path, rel_name: str) -> List[Dict[str, Any]]:
        opened.append(f"zip:{rel_name}")
        return real_read_zip_jsonl(zp, rel_name)

    def spy_compact(row: Dict[str, Any], kind: str, split: str, fallback_id: str) -> Dict[str, Any]:
        compacted.append(lineage_graph._row_session_id(row))
        return real_compact(row, kind, split, fallback_id)

    monkeypatch.setattr(lineage_graph, "read_jsonl", spy_read_jsonl)
    monkeypatch.setattr(lineage_graph, "_read_zip_jsonl", spy_read_zip_jsonl)
    monkeypatch.setattr(lineage_graph, "_compact_train_row", spy_compact)

    frozen = client.get(
        f"/api/datasets/{PACK_HASH}/train",
        params={"session_id": "sess-hit", "kind": "dpo"},
    ).json()
    assert frozen["total"] == 1
    assert frozen["rows"][0]["sample_id"] == "freeze-hit"

    ws = client.get(
        "/api/datasets/workspace/train",
        params={"session_id": "sess-hit", "kind": "dpo"},
    ).json()
    assert ws["total"] == 1
    assert ws["rows"][0]["sample_id"] == "ws-hit"

    # 专用 DPO 会话查询不得加载任何 SFT 文件
    assert not [name for name in opened if "sft" in name.lower()]
    # 只压缩命中会话的行，未命中的行不进入昂贵压缩
    assert compacted == ["sess-hit", "sess-hit"]