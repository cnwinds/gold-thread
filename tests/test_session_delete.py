"""跨阶段会话删除、事务恢复与 API 任务接入测试。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from pipeline import task_registry
from pipeline.common.store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from pipeline.session_delete import _Transaction, delete_session
from server.app import JOBS, JOBS_LOCK, Job, _start, app


SID = "sid-one"
OTHER = "sid-other"


class SessionDeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.state = self.data / "00_state"
        self.raw = self.data / "01_raw"
        self.gold = self.data / "02_gold"
        self.train = self.data / "03_train"
        for path in (
            self.state, self.raw / "positive" / "game_demo", self.gold / "sessions" / "game_demo",
            self.gold / "rejects", self.gold / "annotations", self.train / "exports" / "Qwen",
            self.train / "releases",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.cfg = {"paths": {"data_dir": str(self.data)}}
        self.source = self.root / "source" / f"{SID}.jsonl"
        self.source.parent.mkdir(parents=True)
        self.source.write_text('{"source":true}\n', encoding="utf-8")
        self._make_fixture()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _session_files(self, sid: str) -> dict[str, Path]:
        return {
            "raw_file": self.raw / "positive" / "game_demo" / f"{sid}.jsonl",
            "meta_file": self.raw / "positive" / "game_demo" / f"{sid}.meta.json",
            "gold_file": self.gold / "sessions" / "game_demo" / f"{sid}.gold.jsonl",
            "reject_file": self.gold / "rejects" / f"{sid}.pruned.jsonl",
            "dpo_file": self.gold / "rejects" / f"{sid}.dpo.jsonl",
            "diag_file": self.gold / "rejects" / f"{sid}.diagnostics.jsonl",
            "annotation_file": self.gold / "annotations" / f"{sid}.json",
        }

    def _make_fixture(self) -> None:
        raw_sessions = {}
        gold_sessions = {}
        for sid in (SID, OTHER):
            paths = self._session_files(sid)
            write_jsonl_atomic(paths["raw_file"], [{"session_id": sid}])
            write_json_atomic(paths["meta_file"], {"session_id": sid, "label": "positive"})
            write_jsonl_atomic(paths["gold_file"], [{"session_id": sid, "messages": []}])
            for key in ("reject_file", "dpo_file", "diag_file"):
                write_jsonl_atomic(paths[key], [{"session_id": sid}])
            write_json_atomic(paths["annotation_file"], {"session_id": sid})
            raw_sessions[sid] = {
                "session_id": sid, "label": "positive", "project_group": "game::demo",
                "raw_file": str(paths["raw_file"]), "meta_file": str(paths["meta_file"]),
            }
            gold_sessions[sid] = {
                "session_id": sid, "label": "positive", "project_group": "game::demo",
                **{key: str(paths[key]) for key in ("gold_file", "reject_file", "dpo_file", "diag_file", "annotation_file")},
            }
        write_json_atomic(self.raw / "index.json", {"sessions": raw_sessions, "counts": {"positive": 2}})
        write_json_atomic(self.gold / "index.json", {
            "sessions": gold_sessions, "counts": {"positive": 2}, "dpo_pairs": 2, "diagnostics": 2,
        })
        for name, field in (("dpo_pairs.jsonl", "metadata"), ("diagnostics.jsonl", "top")):
            rows = []
            for sid in (SID, OTHER):
                row = {"sample_id": f"dpo::{sid}::one", "pathologies": ["loop"]}
                if field == "metadata":
                    row["metadata"] = {"session_id": sid, "source": "pruned"}
                else:
                    row["session_id"] = sid
                rows.append(row)
            write_jsonl_atomic(self.gold / name, rows)

        write_json_atomic(self.state / "labels.json", {SID: "positive", OTHER: "negative"})
        write_json_atomic(self.state / "turn_overrides.json", {SID: {"turns": {}}, OTHER: {"turns": {}}})
        write_json_atomic(self.state / "collected.json", {"ids": [SID, OTHER], "items": {SID: {}, OTHER: {}}})
        write_json_atomic(self.state / "stage1.json", {"files": {SID: {}, OTHER: {}}})
        write_json_atomic(self.state / "stage2.json", {"files": {SID: {}, OTHER: {}}})
        write_json_atomic(self.state / "dpo_exclusions.json", {f"dpo::{SID}::one": {}, f"dpo::{OTHER}::one": {}})
        write_json_atomic(self.state / "dataset_draft.json", {
            "version": 1,
            "decisions": {
                key: {"action": "exclude", "reason": "test"}
                for sid in (SID, OTHER)
                for key in (f"step::{sid}::m1", f"traj::{sid}", f"dpo::{sid}::one")
            },
            "split_plan": {"assignments": {"game::demo": "train"}, "pinned": []},
        })
        write_json_atomic(self.gold / "sample_reviews.json", {
            "version": 1,
            "items": {f"sft::{SID}::one": {"decision": "approved"}, f"sft::{OTHER}::one": {"decision": "approved"}},
            "review_snapshot": {"store_items": 2},
        })

        for name in (
            "sft_step_windows_sharegpt.jsonl", "sft_gold_trajectories_sharegpt.jsonl",
            "dpo_preference_pairs.jsonl", "sft.train.jsonl", "exports/Qwen/sft.train.lf.jsonl",
            "releases/sft.train.jsonl",
        ):
            rows = []
            for sid in (SID, OTHER):
                rows.append({
                    "sample_id": f"step::{sid}::m1",
                    "metadata": {
                        "session_id": sid, "sample_id": f"step::{sid}::m1",
                        "project_group": "game::demo", "stage": "opening", "model": "glm",
                        "source": "pruned", "sample_weight": 2,
                    },
                })
            write_jsonl_atomic(self.train / name, rows)
        write_jsonl_atomic(self.train / "negative_diagnostics.jsonl", [
            {"session_id": sid, "pathologies": ["loop"]} for sid in (SID, OTHER)
        ])

        funnels = {
            sid: {
                "session_id": sid, "label": "positive", "project_group": "game::demo",
                "split": "train", "effective_state": "auto_gold",
            } for sid in (SID, OTHER)
        }
        stats = {
            "generated_at": "before", "counts": {"sft_step_windows_sharegpt.jsonl": 2, "sft.train.jsonl": 2},
            "split": {"game::demo": "train"}, "by_project": {"game::demo": 2}, "by_stage": {"opening": 2},
            "by_model": {"glm": 2}, "session_funnels": funnels, "weighted_steps": 4,
            "dpo_by_source": {"pruned": 2}, "diagnostics_by_pathology": {"loop": 2},
            "label_counts": {"positive": 2}, "review_states": {"auto_gold": 2},
        }
        write_json_atomic(self.train / "stats.json", stats)
        write_json_atomic(self.train / "index.json", {"_files": dict(stats["counts"]), "stats": stats})
        write_json_atomic(self.train / "manifest.json", {
            "sessions": [SID, OTHER], "excluded_samples": [f"dpo::{SID}::one", f"dpo::{OTHER}::one"],
            "files": {"sft_step_windows_sharegpt.jsonl": {"count": 2, "sha256": "old"}},
            "counts": {"sft_train": 2},
        })

    def test_delete_removes_only_target_session_across_current_workspace(self) -> None:
        result = delete_session(SID, cfg=self.cfg)

        self.assertTrue(result["ok"])
        self.assertTrue(result["source_preserved"])
        for key, path in self._session_files(SID).items():
            with self.subTest(target=key):
                self.assertFalse(path.exists())
        for path in self._session_files(OTHER).values():
            self.assertTrue(path.exists())
        self.assertTrue(self.source.exists())

        raw_index = read_json(self.raw / "index.json")
        gold_index = read_json(self.gold / "index.json")
        self.assertEqual([OTHER], list(raw_index["sessions"]))
        self.assertEqual([OTHER], list(gold_index["sessions"]))
        self.assertEqual({"positive": 1}, raw_index["counts"])
        self.assertEqual(1, gold_index["dpo_pairs"])
        self.assertEqual(1, gold_index["diagnostics"])

        for name in ("labels.json", "turn_overrides.json", "stage1.json", "stage2.json", "dpo_exclusions.json"):
            data = read_json(self.state / name)
            container = data.get("files", data)
            self.assertFalse(any(SID in str(key).split("::") for key in container), name)
            self.assertTrue(any(OTHER in str(key).split("::") for key in container), name)
        collected = read_json(self.state / "collected.json")
        self.assertEqual([OTHER], collected["ids"])
        self.assertEqual([OTHER], list(collected["items"]))
        draft = read_json(self.state / "dataset_draft.json")
        self.assertEqual(
            {f"step::{OTHER}::m1", f"traj::{OTHER}", f"dpo::{OTHER}::one"},
            set(draft["decisions"]),
        )
        reviews = read_json(self.gold / "sample_reviews.json")
        self.assertEqual([f"sft::{OTHER}::one"], list(reviews["items"]))
        self.assertEqual(1, reviews["review_snapshot"]["store_items"])

        for path in self.train.rglob("*.jsonl"):
            rows = read_jsonl(path)
            if "releases" in path.relative_to(self.train).parts:
                self.assertEqual({SID, OTHER}, {r["metadata"]["session_id"] for r in rows})
            else:
                self.assertEqual([OTHER], [r.get("session_id") or (r.get("metadata") or {}).get("session_id") for r in rows], str(path))

        stats = read_json(self.train / "stats.json")
        self.assertEqual([OTHER], list(stats["session_funnels"]))
        self.assertEqual(1, stats["counts"]["sft_step_windows_sharegpt.jsonl"])
        self.assertEqual(1, stats["counts"]["sft.train.jsonl"])
        self.assertEqual({"game::demo": 1}, stats["by_project"])
        self.assertEqual({"opening": 1}, stats["by_stage"])
        self.assertEqual({"glm": 1}, stats["by_model"])
        self.assertEqual(2, stats["weighted_steps"])
        self.assertEqual({"pruned": 1}, stats["dpo_by_source"])
        self.assertEqual({"loop": 1}, stats["diagnostics_by_pathology"])
        index = read_json(self.train / "index.json")
        self.assertEqual(stats, index["stats"])
        manifest = read_json(self.train / "manifest.json")
        self.assertEqual([OTHER], manifest["sessions"])
        self.assertEqual([f"dpo::{OTHER}::one"], manifest["excluded_samples"])
        self.assertEqual(1, manifest["files"]["sft_step_windows_sharegpt.jsonl"]["count"])
        self.assertEqual(1, manifest["counts"]["sft_train"])
        digest = hashlib.sha256((self.train / "sft_step_windows_sharegpt.jsonl").read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["files"]["sft_step_windows_sharegpt.jsonl"]["sha256"])
        self.assertEqual([], list(self.state.glob(".session-delete-*")))

    def test_delete_drops_unprovable_stats_when_funnels_are_missing(self) -> None:
        stats_path = self.train / "stats.json"
        stats = read_json(stats_path)
        for key in ("session_funnels", "label_counts", "review_states", "split"):
            stats.pop(key)
        write_json_atomic(stats_path, stats)

        delete_session(SID, cfg=self.cfg)
        updated = read_json(stats_path)
        for key in ("label_counts", "review_states", "split"):
            self.assertNotIn(key, updated)
        self.assertEqual(1, updated["counts"]["sft_step_windows_sharegpt.jsonl"])

    def test_delete_rejects_paths_outside_data_root_before_mutating(self) -> None:
        index_path = self.raw / "index.json"
        index = read_json(index_path)
        index["sessions"][SID]["raw_file"] = str(self.source)
        write_json_atomic(index_path, index)

        with self.assertRaisesRegex(ValueError, "越出数据目录"):
            delete_session(SID, cfg=self.cfg)
        self.assertTrue(self.source.exists())
        self.assertTrue(self._session_files(SID)["gold_file"].exists())
        self.assertIn(SID, read_json(index_path)["sessions"])

    def test_transaction_restores_applied_files_on_failure(self) -> None:
        a, b, c = (self.root / name for name in ("a.txt", "b.txt", "c.txt"))
        for path in (a, b, c):
            path.write_text("old", encoding="utf-8")
        tx = _Transaction(self.state)
        tx.write_text(a, "new-a")
        tx.write_text(b, "new-b")
        tx.delete(c)
        real_replace = os.replace

        def fail_on_b(src: Path, dst: Path) -> None:
            if Path(dst) == b:
                raise OSError("injected write failure")
            real_replace(src, dst)

        with patch("pipeline.session_delete.os.replace", side_effect=fail_on_b):
            with self.assertRaisesRegex(OSError, "injected"):
                tx.commit()
        self.assertEqual("old", a.read_text(encoding="utf-8"))
        self.assertEqual("old", b.read_text(encoding="utf-8"))
        self.assertEqual("old", c.read_text(encoding="utf-8"))
        self.assertFalse(tx.root.exists())


class SessionDeleteApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        (self.data / "01_raw").mkdir(parents=True)
        (self.data / "02_gold").mkdir(parents=True)
        self.cfg = {"paths": {"data_dir": str(self.data)}}

    def tearDown(self) -> None:
        self.tmp.cleanup()
        with JOBS_LOCK:
            JOBS.clear()

    @patch("server.app._start")
    @patch("server.app.load_config")
    def test_delete_api_starts_linked_delete_job(self, mock_cfg: MagicMock, mock_start: MagicMock) -> None:
        mock_cfg.return_value = self.cfg
        write_json_atomic(self.data / "01_raw" / "index.json", {"sessions": {SID: {}}})
        mock_start.return_value = Job("delete", False, {"sessions": [SID]})
        response = TestClient(app).delete(f"/api/session/{SID}")
        self.assertEqual(200, response.status_code)
        self.assertIn("job_id", response.json())
        mock_start.assert_called_once_with("delete", False, {"sessions": [SID]})

    @patch("server.app.load_config")
    def test_delete_api_returns_404_for_unknown_session(self, mock_cfg: MagicMock) -> None:
        mock_cfg.return_value = self.cfg
        write_json_atomic(self.data / "01_raw" / "index.json", {"sessions": {}})
        write_json_atomic(self.data / "02_gold" / "index.json", {"sessions": {}})
        response = TestClient(app).delete(f"/api/session/{SID}")
        self.assertEqual(404, response.status_code)

    def test_delete_api_rejects_invalid_sid(self) -> None:
        response = TestClient(app).delete("/api/session/bad%20sid")
        self.assertEqual(400, response.status_code)

    def test_sync_session_writes_are_rejected_while_delete_is_running(self) -> None:
        running = Job("delete", False, {"sessions": [SID]})
        with JOBS_LOCK:
            JOBS[running.id] = running
        client = TestClient(app)
        requests = [
            ("post", f"/api/raw/session/{SID}/relabel", {"label": "positive"}),
            ("put", f"/api/raw/session/{SID}/turns", {}),
            ("put", f"/api/gold/session/{SID}/review/items", {"items": {}}),
            ("post", f"/api/gold/session/{SID}/review/batch", {"action": "reset_all"}),
            ("post", f"/api/gold/session/{SID}/review/decision", {"decision": None}),
            ("post", f"/api/gold/session/{SID}/review/refine", {"items": {}}),
            ("post", f"/api/gold/session/{SID}/finalize", {"finalized": False}),
        ]
        for method, path, body in requests:
            with self.subTest(path=path):
                response = getattr(client, method)(path, json=body)
                self.assertEqual(409, response.status_code)
                self.assertIn("正在删除会话", response.json()["detail"])

    def test_delete_job_conflicts_with_other_writers(self) -> None:
        running = Job("stage1", False)
        with JOBS_LOCK:
            JOBS[running.id] = running
        with self.assertRaises(HTTPException) as ctx:
            _start("delete", False, {"sessions": [SID]})
        self.assertEqual(409, ctx.exception.status_code)
        with JOBS_LOCK:
            JOBS.clear()
            deleting = Job("delete", False, {"sessions": [SID]})
            JOBS[deleting.id] = deleting
        with self.assertRaises(HTTPException) as reverse:
            _start("stage1", False)
        self.assertEqual(409, reverse.exception.status_code)

    def test_delete_task_has_no_link_to_deleted_session(self) -> None:
        job = Job("delete", False, {"sessions": [SID]})
        record = task_registry.task_record_from_job(job)
        self.assertIn("删除会话", record["title"])
        self.assertEqual(["overview", "raw", "gold", "train"], record["affects"])
        self.assertNotIn("sids", record["links"])


if __name__ == "__main__":
    unittest.main()
