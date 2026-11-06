"""任务中心 registry 与 /api/tasks。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pipeline import task_registry
from server.app import Job, _run_job, app


class TaskRegistryTests(unittest.TestCase):
    def test_collect_analyze_title_and_affects(self):
        meta = task_registry.describe_task(
            "collect", False, {"source": "pi", "ids": ["a", "b"], "analyze": True}
        )
        self.assertIn("采集并分析", meta["title"])
        self.assertIn("2", meta["title"])
        self.assertEqual(["overview", "raw", "gold"], meta["affects"])

    def test_collect_ingest_only(self):
        meta = task_registry.describe_task(
            "collect", False, {"source": "cursor", "ids": ["x"], "analyze": False}
        )
        self.assertIn("采集", meta["title"])
        self.assertNotIn("分析", meta["title"])
        self.assertEqual(["overview", "raw"], meta["affects"])

    def test_dataset_evaluation_title_and_affects(self):
        meta = task_registry.describe_task("dataset_evaluate", False, {})
        self.assertIn("DPO", meta["title"])
        self.assertEqual(["overview", "train"], meta["affects"])
        self.assertEqual("train", meta["links"]["open_tab"])

    def test_register_and_sync(self):
        cfg = {"paths": {"data_dir": "data"}}
        job = Job("analyze", False, {"sessions": ["abcd1234efgh"]})
        job.status = "running"
        with patch.object(task_registry, "tasks_path") as tp:
            tp.return_value = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "tasks.json"
            task_registry.register_job(job, cfg)
            job.status = "done"
            job.result = {"sessions": 1}
            job.finished_at = "2026-01-01T00:00:00+08:00"
            task_registry.sync_job(job, cfg)
            listed = task_registry.list_tasks(cfg, limit=10, hours=9999)
            self.assertEqual(0, listed["running"])
            self.assertTrue(any(t["id"] == job.id for t in listed["tasks"]))
            got = task_registry.get_task(job.id, cfg)
            self.assertEqual("done", got["status"])
            self.assertIn("gold", got["affects"])

    def test_sync_persists_logs_list_omits_bodies(self):
        cfg = {"paths": {"data_dir": "data"}}
        job = Job("analyze", False, {"sessions": ["abcd1234efgh"]})
        job.log("▶ 开始 · 模型步级判断")
        job.log("✓ 完成 · 模型步级判断 · 耗时 1.2s")
        job.status = "done"
        job.finished_at = "2026-01-01T00:00:00+08:00"
        with patch.object(task_registry, "tasks_path") as tp:
            tp.return_value = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "tasks.json"
            task_registry.sync_job(job, cfg)
            got = task_registry.get_task(job.id, cfg)
            self.assertEqual(2, got["log_count"])
            self.assertEqual(2, len(got["logs"]))
            self.assertTrue(any("开始" in line for line in got["logs"]))
            listed = task_registry.list_tasks(cfg, limit=10, hours=9999)
            row = next(t for t in listed["tasks"] if t["id"] == job.id)
            self.assertEqual(2, row["log_count"])
            self.assertNotIn("logs", row)

    def test_clip_logs_keeps_tail(self):
        lines = [f"L{i}" for i in range(10)]
        clipped = task_registry.clip_logs(lines, limit=3)
        self.assertEqual(["L7", "L8", "L9"], clipped)


    def test_full_pipeline_evaluates_dpo_before_stage3(self):
        calls = []
        job = Job("all", False, {})
        with patch("server.app.stage1_extract.run", side_effect=lambda **_: calls.append("stage1") or {}), patch(
            "server.app.stage2_gold.run", side_effect=lambda **_: calls.append("stage2") or {}
        ), patch(
            "server.app.stage3_synth.evaluate_dpo_pending",
            side_effect=lambda **_: calls.append("evaluate") or {"pending": 0},
        ), patch(
            "server.app.stage3_synth.run", side_effect=lambda **_: calls.append("stage3") or {}
        ):
            _run_job(job)
        self.assertEqual("done", job.status, job.error)
        self.assertEqual(["stage1", "stage2", "evaluate", "stage3"], calls)

    def test_full_pipeline_aborts_when_dpo_still_pending(self):
        job = Job("all", False, {})
        with patch("server.app.stage1_extract.run", return_value={}), patch(
            "server.app.stage2_gold.run", return_value={}
        ), patch(
            "server.app.stage3_synth.evaluate_dpo_pending",
            return_value={"pending": 2, "failed": 0},
        ), patch("server.app.stage3_synth.run") as stage3:
            _run_job(job)
        self.assertEqual("failed", job.status)
        self.assertIn("DPO 评估未全部完成", job.error or "")
        stage3.assert_not_called()

    def test_review_does_not_hide_dpo_gate_failure(self):
        job = Job("review", False, {})
        with patch("pipeline.review.run_review", return_value={}), patch(
            "server.app.stage3_synth.evaluate_dpo_pending",
            side_effect=RuntimeError("DPO 尚未完成模型复核"),
        ), patch("server.app.stage3_synth.run") as stage3:
            _run_job(job)
        self.assertEqual("failed", job.status)
        self.assertIn("尚未完成模型复核", job.error)
        stage3.assert_not_called()

    def test_review_aborts_when_dpo_evaluation_incomplete(self):
        job = Job("review", False, {})
        with patch("pipeline.review.run_review", return_value={}), patch(
            "server.app.stage3_synth.evaluate_dpo_pending",
            return_value={"pending": 1, "failed": 0},
        ), patch("server.app.stage3_synth.run") as stage3:
            _run_job(job)
        self.assertEqual("failed", job.status)
        self.assertIn("DPO 评估未全部完成", job.error or "")
        stage3.assert_not_called()


class TasksApiTests(unittest.TestCase):
    @patch("server.app._start")
    def test_list_tasks_after_start(self, mock_start):
        job = Job("collect", False, {"source": "pi", "ids": ["1"], "analyze": True})
        mock_start.return_value = job
        client = TestClient(app)
        with patch("server.app.task_registry.list_tasks") as lt:
            lt.return_value = {
                "running": 1,
                "tasks": [{"id": job.id, "title": "采集并分析 1 条", "status": "running"}],
            }
            with patch("server.app.task_registry.reconcile_stale"):
                res = client.get("/api/tasks")
        self.assertEqual(200, res.status_code)
        self.assertEqual(1, res.json()["running"])

    def test_get_task_returns_persisted_logs_when_not_live(self):
        cfg = {"paths": {"data_dir": "data"}}
        job = Job("stage3", False, {})
        job.log("[stage3] 开始")
        job.log("▶ 开始 · 入库门控")
        job.log("✓ 完成 · 入库门控 · 耗时 0.1s")
        job.status = "done"
        job.finished_at = "2026-01-01T00:00:00+08:00"
        with patch.object(task_registry, "tasks_path") as tp:
            store = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "tasks.json"
            tp.return_value = store
            task_registry.sync_job(job, cfg)
            client = TestClient(app)
            with patch("server.app.load_config", return_value=cfg), patch(
                "server.app.task_registry.tasks_path", return_value=store
            ), patch("server.app.JOBS", {}):
                res = client.get(f"/api/tasks/{job.id}")
        self.assertEqual(200, res.status_code)
        body = res.json()
        self.assertFalse(body.get("live"))
        self.assertGreaterEqual(body["log_count"], 3)
        self.assertTrue(any("入库门控" in line for line in body["logs"]))


if __name__ == "__main__":
    unittest.main()
