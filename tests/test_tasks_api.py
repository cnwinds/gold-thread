"""任务中心 registry 与 /api/tasks。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pipeline import task_registry
from server.app import Job, app


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


if __name__ == "__main__":
    unittest.main()
