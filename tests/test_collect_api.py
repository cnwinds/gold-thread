"""POST /api/collect 与 collect 作业的 analyze 开关。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server.app import Job, _run_job, app


class CollectJobTests(unittest.TestCase):
    @patch("server.app._persist_job")
    @patch("server.app.stage2_gold.run")
    @patch("pipeline.collect.ingest_selected")
    @patch("server.app.load_config")
    def test_collect_ingest_only_skips_stage2(
        self, mock_load_cfg, mock_ingest, mock_stage2, _mock_persist
    ):
        mock_load_cfg.return_value = {"paths": {"data_dir": "data"}}
        mock_ingest.return_value = ["sid01"]
        job = Job("collect", False, {"source": "pi", "ids": ["cat1"], "analyze": False})
        _run_job(job)
        mock_stage2.assert_not_called()
        self.assertEqual("done", job.status)
        self.assertEqual({"sessions": ["sid01"]}, job.result)

    @patch("server.app._persist_job")
    @patch("server.app.stage2_gold.run")
    @patch("pipeline.collect.ingest_selected")
    @patch("server.app.load_config")
    def test_collect_with_analyze_runs_stage2(
        self, mock_load_cfg, mock_ingest, mock_stage2, _mock_persist
    ):
        mock_load_cfg.return_value = {"paths": {"data_dir": "data"}}
        mock_ingest.return_value = ["sid01"]
        mock_stage2.return_value = {"sessions": 1}
        job = Job("collect", False, {"source": "pi", "ids": ["cat1"], "analyze": True})
        _run_job(job)
        mock_stage2.assert_called_once()
        self.assertEqual(mock_stage2.call_args.kwargs.get("only"), ["sid01"])
        self.assertEqual(job.result["sessions"], ["sid01"])
        self.assertEqual(job.result["stage2"], {"sessions": 1})


class CollectApiTests(unittest.TestCase):
    @patch("server.app._start")
    def test_post_collect_defaults_analyze_false(self, mock_start):
        mock_start.return_value = MagicMock(id="job1")
        client = TestClient(app)
        res = client.post("/api/collect", json={"source": "pi", "ids": ["a"]})
        self.assertEqual(200, res.status_code)
        mock_start.assert_called_once_with(
            "collect", False, {"source": "pi", "ids": ["a"], "analyze": False}
        )

    @patch("server.app._start")
    def test_post_collect_analyze_true(self, mock_start):
        mock_start.return_value = MagicMock(id="job2")
        client = TestClient(app)
        res = client.post(
            "/api/collect",
            json={"source": "cursor", "ids": ["b"], "analyze": True},
        )
        self.assertEqual(200, res.status_code)
        mock_start.assert_called_once_with(
            "collect", False, {"source": "cursor", "ids": ["b"], "analyze": True}
        )


class CollectConflictTests(unittest.TestCase):
    def tearDown(self) -> None:
        import server.app as sa

        with sa.JOBS_LOCK:
            sa.JOBS.clear()

    def test_second_collect_while_running_returns_409(self) -> None:
        import server.app as sa

        running = Job("collect", False, {"source": "cursor", "ids": ["34c2f50a"]})
        with sa.JOBS_LOCK:
            sa.JOBS[running.id] = running
        client = TestClient(app)
        res = client.post(
            "/api/collect",
            json={"source": "pi", "ids": ["other"], "analyze": False},
        )
        self.assertEqual(409, res.status_code)
        detail = res.json().get("detail") or ""
        self.assertIn("已有作业在运行", detail)
        self.assertIn(running.id, detail)


if __name__ == "__main__":
    unittest.main()
