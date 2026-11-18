"""POST /api/collect 与 collect 作业的 analyze 开关。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server.app import Job, _run_job, app


class CollectJobTests(unittest.TestCase):
    @patch("server.app.task_registry.sync_job")
    @patch("server.app._persist_job")
    @patch("server.app.stage2_gold.run")
    @patch("pipeline.collect.ingest_selected")
    @patch("server.app.load_config")
    def test_collect_ingest_only_skips_stage2(
        self, mock_load_cfg, mock_ingest, mock_stage2, _mock_persist, _mock_sync
    ):
        mock_load_cfg.return_value = {"paths": {"data_dir": "data"}}
        mock_ingest.return_value = ["sid01"]
        job = Job("collect", False, {"source": "pi", "ids": ["cat1"], "analyze": False})
        _run_job(job)
        mock_stage2.assert_not_called()
        self.assertEqual("done", job.status)
        self.assertEqual({"sessions": ["sid01"]}, job.result)

    @patch("server.app.task_registry.sync_job")
    @patch("server.app._persist_job")
    @patch("server.app.stage2_gold.run")
    @patch("pipeline.collect.ingest_selected")
    @patch("server.app.load_config")
    def test_collect_with_analyze_runs_stage2(
        self, mock_load_cfg, mock_ingest, mock_stage2, _mock_persist, _mock_sync
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

    def test_second_collect_while_running_is_queued(self) -> None:
        import server.app as sa
        from unittest.mock import MagicMock, patch

        running = Job("collect", False, {"source": "cursor", "ids": ["34c2f50a"]})
        with sa.JOBS_LOCK:
            sa.JOBS[running.id] = running
        with patch("server.app.task_registry.register_job"), patch(
            "server.app._launch_job_thread"
        ) as thr:
            res = TestClient(app).post(
                "/api/collect",
                json={"source": "pi", "ids": ["other"], "analyze": False},
            )
            thr.assert_not_called()
        self.assertEqual(200, res.status_code, res.text)
        body = res.json()
        self.assertEqual("queued", body["status"])
        self.assertTrue(body.get("job_id"))
        self.assertNotEqual(running.id, body["job_id"])


if __name__ == "__main__":
    unittest.main()
