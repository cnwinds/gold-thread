"""冲突作业入队，结束后自动出队执行。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.app import JOBS, JOBS_LOCK, Job, _jobs_clash, _pump_queue, _start, app


class JobQueueTests(unittest.TestCase):
    def tearDown(self) -> None:
        with JOBS_LOCK:
            JOBS.clear()

    def test_conflicting_job_is_queued_instead_of_409(self) -> None:
        started: list = []

        def capture_thread(target=None, args=(), daemon=None):  # noqa: ANN001
            started.append((target, args[0] if args else None))
            m = MagicMock()
            m.start = MagicMock()
            return m

        with patch("server.app.threading.Thread", side_effect=capture_thread), patch(
            "server.app.task_registry.register_job"
        ), patch("server.app.task_registry.sync_job"):
            first = _start("analyze", True, {"sessions": ["s1"]})
            self.assertEqual("running", first.status)
            self.assertEqual(1, len(started))

            second = _start("stage3", False)
            self.assertEqual("queued", second.status)
            self.assertEqual(1, len(started), "排队作业不应立刻开线程")
            self.assertTrue(any("排队" in line for line in second.logs))

    def test_pump_queue_starts_next_after_finish(self) -> None:
        started: list = []

        def capture_thread(target=None, args=(), daemon=None):  # noqa: ANN001
            started.append(args[0] if args else None)
            m = MagicMock()
            m.start = MagicMock()
            return m

        with patch("server.app.threading.Thread", side_effect=capture_thread), patch(
            "server.app.task_registry.register_job"
        ), patch("server.app.task_registry.sync_job"), patch(
            "server.app.load_config", return_value={}
        ):
            first = _start("analyze", True, {"sessions": ["s1"]})
            second = _start("stage3", False)
            self.assertEqual("queued", second.status)

            first.status = "done"
            _pump_queue()
            self.assertEqual("running", second.status)
            self.assertEqual(2, len(started))
            self.assertIs(started[1], second)

    def test_collect_api_queues_when_busy(self) -> None:
        running = Job("collect", False, {"source": "cursor", "ids": ["34c2f50a"]})
        with JOBS_LOCK:
            JOBS[running.id] = running
        with patch("server.app.task_registry.register_job"), patch(
            "server.app.threading.Thread"
        ) as thr:
            thr.return_value = MagicMock()
            res = TestClient(app).post(
                "/api/collect",
                json={"source": "pi", "ids": ["other"], "analyze": False},
            )
        self.assertEqual(200, res.status_code, res.text)
        body = res.json()
        self.assertEqual("queued", body["status"])
        self.assertTrue(body.get("job_id"))
        with JOBS_LOCK:
            queued = JOBS.get(body["job_id"])
        self.assertIsNotNone(queued)
        self.assertEqual("queued", queued.status)

    def test_all_and_delete_are_exclusive_in_both_orders(self) -> None:
        stages = (
            "stage1", "stage2", "stage3", "build", "dataset_evaluate",
            "analyze", "classify", "collect", "package", "all", "delete",
        )
        for exclusive in ("all", "delete"):
            for stage in stages:
                with self.subTest(exclusive=exclusive, stage=stage):
                    self.assertTrue(_jobs_clash(exclusive, stage))
                    self.assertTrue(_jobs_clash(stage, exclusive))
        self.assertTrue(_jobs_clash("package", "stage3"))
        self.assertFalse(_jobs_clash("package", "stage1"))

    def test_delete_while_writer_running_is_queued(self) -> None:
        running = Job("stage1", False)
        with JOBS_LOCK:
            JOBS[running.id] = running
        with patch("server.app.task_registry.register_job"), patch(
            "server.app.threading.Thread"
        ):
            job = _start("delete", False, {"sessions": ["01a1c68c-aaaa-bbbb-cccc-dddddddddddd"]})
        self.assertEqual("queued", job.status)

    def test_writer_while_delete_queued_is_queued(self) -> None:
        deleting = Job("delete", False, {"sessions": ["x"]})
        deleting.status = "queued"
        with JOBS_LOCK:
            JOBS[deleting.id] = deleting
        with patch("server.app.task_registry.register_job"), patch(
            "server.app.threading.Thread"
        ):
            job = _start("stage1", False)
        self.assertEqual("queued", job.status)

    def test_sync_writes_still_blocked_while_delete_queued(self) -> None:
        deleting = Job("delete", False, {"sessions": ["x"]})
        deleting.status = "queued"
        with JOBS_LOCK:
            JOBS[deleting.id] = deleting
        from server.app import _assert_no_delete_running

        with self.assertRaises(HTTPException) as ctx:
            _assert_no_delete_running()
        self.assertEqual(409, ctx.exception.status_code)
        self.assertIn("删除", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
