"""pipeline.steps begin/finish：可见日志含工作说明与耗时。"""
from __future__ import annotations

import json
import time
import unittest
from typing import List
from unittest import mock

from pipeline import steps as stepcat
from server.app import Job


class FormatElapsedTests(unittest.TestCase):
    def test_sub_second_and_minutes(self):
        self.assertEqual(stepcat.format_elapsed(1.23), "1.2s")
        self.assertEqual(stepcat.format_elapsed(12.4), "12s")
        self.assertEqual(stepcat.format_elapsed(65), "1分05秒")
        self.assertEqual(stepcat.format_elapsed(3661), "1小时01分01秒")


class BeginFinishTests(unittest.TestCase):
    def test_begin_explains_work_and_emits_step(self):
        lines: List[str] = []
        stepcat.begin(lines.append, "s2.judge_steps", msg="会话 abcd1234")
        visible = [x for x in lines if not x.startswith("@@STEP ")]
        self.assertTrue(any(x.startswith("▶ 开始 ·") and "模型步级判断" in x for x in visible))
        self.assertTrue(any("要做：" in x and "keep/prune" in x for x in visible))
        self.assertTrue(any("执行方：模型" in x for x in visible))
        step_lines = [x for x in lines if x.startswith("@@STEP ")]
        self.assertEqual(len(step_lines), 1)
        payload = json.loads(step_lines[0][7:])
        self.assertEqual(payload["id"], "s2.judge_steps")

    def test_finish_includes_elapsed(self):
        lines: List[str] = []
        stepcat.begin(lines.append, "s2.normalize")
        time.sleep(0.05)
        stepcat.finish(lines.append, "s2.normalize", msg="静默 3 步")
        done_lines = [x for x in lines if x.startswith("✓ 完成 ·")]
        self.assertEqual(len(done_lines), 1)
        self.assertIn("机械规整", done_lines[0])
        self.assertIn("静默 3 步", done_lines[0])
        self.assertIn("耗时", done_lines[0])
        step_lines = [x for x in lines if x.startswith("@@STEP ")]
        last = json.loads(step_lines[-1][7:])
        self.assertEqual(last["done"], 1)
        self.assertEqual(last["total"], 1)

    def test_emit_alone_has_no_visible_progress_spam(self):
        lines: List[str] = []
        stepcat.emit(lines.append, "s1.classify", 1, 10, "分类 abcd")
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("@@STEP "))


class JobLogIntegrationTests(unittest.TestCase):
    def test_job_keeps_begin_finish_filters_step_marker(self):
        job = Job("analyze", False, {"sessions": ["abcd"]})
        stepcat.begin(job.log, "s2.apply")
        stepcat.finish(job.log, "s2.apply", done=10, total=10, msg="剪除 2")
        self.assertTrue(any("▶ 开始 ·" in x for x in job.logs))
        self.assertTrue(any("✓ 完成 ·" in x and "耗时" in x for x in job.logs))
        self.assertFalse(any("@@STEP" in x for x in job.logs))
        self.assertEqual(job.progress.get("id"), "s2.apply")
        self.assertIn("s2.apply", job.done_steps)


class Stage2WireTests(unittest.TestCase):
    def test_purify_by_judge_emits_begin_finish_pairs(self):
        from pipeline import stage2_gold

        flow = [
            {"id": "u1", "role": "user", "content": "分析 a.apk", "text": "分析 a.apk"},
            {
                "id": "a1", "role": "assistant", "thinking": "先看文件头", "text": "查看中",
                "tool_calls": [{"id": "t1", "name": "bash", "arguments": {"command": "xxd a.apk | head"}}],
                "tool_results": [],
            },
            {"id": "tr1", "role": "tool", "tool_call_id": "t1", "content": "7f454c46"},
            {
                "id": "a2", "role": "assistant", "thinking": "确认是 ELF", "text": "结论",
                "tool_calls": [], "tool_results": [],
            },
        ]
        lines: List[str] = []
        cfg = {
            "gold": {"silent_intermediate_steps": True},
            "judge": {"enabled": True},
        }

        with mock.patch("pipeline.stage2_gold.judge.decide_steps", return_value={
            "a1": {"action": "keep"},
            "a2": {"action": "keep"},
        }), mock.patch("pipeline.stage2_gold.judge.extract_seams", return_value=[]), mock.patch(
            "pipeline.stage2_gold.judge.audit_seams", return_value={}
        ):
            stage2_gold.purify_by_judge(flow, cfg, session_cwd="/tmp", log=lines.append)

        visible = [x for x in lines if not str(x).startswith("@@STEP ")]
        starts = [x for x in visible if str(x).startswith("▶ 开始 ·")]
        ends = [x for x in visible if str(x).startswith("✓ 完成 ·")]
        # normalize / judge_steps / apply / seams_extract（无接缝则无 judge_seams）
        self.assertGreaterEqual(len(starts), 4)
        self.assertEqual(len(starts), len(ends))
        self.assertTrue(any("模型步级判断" in x for x in starts))
        self.assertTrue(any("耗时" in x for x in ends))


if __name__ == "__main__":
    unittest.main()
