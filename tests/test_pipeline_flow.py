from __future__ import annotations

import copy
import unittest

from pipeline import curation
from pipeline.common.config import DEFAULTS
from pipeline.common.format import to_openai, to_sharegpt
from pipeline.common.pilog import build_flow
from pipeline.common.quality import analyze_step_quality
from pipeline.stage2_gold import purify
from pipeline.stage3_synth import evaluate_session_synthesis


def assistant(mid: str, thinking: str, text: str = "", tool_calls=None, tool_results=None):
    return {
        "id": mid,
        "role": "assistant",
        "model": "deepseek-v4.1-flash-1017",
        "provider": "test",
        "thinking": thinking,
        "text": text,
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "timestamp": "2026-01-01T00:00:00Z",
    }


class FormatTests(unittest.TestCase):
    def test_thinking_tags_are_well_formed(self):
        msg = assistant("a1", "先验证字节头", "继续分析")
        oai = to_openai(msg)
        sgpt = to_sharegpt(msg)[0]
        self.assertIn("<think>\n先验证字节头\n</think>", oai["content"])
        self.assertIn("<think>\n先验证字节头\n</think>", sgpt["value"])
        self.assertNotIn("<think\n", oai["content"])


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["turns"]["root_task_patterns"] = [r"\.xapk", "破解"]
        self.cfg["turns"]["continuation_patterns"] = [r"^继续$"]
        self.cfg["turns"]["correction_patterns"] = ["不要参考其他的目录"]

    def test_root_normalization_continuation_and_correction(self):
        session = {
            "session_id": "s1",
            "cwd": r"D:\crack\jigsawcard",
            "timestamp": "2026-01-01T00:00:00Z",
            "messages": [
                {"id": "u0", "role": "user", "content": "hi", "timestamp": ""},
                assistant("a0", "你好"),
                {"id": "u1", "role": "user", "content": r"帮我破解 D:\x\jigsawcard.xapk", "timestamp": ""},
                assistant("a1", "先看文件"),
                {"id": "u2", "role": "user", "content": "继续", "timestamp": ""},
                assistant("a2", "继续执行"),
                {"id": "u3", "role": "user", "content": "不要参考其他的目录，自己研究", "timestamp": ""},
                assistant("a3", "遵守隔离约束"),
            ],
        }
        flow, marks, stats = build_flow(session, self.cfg)
        users = [m for m in flow if m["role"] == "user"]
        self.assertEqual(1, len(users))
        self.assertIn("com.oakever.jigsawcard", users[0]["content"])
        self.assertEqual("dropped", marks[0]["action"])
        self.assertEqual("folded", marks[2]["action"])
        self.assertEqual("internalized", marks[3]["action"])
        self.assertEqual(2, stats["turns_folded"])
        self.assertIn("不要参考其他的目录", flow[0]["content"])

    def test_manual_override_has_highest_priority(self):
        session = {
            "session_id": "s2",
            "cwd": r"D:\crack\sample",
            "timestamp": "",
            "messages": [
                {"id": "u0", "role": "user", "content": "处理这个任务", "timestamp": ""},
                assistant("a0", "执行"),
                {"id": "u1", "role": "user", "content": "继续", "timestamp": ""},
                assistant("a1", "执行第二段"),
            ],
        }
        _, marks, _ = build_flow(
            session,
            self.cfg,
            annotations={"root_turn": 0, "turns": [{"index": 1, "kind": "continuation", "drop": False}]},
            manual_overrides={"turns": {"1": {"kind": "task"}}},
        )
        self.assertEqual("manual", marks[1]["source"])
        self.assertEqual("task", marks[1]["kind"])
        self.assertEqual("kept", marks[1]["action"])


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["review"]["hold_needs_review"] = True
        self.cfg["synth"]["formats"] = {"sharegpt": True, "openai": True}

    def test_rewritten_pathological_step_is_rescued(self):
        raw_thinking = ("Wait. Actually. " * 10) + ("重复推理。" * 800)
        msg = assistant(
            "a1",
            raw_thinking,
            tool_calls=[{"id": "c1", "name": "bash", "arguments": {"command": "python -c 'print(1)'"}}],
        )
        raw_q = analyze_step_quality(
            raw_thinking,
            "",
            msg["tool_calls"],
            [],
            r"D:\crack\sample",
            max_think_chars=4500,
        )
        self.assertFalse(raw_q["is_clean"])
        msg["raw_quality"] = raw_q
        msg["quality"] = raw_q
        msg["raw_thinking"] = raw_thinking
        record = {
            "session_id": "s3",
            "label": "positive",
            "project_group": "game::sample",
            "cwd": r"D:\crack\sample",
            "dominant_model": "deepseek-v4.1-flash-1017",
            "stats": {"raw_steps": 1, "gold_steps": 1},
            "messages": [
                {"id": "sys", "role": "system", "content": "system"},
                {"id": "root", "role": "user", "content": "分析 sample.apk"},
                msg,
            ],
        }
        rewrite = "工具输出尚未验证。我将用 Python 脚本读取实际字节并打印结构，再据此决定解析方式。"
        annotation = {
            "review_key": "1:abc:6",
            "human_review": {
                "decision": "approved",
                "review_key": "1:abc:6",
                "items": {"a1": {"rewrite": "edited", "edited_text": rewrite}},
                "history": [],
            },
        }
        out = evaluate_session_synthesis(record, annotation, self.cfg)
        self.assertEqual(1, out["funnel"]["admitted_steps"])
        self.assertEqual(1, out["funnel"]["rescued_steps"])
        self.assertEqual("rescued_by_human_edit", out["step_audits"][0]["reason"])
        self.assertIn(rewrite, out["windows_sgpt"][0]["conversations"][-1]["value"])

    def test_stage2_keeps_raw_and_script_thinking(self):
        long_thinking = "事实观察。\n\n" + ("Wait，重新考虑。" * 500) + "\n\n下一步用脚本验证。"
        flow = [
            {"id": "sys", "role": "system", "content": "system"},
            {"id": "root", "role": "user", "content": "分析 a.apk"},
            assistant("a1", long_thinking),
        ]
        gold, _, stats = purify(flow, self.cfg["gold"])
        msg = next(m for m in gold if m["role"] == "assistant")
        self.assertEqual(long_thinking, msg["raw_thinking"])
        self.assertLess(len(msg["thinking"]), len(long_thinking))
        self.assertEqual(1, stats["thinking_compressed"])

    def test_direct_approval_does_not_apply_stale_model_rewrite(self):
        ann = {
            "review_key": "1:new:flow:6",
            "review_rounds_stale": True,
            "review_rounds": [
                {
                    "round": 1,
                    "verdict": "needs_review",
                    "thinking_rewrites": [{"msg_id": "a1", "rewritten": "旧版本改写"}],
                    "bad_step_ids": ["a1"],
                }
            ],
            "human_review": {
                "items": {},
                "decision": None,
                "history": [],
                "note": "",
            },
        }
        curation.decide(ann, "approved")
        effective = curation.resolve(ann, self.cfg)
        self.assertEqual("approved", effective["state"])
        self.assertEqual({}, effective["rewrites"])
        self.assertEqual([], effective["excluded"])


if __name__ == "__main__":
    unittest.main()
