"""DPO 样本必须带 tools 列（与 SFT / dataset_info 声明一致）。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.common.format import default_tools_json, ensure_tools_column
from pipeline.lf_normalize import normalize_sharegpt_row
from pipeline.stage2_gold import _dpo_from_pruned
from pipeline.stage3_synth import _negative_dpo, _review_rewrite_dpo, synthesize_dpo_candidates


def _asst(mid: str, thinking: str, *, pathologies=None, tool_calls=None):
    q = {
        "is_clean": not bool(pathologies),
        "score": 40.0 if pathologies else 100.0,
        "pathologies": list(pathologies or []),
    }
    return {
        "id": mid,
        "role": "assistant",
        "thinking": thinking,
        "raw_thinking": thinking,
        "text": "",
        "tool_calls": tool_calls
        or [{"id": f"tc-{mid}", "name": "bash", "arguments": {"command": "ls"}}],
        "tool_results": [],
        "raw_quality": q,
        "quality": q,
    }


class DpoToolsColumnTests(unittest.TestCase):
    def test_default_tools_json_matches_sft_contract(self):
        tools = json.loads(default_tools_json())
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(["bash", "read", "write", "edit", "codemode"], names)
        self.assertGreater(len(default_tools_json()), 1000)

    def test_stage2_pruned_pairs_include_tools(self):
        sys_msg = {"id": "s", "role": "system", "content": "sys"}
        root = {"id": "u0", "role": "user", "content": "任务"}
        chosen = _asst("a_ok", "正确做法")
        rejected = _asst("a_bad", "弯路做法")
        pruned = [
            {
                "has_dpo": True,
                "reason": "detour",
                "chosen_msg": chosen,
                "rejected_msg": rejected,
                "context_units": [
                    {"type": "user", "msg": root, "tools": []},
                    {"type": "assistant_turn", "msg": rejected, "tools": []},
                ],
            }
        ]
        pairs = _dpo_from_pruned(pruned, sys_msg, root, "sid1", "proj", "model", 8)
        self.assertEqual(1, len(pairs))
        self.assertEqual(default_tools_json(), pairs[0]["tools"])

    def test_stage3_negative_and_review_pairs_include_tools(self):
        long_bad = ("我怀疑这一点是否正确。" * 40)
        msgs = [
            {"id": "s", "role": "system", "content": "sys"},
            {"id": "u0", "role": "user", "content": "任务"},
            _asst("a1", long_bad, pathologies=["over_rumination_doubt_loop"]),
        ]
        with patch("pipeline.stage3_synth.compact_thinking", return_value="压缩后的正确思路。"):
            neg = _negative_dpo(
                msgs, msgs[0], msgs[1], {"min_thinking_chars": 80}, 8, "sid", "proj", "train", "m"
            )
        self.assertTrue(neg)
        self.assertEqual(default_tools_json(), neg[0]["tools"])

        rewritten = _asst("a2", long_bad, pathologies=["over_rumination_length"])
        msgs2 = [msgs[0], msgs[1], rewritten]
        review = _review_rewrite_dpo(
            msgs2, {"a2": "简洁正确的下一步。"}, set(), msgs[0], msgs[1], 8, "sid", "proj", "train", "m"
        )
        self.assertTrue(review)
        self.assertEqual(default_tools_json(), review[0]["tools"])

    def test_normalize_and_synthesize_backfill_missing_tools(self):
        row = {
            "sample_id": "dpo::x",
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "ok"},
                {"from": "human", "value": "继续"},
            ],
            "chosen": {"from": "gpt", "value": "好"},
            "rejected": {"from": "gpt", "value": "差"},
            "metadata": {"source": "stage2_pruned", "session_id": "sid", "project_group": "proj"},
        }
        normalized = normalize_sharegpt_row(row, ranking=True)
        assert normalized is not None
        self.assertEqual(default_tools_json(), normalized["tools"])
        self.assertEqual(default_tools_json(), ensure_tools_column(row)["tools"])

        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "dpo_pairs.jsonl"
            gold.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with patch("pipeline.stage3_synth.stage_dirs", return_value={"gold": Path(tmp)}):
                pairs, _ = synthesize_dpo_candidates(
                    {},
                    {"synth": {"dpo": {"from_pruned_steps": True, "from_review_rewrites": False, "from_negative_steps": False}}},
                    {"proj": "train"},
                )
        self.assertEqual(1, len(pairs))
        self.assertEqual(default_tools_json(), pairs[0]["tools"])


if __name__ == "__main__":
    unittest.main()
