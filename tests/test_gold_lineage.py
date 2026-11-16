"""黄金会话 lineage：DPO「→ N 对」必须等于训练集实写条数。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from server.app import _dpo_written_by_session, _session_lineage


class GoldLineageTests(unittest.TestCase):
    def test_dpo_train_uses_written_pairs_not_candidate_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / "dpo_preference_pairs.jsonl"
            sid = "01a153ca-1a49-77e7-b5d4-5e111955c845"
            rows = [
                {
                    "sample_id": f"dpo::{sid}::pruned{i}",
                    "metadata": {"session_id": sid, "source": "stage2_pruned"},
                }
                for i in (0, 2, 3, 4)
            ]
            pairs.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            written = _dpo_written_by_session(root)
            self.assertEqual(4, written[sid]["total"])
            self.assertEqual({"pruned": 4, "review": 0, "negative": 0}, written[sid]["parts"])

            funnels = {
                sid: {
                    "dpo_pruned": 5,
                    "dpo_review": 0,
                    "dpo_negative": 10,
                    "full_sft_allowed": True,
                    "funnel": {"admitted_steps": 94},
                }
            }
            entry = {"dpo_count": 5, "stats": {"gold_steps": 94, "raw_steps": 99, "pruned_detours": 5}}
            legacy = _session_lineage(sid, entry, funnels)
            self.assertEqual(15, legacy["dpo_train"])  # 旧 funnel 候选合计

            fixed = _session_lineage(sid, entry, funnels, written)
            self.assertEqual(4, fixed["dpo_train"])
            self.assertEqual({"pruned": 4, "review": 0, "negative": 0}, fixed["dpo_parts"])
            self.assertEqual({"pruned": 5, "review": 0, "negative": 10}, fixed["dpo_candidates"])


if __name__ == "__main__":
    unittest.main()
