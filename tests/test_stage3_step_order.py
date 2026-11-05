"""stage3 门控 / 窗口 / 轨迹步骤日志必须串行 begin→finish。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest import mock

from pipeline import stage3_synth


class Stage3StepOrderTests(unittest.TestCase):
    def test_gate_windows_traj_begin_finish_are_serial(self):
        root = Path(tempfile.mkdtemp())
        gold = root / "02_gold"
        train = root / "03_train"
        state = root / "00_state"
        for d in (gold, train, state):
            d.mkdir(parents=True)
        sid = "sess-aabb"
        gold_file = gold / f"{sid}.gold.jsonl"
        gold_file.write_text(
            json.dumps({
                "session_id": sid,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "root"},
                ],
                "dominant_model": "glm-5.3",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (gold / "index.json").write_text(
            json.dumps({
                "sessions": {
                    sid: {
                        "session_id": sid,
                        "label": "positive",
                        "project_group": "game::demo",
                        "gold_file": str(gold_file),
                        "dpo_count": 0,
                    }
                }
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        (gold / "dpo_pairs.jsonl").write_text("", encoding="utf-8")

        eval_res = {
            "effective_state": "auto_gold",
            "full_sft_allowed": True,
            "subsegment_only": False,
            "funnel": {
                "gold_steps": 1, "admitted_steps": 1,
                "rescued_steps": 0, "excluded_steps": 0,
            },
            "step_audits": [{"status": "admitted", "reason": "ok"}],
            "windows_sgpt": [{
                "metadata": {"stage": "explore", "split": "train", "sample_weight": 1},
                "conversations": [],
            }],
            "windows_oai": [{
                "metadata": {"stage": "explore", "split": "train", "sample_weight": 1},
                "messages": [],
            }],
            "gold_traj": {"metadata": {"split": "train"}, "conversations": []},
        }

        lines: List[str] = []
        cfg = {
            "paths": {"data_dir": str(root)},
            "synth": {
                "window_turns": 8,
                "shuffle": False,
                "seed": 1,
                "dpo": {
                    "from_pruned_steps": True,
                    "from_review_rewrites": False,
                    "from_negative_steps": False,
                },
                "formats": {"sharegpt": True, "openai": True},
                "eval_ratio": 0.1,
            },
            "judge": {"enabled": False},
        }

        with mock.patch("pipeline.stage3_synth.stage_dirs", return_value={
            "gold": gold, "train": train, "state": state, "raw": root / "01_raw",
        }), mock.patch(
            "pipeline.stage3_synth.evaluate_session_synthesis", return_value=eval_res
        ), mock.patch("pipeline.stage3_synth.judge.enabled", return_value=False), mock.patch(
            "pipeline.stage3_synth.write_handoff", return_value={}
        ), mock.patch(
            "pipeline.package.build_delivery_zip", return_value={"zip_name": "x.zip"}
        ):
            stage3_synth.run(force=False, log=lines.append, cfg=cfg)

        visible = [x for x in lines if not str(x).startswith("@@STEP ")]
        begins = [i for i, x in enumerate(visible) if str(x).startswith("▶ 开始 ·") and (
            "入库门控" in x or "单步窗口" in x or "完整轨迹" in x
        )]
        ends = [i for i, x in enumerate(visible) if str(x).startswith("✓ 完成 ·") and (
            "入库门控" in x or "单步窗口" in x or "完整轨迹" in x
        )]
        self.assertEqual(3, len(begins))
        self.assertEqual(3, len(ends))
        # 串行：门控 begin→finish → 窗口 begin→finish → 轨迹 begin→finish
        gate_b = next(i for i, x in enumerate(visible) if "▶ 开始 ·" in x and "入库门控" in x)
        gate_e = next(i for i, x in enumerate(visible) if "✓ 完成 ·" in x and "入库门控" in x)
        win_b = next(i for i, x in enumerate(visible) if "▶ 开始 ·" in x and "单步窗口" in x)
        win_e = next(i for i, x in enumerate(visible) if "✓ 完成 ·" in x and "单步窗口" in x)
        traj_b = next(i for i, x in enumerate(visible) if "▶ 开始 ·" in x and "完整轨迹" in x)
        traj_e = next(i for i, x in enumerate(visible) if "✓ 完成 ·" in x and "完整轨迹" in x)
        self.assertLess(gate_b, gate_e)
        self.assertLess(gate_e, win_b)
        self.assertLess(win_b, win_e)
        self.assertLess(win_e, traj_b)
        self.assertLess(traj_b, traj_e)


if __name__ == "__main__":
    unittest.main()
