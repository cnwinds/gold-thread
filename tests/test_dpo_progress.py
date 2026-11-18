"""DPO 评估中途应上报 s3.dpo_review 进度（@@STEP done/total）。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from pipeline import judge, stage3_synth
from pipeline.common.store import write_json_atomic, write_jsonl_atomic
from server.app import Job


def _pair(i: int) -> Dict[str, Any]:
    return {
        "sample_id": f"dpo::s1::p{i}",
        "conversations": [{"from": "human", "value": f"任务{i}"}],
        "chosen": {"from": "gpt", "value": f"chosen-{i}"},
        "rejected": {"from": "gpt", "value": f"rejected-{i}"},
        "metadata": {
            "session_id": "s1",
            "project_group": "game::alpha",
            "source": "stage2_pruned",
            "reason": f"reason-{i}",
        },
    }


class DpoReviewProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.cfg = {
            "paths": {"data_dir": str(self.state / "data")},
            "judge": {
                "enabled": True,
                "model": "test-model",
                "dpo_batch": 2,
                "concurrency": 2,
            },
            "llm": {"model": "test-model"},
        }
        (self.state / "data" / "00_state").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_dpo_emits_incremental_step_progress(self):
        pairs = [_pair(i) for i in range(6)]
        lines: List[str] = []
        call_count = {"n": 0}

        def fake_call(_system, _schema, payload, *_a, **_k):
            call_count["n"] += 1
            return {
                "pairs": [
                    {"sample_id": p["sample_id"], "verdict": "good", "reason": "偏好成立"}
                    for p in payload["pairs"]
                ]
            }

        with patch("pipeline.judge._call", side_effect=fake_call), patch(
            "pipeline.judge._verdict_cache_path",
            return_value=self.state / "data" / "00_state" / "dpo_review_cache.json",
        ), patch("pipeline.judge._model", return_value="test-model"), patch(
            "pipeline.judge.enabled", return_value=True
        ):
            judge.review_dpo(pairs, cfg=self.cfg, log=lines.append)

        steps = [json.loads(x[7:]) for x in lines if str(x).startswith("@@STEP ")]
        self.assertTrue(steps, "应上报 @@STEP 进度")
        self.assertTrue(all(s.get("id") == "s3.dpo_review" for s in steps))
        with_counts = [s for s in steps if s.get("total")]
        self.assertGreaterEqual(len(with_counts), 2)
        self.assertEqual(with_counts[0].get("done"), 0)
        self.assertEqual(with_counts[0].get("total"), 6)
        self.assertEqual(with_counts[-1].get("done"), 6)
        self.assertEqual(with_counts[-1].get("total"), 6)
        dones = [s["done"] for s in with_counts]
        self.assertEqual(dones, sorted(dones))
        self.assertGreater(call_count["n"], 0)
        progress_logs = [str(x) for x in lines if "dpo 进度" in str(x) and not str(x).startswith("@@STEP ")]
        self.assertTrue(progress_logs)
        self.assertTrue(all("耗时" in x for x in progress_logs))

    def test_review_dpo_persists_reason_for_good_and_drop(self):
        pairs = [_pair(0), _pair(1)]
        cache_path = self.state / "data" / "00_state" / "dpo_review_cache.json"

        def fake_call(_system, _schema, payload, *_a, **_k):
            out = []
            for p in payload["pairs"]:
                sid = p["sample_id"]
                if sid.endswith("p0"):
                    out.append({"sample_id": sid, "verdict": "good", "reason": "rejected 是假阴性"})
                else:
                    out.append({"sample_id": sid, "verdict": "drop", "reason": "仅风格差异"})
            return {"pairs": out}

        with patch("pipeline.judge._call", side_effect=fake_call), patch(
            "pipeline.judge._verdict_cache_path", return_value=cache_path,
        ), patch("pipeline.judge._model", return_value="test-model"), patch(
            "pipeline.judge.enabled", return_value=True
        ):
            judge.review_dpo(pairs, cfg=self.cfg, log=lambda _m: None)
            cached = judge.cached_dpo_verdicts(pairs, cfg=self.cfg)

        self.assertEqual("good", cached["dpo::s1::p0"]["verdict"])
        self.assertEqual("rejected 是假阴性", cached["dpo::s1::p0"]["reason"])
        self.assertEqual("drop", cached["dpo::s1::p1"]["verdict"])
        self.assertEqual("仅风格差异", cached["dpo::s1::p1"]["reason"])

    def test_review_dpo_good_with_empty_reason_gets_default_reason(self):
        pairs = [_pair(0)]
        cache_path = self.state / "data" / "00_state" / "dpo_review_cache.json"

        def fake_call(_system, _schema, payload, *_a, **_k):
            return {"pairs": [{"sample_id": p["sample_id"], "verdict": "good", "reason": ""} for p in payload["pairs"]]}

        with patch("pipeline.judge._call", side_effect=fake_call), patch(
            "pipeline.judge._verdict_cache_path", return_value=cache_path,
        ), patch("pipeline.judge._model", return_value="test-model"), patch(
            "pipeline.judge.enabled", return_value=True
        ):
            judge.review_dpo(pairs, cfg=self.cfg, log=lambda _m: None)
            cached = judge.cached_dpo_verdicts(pairs, cfg=self.cfg)

        self.assertEqual("good", cached["dpo::s1::p0"]["verdict"])
        self.assertEqual("偏好成立", cached["dpo::s1::p0"]["reason"])

    def test_review_dpo_omitted_pair_marks_review_failed_not_silent_good(self):
        pairs = [_pair(0), _pair(1)]
        cache_path = self.state / "data" / "00_state" / "dpo_review_cache.json"

        def fake_call(_system, _schema, payload, *_a, **_k):
            # 模型仅返回 p0，漏掉了 p1 -> 绝不静默当 good，拆单重试仍漏返则记 review_failed
            out = []
            for p in payload["pairs"]:
                if p["sample_id"].endswith("p0"):
                    out.append({"sample_id": p["sample_id"], "verdict": "good", "reason": "真实修正"})
            return {"pairs": out}

        with patch("pipeline.judge._call", side_effect=fake_call), patch(
            "pipeline.judge._verdict_cache_path", return_value=cache_path,
        ), patch("pipeline.judge._model", return_value="test-model"), patch(
            "pipeline.judge.enabled", return_value=True
        ):
            out_map = judge.review_dpo(pairs, cfg=self.cfg, log=lambda _m: None)
            cached = judge.cached_dpo_verdicts(pairs, cfg=self.cfg)

        self.assertEqual("good", cached["dpo::s1::p0"]["verdict"])
        self.assertEqual("review_failed", out_map["dpo::s1::p1"]["verdict"])
        self.assertEqual("review_failed", cached["dpo::s1::p1"]["verdict"])

    def test_evaluate_dpo_pending_uses_catalog_step_id(self):
        root = self.state / "data"
        gold = root / "02_gold"
        gold.mkdir(parents=True)
        write_json_atomic(gold / "index.json", {"sessions": {}})
        write_jsonl_atomic(gold / "dpo_pairs.jsonl", [_pair(0), _pair(1)])
        (root / "00_state").mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        cfg = {
            "paths": {"data_dir": str(root)},
            "judge": {"enabled": True, "model": "test-model", "dpo_batch": 1, "concurrency": 1},
            "llm": {"model": "test-model"},
            "synth": {"eval_ratio": 0.0, "seed": 1},
        }

        def fake_call(_system, _schema, payload, *_a, **_k):
            return {
                "pairs": [
                    {"sample_id": p["sample_id"], "verdict": "good", "reason": "ok"}
                    for p in payload["pairs"]
                ]
            }

        with patch("pipeline.judge._call", side_effect=fake_call), patch(
            "pipeline.stage3_synth.synthesize_dpo_candidates",
            return_value=([_pair(0), _pair(1)], {}),
        ), patch("pipeline.stage3_synth._pick_split", return_value={}), patch(
            "pipeline.stage3_synth._session_splits", return_value={}
        ), patch("pipeline.stage3_synth.dataset_draft.load_draft", return_value={}), patch(
            "pipeline.stage3_synth.dataset_draft.exclusion_for",
            return_value={"action": "auto", "reason": "", "scope": ""},
        ), patch("pipeline.stage3_synth._load_dpo_exclusions", return_value={}):
            stage3_synth.evaluate_dpo_pending(cfg=cfg, log=lines.append)

        visible = [x for x in lines if not str(x).startswith("@@STEP ")]
        self.assertTrue(any("DPO 逐对复核" in x and "▶ 开始" in x for x in visible))
        self.assertTrue(any("DPO 逐对复核" in x and "✓ 完成" in x for x in visible))
        self.assertFalse(any("s3.dpo_evaluate" in x for x in lines))
        steps = [json.loads(x[7:]) for x in lines if str(x).startswith("@@STEP ")]
        self.assertTrue(all(s.get("id") == "s3.dpo_review" for s in steps))


class JobProgressLabelTests(unittest.TestCase):
    def test_step_marker_sets_progress_label_and_percent_fields(self):
        job = Job("dataset_evaluate", False, {})
        job.log('@@STEP {"id":"s3.dpo_review","done":3,"total":10,"msg":"已评估 3/10 对"}')
        self.assertEqual(job.progress.get("id"), "s3.dpo_review")
        self.assertEqual(job.progress.get("done"), 3)
        self.assertEqual(job.progress.get("total"), 10)
        self.assertIn("DPO", job.progress.get("label") or "")
        self.assertIn("3/10", job.progress.get("label") or "")


if __name__ == "__main__":
    unittest.main()
