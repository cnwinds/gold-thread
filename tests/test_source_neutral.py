"""会话来源仅作血缘、训练产物筛选与概览分布的 API 测试。"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from pipeline.common.config import DEFAULTS
from pipeline.common.store import write_json_atomic
from server.app import app


class SourceNeutralApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        for name in ("00_state", "01_raw", "02_gold", "03_train"):
            (self.data / name).mkdir(parents=True)
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["paths"]["data_dir"] = str(self.data)

        models = {
            "sid-glm": ("glm-5.3", "positive", 2),
            "sid-qwen": ("qwen3.8", "negative", 4),
            "sid-other": ("deepseek-v4", "pending", 0),
        }
        raw_sessions = {}
        gold_sessions = {}
        funnels = {}
        for sid, (model, label, steps) in models.items():
            raw_sessions[sid] = {
                "session_id": sid, "label": label, "project_group": "game::demo",
                "models": {model: steps}, "stats": {"assistant_steps": steps},
            }
            gold_sessions[sid] = {
                "session_id": sid, "label": label, "dominant_model": model,
                "project_group": "game::demo", "annotation_file": str(self.data / "missing.json"),
                "stats": {"gold_steps": steps},
            }
        funnels["sid-glm"] = {
            "label": "positive", "project_group": "game::demo", "split": "train",
            "funnel": {"admitted_steps": 2}, "dpo_pruned": 0, "dpo_negative": 0, "dpo_review": 0,
        }
        funnels["sid-qwen"] = {
            "label": "negative", "project_group": "game::demo", "split": "train",
            "funnel": {"admitted_steps": 4}, "dpo_pruned": 1, "dpo_negative": 2, "dpo_review": 0,
        }
        write_json_atomic(self.data / "01_raw" / "index.json", {"sessions": {
            **raw_sessions,
            "sid-new": {
                "session_id": "sid-new", "label": "pending", "project_group": "game::demo",
                "models": {"new-model": 1}, "stats": {"assistant_steps": 1},
            },
        }})
        write_json_atomic(self.data / "02_gold" / "index.json", {"sessions": gold_sessions})
        write_json_atomic(self.data / "03_train" / "stats.json", {"session_funnels": funnels, "counts": {}})
        write_json_atomic(self.data / "00_state" / "stage1.json", {})
        write_json_atomic(self.data / "00_state" / "stage2.json", {})
        write_json_atomic(self.data / "00_state" / "runs.json", [])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_raw_filters_by_analysis_state_and_dominant_model(self) -> None:
        with patch("server.app.load_config", return_value=self.cfg):
            client = TestClient(app)
            qwen = client.get("/api/raw/sessions?model=qwen3.8").json()
            self.assertEqual(["sid-qwen"], [r["session_id"] for r in qwen["sessions"]])
            self.assertEqual({"glm-5.3", "qwen3.8", "deepseek-v4", "new-model"}, set(qwen["source_models"]))
            unanalyzed = client.get("/api/raw/sessions?adopted=no").json()
            self.assertEqual(["sid-new"], [r["session_id"] for r in unanalyzed["sessions"]])

    def test_gold_filters_by_actual_sft_and_dpo_outputs(self) -> None:
        with patch("server.app.load_config", return_value=self.cfg):
            client = TestClient(app)
            sft = client.get("/api/gold/sessions?output=sft").json()
            self.assertEqual({"sid-glm", "sid-qwen"}, {r["session_id"] for r in sft["sessions"]})
            dpo = client.get("/api/gold/sessions?output=dpo").json()
            self.assertEqual(["sid-qwen"], [r["session_id"] for r in dpo["sessions"]])
            unwritten = client.get("/api/gold/sessions?output=unwritten").json()
            self.assertEqual(["sid-other"], [r["session_id"] for r in unwritten["sessions"]])
            qwen = client.get("/api/gold/sessions?model=qwen3.8").json()
            self.assertEqual(["sid-qwen"], [r["session_id"] for r in qwen["sessions"]])

    def test_overview_exposes_source_model_distribution(self) -> None:
        with patch("server.app.load_config", return_value=self.cfg):
            data = TestClient(app).get("/api/overview").json()
        self.assertEqual({"glm-5.3": 1, "qwen3.8": 1, "deepseek-v4": 1, "new-model": 1}, data["source_models"])


if __name__ == "__main__":
    unittest.main()
