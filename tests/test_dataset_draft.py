"""数据集草稿的公开 API 与 Stage 3 构建行为。"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from pipeline import stage3_synth
from pipeline.common.config import DEFAULTS, config_hash, stage_dirs
from pipeline.common.store import read_jsonl, write_json_atomic, write_jsonl_atomic
from server.app import app


def _assistant(mid: str = "a1"):
    return {
        "id": mid,
        "role": "assistant",
        "model": "deepseek-test",
        "thinking": "先读取文件头，再用脚本验证实际字节。",
        "text": "",
        "tool_calls": [{"id": "call1", "name": "bash", "arguments": {"command": "python check.py"}}],
        "tool_results": [],
    }


def _good_dpo_verdict(_system, _schema, payload, _cfg, _j, _tag, _log, **_kwargs):
    return {"pairs": [{"sample_id": p["sample_id"], "verdict": "good", "reason": "ok"} for p in payload["pairs"]]}


class DatasetDraftApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["paths"]["data_dir"] = str(self.root / "data")
        self.cfg["synth"]["eval_ratio"] = 0.0
        self.cfg["synth"]["shuffle"] = False
        self.cfg["judge"] = {"enabled": False}
        self.dirs = stage_dirs(self.cfg)
        self._write_session("s1", "game::alpha")
        self._write_session("s2", "game::beta")
        self.client = TestClient(app)
        self.cfg_patch = patch("server.app.load_config", return_value=self.cfg)
        self.cfg_patch.start()
        self.handoff_patch = patch("pipeline.stage3_synth.write_handoff", return_value={})
        self.handoff_patch.start()
        self.package_patch = patch("pipeline.package.build_delivery_zip", return_value={"zip_name": "test.zip"})
        self.package_mock = self.package_patch.start()

    def tearDown(self):
        self.package_patch.stop()
        self.handoff_patch.stop()
        self.cfg_patch.stop()
        self.tmp.cleanup()

    def _write_session(self, sid: str, project: str):
        gold_file = self.dirs["gold"] / "sessions" / f"{sid}.gold.jsonl"
        annotation_file = self.dirs["gold"] / "annotations" / f"{sid}.json"
        report = {"verdict": "trainable", "confidence": 0.95}
        record = {
            "session_id": sid,
            "label": "positive",
            "project_group": project,
            "cwd": f"D:\\crack\\{project.split('::')[-1]}",
            "dominant_model": "deepseek-test",
            "model_report": report,
            "stats": {"raw_steps": 1, "gold_steps": 1},
            "messages": [
                {"id": "sys", "role": "system", "content": "system"},
                {"id": "root", "role": "user", "content": "分析目标文件"},
                _assistant(),
            ],
        }
        write_jsonl_atomic(gold_file, [record])
        write_json_atomic(annotation_file, {"model_report": report, "review_key": f"1:{sid}:flow:1"})
        index_path = self.dirs["gold"] / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"sessions": {}}
        index["sessions"][sid] = {
            "project_group": project,
            "label": "positive",
            "dominant_model": "deepseek-test",
            "gold_file": str(gold_file),
            "annotation_file": str(annotation_file),
            "diag_file": "",
            "dpo_count": 0,
        }
        write_json_atomic(index_path, index)
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [])

    def _draft_row(self, sample_id: str):
        response = self.client.get("/api/dataset/draft", params={"limit": 500})
        self.assertEqual(200, response.status_code, response.text)
        return next(row for row in response.json()["rows"] if row["sample_id"] == sample_id)

    def _set_decision(self, sample_id: str, action: str):
        response = self.client.put(
            "/api/dataset/draft/decisions",
            json={"decisions": [{"sample_id": sample_id, "action": action, "reason": "测试决策"}]},
        )
        self.assertEqual(200, response.status_code, response.text)

    def test_exclude_survives_rebuild_and_auto_restores_sample(self):
        sample_id = "step::s1::a1"
        original = self._draft_row(sample_id)
        self.assertTrue(original["included"])
        self.assertEqual("auto", original["decision"])

        self._set_decision(sample_id, "exclude")
        excluded = self._draft_row(sample_id)
        self.assertFalse(excluded["included"])
        self.assertEqual("exclude", excluded["decision"])
        self.assertEqual("测试决策", excluded["decision_reason"])

        for _ in range(2):
            stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
            sharegpt = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
            openai = read_jsonl(self.dirs["train"] / "sft_step_windows_openai.jsonl")
            self.assertNotIn(sample_id, {row["sample_id"] for row in sharegpt})
            self.assertNotIn(sample_id, {row["sample_id"] for row in openai})
            self.assertIn("step::s2::a1", {row["sample_id"] for row in sharegpt})
            # 排除单步窗口不会误删同会话的完整轨迹。
            trajectories = read_jsonl(self.dirs["train"] / "sft_gold_trajectories_sharegpt.jsonl")
            self.assertIn("traj::s1", {row["sample_id"] for row in trajectories})

        self._set_decision(sample_id, "auto")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        restored = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
        self.assertIn(sample_id, {row["sample_id"] for row in restored})
        self.assertTrue(self._draft_row(sample_id)["included"])

    def _set_scope(self, scope: str, ident: str, action: str):
        response = self.client.put(
            "/api/dataset/draft/scopes",
            json={"updates": [{"scope": scope, "id": ident, "action": action, "reason": "测试层级停用"}]},
        )
        self.assertEqual(200, response.status_code, response.text)

    def test_session_scope_exclusion_cascades_to_samples_and_build(self):
        self._set_scope("session", "s1", "exclude")
        row = self._draft_row("step::s1::a1")
        self.assertFalse(row["included"])
        self.assertEqual("exclude", row["decision"])
        self.assertEqual("session", row["decision_scope"])
        self.assertTrue(self._draft_row("step::s2::a1")["included"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        sharegpt = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
        self.assertNotIn("step::s1::a1", {r["sample_id"] for r in sharegpt})
        self.assertIn("step::s2::a1", {r["sample_id"] for r in sharegpt})
        trajectories = read_jsonl(self.dirs["train"] / "sft_gold_trajectories_sharegpt.jsonl")
        self.assertNotIn("traj::s1", {r["sample_id"] for r in trajectories})

        self._set_scope("session", "s1", "auto")
        self.assertTrue(self._draft_row("step::s1::a1")["included"])
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        restored = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
        self.assertIn("step::s1::a1", {r["sample_id"] for r in restored})

    def _write_s1_dpo_pair(self, sample_id: str = "dpo::s1::pair"):
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [{
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜"},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "source": "stage2_pruned",
                "msg_id": "a1",
            },
        }])
        index_path = self.dirs["gold"] / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["sessions"]["s1"]["dpo_count"] = 1
        write_json_atomic(index_path, index)

    def test_session_sft_scope_does_not_disable_dpo(self):
        self._write_s1_dpo_pair()
        self._set_scope("session_sft", "s1", "exclude")
        sft = self._draft_row("step::s1::a1")
        self.assertFalse(sft["included"])
        self.assertEqual("session_sft", sft["decision_scope"])
        # 同会话 DPO 不受 session_sft 影响
        preview = self.client.get("/api/dataset/draft").json()
        dpo_rows = [r for r in preview["rows"] if r.get("kind") == "dpo" and r.get("session_id") == "s1"]
        self.assertTrue(dpo_rows)
        self.assertTrue(all(r.get("included") for r in dpo_rows))

        gold = self.client.get("/api/gold/sessions").json()["sessions"]
        by_id = {s["session_id"]: s for s in gold}
        self.assertTrue(by_id["s1"]["sft_excluded"])
        self.assertFalse(by_id["s1"]["dpo_excluded"])
        self.assertFalse(by_id["s1"]["dataset_excluded"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        sharegpt = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
        self.assertNotIn("step::s1::a1", {r["sample_id"] for r in sharegpt})
        dpo = read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")
        self.assertTrue(any(r.get("metadata", {}).get("session_id") == "s1" for r in dpo))

        self._set_scope("session_sft", "s1", "auto")
        self.assertTrue(self._draft_row("step::s1::a1")["included"])

    def test_session_dpo_scope_does_not_disable_sft(self):
        self._write_s1_dpo_pair()
        self._set_scope("session_dpo", "s1", "exclude")
        self.assertTrue(self._draft_row("step::s1::a1")["included"])
        preview = self.client.get("/api/dataset/draft").json()
        dpo_rows = [r for r in preview["rows"] if r.get("kind") == "dpo" and r.get("session_id") == "s1"]
        self.assertTrue(dpo_rows)
        self.assertTrue(all(not r.get("included") for r in dpo_rows))
        self.assertTrue(all(r.get("decision_scope") == "session_dpo" for r in dpo_rows))

        gold = {s["session_id"]: s for s in self.client.get("/api/gold/sessions").json()["sessions"]}
        self.assertTrue(gold["s1"]["dpo_excluded"])
        self.assertFalse(gold["s1"]["sft_excluded"])

        detail = self.client.get("/api/gold/session/s1").json()
        self.assertTrue(detail["dpo_excluded"])
        self.assertFalse(detail["sft_excluded"])
        self.assertFalse(detail["dataset_excluded"])

    def test_project_scope_exclusion_covers_its_sessions_only(self):
        self._set_scope("project", "game::alpha", "exclude")
        row = self._draft_row("step::s1::a1")
        self.assertFalse(row["included"])
        self.assertEqual("project", row["decision_scope"])
        self.assertTrue(self._draft_row("step::s2::a1")["included"])

        gold = self.client.get("/api/gold/sessions").json()["sessions"]
        by_id = {s["session_id"]: s for s in gold}
        self.assertTrue(by_id["s1"]["dataset_excluded"])
        self.assertEqual("project", by_id["s1"]["exclusion_scope"])
        self.assertFalse(by_id["s2"]["dataset_excluded"])

    def test_session_split_override_moves_only_that_session(self):
        response = self.client.put(
            "/api/dataset/draft/split",
            json={"session_assignments": {"s1": "eval"}},
        )
        self.assertEqual(200, response.status_code, response.text)
        row = self._draft_row("step::s1::a1")
        self.assertEqual("eval", row["split"])
        self.assertEqual("session", row["split_source"])
        self.assertEqual("train", self._draft_row("step::s2::a1")["split"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        eval_rows = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.eval.jsonl")
        train_rows = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.train.jsonl")
        self.assertIn("step::s1::a1", {r["sample_id"] for r in eval_rows})
        self.assertNotIn("step::s1::a1", {r["sample_id"] for r in train_rows})
        self.assertIn("step::s2::a1", {r["sample_id"] for r in train_rows})

        cleared = self.client.put(
            "/api/dataset/draft/split",
            json={"session_assignments": {"s1": "inherit"}},
        )
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertEqual("train", self._draft_row("step::s1::a1")["split"])

    def test_tree_sft_samples_follow_trajectory_step_order(self):
        """SFT 应按轨迹步序排列，不能按 msg UUID 字典序（否则看起来像打乱）。"""
        gold_file = self.dirs["gold"] / "sessions" / "s1.gold.jsonl"
        record = read_jsonl(gold_file)[0]
        # 后出现的步用字典序更靠前的 id，若按 sample_id 排序会颠倒。
        record["messages"] = [
            {"id": "sys", "role": "system", "content": "system"},
            {"id": "root", "role": "user", "content": "分析目标文件"},
            _assistant("zz_first"),
            {
                **_assistant("aa_second"),
                "thinking": "并行核查关键偏移并落盘。",
                "tool_calls": [
                    {"id": "c1", "name": "bash", "arguments": {"command": "python a.py"}},
                    {"id": "c2", "name": "bash", "arguments": {"command": "python b.py"}},
                ],
            },
        ]
        record["stats"] = {"raw_steps": 2, "gold_steps": 2}
        write_jsonl_atomic(gold_file, [record])

        tree = self.client.get("/api/dataset/draft/tree").json()
        alpha = next(g for g in tree["groups"] if g["key"] == "game::alpha")
        sft = alpha["sessions"][0]["trajectories"][0]["kinds"][0]["samples"]
        self.assertGreaterEqual(len(sft), 2)
        self.assertEqual([1, 2], [r.get("ref_step") for r in sft[:2]])
        self.assertTrue(str(sft[0]["title"]).startswith("第1步"))
        self.assertTrue(str(sft[1]["title"]).startswith("第2步"))
        self.assertEqual(
            ["step::s1::zz_first", "step::s1::aa_second"],
            [r["sample_id"] for r in sft[:2]],
        )

    def test_tree_actionable_samples_sort_before_inert(self):
        """能点恢复/停用的在前；样本层手动停用仍可操作，排在门禁丢弃等无按钮项之前。"""
        gold_file = self.dirs["gold"] / "sessions" / "s1.gold.jsonl"
        record = read_jsonl(gold_file)[0]
        record["messages"] = [
            {"id": "sys", "role": "system", "content": "system"},
            {"id": "root", "role": "user", "content": "分析目标文件"},
            _assistant("step_a"),
            _assistant("step_b"),
            _assistant("step_c"),
        ]
        record["stats"] = {"raw_steps": 3, "gold_steps": 3}
        write_jsonl_atomic(gold_file, [record])

        self._set_decision("step::s1::step_b", "exclude")
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [
            {
                "sample_id": "dpo::s1::good",
                "conversations": [{"from": "human", "value": "分析文件"}],
                "chosen": {"from": "gpt", "value": "先验证"},
                "rejected": {"from": "gpt", "value": "直接猜"},
                "metadata": {
                    "session_id": "s1",
                    "project_group": "game::alpha",
                    "source": "stage2_pruned",
                    "msg_id": "step_a",
                },
            },
            {
                "sample_id": "dpo::s1::manual_off",
                "conversations": [{"from": "human", "value": "分析文件"}],
                "chosen": {"from": "gpt", "value": "稳健做法"},
                "rejected": {"from": "gpt", "value": "草率做法"},
                "metadata": {
                    "session_id": "s1",
                    "project_group": "game::alpha",
                    "source": "stage2_pruned",
                    "msg_id": "step_c",
                },
            },
            {
                "sample_id": "dpo_neg::s1::empty",
                "conversations": [{"from": "human", "value": "分析文件"}],
                "chosen": {"from": "gpt", "value": ""},
                "rejected": {"from": "gpt", "value": ""},
                "metadata": {
                    "session_id": "s1",
                    "project_group": "game::alpha",
                    "source": "negative_step",
                    "msg_id": "step_b",
                    "reason": "over_rumination_length",
                },
            },
        ])
        self.cfg["judge"] = {"enabled": False}
        self._set_decision("dpo::s1::manual_off", "exclude")

        tree = self.client.get("/api/dataset/draft/tree").json()
        alpha = next(g for g in tree["groups"] if g["key"] == "game::alpha")
        kinds = {k["kind"]: k["samples"] for k in alpha["sessions"][0]["trajectories"][0]["kinds"]}
        sft = kinds["sft"]
        dpo = kinds["dpo"]

        # SFT：已纳入 → 样本层可恢复停用；三者都能操作，按纳入优先 + 步序。
        self.assertEqual(
            ["step::s1::step_a", "step::s1::step_c", "step::s1::step_b"],
            [r["sample_id"] for r in sft if r["sample_id"].startswith("step::s1::step_")],
        )
        self.assertTrue(sft[0]["included"] and sft[1]["included"])
        self.assertEqual("exclude", sft[2]["decision"])
        self.assertIn(sft[2].get("decision_scope") or "sample", ("", "sample"))

        # DPO：可停用的 good、可恢复的 manual_off 在前；空坏步无按钮沉底。
        dpo_ids = [r["sample_id"] for r in dpo]
        self.assertEqual(
            ["dpo::s1::good", "dpo::s1::manual_off", "dpo_neg::s1::empty"],
            dpo_ids,
        )
        self.assertTrue(dpo[0]["included"])
        self.assertEqual("exclude", dpo[1]["decision"])
        self.assertFalse(dpo[2]["included"])
        self.assertNotEqual("exclude", dpo[2].get("decision"))

    def test_tree_groups_gold_sessions_then_trajectories_then_samples(self):
        self._set_scope("session", "s2", "exclude")
        response = self.client.get("/api/dataset/draft/tree")
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        groups = {g["key"]: g for g in body["groups"]}
        self.assertEqual({"game::alpha", "game::beta"}, set(groups))
        alpha = groups["game::alpha"]
        self.assertEqual(1, len(alpha["sessions"]))
        session = alpha["sessions"][0]
        self.assertEqual("s1", session["session_id"])
        self.assertNotIn("samples", session)
        self.assertTrue(session["trajectories"])
        traj = session["trajectories"][0]
        self.assertEqual("traj::s1", traj["trajectory_id"])
        self.assertIsNotNone(traj["sample"])
        self.assertEqual("trajectory", traj["sample"]["kind"])
        self.assertEqual(["sft", "dpo"], [b["kind"] for b in traj["kinds"]])
        sft_bucket = traj["kinds"][0]
        dpo_bucket = traj["kinds"][1]
        self.assertTrue(any(r["kind"] == "sft" for r in sft_bucket["samples"]))
        self.assertTrue(all(r["kind"] == "sft" for r in sft_bucket["samples"]))
        self.assertTrue(all(r["kind"] == "dpo" for r in dpo_bucket["samples"]))
        flat = ([traj["sample"]] if traj.get("sample") else []) + [
            r for bucket in traj["kinds"] for r in bucket["samples"]
        ]
        self.assertEqual(session["counts"]["candidates"], len(flat))
        self.assertEqual(traj["counts"]["candidates"], len(flat))
        beta_session = groups["game::beta"]["sessions"][0]
        self.assertTrue(beta_session["excluded"])
        self.assertEqual(0, beta_session["counts"]["included"])

        by_split = self.client.get("/api/dataset/draft/tree", params={"group_by": "split"})
        self.assertEqual(200, by_split.status_code, by_split.text)
        self.assertEqual({"train"}, {g["key"] for g in by_split.json()["groups"]})

    def test_existing_projects_keep_split_when_a_new_project_appears(self):
        self.cfg["synth"]["eval_ratio"] = 0.5
        initial_response = self.client.get("/api/dataset/draft")
        self.assertEqual(200, initial_response.status_code, initial_response.text)
        initial = initial_response.json()["split"]
        self.assertEqual(1, sum(value == "eval" for value in initial.values()))

        self._write_session("s3", "game::aardvark")
        expanded_response = self.client.get("/api/dataset/draft")
        self.assertEqual(200, expanded_response.status_code, expanded_response.text)
        expanded = expanded_response.json()["split"]
        self.assertEqual({"game::alpha", "game::beta"}, set(initial))
        self.assertIn("game::aardvark", expanded)
        for project, split in initial.items():
            self.assertEqual(split, expanded[project], f"{project} 不应因新增项目而换组")

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        generated = read_jsonl(self.dirs["train"] / "sft_step_windows_sharegpt.jsonl")
        self.assertEqual(3, len(generated))
        for row in generated:
            project = row["metadata"]["project_group"]
            self.assertEqual(expanded[project], row["metadata"]["split"])

    def test_pinned_project_survives_explicit_replan(self):
        self.cfg["synth"]["eval_ratio"] = 0.5
        self.assertEqual(200, self.client.get("/api/dataset/draft").status_code)
        pin = self.client.put(
            "/api/dataset/draft/split",
            json={"assignments": {"game::alpha": "eval"}},
        )
        self.assertEqual(200, pin.status_code, pin.text)

        replan = self.client.post(
            "/api/dataset/draft/split/replan",
            json={"eval_ratio": 0.0, "seed": 7},
        )
        self.assertEqual(200, replan.status_code, replan.text)
        body = self.client.get("/api/dataset/draft").json()
        self.assertEqual("eval", body["split"]["game::alpha"])
        self.assertEqual("train", body["split"]["game::beta"])
        self.assertIn("game::alpha", body["split_plan"]["pinned"])
        self.assertEqual(0.0, body["split_plan"]["eval_ratio"])
        self.assertEqual(7, body["split_plan"]["seed"])

    def test_generic_decision_overrides_legacy_dpo_exclusion(self):
        sample_id = "dpo::s1::pair1"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "model": "deepseek-test",
                "source": "stage2_pruned",
                "reason": "测试偏好对",
            },
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        write_json_atomic(
            self.dirs["state"] / "dpo_exclusions.json",
            {sample_id: {"reason": "旧人工排除", "by": "user", "at": "2026-01-01"}},
        )

        self._set_decision(sample_id, "auto")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        restored = read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")
        self.assertIn(sample_id, {row["sample_id"] for row in restored})

        self._set_decision(sample_id, "exclude")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        excluded = read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")
        self.assertNotIn(sample_id, {row["sample_id"] for row in excluded})

    def test_build_status_tracks_draft_changes(self):
        initial = self.client.get("/api/dataset/draft").json()
        self.assertTrue(initial["dirty"])
        self.assertTrue(initial["plan_hash"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        built = self.client.get("/api/dataset/draft").json()
        self.assertFalse(built["dirty"])
        self.assertEqual(initial["plan_hash"], built["last_build"]["plan_hash"])

        self._set_decision("step::s1::a1", "exclude")
        changed = self.client.get("/api/dataset/draft").json()
        self.assertTrue(changed["dirty"])
        self.assertNotEqual(initial["plan_hash"], changed["plan_hash"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        rebuilt = self.client.get("/api/dataset/draft").json()
        self.assertFalse(rebuilt["dirty"])
        self.assertEqual(changed["plan_hash"], rebuilt["last_build"]["plan_hash"])

    def test_draft_sample_detail_keeps_content_and_decision(self):
        sample_id = "step::s1::a1"
        response = self.client.get("/api/dataset/draft/sample", params={"sample_id": sample_id})
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual(sample_id, body["sample_id"])
        self.assertTrue(body["conversations"])
        self.assertTrue(body["_draft"]["included"])

        self._set_decision(sample_id, "exclude")
        excluded = self.client.get("/api/dataset/draft/sample", params={"sample_id": sample_id}).json()
        self.assertFalse(excluded["_draft"]["included"])
        self.assertEqual("测试决策", excluded["_draft"]["decision_reason"])
        missing = self.client.get("/api/dataset/draft/sample", params={"sample_id": "step::missing::a1"})
        self.assertEqual(404, missing.status_code)

    def test_replan_dry_run_previews_changes_without_saving(self):
        self.assertEqual(200, self.client.get("/api/dataset/draft").status_code)
        self.client.put(
            "/api/dataset/draft/split",
            json={"assignments": {"game::alpha": "eval"}},
        )
        response = self.client.post(
            "/api/dataset/draft/split/replan",
            json={"eval_ratio": 1.0, "seed": 99, "dry_run": True},
        )
        self.assertEqual(200, response.status_code, response.text)
        changes = response.json()["changes"]
        self.assertIn("game::beta", changes)
        self.assertEqual("eval", changes["game::beta"]["after"])

        current = self.client.get("/api/dataset/draft").json()
        self.assertEqual("train", current["split"]["game::beta"])
        self.assertEqual(42, current["split_plan"]["seed"])
        self.assertEqual(0.0, current["split_plan"]["eval_ratio"])

    def test_excluding_last_dpo_removes_stale_split_and_registration(self):
        sample_id = "dpo::s1::only"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        split_path = self.dirs["train"] / "dpo_preference_pairs.train.jsonl"
        self.assertIn(sample_id, {row["sample_id"] for row in read_jsonl(split_path)})
        info_path = self.dirs["train"] / "dataset_info.json"
        self.assertIn("pi_dpo_preference_pairs_train", json.loads(info_path.read_text(encoding="utf-8")))

        self._set_decision(sample_id, "exclude")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual([], read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl"))
        self.assertFalse(split_path.exists(), "空分片不能保留上次构建的样本")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.assertNotIn("pi_dpo_preference_pairs_train", info)

    def test_legacy_dpo_exclusion_change_marks_draft_dirty(self):
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        before = self.client.get("/api/dataset/draft").json()
        self.assertFalse(before["dirty"])
        write_json_atomic(
            self.dirs["state"] / "dpo_exclusions.json",
            {"dpo::legacy": {"reason": "旧账本新增排除"}},
        )
        after = self.client.get("/api/dataset/draft").json()
        self.assertTrue(after["dirty"])
        self.assertNotEqual(before["plan_hash"], after["plan_hash"])

    def test_first_replan_dry_run_does_not_create_draft_state(self):
        state_path = self.dirs["state"] / "dataset_draft.json"
        self.assertFalse(state_path.exists())
        response = self.client.post(
            "/api/dataset/draft/split/replan",
            json={"eval_ratio": 1.0, "seed": 99, "dry_run": True},
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertFalse(state_path.exists(), "首次 dry-run 不能初始化或写入草稿")

    def test_draft_rejects_negative_pagination(self):
        for params in ({"offset": -1}, {"limit": 0}, {"limit": -2}):
            with self.subTest(params=params):
                response = self.client.get("/api/dataset/draft", params=params)
                self.assertEqual(422, response.status_code)

    def test_out_of_range_offset_clamps_to_last_page(self):
        response = self.client.get("/api/dataset/draft", params={"offset": 99, "limit": 2})
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual(4, body["total"])
        self.assertEqual(2, body["offset"])
        self.assertEqual(2, len(body["rows"]))

    def test_concurrent_decisions_do_not_overwrite_each_other(self):
        for index in range(8):
            self._write_session(f"extra{index}", f"game::extra{index}")

        def exclude(index):
            sample_id = f"step::extra{index}::a1" if index >= 0 else "step::s1::a1"
            return self.client.put(
                "/api/dataset/draft/decisions",
                json={"decisions": [{"sample_id": sample_id, "action": "exclude", "reason": "并发测试"}]},
            )

        with ThreadPoolExecutor(max_workers=9) as pool:
            responses = list(pool.map(exclude, range(-1, 8)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        rows = self.client.get("/api/dataset/draft", params={"limit": 500}).json()["rows"]
        expected = {"step::s1::a1", *(f"step::extra{i}::a1" for i in range(8))}
        excluded = {row["sample_id"] for row in rows if row["decision"] == "exclude"}
        self.assertEqual(expected, excluded)

    def test_replan_confirmation_rejects_stale_preview(self):
        self.assertEqual(200, self.client.get("/api/dataset/draft").status_code)
        preview = self.client.post(
            "/api/dataset/draft/split/replan",
            json={"eval_ratio": 1.0, "seed": 99, "dry_run": True},
        ).json()
        self.assertTrue(preview["base_hash"])
        self.client.put(
            "/api/dataset/draft/split",
            json={"assignments": {"game::beta": "train"}},
        )
        state_path = self.dirs["state"] / "dataset_draft.json"
        before_stale = state_path.read_text(encoding="utf-8")
        self._write_session("s3", "game::new-during-confirm")
        stale = self.client.post(
            "/api/dataset/draft/split/replan",
            json={
                "eval_ratio": 1.0,
                "seed": 99,
                "dry_run": False,
                "expected_hash": preview["base_hash"],
            },
        )
        self.assertEqual(409, stale.status_code, stale.text)
        self.assertEqual(before_stale, state_path.read_text(encoding="utf-8"), "409 不能落盘任何项目变化")
        self.assertEqual(42, self.client.get("/api/dataset/draft").json()["split_plan"]["seed"])

    def test_draft_stats_separate_excluded_sample_types(self):
        sample_id = "dpo::s1::stats"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        for excluded_id in ("step::s1::a1", "traj::s2", sample_id):
            self._set_decision(excluded_id, "exclude")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        stats = json.loads((self.dirs["train"] / "stats.json").read_text(encoding="utf-8"))
        draft = stats["draft"]
        self.assertEqual(1, draft["sft_excluded"])
        self.assertEqual(1, draft["trajectory_excluded"])
        self.assertEqual(1, draft["dpo_excluded"])
        self.assertEqual(1, draft["sft_included"])
        self.assertEqual(1, draft["trajectory_included"])
        self.assertEqual(0, draft["dpo_included"])
        self.assertNotIn("draft_excluded", stats["drop_reasons"])

    def test_project_stats_include_sessions_without_candidates(self):
        self._write_session("blocked", "game::blocked")
        gold_file = self.dirs["gold"] / "sessions" / "blocked.gold.jsonl"
        record = read_jsonl(gold_file)[0]
        record["model_report"] = {"verdict": "discard", "confidence": 0.9}
        write_jsonl_atomic(gold_file, [record])
        annotation_file = self.dirs["gold"] / "annotations" / "blocked.json"
        write_json_atomic(annotation_file, {"model_report": record["model_report"]})

        body = self.client.get("/api/dataset/draft").json()
        stat = body["project_stats"]["game::blocked"]
        self.assertEqual(1, stat["sessions"])
        self.assertEqual(0, stat["candidates"])
        self.assertIn("game::blocked", body["projects"])
        self.assertEqual("train", body["split"]["game::blocked"])

    def test_dpo_evaluation_cache_change_marks_existing_build_dirty(self):
        pair = {
            "sample_id": "dpo::s1::reviewed",
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}
        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict):
            stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertFalse(self.client.get("/api/dataset/draft").json()["dirty"])

        write_json_atomic(self.dirs["state"] / "dpo_review_cache.json", {"changed": True})
        self.assertTrue(self.client.get("/api/dataset/draft").json()["dirty"])

    def test_input_change_during_build_fails_instead_of_marking_stale_output_clean(self):
        sample_id = "dpo::s1::race"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}
        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict):
            stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        cached = stage3_synth.judge.resolve_dpo_reviews

        def change_legacy_input(pairs, cfg):
            result = cached(pairs, cfg)
            write_json_atomic(
                self.dirs["state"] / "dpo_exclusions.json",
                {sample_id: {"reason": "构建期间新增排除"}},
            )
            return result

        with patch("pipeline.stage3_synth.judge.resolve_dpo_reviews", side_effect=change_legacy_input):
            with self.assertRaisesRegex(RuntimeError, "构建期间.*变化"):
                stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertFalse((self.dirs["train"] / "dpo_preference_pairs.jsonl").exists())
        self.assertTrue(self.client.get("/api/dataset/draft").json()["dirty"])

    def test_input_change_between_final_hash_and_consistency_check_fails(self):
        sample_id = "dpo::s1::gap"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        original = stage3_synth.dataset_draft.plan_fingerprint

        def mutate_during_final_hash(cfg, draft, include_review_cache=True, include_aux_outputs=False, revisions=None):
            result = original(
                cfg,
                draft,
                include_review_cache=include_review_cache,
                include_aux_outputs=include_aux_outputs,
                revisions=revisions,
            )
            if include_review_cache:
                write_json_atomic(
                    self.dirs["state"] / "dpo_exclusions.json",
                    {sample_id: {"reason": "最终指纹期间新增排除"}},
                )
            return result

        with patch("pipeline.stage3_synth.dataset_draft.plan_fingerprint", side_effect=mutate_during_final_hash):
            with self.assertRaisesRegex(RuntimeError, "构建期间.*变化"):
                stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertFalse((self.dirs["train"] / "dpo_preference_pairs.jsonl").exists())

    def test_dpo_requires_explicit_evaluation_before_matching_build(self):
        sample_id = "dpo::s1::explicit-evaluation"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "source": "stage2_pruned",
                "rejected_msg_id": "a1",
                "chosen_msg_id": "a1",
            },
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}

        pending = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()
        self.assertEqual(1, pending["summary"]["pending_evaluations"])
        row = pending["rows"][0]
        self.assertEqual(sample_id, row["sample_id"])
        self.assertFalse(row["included"])
        self.assertEqual("pending", row["gate_status"])
        self.assertEqual("尚未完成模型复核", row["decision_reason"])
        self.assertEqual("第1步（弯路）", row["title"])
        self.assertEqual(1, row["ref_step"])
        self.assertTrue(row["preview"]["chosen_gist"])
        self.assertTrue(row["preview"]["rejected_gist"])
        pending_detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": sample_id}).json()
        self.assertEqual("pending", pending_detail["_draft"]["gate_status"])
        self.assertEqual("尚未完成模型复核", pending_detail["_draft"]["decision_reason"])
        self.assertEqual("第1步（弯路）", pending_detail["_draft"]["title"])
        self.assertEqual(1, pending_detail["_draft"]["ref_step"])
        with self.assertRaisesRegex(RuntimeError, "尚未完成模型复核"):
            stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)

        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict):
            evaluated = stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual(1, evaluated["evaluated"])
        self.assertEqual(0, evaluated["pending"])

        ready = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()
        self.assertTrue(ready["rows"][0]["included"])
        self.assertEqual("good", ready["rows"][0]["gate_status"])
        with patch("pipeline.stage3_synth.judge.review_dpo", side_effect=AssertionError("Stage 3 不应再调用模型")):
            stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        generated = read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")
        self.assertIn(sample_id, {row["sample_id"] for row in generated})

    def test_dpo_good_swap_drop_and_manual_exclude_have_distinct_outcomes(self):
        verdicts = {"good": "good", "drop": "drop", "swap": "swap"}
        pairs = []
        for kind in ("good", "drop", "swap"):
            pairs.append({
                "sample_id": f"dpo::s1::{kind}",
                "conversations": [{"from": "human", "value": "分析文件"}],
                "chosen": {"from": "gpt", "value": f"chosen-{kind}"},
                "rejected": {"from": "gpt", "value": f"rejected-{kind}"},
                "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
            })
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", pairs)
        self.cfg["judge"] = {"enabled": True}

        def verdict_by_id(_system, _schema, payload, _cfg, _j, _tag, _log, **_kwargs):
            return {"pairs": [
                {"sample_id": p["sample_id"], "verdict": verdicts[p["sample_id"].rsplit("::", 1)[-1]], "reason": "test"}
                for p in payload["pairs"]
            ]}

        with patch("pipeline.judge._call", side_effect=verdict_by_id):
            stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        rows = {r["sample_id"].rsplit("::", 1)[-1]: r for r in self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"]}
        self.assertTrue(rows["good"]["included"])
        self.assertTrue(rows["swap"]["included"])
        self.assertFalse(rows["drop"]["included"])
        self.assertEqual("test", rows["good"]["decision_reason"])
        self.assertEqual("test", rows["drop"]["decision_reason"])
        detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": "dpo::s1::swap"}).json()
        self.assertEqual("rejected-swap", detail["chosen"]["value"])
        self.assertEqual("swap", detail["_draft"]["gate_status"])
        self.assertEqual("test", detail["_draft"]["decision_reason"])
        drop_detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": "dpo::s1::drop"}).json()
        self.assertFalse(drop_detail["_draft"]["included"])
        self.assertEqual("drop", drop_detail["_draft"]["gate_status"])
        self.assertEqual("test", drop_detail["_draft"]["decision_reason"])
        good_detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": "dpo::s1::good"}).json()
        self.assertTrue(good_detail["_draft"]["included"])
        self.assertEqual("good", good_detail["_draft"]["gate_status"])
        self.assertEqual("test", good_detail["_draft"]["decision_reason"])

        self._set_decision("dpo::s1::good", "exclude")
        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        generated = {r["sample_id"]: r for r in read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")}
        self.assertEqual({"dpo::s1::swap"}, set(generated))
        self.assertEqual("rejected-swap", generated["dpo::s1::swap"]["chosen"]["value"])

    def test_empty_dpo_chosen_rejected_are_auto_disabled(self):
        sample_id = "dpo_neg::s1::empty"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": ""},
            "rejected": {"from": "gpt", "value": ""},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "source": "negative_step",
                "msg_id": "a1",
                "reason": "over_rumination_length",
            },
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": False}

        row = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]
        self.assertEqual(sample_id, row["sample_id"])
        self.assertFalse(row["included"])
        self.assertEqual("invalid", row["gate_status"])
        self.assertEqual("chosen/rejected 均为空", row["decision_reason"])
        self.assertEqual("第1步（坏步）", row["title"])

        detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": sample_id}).json()
        self.assertFalse(detail["_draft"]["included"])
        self.assertEqual("invalid", detail["_draft"]["gate_status"])
        self.assertEqual("chosen/rejected 均为空", detail["_draft"]["decision_reason"])

        stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        generated = read_jsonl(self.dirs["train"] / "dpo_preference_pairs.jsonl")
        self.assertNotIn(sample_id, {r["sample_id"] for r in generated})

    def test_negative_step_same_action_only_think_compress_is_auto_disabled(self):
        sample_id = "dpo_neg::s1::same-action"
        tools = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": f"<think>\n短思考。\n</think>\n\n{tools}"},
            "rejected": {"from": "gpt", "value": f"<think>\n很长很长的反复自我怀疑思考。\n</think>\n\n{tools}"},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "source": "negative_step",
                "msg_id": "a1",
                "reason": "over_rumination_length",
                "pathologies": ["over_rumination_length"],
            },
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": False}

        row = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]
        self.assertFalse(row["included"])
        self.assertEqual("invalid", row["gate_status"])
        self.assertEqual("同动作，仅压缩思维链", row["decision_reason"])
        self.assertTrue(row["preview"]["same_action"])
        self.assertIn("短思考", row["preview"]["chosen_gist"])
        self.assertNotIn("<tool_call>", row["preview"]["chosen_gist"])
        detail = self.client.get("/api/dataset/draft/sample", params={"sample_id": sample_id}).json()
        self.assertEqual("dpo", detail.get("kind"))
        self.assertIn("短思考", detail["chosen"]["value"])

    def test_dpo_preview_gist_excludes_tools_when_think_missing(self):
        """无思考内容时，摘要/预览不得把 tool 正文塞进 chosen/rejected 思考。"""
        tools = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
        pair = {
            "sample_id": "dpo::s1::tool-only",
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": f"<think>\n\n</think>\n\n{tools}"},
            "rejected": {"from": "gpt", "value": tools},
            "metadata": {
                "session_id": "s1",
                "project_group": "game::alpha",
                "source": "stage2_pruned",
                "msg_id": "a1",
            },
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": False}

        row = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]
        self.assertEqual("", row["preview"]["chosen_gist"])
        self.assertEqual("", row["preview"]["rejected_gist"])
        self.assertNotIn("tool_call", row["preview"]["chosen_gist"])
        self.assertNotIn("bash", row["preview"]["rejected_gist"])
        self.assertNotIn("ls -la", json.dumps(row["preview"], ensure_ascii=False))

        from pipeline.stage3_synth import _think_display_text, _gist_text
        self.assertEqual("", _think_display_text(tools))
        self.assertEqual("", _think_display_text(f"<think>\n\n</think>\n\n{tools}"))
        self.assertEqual("先验证再改", _think_display_text("先验证再改"))
        self.assertEqual("", _gist_text(tools, prefer_think=True))
        self.assertIn("短思考", _gist_text("<think>\n短思考\n</think>\n\n" + tools, prefer_think=True))
        self.assertNotIn("tool_call", _gist_text("<think>\n短思考\n</think>\n\n" + tools, prefer_think=True))

    def test_failed_dpo_evaluation_is_visible_and_blocks_build_until_retried(self):
        pair = {
            "sample_id": "dpo::s1::failed",
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}
        with patch("pipeline.judge._call", return_value={}):
            result = stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual(1, result["failed"])
        row = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]
        self.assertEqual("review_failed", row["gate_status"])
        self.assertFalse(row["included"])
        with self.assertRaisesRegex(RuntimeError, "尚未完成模型复核"):
            stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)

        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict):
            stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        retried = self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]
        self.assertEqual("good", retried["gate_status"])
        self.assertTrue(retried["included"])

    def test_dpo_context_and_unknown_cache_entries_are_reevaluated(self):
        sample_id = "dpo::s1::context"
        pair = {
            "sample_id": sample_id,
            "conversations": [{"from": "human", "value": "原始上下文"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned", "reason": "原始原因"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}
        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict):
            stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual("good", self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]["gate_status"])

        pair["conversations"][0]["value"] = "上下文已变化，但 chosen/rejected 不变"
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.assertEqual("pending", self.client.get("/api/dataset/draft", params={"kind": "dpo"}).json()["rows"][0]["gate_status"])
        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict) as call:
            result = stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual(1, result["evaluated"])
        self.assertTrue(call.called)

        cache_path = self.dirs["state"] / "dpo_review_cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for key in list(cache):
            cache[key] = {"verdict": "error", "reason": "旧格式"}
        write_json_atomic(cache_path, cache)
        with patch("pipeline.judge._call", side_effect=_good_dpo_verdict) as retry:
            result = stage3_synth.evaluate_dpo_pending(cfg=self.cfg, log=lambda _msg: None)
        self.assertEqual(1, result["evaluated"])
        self.assertTrue(retry.called)

    def test_diag_change_creates_a_new_build(self):
        first = stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        index = json.loads((self.dirs["gold"] / "index.json").read_text(encoding="utf-8"))
        diag_file = self.dirs["gold"] / "rejects" / "s1.diagnostics.jsonl"
        write_jsonl_atomic(diag_file, [{"session_id": "s1", "pathologies": ["new_diag"]}])
        index["sessions"]["s1"]["diag_file"] = str(diag_file)
        write_json_atomic(self.dirs["gold"] / "index.json", index)

        second = stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertNotEqual(first["build_id"], second["build_id"])
        rows = read_jsonl(self.dirs["root"] / "04_release" / "builds" / second["build_id"] / "negative_diagnostics.jsonl")
        self.assertIn("new_diag", rows[0]["pathologies"])

    def test_stage3_does_not_auto_package(self):
        self.package_mock.reset_mock()
        result = stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        self.assertTrue(result["build_id"])
        self.assertIsNone(result.get("package"))
        self.package_mock.assert_not_called()

    def test_missing_build_config_prevents_package_fallback(self):
        result = stage3_synth.run(cfg=self.cfg, log=lambda _msg: None)
        build_dir = self.dirs["root"] / "04_release" / "builds" / result["build_id"]
        (build_dir / "build_config.json").unlink()
        from server.app import Job, _run_job
        job = Job("package", True, {"build_id": result["build_id"], "profile": None})
        _run_job(job)
        self.assertEqual("failed", job.status)
        self.assertIn("配置快照", job.error)

    def test_failed_dpo_evaluation_marks_job_failed(self):
        pair = {
            "sample_id": "dpo::s1::job-failed",
            "conversations": [{"from": "human", "value": "分析文件"}],
            "chosen": {"from": "gpt", "value": "先验证"},
            "rejected": {"from": "gpt", "value": "直接猜测"},
            "metadata": {"session_id": "s1", "project_group": "game::alpha", "source": "stage2_pruned"},
        }
        write_jsonl_atomic(self.dirs["gold"] / "dpo_pairs.jsonl", [pair])
        self.cfg["judge"] = {"enabled": True}
        from server.app import Job, _run_job
        job = Job("dataset_evaluate", False, {})
        with patch("pipeline.judge._call", return_value={}):
            _run_job(job)
        self.assertEqual("failed", job.status)
        self.assertIn("评估未全部完成", job.error)


if __name__ == "__main__":
    unittest.main()
