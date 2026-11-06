"""发布、激活、回滚，以及历史发布迁入 Build。"""
from __future__ import annotations

import copy
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from fastapi.testclient import TestClient

from pipeline import release
from pipeline.common.config import DEFAULTS, config_hash, stage_dirs
from pipeline.common.store import sha256_file, write_json_atomic
from pipeline.dataset_build import load_build_config, resolve_build
from server.app import app

LEGACY = "v20261027-1736-2715fdbe"
BUILD_A = "a" * 16
BUILD_B = "b" * 16


def _cfg(root: Path) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["paths"]["data_dir"] = str(root)
    return cfg


def _seed_build(root: Path, build_id: str, row: str) -> None:
    build = root / "04_release" / "builds" / build_id
    build.mkdir(parents=True)
    data = build / "sft.train.jsonl"
    data.write_text(row, encoding="utf-8")
    snapshot = {"marker": build_id}
    write_json_atomic(build / "build_config.json", snapshot)
    write_json_atomic(build / "BUILD.json", {
        "version": 1,
        "build_id": build_id,
        "plan_hash": build_id,
        "created_at": "2026-11-06T10:00:00+08:00",
        "config_hash": config_hash(snapshot),
        "files": [
            {
                "path": "sft.train.jsonl",
                "sha256": sha256_file(data),
                "size": data.stat().st_size,
                "rows": 1,
            },
            {
                "path": "build_config.json",
                "sha256": sha256_file(build / "build_config.json"),
                "size": (build / "build_config.json").stat().st_size,
            },
        ],
        "counts": {"sft.train.jsonl": 1},
    })
    write_json_atomic(
        root / "04_release" / "builds" / "LATEST_BUILD.json",
        {"build_id": build_id},
    )


def _seed_legacy(root: Path) -> None:
    folder = root / "04_release" / "releases" / LEGACY
    folder.mkdir(parents=True)
    data = folder / "sft.train.jsonl"
    data.write_text('{"sample_id":"old"}\n', encoding="utf-8")
    write_json_atomic(folder / "manifest.json", {
        "version": LEGACY,
        "created_at": "2026-10-27T17:36:59+08:00",
        "config_hash": "b3267c89eaca12e9",
        "files": {
            "sft.train.jsonl": {"count": 1, "sha256": sha256_file(data)},
        },
        "counts": {"sft_train": 1},
    })
    (root / "04_release" / "ACTIVE_RELEASE.txt").write_text(
        LEGACY + "\nsynced_at=2026-10-29T15:45:47+08:00\n",
        encoding="utf-8",
    )


class ReleaseFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = _cfg(self.root)
        _seed_build(self.root, BUILD_A, "alpha\n")
        _seed_legacy(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_does_not_change_the_active_release(self):
        before = (self.root / "04_release" / "ACTIVE_RELEASE.txt").read_text(encoding="utf-8")
        published = release.publish(self.cfg, BUILD_A)
        self.assertFalse(published["activated"])
        self.assertFalse(published["active"])
        self.assertTrue(published["version"].endswith(BUILD_A[:8]))
        after = (self.root / "04_release" / "ACTIVE_RELEASE.txt").read_text(encoding="utf-8")
        self.assertEqual(before, after)
        copied = self.root / "04_release" / "releases" / published["version"] / "sft.train.jsonl"
        self.assertEqual("alpha\n", copied.read_text(encoding="utf-8"))

        again = release.publish(self.cfg, BUILD_A)
        self.assertTrue(again["reused"])
        self.assertEqual(published["version"], again["version"])
        self.assertEqual(before, (self.root / "04_release" / "ACTIVE_RELEASE.txt").read_text(encoding="utf-8"))

    def test_activate_then_rollback_restores_the_previous_version(self):
        _seed_build(self.root, BUILD_B, "beta\n")
        first = release.publish(self.cfg, BUILD_A)
        second = release.publish(self.cfg, BUILD_B)
        release.activate(self.cfg, first["version"])
        release.activate(self.cfg, second["version"])
        self.assertEqual(second["version"], release.read_active_version(self.cfg))

        restored = release.rollback(self.cfg)
        self.assertEqual(first["version"], restored["active_version"])
        self.assertEqual(first["version"], release.read_active_version(self.cfg))
        older = release.rollback(self.cfg)
        self.assertEqual(LEGACY, older["active_version"])
        self.assertNotEqual(second["version"], release.read_active_version(self.cfg))
        with self.assertRaisesRegex(RuntimeError, "没有可回滚的上一版本"):
            release.rollback(self.cfg)
        self.assertEqual(LEGACY, release.read_active_version(self.cfg))

    def test_legacy_import_creates_a_build_without_moving_pointers(self):
        latest = self.root / "04_release" / "builds" / "LATEST_BUILD.json"
        before_latest = latest.read_text(encoding="utf-8")
        before_active = release.read_active_version(self.cfg)
        imported = release.import_legacy(self.cfg, LEGACY)
        self.assertFalse(imported["activated"])
        self.assertEqual(16, len(imported["build_id"]))
        self.assertEqual(before_latest, latest.read_text(encoding="utf-8"))
        self.assertEqual(before_active, release.read_active_version(self.cfg))

        _path, manifest = resolve_build(self.cfg, imported["build_id"])
        self.assertEqual("legacy_release", manifest["origin"])
        self.assertEqual(LEGACY, manifest["legacy_version"])
        snapshot = load_build_config(_path, self.cfg)
        self.assertEqual(LEGACY, snapshot["legacy_import"]["version"])
        _latest_path, latest_manifest = resolve_build(self.cfg)
        self.assertEqual(BUILD_A, latest_manifest["build_id"])

        again = release.import_legacy(self.cfg, LEGACY)
        self.assertTrue(again["reused"])
        self.assertEqual(imported["build_id"], again["build_id"])
        listed = release.list_releases(self.cfg)
        legacy = next(row for row in listed["releases"] if row["version"] == LEGACY)
        self.assertEqual(imported["build_id"], legacy["imported_build_id"])
        self.assertTrue(legacy["active"])


class ReleaseApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = _cfg(self.root)
        _seed_build(self.root, BUILD_A, "alpha\n")
        _seed_legacy(self.root)
        self.client = TestClient(app)
        self.patch = unittest.mock.patch("server.app.load_config", return_value=self.cfg)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_http_publish_activate_and_failed_rollback(self):
        listed = self.client.get("/api/releases")
        self.assertEqual(200, listed.status_code)
        self.assertEqual(LEGACY, listed.json()["active_version"])
        self.assertFalse(listed.json()["can_rollback"])

        published = self.client.post("/api/releases", json={"build_id": BUILD_A})
        self.assertEqual(200, published.status_code, published.text)
        body = published.json()
        self.assertFalse(body["activated"])
        self.assertEqual(LEGACY, release.read_active_version(self.cfg))

        activated = self.client.post(f"/api/releases/{body['version']}/activate")
        self.assertEqual(200, activated.status_code, activated.text)
        self.assertEqual(body["version"], release.read_active_version(self.cfg))

        rolled = self.client.post("/api/releases/rollback")
        self.assertEqual(200, rolled.status_code, rolled.text)
        self.assertEqual(LEGACY, rolled.json()["active_version"])
        again = self.client.post("/api/releases/rollback")
        self.assertEqual(400, again.status_code)
        self.assertIn("没有可回滚的上一版本", again.json()["detail"])

        migrated = self.client.post(f"/api/releases/{LEGACY}/import-build")
        self.assertEqual(200, migrated.status_code, migrated.text)
        self.assertFalse(migrated.json()["activated"])
        self.assertEqual(LEGACY, release.read_active_version(self.cfg))
        dirs = stage_dirs(self.cfg)
        self.assertTrue((dirs["root"] / "04_release" / "builds" / migrated.json()["build_id"] / "BUILD.json").is_file())


class ExportToolstyleReportTests(unittest.TestCase):
    def test_unrecognized_tool_call_is_not_also_counted_as_hermes(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "tools" / "export_toolstyle.py"
        spec = importlib.util.spec_from_file_location("export_toolstyle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hermes = '<tool_call>\n{"name": "bash", "arguments": "ls"}\n</tool_call>'
        unknown = "<tool_call>\nnot-a-known-style\n</tool_call>"
        xml = "<tool_call>\n<function=bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>"
        report = module._style_report([
            {"conversations": [
                {"from": "gpt", "value": hermes},
                {"from": "gpt", "value": unknown},
            ]},
            {"conversations": [
                {"from": "gpt", "value": hermes},
                {"from": "gpt", "value": xml},
            ]},
            {"conversations": [{"from": "gpt", "value": hermes}]},
        ])
        self.assertEqual({"pure_hermes_rows": 1, "pure_xml_rows": 0, "mixed_or_unparsed_rows": 2, "no_toolcall_rows": 0}, report)
