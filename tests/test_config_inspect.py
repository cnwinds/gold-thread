"""`gt inspect config` 与 describe_config 的只读行为回归测试。"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from goldthread import cli
from pipeline.common import config as config_module
from pipeline.common import config_info
from pipeline.common.config import DEFAULTS, load_config
from pipeline.common.config_schema import ConfigError


def _get_path(data, path):
    current = data
    for part in path.split("."):
        current = current[part]
    return current


class DescribeConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "_config_path": "/tmp/private.yaml",
            "llm": {"model": "file-model"},
            "admission": {"workspace_prefixes": ["--tmp-"]},
            "synth": {"weights": {"opening": 5}},
            "judge": {"enabled": True, "task_models": {"seams": "seam-model"}},
            "review": {"enabled": True},
            "domain": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                            },
                        },
                    }
                ]
            },
        }
        self.entries = config_info.describe_config(self.raw)
        self.by_path = {entry["path"]: entry for entry in self.entries}

    def test_entries_are_sorted_and_have_exact_public_shape(self):
        paths = [entry["path"] for entry in self.entries]
        self.assertEqual(sorted(paths), paths)
        self.assertEqual(len(paths), len(set(paths)))
        for entry in self.entries:
            self.assertEqual({"path", "value", "source", "stage"}, set(entry))
        self.assertNotIn("_config_path", paths)
        self.assertFalse(any(path.startswith("_") for path in paths))

    def test_source_distinguishes_explicit_file_values_from_defaults(self):
        self.assertEqual("file", self.by_path["llm.model"]["source"])
        self.assertEqual("file-model", self.by_path["llm.model"]["value"])
        self.assertEqual("default", self.by_path["llm.temperature"]["source"])
        self.assertEqual("file", self.by_path["synth.weights.opening"]["source"])
        self.assertEqual(5, self.by_path["synth.weights.opening"]["value"])
        self.assertEqual("default", self.by_path["synth.weights.routine"]["source"])
        self.assertEqual("file", self.by_path["judge.task_models.seams"]["source"])
        self.assertNotIn("judge.task_models", self.by_path)

    def test_lists_replace_and_empty_mappings_do_not_clear_defaults(self):
        self.assertEqual(["--tmp-"], self.by_path["admission.workspace_prefixes"]["value"])
        merged = config_info.describe_config({"llm": {}})
        llm_model = next(item for item in merged if item["path"] == "llm.model")
        self.assertEqual(DEFAULTS["llm"]["model"], llm_model["value"])
        self.assertEqual("default", llm_model["source"])

    def test_dynamic_tool_schema_is_one_leaf_not_public_properties(self):
        self.assertIn("domain.tools", self.by_path)
        self.assertEqual(self.raw["domain"]["tools"], self.by_path["domain.tools"]["value"])
        self.assertFalse(any(path.startswith("domain.tools.") for path in self.by_path))
        self.assertFalse(any("properties" in path for path in self.by_path))

    def test_stage_mapping_marks_core_legacy_auxiliary_and_ops(self):
        self.assertIn("stage1", self.by_path["admission.workspace_prefixes"]["stage"])
        self.assertIn("stage2", self.by_path["gold.min_gold_steps"]["stage"])
        self.assertIn("stage3", self.by_path["synth.window_turns"]["stage"])
        self.assertIn("legacy-only", self.by_path["review.enabled"]["stage"])
        self.assertIn("auxiliary", self.by_path["llm.model"]["stage"])
        self.assertIn("ops", self.by_path["turns.use_model"]["stage"])
        self.assertIn("ops", self.by_path["judge.enabled"]["stage"])
        self.assertIn("auxiliary", self.by_path["judge.enabled"]["stage"])

    def test_offline_forced_flags_describe_real_consumption(self):
        turns_stage = self.by_path["turns.use_model"]["stage"]
        judge_stage = self.by_path["judge.enabled"]["stage"]
        for stage in (turns_stage, judge_stage):
            self.assertIn("强制 False", stage)
            self.assertIn("assist", stage)
        self.assertIn("assist turns", turns_stage)
        self.assertIn("assist judge", judge_stage)
        self.assertNotIn("默认 false", judge_stage)

    def test_dotted_and_underscore_dynamic_keys_keep_file_provenance(self):
        raw = {
            "turns": {
                "root_normalizations": {
                    "proj.with.dot": "dot prompt",
                    "_private_proj": "underscore prompt",
                }
            }
        }
        by_path = {entry["path"]: entry for entry in config_info.describe_config(raw)}

        dotted = by_path["turns.root_normalizations.proj.with.dot"]
        self.assertEqual("file", dotted["source"])
        self.assertEqual("dot prompt", dotted["value"])
        self.assertIn("stage1", dotted["stage"])

        private = by_path["turns.root_normalizations._private_proj"]
        self.assertEqual("file", private["source"])
        self.assertEqual("underscore prompt", private["value"])

        # 未显式覆盖的内建动态键仍是 default，不会被带点的显式键误标成 file。
        self.assertEqual("default", by_path["turns.root_normalizations.jigsawcard"]["source"])
        # 顶层来源标记依旧不是配置叶值。
        self.assertNotIn("_config_path", by_path)
        self.assertFalse(any(path.startswith("_config_path") for path in by_path))

    def test_merge_matches_load_config_effective_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            raw = {
                "llm": {"model": "file-model"},
                "synth": {"weights": {"opening": 5}, "formats": {"sharegpt": False}},
                "admission": {"workspace_prefixes": ["--tmp-"]},
                "review": {"enabled": True},
            }
            path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            loaded = load_config(path)
            for entry in config_info.describe_config(raw):
                self.assertEqual(_get_path(loaded, entry["path"]), entry["value"], entry["path"])


class ConfigSnapshotTests(unittest.TestCase):
    def test_snapshot_uses_canonical_parser_without_guessed_imports(self):
        self.assertIs(config_info.parse_config_text, config_module.parse_config_text)
        self.assertFalse(hasattr(config_info, "_schema_parser"))
        self.assertFalse(hasattr(config_info, "parse_explicit_config_text"))
        self.assertFalse(hasattr(config_info, "_SCHEMA_MODULES"))

    def test_snapshot_parses_each_file_exactly_once(self):
        seen = []
        real = config_module.parse_config_text

        def counting(text):
            seen.append(text)
            return real(text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text("llm:\n  model: snapshot-model\n", encoding="utf-8")
            with mock.patch.object(config_info, "parse_config_text", counting):
                raw, target, exists = config_info.load_explicit_config(path)
        self.assertEqual(["llm:\n  model: snapshot-model\n"], seen)
        self.assertEqual({"llm": {"model": "snapshot-model"}}, raw)
        self.assertEqual(path, target)
        self.assertTrue(exists)

    def test_invalid_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text("synth:\n  window_turn: 3\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                config_info.load_explicit_config(path)
        self.assertIn("synth.window_turn", str(ctx.exception))

    def test_invalid_yaml_fails_closed_without_safe_load_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text("llm: [unclosed\n", encoding="utf-8")
            boom = AssertionError("permissive yaml.safe_load fallback must not be used")
            with mock.patch("yaml.safe_load", side_effect=boom):
                with self.assertRaises(ConfigError) as ctx:
                    config_info.load_explicit_config(path)
        self.assertIn("invalid YAML", str(ctx.exception))

    def test_non_mapping_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                config_info.load_explicit_config(path)
        self.assertIn("object", str(ctx.exception))

    def test_explicit_missing_file_fails_but_default_missing_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                config_info.load_explicit_config(missing)
            with mock.patch.object(config_info, "CONFIG_PATH", missing):
                raw, path, exists = config_info.load_explicit_config(None)
            self.assertEqual({}, raw)
            self.assertEqual(missing, path)
            self.assertFalse(exists)


class InspectConfigCliTests(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_json_cli_reads_explicit_file_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "llm": {"model": "cli-model"},
                        "admission": {"workspace_prefixes": ["--cli-"]},
                        "judge": {"enabled": True},
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            code, out, err = self._run(
                ["inspect", "config", "--config", str(path), "--json"]
            )
        self.assertEqual(0, code, err)
        entries = json.loads(out)
        self.assertIsInstance(entries, list)
        by_path = {entry["path"]: entry for entry in entries}
        self.assertEqual("file", by_path["llm.model"]["source"])
        self.assertEqual("cli-model", by_path["llm.model"]["value"])
        self.assertEqual("default", by_path["turns.use_model"]["source"])
        self.assertIn("ops", by_path["turns.use_model"]["stage"])

    def test_cli_reads_and_validates_the_file_once(self):
        seen = []
        real = config_module.parse_config_text

        def counting(text):
            seen.append(text)
            return real(text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text(
                yaml.safe_dump({"llm": {"model": "cli-model"}}, allow_unicode=True),
                encoding="utf-8",
            )
            with mock.patch.object(config_module, "parse_config_text", counting), mock.patch.object(
                config_info, "parse_config_text", counting
            ):
                code, out, err = self._run(
                    ["inspect", "config", "--config", str(path), "--json"]
                )
        self.assertEqual(0, code, err)
        # 值 + 来源来自同一份校验快照，文件只解析一次。
        self.assertEqual(1, len(seen))
        entries = json.loads(out)
        by_path = {entry["path"]: entry for entry in entries}
        self.assertEqual("file", by_path["llm.model"]["source"])
        self.assertEqual("cli-model", by_path["llm.model"]["value"])

    def test_invalid_config_cli_matches_direct_snapshot_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.yaml"
            path.write_text("synth:\n  window_turn: 3\n", encoding="utf-8")
            code, _out, err = self._run(["inspect", "config", "--config", str(path)])
            json_code, json_out, _json_err = self._run(
                ["inspect", "config", "--config", str(path), "--json"]
            )
            with self.assertRaises(ConfigError) as ctx:
                config_info.load_explicit_config(path)
        self.assertNotEqual(0, code)
        self.assertIn("synth.window_turn", err)
        self.assertNotEqual(0, json_code)
        payload = json.loads(json_out)
        self.assertFalse(payload["ok"])
        self.assertEqual("ConfigError", payload["error_type"])
        self.assertIn("synth.window_turn", payload["error"])
        self.assertIn("synth.window_turn", str(ctx.exception))

    def test_explicit_missing_file_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.yaml"
            code, _out, err = self._run(
                ["inspect", "config", "--config", str(missing)]
            )
        self.assertNotEqual(0, code)
        self.assertIn("配置文件不存在", err)

    def test_explicit_missing_file_json_error_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.yaml"
            code, out, _err = self._run(
                ["inspect", "config", "--config", str(missing), "--json"]
            )
        self.assertNotEqual(0, code)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertIn("配置文件不存在", payload["error"])

    def test_ordinary_default_file_missing_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "pipeline.yaml"
            with mock.patch.object(config_module, "CONFIG_PATH", missing), mock.patch.object(
                config_info, "CONFIG_PATH", missing
            ):
                code, out, err = self._run(["inspect", "config", "--json"])
        self.assertEqual(0, code, err)
        entries = json.loads(out)
        self.assertTrue(entries)
        self.assertTrue(all(entry["source"] == "default" for entry in entries))
        by_path = {entry["path"]: entry for entry in entries}
        self.assertEqual(DEFAULTS["llm"]["model"], by_path["llm.model"]["value"])


if __name__ == "__main__":
    unittest.main()