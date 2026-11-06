"""交付 zip 必须解压后就能按 YAML 训练：键、文件、dataset_dir 三者一致。"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline import dataset_build

from pipeline.common.config import load_config
from pipeline.common.format import _parse_tc_block, convert_toolcall_text, export_profiles_public
from pipeline.lf_normalize import TOOL_RESPONSE_JOIN, normalize_messages, prepare_datasets
from pipeline.package import build_delivery_zip, compute_package_hash
from pipeline import length_gate as length_gate_mod

THINK = "<" + "think>" + "\nok\n" + "<" + "/think>" + "\n\n"


def _msg(role: str, value: str):
    return {"from": role, "value": value}


def _sft(sample_id: str, *, illegal: bool = False):
    conv = [
        _msg("system", "sys"),
        _msg("human", "do the task"),
        _msg("gpt", "call"),
        _msg("observation", "part-a" + TOOL_RESPONSE_JOIN + "part-b"),
        _msg("observation", "part-c"),
        _msg("gpt", THINK + "done"),
    ]
    if illegal:
        conv.insert(-1, _msg("human", "打断"))
    return {"sample_id": sample_id, "conversations": conv, "tools": "[]"}


def _dpo(sample_id: str):
    return {
        "sample_id": sample_id,
        "conversations": [
            _msg("system", "sys"),
            _msg("human", "do the task"),
            _msg("gpt", "first"),
            _msg("gpt", "second"),
            _msg("observation", "out"),
        ],
        "chosen": {"from": "gpt", "value": THINK + "good"},
        "rejected": {"from": "gpt", "value": THINK + "bad"},
        "tools": "[]",
    }


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _yaml(stage: str, dataset: str, eval_dataset: str) -> str:
    return (
        f"### method\nstage: {stage}\n### dataset\n"
        f"dataset_dir: /data/goldthread/03_train\n"
        f"dataset: {dataset}\n"
        f"eval_dataset: {eval_dataset}\n"
    )


def _scaffold(root: Path) -> None:
    (root / "handoff.md").write_text("# handoff\n\n旧文档仍写着 pi_sft_step_windows_train。\n", encoding="utf-8")
    (root / "train_sft.yaml").write_text(
        _yaml("sft", "pi_sft_step_windows_train", "pi_sft_step_windows_eval"),
        encoding="utf-8",
    )
    (root / "train_sft_qlora.yaml").write_text(
        _yaml("sft", "gt_sft_train", "gt_sft_eval"),
        encoding="utf-8",
    )
    (root / "train_dpo.yaml").write_text(
        _yaml("dpo", "pi_dpo_preference_pairs_train", "pi_dpo_preference_pairs_eval"),
        encoding="utf-8",
    )
    (root / "eval_sft.yaml").write_text(
        "### method\nstage: sft\n### dataset\n"
        "dataset_dir: /abs/03_train\n"
        "eval_dataset: gt_sft_eval_lf\n",
        encoding="utf-8",
    )


PACK = "goldthread-train"


def _at(root: Path, *parts: str) -> Path:
    return root.joinpath(PACK, *parts)


def _unzip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def _check_data(pack_root: Path):
    return subprocess.run(
        [sys.executable, str(pack_root / "tools" / "check_data.py")],
        cwd=pack_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class NormalizeTests(unittest.TestCase):
    def test_join_and_consecutive_roles_collapse(self):
        messages = normalize_messages(_sft("x")["conversations"])
        roles = [m["from"] for m in messages]
        self.assertEqual(["system", "human", "gpt", "observation", "gpt"], roles)
        obs = messages[3]["value"]
        self.assertNotIn("tool_response", obs)
        self.assertIn("part-a\n\npart-b\n\npart-c", obs)


class DeliveryZipTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def _build(self, root: Path):
        return build_delivery_zip(root, self.cfg, log=lambda *_: None)

    def test_build_package_ignores_workspace_legacy_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg)
            cfg["paths"]["data_dir"] = str(Path(tmp) / "data")
            root = Path(cfg["paths"]["data_dir"]) / "03_train"
            root.mkdir(parents=True)
            _scaffold(root)
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("from-v2")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("from-v2-eval")])
            _write_jsonl(root / "sft.train.lf.jsonl", [_sft("from-old-lf")])
            _write_jsonl(root / "sft.eval.lf.jsonl", [_sft("from-old-lf-eval")])
            _write_jsonl(root / "sft_step_windows_sharegpt.train.jsonl", [_sft("from-stage3")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.jsonl", [_sft("from-stage3-eval")])
            counts = {
                "sft_step_windows_sharegpt.train.jsonl": 1,
                "sft_step_windows_sharegpt.eval.jsonl": 1,
            }
            draft = {"split_plan": {"assignments": {"game::demo": "train"}}, "decisions": {}}
            build_dir, _manifest = dataset_build.create_build(root, cfg, "a" * 16, draft, counts)
            result = build_delivery_zip(
                build_dir,
                cfg,
                log=lambda *_: None,
            )
            self.assertEqual(
                Path(cfg["paths"]["data_dir"]) / "04_release" / "delivery",
                Path(result["zip_path"]).parent,
            )
            unzipped = Path(tmp) / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            rows = _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")
            self.assertEqual(["from-stage3"], [row["sample_id"] for row in rows])
            eval_rows = _load_jsonl(_at(unzipped, "data") / "sft.eval.lf.jsonl")
            self.assertEqual(["from-stage3-eval"], [row["sample_id"] for row in eval_rows])
            manifest = json.loads((_at(unzipped, "data", "MANIFEST.json")).read_text(encoding="utf-8"))
            self.assertEqual("a" * 16, manifest["build_id"])
            self.assertEqual("a" * 16, result["build_id"])

            _write_jsonl(root / "sft_step_windows_sharegpt.train.jsonl", [_sft("changed-without-new-id")])
            with self.assertRaisesRegex(RuntimeError, "产物内容不同"):
                dataset_build.create_build(root, cfg, "a" * 16, draft, counts)

    def test_prefers_v2_and_locks_yaml_to_lf_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("from-v2")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("from-v2-eval")])
            _write_jsonl(root / "sft_step_windows_sharegpt.train.jsonl", [_sft("from-stage3")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.jsonl", [_sft("from-stage3-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.v2.jsonl", [_dpo("dpo-v2")])
            _write_jsonl(root / "dpo_preference_pairs.eval.v2.jsonl", [_dpo("dpo-v2-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.jsonl", [_dpo("dpo-stage3")])
            _write_jsonl(
                root / "sft_step_windows_openai.train.jsonl",
                [_sft("openai-should-not-pack")],
            )

            result = self._build(root)
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)

            names = {n for n in zipfile.ZipFile(result["zip_path"]).namelist()}
            self.assertIn(f"{PACK}/data/sft.train.lf.jsonl", names)
            self.assertIn(f"{PACK}/data/dpo.train.lf.jsonl", names)
            self.assertIn(f"{PACK}/data/dataset_info.json", names)
            self.assertNotIn("dataset_info.json", names)
            self.assertNotIn(f"{PACK}/data/sft_step_windows_sharegpt.train.jsonl", names)
            self.assertNotIn(f"{PACK}/data/sft_step_windows_openai.train.jsonl", names)
            self.assertIn(f"{PACK}/autotrain.py", names)
            self.assertIn(f"{PACK}/tools/train_lowmem.py", names)
            self.assertIn(f"{PACK}/tools/lowmem_dpo.py", names)
            self.assertIn(f"{PACK}/README.md", names)
            self.assertIn(f"{PACK}/tools/check_data.py", names)
            self.assertIn(f"{PACK}/tools/filter_by_cutoff.py", names)
            self.assertNotIn("START_HERE.md", names)

            # zip stem is package_hash (content fingerprint), not config_hash[:8]
            self.assertIn("package_hash", result)
            self.assertIn("config_hash", result)
            self.assertEqual(
                result["zip_name"],
                f"goldthread-train-{result['package_hash']}.zip",
            )
            self.assertEqual(8, len(result["package_hash"]))
            man = json.loads((_at(unzipped, "data", "MANIFEST.json")).read_text(encoding="utf-8"))
            self.assertEqual(result["package_hash"], man["package_hash"])
            self.assertEqual(result["config_hash"], man["config_hash"])
            self.assertIn("length_gate_version", man)
            self.assertTrue(
                str(man["length_gate_version"]).startswith(length_gate_mod.LENGTH_GATE_VERSION)
            )

            info = json.loads((_at(unzipped, "data") / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual("sft.train.lf.jsonl", info["gt_sft_train_lf"]["file_name"])
            self.assertEqual("dpo.train.lf.jsonl", info["gt_dpo_train_lf"]["file_name"])
            self.assertFalse(info["gt_sft_train_lf"]["file_name"].startswith("data/"))
            self.assertTrue(info["gt_dpo_train_lf"]["ranking"])

            sft_rows = _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")
            self.assertEqual(["from-v2"], [r["sample_id"] for r in sft_rows])
            self.assertNotIn("tool_response", json.dumps(sft_rows, ensure_ascii=False))
            dpo_rows = _load_jsonl(_at(unzipped, "data") / "dpo.train.lf.jsonl")
            self.assertEqual(["dpo-v2"], [r["sample_id"] for r in dpo_rows])
            self.assertIsInstance(dpo_rows[0]["chosen"], dict)
            self.assertEqual(
                ["system", "human", "gpt", "observation"],
                [m["from"] for m in dpo_rows[0]["conversations"]],
            )

            settings = _at(unzipped, "settings.yaml").read_text(encoding="utf-8")
            self.assertRegex(settings, r"(?m)^  template: qwen3_8\b")
            self.assertIn("gt_sft_train_lf", settings)
            self.assertIn("gt_dpo_train_lf", settings)
            readme = _at(unzipped, "README.md").read_text(encoding="utf-8")
            self.assertIn("python autotrain.py sft", readme)
            self.assertIn("qwen3_coder", readme)
            self.assertIn("| 1 |", readme)

            proc = _check_data(_at(unzipped))
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_template_trains_through_autotrain_not_workspace_yaml(self):
        """工作区里的旧 merge/train yaml 不进包；训练入口是模板里的 autotrain。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            (root / "merge_lora.yaml").write_text("export_dir: /should/not/pack\n", encoding="utf-8")
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.v2.jsonl", [_dpo("dpo")])
            _write_jsonl(root / "dpo_preference_pairs.eval.v2.jsonl", [_dpo("dpo-eval")])

            result = self._build(root)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn(f"{PACK}/autotrain.py", names)
            self.assertIn(f"{PACK}/configs/sft.yaml", names)
            self.assertIn(f"{PACK}/configs/dpo.yaml", names)
            self.assertNotIn(f"{PACK}/merge_lora.yaml", names)
            self.assertFalse(any("should/not/pack" in zipfile.ZipFile(result["zip_path"]).read(n).decode("utf-8", "ignore") for n in names if n.endswith((".yaml", ".yml", ".md"))))

            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            sft_cfg = _at(unzipped, "configs", "sft.yaml").read_text(encoding="utf-8")
            dpo_cfg = _at(unzipped, "configs", "dpo.yaml").read_text(encoding="utf-8")
            self.assertIn("stage: sft", sft_cfg)
            self.assertIn("learning_rate: 1.0e-5", dpo_cfg)
            self.assertIn("template: null", sft_cfg)
            readme = _at(unzipped, "README.md").read_text(encoding="utf-8")
            self.assertIn("python autotrain.py sft", readme)
            self.assertIn("python autotrain.py dpo", readme)

    def test_settings_omit_dpo_keys_without_dpo_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])

            result = self._build(root)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertNotIn(f"{PACK}/data/dpo.train.lf.jsonl", names)
            man = json.loads(zipfile.ZipFile(result["zip_path"]).read(f"{PACK}/data/MANIFEST.json").decode("utf-8"))
            self.assertNotIn("gt_dpo_train_lf", man["datasets"])
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            settings = _at(unzipped, "settings.yaml").read_text(encoding="utf-8")
            self.assertNotIn("gt_dpo_train_lf", settings)
            proc = _check_data(_at(unzipped))
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_workspace_yaml_does_not_change_package_hash(self):
        """模板外的工作区 YAML 不再决定包内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])
            (root / "merge_lora.yaml").write_text("export_dir: /data/models/a\n", encoding="utf-8")
            first = self._build(root)
            (root / "merge_lora.yaml").write_text("export_dir: /data/models/b\n", encoding="utf-8")
            changed = self._build(root)
            self.assertEqual(first["package_hash"], changed["package_hash"])

    def test_stage3_only_drops_illegal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            _write_jsonl(
                root / "sft_step_windows_sharegpt.train.jsonl",
                [_sft("keep"), _sft("drop-me", illegal=True)],
            )
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.jsonl", [_sft("keep-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.jsonl", [_dpo("dpo-keep")])
            _write_jsonl(root / "dpo_preference_pairs.eval.jsonl", [_dpo("dpo-eval")])
            result = self._build(root)
            prepared = prepare_datasets(root)
            self.assertEqual(["keep"], [r["sample_id"] for r in prepared.files["sft.train.lf.jsonl"]])
            self.assertTrue(any("drop-me" in item for item in prepared.dropped["sft.train"]))
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            proc = _check_data(_at(unzipped))
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_existing_aligned_lf_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            kept = _sft("hand-tuned")
            kept["conversations"] = normalize_messages(kept["conversations"])
            kept_eval = _sft("hand-tuned-eval")
            kept_eval["conversations"] = normalize_messages(kept_eval["conversations"])
            _write_jsonl(root / "sft.train.lf.jsonl", [kept])
            _write_jsonl(root / "sft.eval.lf.jsonl", [kept_eval])
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("from-v2")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("from-v2-eval")])
            _write_jsonl(root / "dpo.train.lf.jsonl", [_dpo_aligned("dpo-kept")])
            _write_jsonl(root / "dpo.eval.lf.jsonl", [_dpo_aligned("dpo-kept-eval")])
            result = self._build(root)
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            ids = [r["sample_id"] for r in _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")]
            self.assertEqual(["hand-tuned"], ids)

    def test_unaligned_existing_lf_is_rebuilt_from_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            _write_jsonl(root / "sft.train.lf.jsonl", [_sft("broken", illegal=True)])
            _write_jsonl(root / "sft.eval.lf.jsonl", [_sft("broken-eval", illegal=True)])
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("from-v2")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("from-v2-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.v2.jsonl", [_dpo("dpo-v2")])
            _write_jsonl(root / "dpo_preference_pairs.eval.v2.jsonl", [_dpo("dpo-v2-eval")])
            result = self._build(root)
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            ids = [r["sample_id"] for r in _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")]
            self.assertEqual(["from-v2"], ids)
            proc = _check_data(_at(unzipped))
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_stale_subset_lf_is_rebuilt_from_v2(self):
        """盘上 lf 条数明显少于 v2（旧 release 子集）时必须按 v2 重生成。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            _scaffold(root)
            v2 = [_sft(f"s{i}") for i in range(10)]
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", v2)
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", v2[:2])
            _write_jsonl(root / "sft.train.lf.jsonl", v2[:4])
            _write_jsonl(root / "sft.eval.lf.jsonl", v2[:1])
            from pipeline.lf_normalize import prepare_datasets
            prep = prepare_datasets(root)
            self.assertEqual(prep.counts.get("sft.train.lf.jsonl"), 10)
            self.assertTrue(str(prep.sources["sft.train"]).endswith(".v2.jsonl"))


def _dpo_aligned(sample_id: str):
    row = _dpo(sample_id)
    row["conversations"] = normalize_messages(row["conversations"])
    return row


def _hermes(command: str) -> str:
    body = json.dumps({"name": "bash", "arguments": {"command": command}}, ensure_ascii=False)
    return "<tool_call>\n" + body + "\n</tool_call>"


def _sft_tool(sample_id: str = "with-tool"):
    row = _sft(sample_id)
    row["conversations"][2]["value"] = _hermes("ls -la")
    row["conversations"][3]["value"] = "tool said " + _hermes("echo keep") + " and <tool_call>garbage</tool_call>"
    row["conversations"][5]["value"] = THINK + _hermes("pwd")
    return row


def _dpo_tool(sample_id: str = "dpo-tool"):
    row = _dpo(sample_id)
    row["chosen"]["value"] = THINK + _hermes("good-cmd")
    row["rejected"]["value"] = THINK + _hermes("bad-cmd")
    return row


class ExportFormatTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def _seed(self, root: Path):
        _scaffold(root)
        (root / "train_sft_qwen3_8b.yaml").write_text(
            _yaml("sft", "gt_sft_train", "gt_sft_eval") + "template: qwen3\n",
            encoding="utf-8",
        )
        _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft_tool("from-v2")])
        _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft_tool("from-v2-eval")])
        _write_jsonl(root / "dpo_preference_pairs.train.v2.jsonl", [_dpo_tool("dpo-v2")])
        _write_jsonl(root / "dpo_preference_pairs.eval.v2.jsonl", [_dpo_tool("dpo-v2-eval")])

    def test_qwen_xml_rewrites_assistant_and_locks_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            self._seed(root)
            result = build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="Qwen3.8-27B")
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)

            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn(f"{PACK}/data/MANIFEST.json", names)
            self.assertNotIn(f"{PACK}/train_sft_qwen3_8b.yaml", names)
            self.assertFalse(any(n.endswith("qwen3_8b.yaml") for n in names))

            sft = _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")[0]
            gpts = [m["value"] for m in sft["conversations"] if m["from"] == "gpt"]
            gpt, target = gpts[0], gpts[-1]
            obs = next(m["value"] for m in sft["conversations"] if m["from"] == "observation")
            self.assertIn("<function=bash>", gpt)
            self.assertIn("<parameter=command>", gpt)
            self.assertIn("ls -la", gpt)
            self.assertNotIn('"arguments"', gpt)
            self.assertIn("<function=bash>", target)
            self.assertIn("pwd", target)
            self.assertIn('"arguments"', obs)
            self.assertIn("<tool_call>garbage</tool_call>", obs)

            dpo = _load_jsonl(_at(unzipped, "data") / "dpo.train.lf.jsonl")[0]
            self.assertIn("good-cmd", dpo["chosen"]["value"])
            self.assertIn("<function=bash>", dpo["chosen"]["value"])
            self.assertIn("bad-cmd", dpo["rejected"]["value"])
            self.assertNotIn('"arguments"', dpo["rejected"]["value"])

            settings = _at(unzipped, "settings.yaml").read_text(encoding="utf-8")
            self.assertRegex(settings, r"(?m)^  template: qwen3_8\b")
            readme = _at(unzipped, "README.md").read_text(encoding="utf-8")
            self.assertIn("--tool-call-parser qwen3_coder", readme)
            man = json.loads((_at(unzipped, "data", "MANIFEST.json")).read_text(encoding="utf-8"))
            contract = man["format_contract"]
            self.assertEqual("Qwen3.8-27B", contract["profile"])
            self.assertEqual("coder-xml", contract["tool_style"])
            self.assertEqual("qwen3_8", contract["train_template"])
            self.assertEqual("qwen3_coder", contract["serve_tool_parser"])
            self.assertGreater(contract["blocks_converted"], 0)
            self.assertEqual("coder-xml", man["format"]["tool_call"])

    def test_hermes_keeps_json_and_includes_8b_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            self._seed(root)
            result = build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="hermes-json")
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertFalse(any(n.endswith("qwen3_8b.yaml") for n in names))

            sft = _load_jsonl(_at(unzipped, "data") / "sft.train.lf.jsonl")[0]
            gpt = sft["conversations"][2]["value"]
            self.assertIn('"arguments"', gpt)
            self.assertNotIn("<function=bash>", gpt)
            self.assertIn("<tool_call>garbage</tool_call>", sft["conversations"][3]["value"])

            settings = _at(unzipped, "settings.yaml").read_text(encoding="utf-8")
            self.assertRegex(settings, r"(?m)^  template: qwen3\b")
            self.assertIn("gt_sft_train_lf", settings)
            readme = _at(unzipped, "README.md").read_text(encoding="utf-8")
            self.assertIn("--tool-call-parser hermes", readme)
            self.assertNotIn("qwen3_coder", readme)
            public = {item["id"]: item for item in export_profiles_public()}
            self.assertFalse(public["Qwen3.8-8B"]["pack_8b_yaml"])
            self.assertIn("autotrain.py", public["Qwen3.8-8B"]["summary"])
            self.assertIn("qwen3_8", public["Qwen3.8-27B"]["summary"])

    def test_unparsed_assistant_tool_call_aborts_before_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            self._seed(root)
            bad = _sft("bad")
            bad["conversations"][5]["value"] = "<tool_call>nope</tool_call>"
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [bad])
            delivery = root.parent / "04_release" / "delivery"
            before = {p.name for p in delivery.glob("*.zip")} if delivery.exists() else set()
            with self.assertRaises(ValueError) as caught:
                build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="qwen-xml")
            self.assertIn("无法识别", str(caught.exception))
            after = {p.name for p in delivery.glob("*.zip")} if delivery.exists() else set()
            self.assertEqual(before, after)

    def test_stage3_repack_remembers_the_last_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            root.mkdir()
            self._seed(root)
            first = build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="Qwen3.8-8B")
            second = build_delivery_zip(root, self.cfg, log=lambda *_: None)
            self.assertEqual("Qwen3.8-8B", first["format"]["profile"])
            self.assertEqual("Qwen3.8-8B", second["format"]["profile"])
            self.assertEqual(first["package_hash"], second["package_hash"])

    def test_xml_roundtrip_keeps_hex_offsets_and_trailing_whitespace(self):
        command = (
            "cd /d/crack/jigsawcard/30-config/textassets-binary && "
            "xxd -s 0x940 -l 0x90 *ab__config.bin; echo; xxd -s 0x60 -l 0x60 *ab__config.bin"
        )
        script = "print(rb'[\\x20-\\x7e]{4,}')\n"
        padded = "grep flatbuffers "
        src = _hermes(command) + "\n" + _hermes(script) + "\n" + _hermes(padded)
        xml, converted, unparsed = convert_toolcall_text(src, "coder-xml")
        self.assertEqual(3, converted)
        self.assertEqual(0, unparsed)
        self.assertIn("xxd -s 0x940 -l 0x90", xml)
        self.assertIn("rb'[\\x20-\\x7e]{4,}'", xml)
        back, converted_back, unparsed_back = convert_toolcall_text(xml, "hermes-json")
        self.assertEqual(3, converted_back)
        self.assertEqual(0, unparsed_back)

        def commands(text):
            out = []
            for block in text.split("</tool_call>"):
                if "<tool_call>" not in block:
                    continue
                tc = _parse_tc_block(block[block.index("<tool_call>"):] + "</tool_call>")
                out.append(tc["arguments"]["command"])
            return out

        self.assertEqual([command, script, padded], commands(src))
        self.assertEqual([command, script, padded], commands(xml))
        self.assertEqual([command, script, padded], commands(back))

    def test_profile_changes_package_hash(self):
        rows = [_sft_tool("a")]
        fp = length_gate_mod.length_gate_fingerprint()
        h1 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, fp, "Qwen3.8-27B")
        h2 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, fp, "Qwen3.8-8B")
        h3 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, fp, "Qwen3.8-27B")
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, h3)


class PackageHashTests(unittest.TestCase):
    def test_content_change_renames_hash(self):
        rows_a = [{"sample_id": "a", "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]}]
        rows_b = [{"sample_id": "b", "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]}]
        fp = length_gate_mod.length_gate_fingerprint()
        h1 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows_a}, fp)
        h2 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows_b}, fp)
        self.assertEqual(8, len(h1))
        self.assertNotEqual(h1, h2)

    def test_length_gate_version_changes_hash(self):
        rows = [{"sample_id": "a", "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]}]
        h1 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, "obs-soft-v1")
        h2 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, "obs-soft-v2")
        self.assertNotEqual(h1, h2)

    def test_stable_for_same_inputs(self):
        rows = [{"sample_id": "a", "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]}]
        fp = length_gate_mod.length_gate_fingerprint()
        h1 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, fp)
        h2 = compute_package_hash("cfgabc", {"sft.train.lf.jsonl": rows}, fp)
        self.assertEqual(h1, h2)


class DeliveryListingTests(unittest.TestCase):
    def test_console_reads_new_and_legacy_zip_manifests(self):
        from server.app import _delivery_zips, _package_hash_from_name, _zip_manifest

        with tempfile.TemporaryDirectory() as tmp:
            delivery = Path(tmp)
            old = delivery / "goldthread-delivery-aaaaaaaa.zip"
            new = delivery / "goldthread-train-bbbbbbbb.zip"
            (delivery / "notes.zip").write_bytes(b"not a training pack")
            with zipfile.ZipFile(old, "w") as zf:
                zf.writestr(
                    "MANIFEST.json",
                    json.dumps({"package_hash": "aaaaaaaa", "counts": {"sft.train.lf.jsonl": 2}}),
                )
            with zipfile.ZipFile(new, "w") as zf:
                zf.writestr(
                    "goldthread-train/data/MANIFEST.json",
                    json.dumps({"package_hash": "bbbbbbbb", "counts": {"sft.train.lf.jsonl": 4}}),
                )
            names = [p.name for p in _delivery_zips(delivery)]
            self.assertEqual(
                ["goldthread-train-bbbbbbbb.zip", "goldthread-delivery-aaaaaaaa.zip"],
                names,
            )
            self.assertEqual("aaaaaaaa", _zip_manifest(old)["package_hash"])
            self.assertEqual(4, _zip_manifest(new)["counts"]["sft.train.lf.jsonl"])
            self.assertEqual("bbbbbbbb", _package_hash_from_name(new.stem))
            self.assertEqual("aaaaaaaa", _package_hash_from_name(old.stem))
        from server.app import _package_layout_note
        self.assertEqual("goldthread-train 模板", _package_layout_note("goldthread-train-bbbbbbbb.zip"))
        self.assertEqual("交付包", _package_layout_note("goldthread-delivery-aaaaaaaa.zip"))


if __name__ == "__main__":
    unittest.main()
