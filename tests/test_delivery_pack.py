"""交付 zip 必须解压后就能按 YAML 训练：键、文件、dataset_dir 三者一致。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline.common.config import load_config
from pipeline.common.format import _parse_tc_block, convert_toolcall_text
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


def _unzip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


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

    def test_prefers_v2_and_locks_yaml_to_lf_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            self.assertIn("data/sft.train.lf.jsonl", names)
            self.assertIn("data/dpo.train.lf.jsonl", names)
            self.assertIn("data/dataset_info.json", names)
            self.assertNotIn("dataset_info.json", names)
            self.assertNotIn("data/sft_step_windows_sharegpt.train.jsonl", names)
            self.assertNotIn("data/sft_step_windows_openai.train.jsonl", names)
            self.assertIn("train_lowmem.py", names)
            self.assertIn("lowmem_dpo.py", names)
            self.assertIn("START_HERE.md", names)
            self.assertIn("check_data.py", names)
            self.assertIn("filter_by_cutoff.py", names)

            # zip stem is package_hash (content fingerprint), not config_hash[:8]
            self.assertIn("package_hash", result)
            self.assertIn("config_hash", result)
            self.assertEqual(
                result["zip_name"],
                f"goldthread-delivery-{result['package_hash']}.zip",
            )
            self.assertEqual(8, len(result["package_hash"]))
            man = json.loads((unzipped / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(result["package_hash"], man["package_hash"])
            self.assertEqual(result["config_hash"], man["config_hash"])
            self.assertIn("length_gate_version", man)
            self.assertTrue(
                str(man["length_gate_version"]).startswith(length_gate_mod.LENGTH_GATE_VERSION)
            )

            info = json.loads((unzipped / "data" / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual("sft.train.lf.jsonl", info["gt_sft_train_lf"]["file_name"])
            self.assertEqual("dpo.train.lf.jsonl", info["gt_dpo_train_lf"]["file_name"])
            self.assertFalse(info["gt_sft_train_lf"]["file_name"].startswith("data/"))
            self.assertTrue(info["gt_dpo_train_lf"]["ranking"])

            sft_rows = _load_jsonl(unzipped / "data" / "sft.train.lf.jsonl")
            self.assertEqual(["from-v2"], [r["sample_id"] for r in sft_rows])
            self.assertNotIn("tool_response", json.dumps(sft_rows, ensure_ascii=False))
            dpo_rows = _load_jsonl(unzipped / "data" / "dpo.train.lf.jsonl")
            self.assertEqual(["dpo-v2"], [r["sample_id"] for r in dpo_rows])
            self.assertIsInstance(dpo_rows[0]["chosen"], dict)
            self.assertEqual(
                ["system", "human", "gpt", "observation"],
                [m["from"] for m in dpo_rows[0]["conversations"]],
            )

            for name in ("train_sft.yaml", "train_sft_qlora.yaml", "train_dpo.yaml", "eval_sft.yaml"):
                text = (unzipped / name).read_text(encoding="utf-8")
                self.assertIn("dataset_dir: ./data", text)
                self.assertNotIn("pi_sft", text)
                self.assertNotIn("pi_dpo", text)
            self.assertIn("dataset: gt_sft_train_lf", (unzipped / "train_sft.yaml").read_text(encoding="utf-8"))
            self.assertIn("eval_dataset: gt_sft_eval_lf", (unzipped / "train_sft_qlora.yaml").read_text(encoding="utf-8"))
            self.assertIn("dataset: gt_dpo_train_lf", (unzipped / "train_dpo.yaml").read_text(encoding="utf-8"))
            self.assertIn("CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml", (unzipped / "START_HERE.md").read_text(encoding="utf-8"))
            self.assertIn("gt_sft_train_lf", (unzipped / "handoff.md").read_text(encoding="utf-8")[:600])

            proc = subprocess.run(
                [sys.executable, str(unzipped / "check_data.py")],
                cwd=unzipped,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_merge_yaml_chain_is_packed(self):
        """训练→合并→DPO 的链路要能只靠包内的 yaml 跑完。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold(root)
            (root / "merge_lora.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B\n"
                "adapter_name_or_path: /data/saves/qwen38-27b-goldthread-sft-ab56c1d1-lf\n"
                "template: qwen3_8\nfinetuning_type: lora\n"
                "export_dir: /data/models/Qwen3.8-27B-merged\nexport_size: 5\n",
                encoding="utf-8",
            )
            (root / "merge_dpo.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B-merged\n"
                "adapter_name_or_path: /data/saves/qwen38-27b-goldthread-dpo-lf\n"
                "template: qwen3_8\nfinetuning_type: lora\n"
                "export_dir: /data/models/Qwen3.8-27B-dpo-merged\nexport_size: 5\n",
                encoding="utf-8",
            )
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])
            _write_jsonl(root / "dpo_preference_pairs.train.v2.jsonl", [_dpo("dpo")])
            _write_jsonl(root / "dpo_preference_pairs.eval.v2.jsonl", [_dpo("dpo-eval")])

            result = self._build(root)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn("merge_lora.yaml", names)
            self.assertIn("merge_dpo.yaml", names)

            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            sft_merge = (unzipped / "merge_lora.yaml").read_text(encoding="utf-8")
            dpo_merge = (unzipped / "merge_dpo.yaml").read_text(encoding="utf-8")
            train_sft = (unzipped / "train_sft.yaml").read_text(encoding="utf-8")
            self.assertIn("export_dir: /data/models/Qwen3.8-27B-merged", sft_merge)
            # DPO 底座 = SFT 合并产物；DPO 合并底座 = SFT 合并产物
            self.assertIn("model_name_or_path: /data/models/Qwen3.8-27B-merged", dpo_merge)
            self.assertIn("export_dir: /data/models/Qwen3.8-27B-dpo-merged", dpo_merge)
            # 合并 yaml 的 template 必须与数据/训练 yaml 同一套
            want_template = next(ln for ln in train_sft.splitlines() if ln.startswith("template:"))
            self.assertIn(want_template, sft_merge)
            self.assertIn(want_template, dpo_merge)
            self.assertNotIn("dataset:", sft_merge)
            start = (unzipped / "START_HERE.md").read_text(encoding="utf-8")
            self.assertIn("llamafactory-cli export merge_lora.yaml", start)
            self.assertIn("llamafactory-cli export merge_dpo.yaml", start)

    def test_merge_dpo_skipped_without_dpo_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold(root)
            (root / "merge_lora.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B\ntemplate: qwen3_8\n"
                "export_dir: /data/models/Qwen3.8-27B-merged\n",
                encoding="utf-8",
            )
            (root / "merge_dpo.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B-merged\ntemplate: qwen3_8\n"
                "export_dir: /data/models/Qwen3.8-27B-dpo-merged\n",
                encoding="utf-8",
            )
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])

            result = self._build(root)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn("merge_lora.yaml", names)
            self.assertNotIn("merge_dpo.yaml", names)
            man = json.loads(zipfile.ZipFile(result["zip_path"]).read("MANIFEST.json").decode("utf-8"))
            self.assertEqual(["merge_lora.yaml"], man["merge_yaml"])

    def test_package_hash_covers_packed_yaml(self):
        """改包内 YAML（不改数据）也要换 zip 名，避免同名不同内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold(root)
            _write_jsonl(root / "sft_step_windows_sharegpt.train.v2.jsonl", [_sft("sft")])
            _write_jsonl(root / "sft_step_windows_sharegpt.eval.v2.jsonl", [_sft("sft-eval")])
            (root / "merge_lora.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B\ntemplate: qwen3_8\n"
                "export_dir: /data/models/Qwen3.8-27B-merged\n",
                encoding="utf-8",
            )
            first = self._build(root)
            again = self._build(root)
            self.assertEqual(first["package_hash"], again["package_hash"])

            # 只改合并配置的 export_dir
            (root / "merge_lora.yaml").write_text(
                "model_name_or_path: /data/models/Qwen3.8-27B\ntemplate: qwen3_8\n"
                "export_dir: /data/models/Qwen3.8-27B-merged-b\n",
                encoding="utf-8",
            )
            changed = self._build(root)
            self.assertNotEqual(first["package_hash"], changed["package_hash"])

    def test_stage3_only_drops_illegal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            proc = subprocess.run(
                [sys.executable, str(unzipped / "check_data.py")],
                cwd=unzipped,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_existing_aligned_lf_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            ids = [r["sample_id"] for r in _load_jsonl(unzipped / "data" / "sft.train.lf.jsonl")]
            self.assertEqual(["hand-tuned"], ids)

    def test_unaligned_existing_lf_is_rebuilt_from_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            ids = [r["sample_id"] for r in _load_jsonl(unzipped / "data" / "sft.train.lf.jsonl")]
            self.assertEqual(["from-v2"], ids)
            proc = subprocess.run(
                [sys.executable, str(unzipped / "check_data.py")],
                cwd=unzipped,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_stale_subset_lf_is_rebuilt_from_v2(self):
        """盘上 lf 条数明显少于 v2（旧 release 子集）时必须按 v2 重生成。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            root = Path(tmp)
            self._seed(root)
            result = build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="Qwen3.8-27B")
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)

            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn("FORMAT_CONTRACT.json", names)
            self.assertNotIn("train_sft_qwen3_8b.yaml", names)

            sft = _load_jsonl(unzipped / "data" / "sft.train.lf.jsonl")[0]
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

            dpo = _load_jsonl(unzipped / "data" / "dpo.train.lf.jsonl")[0]
            self.assertIn("good-cmd", dpo["chosen"]["value"])
            self.assertIn("<function=bash>", dpo["chosen"]["value"])
            self.assertIn("bad-cmd", dpo["rejected"]["value"])
            self.assertNotIn('"arguments"', dpo["rejected"]["value"])

            sft_yaml = (unzipped / "train_sft.yaml").read_text(encoding="utf-8")
            self.assertRegex(sft_yaml, r"(?m)^template: qwen3_8$")
            start = (unzipped / "START_HERE.md").read_text(encoding="utf-8")
            self.assertIn("--tool-call-parser qwen3_coder", start)
            self.assertIn("没有打入", start)
            contract = json.loads((unzipped / "FORMAT_CONTRACT.json").read_text(encoding="utf-8"))
            self.assertEqual("Qwen3.8-27B", contract["profile"])
            self.assertEqual("coder-xml", contract["tool_style"])
            self.assertEqual("qwen3_8", contract["train_template"])
            self.assertEqual("qwen3_coder", contract["serve_tool_parser"])
            self.assertGreater(contract["blocks_converted"], 0)
            man = json.loads((unzipped / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(contract["profile"], man["format_contract"]["profile"])
            self.assertIn("工具调用格式已锁定", (unzipped / "handoff.md").read_text(encoding="utf-8")[:2000])

    def test_hermes_keeps_json_and_includes_8b_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            result = build_delivery_zip(root, self.cfg, log=lambda *_: None, profile="hermes-json")
            unzipped = root / "out"
            _unzip(Path(result["zip_path"]), unzipped)
            names = set(zipfile.ZipFile(result["zip_path"]).namelist())
            self.assertIn("train_sft_qwen3_8b.yaml", names)

            sft = _load_jsonl(unzipped / "data" / "sft.train.lf.jsonl")[0]
            gpt = sft["conversations"][2]["value"]
            self.assertIn('"arguments"', gpt)
            self.assertNotIn("<function=bash>", gpt)
            self.assertIn("<tool_call>garbage</tool_call>", sft["conversations"][3]["value"])

            sft_yaml = (unzipped / "train_sft.yaml").read_text(encoding="utf-8")
            self.assertRegex(sft_yaml, r"(?m)^template: qwen3$")
            eight = (unzipped / "train_sft_qwen3_8b.yaml").read_text(encoding="utf-8")
            self.assertRegex(eight, r"(?m)^template: qwen3$")
            self.assertIn("dataset: gt_sft_train_lf", eight)
            start = (unzipped / "START_HERE.md").read_text(encoding="utf-8")
            self.assertIn("--tool-call-parser hermes", start)
            self.assertNotIn("qwen3_coder", start)

    def test_unparsed_assistant_tool_call_aborts_before_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            root = Path(tmp)
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


if __name__ == "__main__":
    unittest.main()
