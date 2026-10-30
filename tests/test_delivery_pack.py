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
            )
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

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
