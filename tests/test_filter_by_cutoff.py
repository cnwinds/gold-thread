"""长度过滤的 tokenizer 不能写死训练机路径。"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "tools" / "filter_by_cutoff.py"
    spec = importlib.util.spec_from_file_location("filter_by_cutoff_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TokenizerResolveTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self._prev = os.environ.get("GT_LENGTH_TOKENIZER")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GT_LENGTH_TOKENIZER", None)
        else:
            os.environ["GT_LENGTH_TOKENIZER"] = self._prev

    def test_default_is_huggingface_id(self):
        os.environ.pop("GT_LENGTH_TOKENIZER", None)
        self.assertEqual("Qwen/Qwen2.5-0.5B", self.mod.resolve_tokenizer_id())
        self.assertEqual("Qwen/Qwen2.5-0.5B", self.mod.resolve_tokenizer_id("  "))

    def test_env_and_cli_override(self):
        os.environ["GT_LENGTH_TOKENIZER"] = "/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct"
        self.assertEqual(
            "/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct",
            self.mod.resolve_tokenizer_id(),
        )
        self.assertEqual("/other/tokenizer", self.mod.resolve_tokenizer_id("/other/tokenizer"))


if __name__ == "__main__":
    unittest.main()
