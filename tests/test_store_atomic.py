"""原子写入在 Windows 占用场景下的重试 / 回退。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.common.store import replace_atomic, write_json_atomic, write_jsonl_atomic


class ReplaceAtomicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_replace_retries_then_succeeds(self) -> None:
        dest = self.root / "out.jsonl"
        dest.write_text("old\n", encoding="utf-8")
        src = self.root / "new.tmp"
        src.write_text("new\n", encoding="utf-8")
        calls = {"n": 0}
        real = os.replace

        def flaky(a, b):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] < 3:
                err = PermissionError(13, "Access is denied")
                err.winerror = 5  # type: ignore[attr-defined]
                raise err
            real(a, b)

        with patch("pipeline.common.store.time.sleep"), patch(
            "pipeline.common.store.os.replace", side_effect=flaky
        ):
            replace_atomic(src, dest, retries=5)
        self.assertEqual("new\n", dest.read_text(encoding="utf-8"))
        self.assertEqual(3, calls["n"])

    def test_replace_falls_back_to_unlink_rename(self) -> None:
        dest = self.root / "out.jsonl"
        dest.write_text("old\n", encoding="utf-8")
        src = self.root / "new.tmp"
        src.write_text("new\n", encoding="utf-8")

        def always_busy(a, b):  # noqa: ANN001
            err = OSError(13, "Access is denied")
            err.winerror = 5  # type: ignore[attr-defined]
            raise err

        with patch("pipeline.common.store.time.sleep"), patch(
            "pipeline.common.store.os.replace", side_effect=always_busy
        ):
            replace_atomic(src, dest, retries=2)
        self.assertEqual("new\n", dest.read_text(encoding="utf-8"))
        self.assertFalse(src.exists())

    def test_write_jsonl_atomic_overwrites(self) -> None:
        path = self.root / "rows.jsonl"
        write_jsonl_atomic(path, [{"a": 1}])
        write_jsonl_atomic(path, [{"a": 2}, {"a": 3}])
        text = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(2, len(text))
        self.assertIn('"a": 2', text[0])

    def test_write_json_atomic_overwrites(self) -> None:
        path = self.root / "meta.json"
        write_json_atomic(path, {"v": 1})
        write_json_atomic(path, {"v": 2})
        self.assertIn('"v": 2', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
