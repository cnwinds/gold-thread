"""README 不与已删除的控制台「人工复核页」叙述冲突。"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


class ReadmeAlignmentTests(unittest.TestCase):
    def test_readme_does_not_describe_human_review_page_as_current_ui(self):
        text = README.read_text(encoding="utf-8")
        self.assertNotIn("人工复核（黄金数据页）", text)
        self.assertNotIn("采纳/不采纳/手改/排除 → 定稿", text)
        self.assertIn("CONTEXT.md", text)
        self.assertIn("Legacy", text)

    def test_readme_console_mentions_collect_not_review_sidebar(self):
        text = README.read_text(encoding="utf-8")
        section = text.split("## 控制台", 1)[-1].split("## ", 1)[0]
        self.assertIn("采集", section)
        self.assertNotIn("审核页", section)
        self.assertIn("无独立「审核」侧栏", section)


if __name__ == "__main__":
    unittest.main()
