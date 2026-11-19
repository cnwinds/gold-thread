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

    def test_core_docs_reflect_virtual_datasets_cas_and_agent_first_cli(self):
        readme = README.read_text(encoding="utf-8")
        context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
        skill = (ROOT / "skills" / "goldthread" / "SKILL.md").read_text(encoding="utf-8")

        # 已废弃的旧叙述不再残留在 README 中
        self.assertNotIn("改数据集角色（positive/negative/pending）", readme)
        self.assertNotIn("人工在 UI 里改", readme)
        self.assertNotIn("上机训练只认 `releases/<版本>/`", readme)

        # ADR-0008 / ADR-0009 核心能力在三大主文档均已同步
        for marker in ("inspect graph", "lineage_graph.py", "cas_store.py", "Variant A"):
            self.assertIn(marker, readme)
        for marker in ("虚拟数据集条目", "CAS 快照池", "parent_pack", "ADR-0009"):
            self.assertIn(marker, context)
        for marker in ("inspect graph", "--parent", "04_release/cas/"):
            self.assertIn(marker, skill)


if __name__ == "__main__":
    unittest.main()
