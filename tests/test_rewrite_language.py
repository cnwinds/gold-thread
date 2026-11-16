"""改写语种不变量：thinking 改写不得把主导语种翻成另一种。"""
from __future__ import annotations

import unittest

from pipeline.common.quality import (
    dominant_thinking_lang,
    rewrite_preserves_language,
)
from pipeline.stage2_gold import _rewrite_ok


class DominantThinkingLangTests(unittest.TestCase):
    def test_english_thinking(self):
        text = (
            "The sparse packed container stores offsets as uint32. "
            "I will print the first 64 bytes and confirm the header layout."
        )
        self.assertEqual("en", dominant_thinking_lang(text))

    def test_chinese_thinking(self):
        text = (
            "稀疏打包容器用 uint32 存偏移。"
            "我先打印前 64 字节，再确认文件头结构。"
        )
        self.assertEqual("zh", dominant_thinking_lang(text))

    def test_short_unknown(self):
        self.assertEqual("unknown", dominant_thinking_lang("ok 0x1e"))


class RewriteLanguageGateTests(unittest.TestCase):
    def test_same_language_ok(self):
        old = "I need to inspect the binary header before guessing field sizes."
        new = "Inspect the binary header first, then decide field sizes from evidence."
        self.assertTrue(rewrite_preserves_language(old, new))

    def test_english_to_chinese_rejected(self):
        old = (
            "The parser assumed little-endian floats. I should verify with a Python "
            "struct unpack against the raw bytes instead of mental math."
        )
        new = (
            "解析器此前假设了小端浮点。我应该用 Python 的 struct 对照原始字节验证，"
            "而不是心算结论。"
        )
        self.assertFalse(rewrite_preserves_language(old, new))

    def test_chinese_to_english_rejected(self):
        old = "需要先读源码确认字段含义，再写探针打印实际偏移。"
        new = "I should read the source first, then write a probe that prints offsets."
        self.assertFalse(rewrite_preserves_language(old, new))

    def test_mixed_original_allows_rewrite(self):
        old = (
            "需要先确认存档头字段含义与偏移布局。"
            "The header stores uint32 offsets. "
            "然后再用脚本打印实测值核对。"
        )
        self.assertEqual("mixed", dominant_thinking_lang(old))
        new = (
            "需要先确认存档头字段含义与偏移布局。"
            "Header 使用 uint32 offsets。"
            "然后再用脚本打印实测值核对。"
        )
        self.assertTrue(rewrite_preserves_language(old, new))

    def test_rewrite_ok_rejects_language_flip(self):
        msg = {
            "thinking": (
                "Hold on, the previous offset guess was wrong. I will dump the "
                "bytes with Python and rebuild the parse table from evidence."
            ),
            "raw_thinking": (
                "Hold on, the previous offset guess was wrong. I will dump the "
                "bytes with Python and rebuild the parse table from evidence."
            ),
        }
        flipped = (
            "刚才的偏移猜测不对。我将用 Python 打印字节，并根据实测结果重建解析表。"
        )
        self.assertFalse(_rewrite_ok(flipped, msg))

    def test_rewrite_ok_keeps_same_language(self):
        msg = {
            "thinking": (
                "Hold on, the previous offset guess was wrong. I will dump the "
                "bytes with Python and rebuild the parse table from evidence."
            ),
        }
        rewritten = (
            "The previous offset guess was wrong. Dump the bytes with Python, "
            "then rebuild the parse table from evidence."
        )
        self.assertTrue(_rewrite_ok(rewritten, msg))


if __name__ == "__main__":
    unittest.main()
