"""并行工具调用加权的单元测试（stage3 `apply_parallel_boost`）。

背景：训练集里并行步（一步 >=2 个 tool_call）只占 5.3%，而被整包留出的 eval 占 26.2%，
不加权模型会退化成「一次一个调用」。这条规则很容易写错（银行家舍入、把 highlight 也乘了、
乘完反而变小），所以单独测。
"""
from __future__ import annotations

import unittest

from pipeline.stage3_synth import apply_parallel_boost


class TestParallelBoost(unittest.TestCase):
    def test_routine_step_is_boosted(self):
        # routine=1, ×1.5 -> 2
        self.assertEqual(apply_parallel_boost(1, 2, "routine", 1.5), (2, True))

    def test_key_decision_step(self):
        self.assertEqual(apply_parallel_boost(2, 2, "key_decision", 1.5), (3, True))

    def test_closing_step_not_rounded_down(self):
        # 3 × 1.5 = 4.5：round() 会因银行家舍入给 4，这里必须给 5
        self.assertEqual(apply_parallel_boost(3, 2, "closing", 1.5), (5, True))

    def test_single_call_untouched(self):
        for stage in ("routine", "key_decision", "closing", "opening"):
            self.assertEqual(apply_parallel_boost(2, 1, stage, 1.5), (2, False))

    def test_zero_calls_untouched(self):
        # 纯文本回答的步不该被加成
        self.assertEqual(apply_parallel_boost(3, 0, "closing", 1.5), (3, False))

    def test_highlight_never_boosted(self):
        # highlight 的权重是模型给的分，再乘就是双重计分
        self.assertEqual(apply_parallel_boost(2, 3, "highlight", 2.0), (2, False))

    def test_factor_one_disables(self):
        self.assertEqual(apply_parallel_boost(2, 3, "routine", 1.0), (2, False))

    def test_factor_below_one_never_downgrades(self):
        # 即使配了 0.8，也只会保持原值（只乘不降）
        self.assertEqual(apply_parallel_boost(3, 2, "closing", 0.8), (3, False))

    def test_three_parallel_calls(self):
        self.assertEqual(apply_parallel_boost(2, 3, "key_decision", 2.0), (4, True))


if __name__ == "__main__":
    unittest.main()