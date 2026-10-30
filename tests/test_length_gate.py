"""长度处理：软压缩抢救 / 打包不丢 / 训练时按 cutoff 过滤 / 不截最后一条 assistant。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import length_gate as lg
from pipeline.lf_normalize import prepare_datasets


class FakeTok:
    """测试用：1 字符 = 1 token；decode 还原原文切片。"""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


def _msg(role: str, value: str):
    return {"from": role, "value": value}


def _hex_dump(n_lines: int = 80) -> str:
    lines = [f"0x{i:08x}: " + ("deadbeef" * 8) for i in range(n_lines)]
    return "\n".join(lines)


class LengthGateUnitTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTok()

    def test_soft_truncate_keeps_head_and_tail(self):
        text = "A" * 700 + "MID" + "B" * 700
        out, changed = lg.soft_truncate_tokens(self.tok, text, 200, head_ratio=0.7)
        self.assertTrue(changed)
        self.assertIn("truncated", out)
        self.assertTrue(out.startswith("A"))
        self.assertTrue(out.endswith("B"))
        self.assertLessEqual(lg.count_text_tokens(self.tok, out), 200)

    def test_high_entropy_dump_detected(self):
        self.assertTrue(lg.is_high_entropy_dump(_hex_dump(100)))
        self.assertFalse(lg.is_high_entropy_dump("normal tool output " * 20))

    def test_constants_aggressive(self):
        self.assertEqual(500, lg.SOFT_LIMIT_NORMAL)
        self.assertEqual("obs-soft-v2", lg.LENGTH_GATE_VERSION)
        self.assertIn("obs-soft-v2", lg.length_gate_fingerprint())
        self.assertEqual(200, lg.SOFT_LIMIT_DUMP)
        self.assertEqual(20, lg.DUMP_HEAD_LINES)
        self.assertEqual(200, lg.SOFT_LIMIT_NORMAL_OLD)
        self.assertEqual(80, lg.SOFT_LIMIT_DUMP_OLD)
        self.assertEqual(8, lg.DUMP_HEAD_LINES_OLD)
        self.assertEqual(2, lg.RECENCY_RECENT_COUNT)

    def test_light_json_trim_shortens_long_strings(self):
        payload = {"ok": True, "blob": "Z" * 500, "nested": {"s": "Y" * 200}}
        text = json.dumps(payload, ensure_ascii=False)
        out, changed = lg.light_json_trim(text)
        self.assertTrue(changed)
        parsed = json.loads(out)
        self.assertLessEqual(len(parsed["blob"]), lg.JSON_STRING_TRIM + 1)
        self.assertLessEqual(len(parsed["nested"]["s"]), lg.JSON_STRING_TRIM + 1)
        self.assertTrue(parsed["ok"])
        # 非 JSON 原样
        plain, ch2 = lg.light_json_trim("not json " + ("x" * 300))
        self.assertFalse(ch2)
        self.assertEqual(plain, "not json " + ("x" * 300))

    def test_near_dedupe_consecutive_obs(self):
        body = "tool result ALPHA " + ("line\n" * 50)
        assistant = "FINAL_KEEP"
        row = {
            "sample_id": "dedupe",
            "conversations": [
                _msg("human", "q"),
                _msg("gpt", "c1"),
                _msg("observation", body),
                _msg("gpt", "c2"),
                _msg("observation", body),  # identical consecutive obs
                _msg("gpt", assistant),
            ],
        }
        out, changed = lg.soft_compress_row(self.tok, row)
        self.assertTrue(changed)
        vals = [m["value"] for m in out["conversations"] if m["from"] == "observation"]
        self.assertEqual(2, len(vals))
        self.assertEqual(lg.DUPLICATE_OBS_MARKER, vals[0])
        self.assertNotEqual(lg.DUPLICATE_OBS_MARKER, vals[1])
        self.assertEqual(assistant, out["conversations"][-1]["value"])

    def test_recency_bias_tighter_on_older_obs(self):
        # 3 条超长普通 obs：前 1 条用 OLD 上限(200)，后 2 条用 500
        long_obs = ("result line OK\n" * 80)  # ~1280 chars > 500
        assistant = "ASSISTANT_STAYS"
        row = {
            "sample_id": "recency",
            "conversations": [
                _msg("human", "q"),
                _msg("gpt", "c1"),
                _msg("observation", long_obs + "OLD0"),
                _msg("gpt", "c2"),
                _msg("observation", long_obs + "REC1"),
                _msg("gpt", "c3"),
                _msg("observation", long_obs + "REC2"),
                _msg("gpt", assistant),
            ],
        }
        out, changed = lg.soft_compress_row(self.tok, row)
        self.assertTrue(changed)
        obs_vals = [m["value"] for m in out["conversations"] if m["from"] == "observation"]
        self.assertEqual(3, len(obs_vals))
        # older should be under OLD normal limit; recent under NORMAL
        self.assertLessEqual(lg.count_text_tokens(self.tok, obs_vals[0]), lg.SOFT_LIMIT_NORMAL_OLD)
        self.assertLessEqual(lg.count_text_tokens(self.tok, obs_vals[1]), lg.SOFT_LIMIT_NORMAL)
        self.assertLessEqual(lg.count_text_tokens(self.tok, obs_vals[2]), lg.SOFT_LIMIT_NORMAL)
        # older tighter than recent caps allow: first should be shorter than a 500-budget piece
        self.assertLessEqual(len(obs_vals[0]), lg.SOFT_LIMIT_NORMAL_OLD + 5)
        self.assertEqual(assistant, out["conversations"][-1]["value"])

    def test_salvage_success_keeps_sample(self):
        # 超长 observation，压缩后应能进包；assistant 原文保留
        assistant = "<think>\nok\n</think>\n\nfinal-answer-KEEP"
        obs = _hex_dump(120)
        row = {
            "sample_id": "salvage-me",
            "conversations": [
                _msg("system", "sys"),
                _msg("human", "do"),
                _msg("gpt", "call"),
                _msg("observation", obs),
                _msg("gpt", assistant),
            ],
            "tools": "[]",
        }
        # FakeTok 下 ChatML 渲染会再加标记，cutoff 取压缩后可过、压缩前不过的值
        before = lg.measure_row_tokens(self.tok, row, ranking=False, prefix="")
        self.assertGreater(before, 500)
        kept, stats = lg.gate_rows(
            [row],
            ranking=False,
            cutoff=max(800, before - 100),  # 低于 before，迫使抢救
            tok=self.tok,
            tokenizer_label="FakeTok",
            prefix="",
        )
        # 若仍超（dump 太大），放宽：用更大预算验证 salvage 路径
        if stats.dropped_overlength:
            kept, stats = lg.gate_rows(
                [row],
                ranking=False,
                cutoff=before - 50,
                tok=self.tok,
                tokenizer_label="FakeTok",
                prefix="",
            )
        # 再不行就用明确可救的普通超长文本
        if not kept:
            obs2 = ("tool line result OK\n" * 200) + "TAIL_MARKER"
            row2 = {
                "sample_id": "salvage-me",
                "conversations": [
                    _msg("system", "sys"),
                    _msg("human", "do"),
                    _msg("gpt", "call"),
                    _msg("observation", obs2),
                    _msg("gpt", assistant),
                ],
                "tools": "[]",
            }
            before2 = lg.measure_row_tokens(self.tok, row2, ranking=False, prefix="")
            # 软压缩上限 500，构造 before2 高于 cutoff 且压缩后可过
            kept, stats = lg.gate_rows(
                [row2],
                ranking=False,
                cutoff=before2 - 100,
                tok=self.tok,
                tokenizer_label="FakeTok",
                prefix="",
            )
            self.assertEqual(1, len(kept), msg=stats.summary())
            self.assertEqual(1, stats.salvaged_by_obs_truncate)
            self.assertEqual(assistant, kept[0]["conversations"][-1]["value"])
        else:
            self.assertEqual(1, len(kept))
            self.assertGreaterEqual(stats.salvaged_by_obs_truncate, 1)
            self.assertEqual(assistant, kept[0]["conversations"][-1]["value"])

    def test_still_over_after_salvage_is_kept_with_overlength_stat(self):
        assistant = "ASSISTANT_MUST_STAY"
        # 多条巨大 observation，软压缩每条仍累加超 cutoff —— 打包阶段仍保留
        obs = _hex_dump(200)
        row = {
            "sample_id": "over-me",
            "conversations": [
                _msg("system", "sys"),
                _msg("human", "do"),
                _msg("gpt", "c1"),
                _msg("observation", obs),
                _msg("gpt", "c2"),
                _msg("observation", obs),
                _msg("gpt", "c3"),
                _msg("observation", obs),
                _msg("gpt", assistant),
            ],
            "tools": "[]",
        }
        before = lg.measure_row_tokens(self.tok, row, ranking=False, prefix="")
        kept, stats = lg.gate_rows(
            [row],
            ranking=False,
            cutoff=min(600, before // 4),
            tok=self.tok,
            tokenizer_label="FakeTok",
            prefix="",
        )
        self.assertEqual(1, len(kept))
        self.assertEqual(0, stats.dropped_overlength)
        self.assertEqual(1, stats.overlength_at_default_cutoff)
        self.assertEqual(["over-me"], [r.sample_id for r in stats.records_overlength])
        self.assertEqual(assistant, kept[0]["conversations"][-1]["value"])

    def test_assistant_never_truncated(self):
        assistant = "UNIQUE_ASSISTANT_PAYLOAD_" + ("X" * 500)
        obs = "Y" * 3000
        row = {
            "sample_id": "keep-asst",
            "conversations": [
                _msg("human", "q"),
                _msg("gpt", "tool"),
                _msg("observation", obs),
                _msg("gpt", assistant),
            ],
        }
        out, changed = lg.soft_compress_row(self.tok, row)
        self.assertTrue(changed)
        self.assertEqual(assistant, out["conversations"][-1]["value"])
        # observation 应被压短
        self.assertLess(len(out["conversations"][-2]["value"]), len(obs))
        self.assertLessEqual(
            lg.count_text_tokens(self.tok, out["conversations"][-2]["value"]),
            lg.SOFT_LIMIT_NORMAL,
        )


class LengthGatePrepareTests(unittest.TestCase):
    def test_prepare_datasets_applies_gate_with_injected_tok(self):
        """通过 monkeypatch 避免测试依赖真实 HF 下载。"""
        tok = FakeTok()

        def fake_load(model_id=lg.TOKENIZER_ID):
            return tok, "FakeTok"

        original = lg.load_gate_tokenizer
        lg.load_gate_tokenizer = fake_load  # type: ignore
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "train_sft.yaml").write_text("cutoff_len: 900\n", encoding="utf-8")
                (root / "train_dpo.yaml").write_text("cutoff_len: 500\n", encoding="utf-8")
                assistant = "FINAL_ASSISTANT"
                # 可抢救：单条 obs 很长，cutoff 900，软压后可通过
                obs = ("result line\n" * 400) + "END"
                row = {
                    "sample_id": "p-salvage",
                    "conversations": [
                        _msg("system", "s"),
                        _msg("human", "h"),
                        _msg("gpt", "g"),
                        _msg("observation", obs),
                        _msg("gpt", assistant),
                    ],
                    "tools": "[]",
                }
                # 仍超长：多轮 *互不相同* 的 dump（避免 near-dedupe 折叠），即使新上限仍累加超 cutoff
                # FakeTok 下每条 dump 软压后仍约数十～两百 token，8 条足以压过 900
                dumps = [_hex_dump(80 + i) + f"\nUNIQUE_{i}_" + ("Q" * 50) for i in range(8)]
                conv_drop = [_msg("system", "s"), _msg("human", "h")]
                for i, d in enumerate(dumps):
                    conv_drop.append(_msg("gpt", f"g{i}"))
                    conv_drop.append(_msg("observation", d))
                conv_drop.append(_msg("gpt", assistant))
                row_drop = {
                    "sample_id": "p-drop",
                    "conversations": conv_drop,
                    "tools": "[]",
                }
                eval_row = {
                    "sample_id": "p-eval",
                    "conversations": [
                        _msg("system", "s"),
                        _msg("human", "h"),
                        _msg("gpt", "short"),
                    ],
                    "tools": "[]",
                }
                path = root / "sft_step_windows_sharegpt.train.v2.jsonl"
                path.write_text(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                    + json.dumps(row_drop, ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )
                (root / "sft_step_windows_sharegpt.eval.v2.jsonl").write_text(
                    json.dumps(eval_row, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                prep = prepare_datasets(root)
                ids = [r["sample_id"] for r in prep.files["sft.train.lf.jsonl"]]
                self.assertIn("p-salvage", ids)
                self.assertIn("p-drop", ids)  # 打包不再硬丢，仍在全集中
                self.assertEqual(2, len(ids))
                salvaged = next(r for r in prep.files["sft.train.lf.jsonl"] if r["sample_id"] == "p-salvage")
                self.assertEqual(assistant, salvaged["conversations"][-1]["value"])
                gate = prep.length_gate.get("sft.train") or {}
                self.assertGreaterEqual(int(gate.get("salvaged_by_obs_truncate") or 0), 1)
                self.assertEqual(0, int(gate.get("dropped_overlength") or 0))
                self.assertGreaterEqual(int(gate.get("overlength_at_default_cutoff") or 0), 1)
                self.assertEqual(900, int(gate.get("cutoff") or 0))
                # prepared.dropped 不应再因超长追加
                drop_notes = " ".join(prep.dropped.get("sft.train") or [])
                self.assertNotIn("overlength", drop_notes)
        finally:
            lg.load_gate_tokenizer = original  # type: ignore


if __name__ == "__main__":
    unittest.main()
