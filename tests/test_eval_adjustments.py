"""针对 goldthread-d9dbbbd1 训练效果评估报告发现的数据与管线根因修复回归测试。"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import curation, stage1_extract, stage2_gold, stage3_synth
from pipeline.common import pilog
from pipeline.common.config import DEFAULTS, load_config
from pipeline.common.store import write_json_atomic, write_jsonl_atomic


def _asst(mid: str, thinking: str, text: str = "", tool_calls=None, tool_results=None, **extra):
    msg = {
        "id": mid,
        "role": "assistant",
        "model": "glm-5.3",
        "provider": "test",
        "thinking": thinking,
        "text": text,
        "tool_calls": tool_calls if tool_calls is not None else [],
        "tool_results": tool_results if tool_results is not None else [],
        "timestamp": "2026-11-18T00:00:00Z",
    }
    msg.update(extra)
    return msg


class EvalAdjustmentsTests(unittest.TestCase):
    def test_unwrap_user_text_strips_file_and_user_query_tags_in_parse_and_flow(self) -> None:
        wrapped = (
            '<file name="D:\\crack_test\\teacher\\prompt.txt">\n'
            "请完整逆向 com.tinydobbins.monkeymart_1.6.20.xapk，恢复资源与脚本。\n"
            "</file>\n"
            "<timestamp>2026-11-01</timestamp>"
        )
        clean = pilog.unwrap_user_text(wrapped)
        self.assertNotIn("<file", clean)
        self.assertNotIn("<timestamp>", clean)
        self.assertIn("com.tinydobbins.monkeymart_1.6.20.xapk", clean)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sess.jsonl"
            events = [
                {"type": "session", "id": "s_unwrap", "cwd": r"D:\crack_test\teacher\com.tinydobbins.monkeymart_1.6.20"},
                {
                    "type": "message",
                    "id": "u0",
                    "message": {"role": "user", "content": [{"type": "text", "text": wrapped}]},
                },
                {
                    "type": "message",
                    "id": "a0",
                    "message": {
                        "role": "assistant",
                        "model": "glm-5",
                        "content": [
                            {"type": "thinking", "thinking": "先解压 XAPK。"},
                            {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls -la"}},
                        ],
                    },
                },
            ]
            p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
            parsed = pilog.parse_session(p)
            self.assertNotIn("<file", parsed["messages"][0]["content"])
            self.assertIn("com.tinydobbins.monkeymart_1.6.20.xapk", parsed["messages"][0]["content"])

    def test_infer_project_group_canonicalizes_teacher_and_student_paths(self) -> None:
        cases = [
            (r"D:\crack\jigsawcard", "x/a.jsonl", "game::jigsawcard"),
            (r"D:\crack\jigsawcard\ds", "x/a.jsonl", "game::jigsawcard"),
            (r"D:\crack\jigsawcard-qwen3.8", "x/a.jsonl", "game::jigsawcard"),
            (r"D:\crack\meowdoku-qwen3.8-27b", "x/a.jsonl", "game::meowdoku"),
            (r"D:\crack_test\teacher\com.oakever.meowdoku_1.21.0\rerun_glm", "x/a.jsonl", "game::meowdoku"),
            (r"D:\crack_test\teacher\_retry\com.biglime.cookingmadness_3.1.8\rerun_glm", "x/a.jsonl", "game::cookingmadness"),
            (r"D:\crack_test\teacher\com.fugo.words_5.6.0\from_crack_puzzle", "x/a.jsonl", "game::puzzle"),
            (r"D:\crack_test\teacher\com.easybrain.block.puzzle.games_3.32.0", "x/a.jsonl", "game::puzzle"),
            (r"D:\crack_test\teacher\net.radle.unblock3d.remastered_3.0.5104", "x/a.jsonl", "game::unblock3d"),
            (r"D:\crack_test\teacher\air.com.tedven.youmustescape2_2.0.6-rc8", "x/a.jsonl", "game::youmustescape2"),
            (r"D:\crack_test\teacher\com.dopuz.klotski.riddle_1.62\from_crack_minimax", "x/a.jsonl", "game::riddle"),
            (r"D:\crack_test\teacher\draziw.button.mines_1.49.1\rerun_glm", "x/a.jsonl", "game::minesweeper"),
            (r"D:\crack_test\teacher\air.com.FDGEntertainment.BananaKong.gp_1.9.16.16\rerun_glm", "x/a.jsonl", "game::bananakong"),
            (r"D:\crack_test\student\goldthread-d9dbbbd1\riddle-27b-dpo-q6", "x/a.jsonl", "game::riddle"),
            (r"D:\other_workspace\my_service", "x/a.jsonl", "workspace::my_service"),
        ]
        for cwd, rel, expected in cases:
            with self.subTest(cwd=cwd):
                self.assertEqual(expected, stage1_extract.infer_project_group(cwd, rel))

    def test_curation_resolve_unblocks_stale_review_rounds_when_model_report_present(self) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        ann = {
            "review_key": "1:new:flow:6",
            "review_rounds_stale": True,
            "review_rounds": [
                {
                    "round": 1,
                    "verdict": "needs_review",
                    "confidence": 0.65,
                    "thinking_rewrites": [{"msg_id": "a1", "rewritten": "过期重写不应生效"}],
                    "bad_step_ids": ["a1"],
                }
            ],
            "model_report": {"verdict": "trainable", "confidence": 0.85},
            "human_review": {"items": {}, "decision": None, "history": []},
        }
        eff = curation.resolve(ann, cfg)
        self.assertEqual("auto_gold", eff["state"])
        self.assertEqual({}, eff["rewrites"], "过期 review_rounds 的改写不应混入")
        self.assertEqual([], eff["excluded"], "过期 review_rounds 的 bad_step_ids 不应混入")

    def test_continuation_folds_over_model_task_and_emits_premature_stop_dpo(self) -> None:
        cfg = load_config()
        session = {
            "session_id": "s_mahjong",
            "cwd": r"D:\crack\mahjong",
            "timestamp": "2026-11-18T00:00:00Z",
            "messages": [
                {"id": "u0", "role": "user", "content": "帮我破解 mahjong.xapk，获取全部资源与代码", "timestamp": ""},
                _asst("a1", "先用 apktool 解包。", tool_calls=[{"id": "c1", "name": "bash", "arguments": {"command": "apktool d mahjong.apk"}}]),
                {"id": "tr1", "role": "tool", "tool_call_id": "c1", "name": "bash", "content": "I: Using Apktool"},
                _asst("a_premature", "已完成初步解包，准备汇报。", text="# 🎉 破解完成！已提取基础图片。"),
                {
                    "id": "u1",
                    "role": "user",
                    "content": "函数名即刻恢复。下载在线资源，全部解密。配合mumu模拟器，直接dump metadata实现全破解。",
                    "timestamp": "",
                },
                _asst("a2", "连接 MuMu 模拟器 dump global-metadata.dat。", tool_calls=[{"id": "c2", "name": "bash", "arguments": {"command": "adb devices"}}]),
                {"id": "tr2", "role": "tool", "tool_call_id": "c2", "name": "bash", "content": "127.0.0.1:16384 device"},
                _asst("a3", "全部函数名与加密资产已验证完毕。", text="最终完整报告。"),
            ],
        }
        # 模拟 turn_annotations 曾把 u1 误标为 task
        annotations = {
            "root_turn": 0,
            "turns": [
                {"index": 0, "kind": "root", "drop": False},
                {"index": 1, "kind": "task", "drop": False},
            ],
        }
        flow, marks, _ = pilog.build_flow(session, cfg, annotations=annotations)
        self.assertEqual("folded", marks[1]["action"])
        self.assertTrue(marks[1]["popped_await_msg"])
        asst_ids = [m["id"] for m in flow if m["role"] == "assistant"]
        self.assertEqual(["a1", "a2", "a3"], asst_ids, "中途提前完成播报步应被弹出")

        gold_msgs, pruned, _ = stage2_gold.purify(flow, cfg["gold"], session["cwd"])
        premature_dpo = [p for p in pruned if p.get("reason") == "premature_stop_folded"]
        self.assertEqual(1, len(premature_dpo))
        self.assertEqual("a2", premature_dpo[0]["chosen_msg"]["id"])
        self.assertEqual("a_premature", premature_dpo[0]["rejected_msg"]["id"])

    def test_evaluate_session_synthesis_drops_intermediate_no_tool_stop_steps(self) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["synth"]["formats"] = {"sharegpt": True, "openai": True}
        record = {
            "session_id": "s_stop_gate",
            "label": "positive",
            "project_group": "game::demo",
            "cwd": r"D:\crack\demo",
            "dominant_model": "glm-5",
            "stats": {"raw_steps": 3, "gold_steps": 3},
            "messages": [
                {"id": "sys", "role": "system", "content": "sys"},
                {"id": "u0", "role": "user", "content": "破解 demo.apk"},
                _asst(
                    "a1_tool",
                    "先用脚本检查 APK 目录结构与加密头部特征。",
                    tool_calls=[{"id": "c1", "name": "bash", "arguments": {"command": "ls -la"}}],
                ),
                {"id": "tr1", "role": "tool", "tool_call_id": "c1", "name": "bash", "content": "assets/"},
                _asst(
                    "a2_mid_stop",
                    "阶段性总结一下当前进度，问问用户选方案 1 还是 2。",
                    text="已完成初步解压，请选择下一步：1. 解密配置 2. 逆向 so",
                    tool_calls=[],
                ),
                {"id": "u1", "role": "user", "content": "特殊追加任务"},
                _asst(
                    "a3_final_stop",
                    "所有配置与脚本均已用 Python 核验通过，输出最终收尾报告。",
                    text="全部逆向完成。",
                    tool_calls=[],
                ),
            ],
        }
        ann = {"review_key": "1:x:6", "model_report": {"verdict": "trainable", "confidence": 0.9}}
        res = stage3_synth.evaluate_session_synthesis(record, ann, cfg)
        audits = {a["msg_id"]: a for a in res["step_audits"]}
        self.assertEqual("admitted", audits["a1_tool"]["status"])
        self.assertEqual("dropped", audits["a2_mid_stop"]["status"])
        self.assertEqual("premature_stop_without_tool", audits["a2_mid_stop"]["reason"])
        self.assertEqual("admitted", audits["a3_final_stop"]["status"])
        sample_ids = [w["sample_id"] for w in res["windows_sgpt"]]
        self.assertIn("step::s_stop_gate::a1_tool", sample_ids)
        self.assertNotIn("step::s_stop_gate::a2_mid_stop", sample_ids)
        self.assertIn("step::s_stop_gate::a3_final_stop", sample_ids)

    def test_synthesize_dpo_candidates_includes_stage2_judge_thinking_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            for sub in ("00_state", "01_raw", "02_gold", "03_train"):
                (data / sub).mkdir(parents=True)
            cfg = copy.deepcopy(DEFAULTS)
            cfg["paths"]["data_dir"] = str(data)

            raw_bad_think = ("Wait, is that right? Actually let me reconsider. " * 25)
            clean_think = "通过 Python 脚本直接读取前 16 字节验证魔数与异或密钥。"
            msg = _asst(
                "a1",
                clean_think,
                tool_calls=[{"id": "c1", "name": "bash", "arguments": {"command": "python -c 'print(1)'"}}],
                raw_thinking=raw_bad_think,
                flags=["thinking_compressed"],
                judge_reason="压缩反刍循环",
                raw_quality={"is_clean": False, "score": 55.0, "pathologies": ["over_rumination_doubt_loop"]},
                quality={"is_clean": True, "score": 95.0, "pathologies": []},
            )
            gf = data / "02_gold" / "s_rw.gold.jsonl"
            af = data / "02_gold" / "s_rw.ann.json"
            write_jsonl_atomic(gf, [{
                "session_id": "s_rw",
                "label": "positive",
                "project_group": "game::demo",
                "cwd": r"D:\crack\demo",
                "dominant_model": "deepseek-v3.2",
                "messages": [
                    {"id": "sys", "role": "system", "content": "sys"},
                    {"id": "u0", "role": "user", "content": "分析 demo.apk"},
                    msg,
                ],
            }])
            write_json_atomic(af, {"review_key": "1:x:6", "model_report": {"verdict": "trainable", "confidence": 0.9}})
            write_jsonl_atomic(data / "02_gold" / "dpo_pairs.jsonl", [])
            records = {
                "s_rw": {
                    "session_id": "s_rw",
                    "label": "positive",
                    "project_group": "game::demo",
                    "dominant_model": "deepseek-v3.2",
                    "gold_file": str(gf),
                    "annotation_file": str(af),
                }
            }
            pairs, counts = stage3_synth.synthesize_dpo_candidates(records, cfg, {"game::demo": "train"})
            self.assertEqual(1, counts["s_rw"]["review"])
            self.assertEqual(1, len(pairs))
            self.assertEqual("review_rewrite", pairs[0]["metadata"]["source"])
            self.assertIn(clean_think, pairs[0]["chosen"]["value"])
            self.assertIn("Wait, is that right?", pairs[0]["rejected"]["value"])


if __name__ == "__main__":
    unittest.main()
