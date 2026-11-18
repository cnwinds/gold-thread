"""针对全量代码审核报告（P0 / P1 / P2 / P3）各项修复的回归测试套件。"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pipeline import curation, dataset_build, judge, length_gate, stage1_extract, stage2_gold
from pipeline.collect import _cursor_to_pi
from pipeline.common import pilog
from pipeline.common.config import DEFAULTS
from pipeline.common.format import (
    _parse_tc_block,
    convert_sharegpt_row,
    convert_toolcall_text,
    default_tools_json,
    to_openai,
    to_sharegpt,
)
from pipeline.common.quality import analyze_step_quality
from pipeline.common.store import read_json, write_json_atomic, write_jsonl_atomic
from pipeline.lf_normalize import prepare_datasets
from pipeline.package import build_delivery_zip
from pipeline.session_delete import delete_session

ROOT = Path(__file__).resolve().parents[1]


class P0SampleWeightTests(unittest.TestCase):
    """P0-1: metadata.sample_weight 在 sft.train.lf.jsonl 中确定性展开，且不污染 eval。"""

    def test_sample_weight_expands_sft_train_lf_and_keeps_eval_1x(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "03_train"
            root.mkdir(parents=True)
            train_rows = [
                {
                    "sample_id": "step::s1::open",
                    "conversations": [
                        {"from": "system", "value": "sys"},
                        {"from": "human", "value": "task"},
                        {"from": "gpt", "value": "<think>\n开局侦察\n</think>\ndone"},
                    ],
                    "metadata": {"session_id": "s1", "stage": "opening", "sample_weight": 3},
                },
                {
                    "sample_id": "step::s1::routine",
                    "conversations": [
                        {"from": "system", "value": "sys"},
                        {"from": "human", "value": "task"},
                        {"from": "gpt", "value": "<think>\n常规推进\n</think>\nok"},
                    ],
                    "metadata": {"session_id": "s1", "stage": "routine", "sample_weight": 1},
                },
            ]
            eval_rows = [
                {
                    "sample_id": "step::s2::eval_open",
                    "conversations": [
                        {"from": "system", "value": "sys"},
                        {"from": "human", "value": "eval task"},
                        {"from": "gpt", "value": "<think>\n验证集思考\n</think>\neval"},
                    ],
                    "metadata": {"session_id": "s2", "stage": "opening", "sample_weight": 3},
                }
            ]
            write_jsonl_atomic(root / "sft_step_windows_sharegpt.train.jsonl", train_rows)
            write_jsonl_atomic(root / "sft_step_windows_sharegpt.eval.jsonl", eval_rows)
            # 预先放一份未展开权重的旧 sft.train.lf.jsonl（2 条 < 期望 4 条），应自动触发重生成
            write_jsonl_atomic(root / "sft.train.lf.jsonl", train_rows)

            prep = prepare_datasets(root, apply_length_gate=False)
            lf_train = prep.files["sft.train.lf.jsonl"]
            lf_eval = prep.files["sft.eval.lf.jsonl"]

            self.assertEqual(4, len(lf_train))
            self.assertEqual(
                ["step::s1::open", "step::s1::open#w2", "step::s1::open#w3", "step::s1::routine"],
                [r["sample_id"] for r in lf_train],
            )
            self.assertEqual("step::s1::open", lf_train[1]["metadata"]["origin_sample_id"])
            self.assertEqual(1, len(lf_eval), "eval 切分严禁过采样复制")
            self.assertEqual(
                {"unique_rows": 2, "expanded_rows": 4, "boosted_unique_rows": 1, "extra_rows": 2},
                prep.weighted_expansion["sft.train"],
            )

            cfg = copy.deepcopy(DEFAULTS)
            cfg["paths"]["data_dir"] = str(Path(tmp))
            with patch("pipeline.length_gate.load_gate_tokenizer", return_value=(None, "skip:test")):
                pack = build_delivery_zip(root, cfg, log=lambda *_: None)
            with zipfile.ZipFile(pack["zip_path"]) as zf:
                man = json.loads(zf.read("goldthread-train/data/MANIFEST.json").decode("utf-8"))
            self.assertEqual(4, man["datasets"]["gt_sft_train_lf"]["rows"])
            self.assertEqual(1, man["datasets"]["gt_sft_eval_lf"]["rows"])
            self.assertIn("weighted_expansion", man)


class P0ConsoleDrawerRuntimeTests(unittest.TestCase):
    """P0-2: web/index.html 内联脚本语法与会话抽屉底栏运行时无 ReferenceError。"""

    @unittest.skipUnless(shutil.which("node"), "需要 node 执行前端脚本运行时验证")
    def test_session_drawer_footer_executes_without_reference_error(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        m = re.search(r"<script>([\s\S]*?)</script>", html)
        self.assertIsNotNone(m)
        script = m.group(1)

        with tempfile.TemporaryDirectory() as tmp:
            js_file = Path(tmp) / "console.js"
            harness = Path(tmp) / "harness.js"
            js_file.write_text(script, encoding="utf-8")

            syntax = subprocess.run(["node", "--check", str(js_file)], capture_output=True, text=True)
            self.assertEqual(0, syntax.returncode, syntax.stderr)

            harness.write_text(
                """
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync(process.argv[2], "utf8");
const stubEl = () => ({
  innerHTML: "", textContent: "", className: "", hidden: false, value: "",
  scrollTop: 0, clientHeight: 0, scrollHeight: 0,
  classList: { add(){}, remove(){}, toggle(){}, contains(){ return true; } },
  addEventListener(){}, appendChild(){}, querySelector(){ return stubEl(); },
  querySelectorAll(){ return []; }, setAttribute(){}
});
const sandbox = {
  console, setTimeout: () => 1, clearTimeout(){}, setInterval: () => 1, clearInterval(){},
  fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }),
  localStorage: { getItem(){ return null; }, setItem(){} },
  location: { hash: "#overview" },
  window: { addEventListener(){}, confirm(){ return false; } },
  document: {
    querySelector(){ return stubEl(); },
    querySelectorAll(){ return []; },
    createElement(){ return stubEl(); },
    addEventListener(){}
  }
};
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const idleRaw = vm.runInContext('RAW_DRAWER.sid="s1"; RAW_DRAWER.adopted=true; renderSessionActionsFoot("raw", "s1")', sandbox);
const idleGold = vm.runInContext('renderSessionActionsFoot("gold", "s1")', sandbox);
const busyRaw = vm.runInContext('JOB_ID="j1"; RAW_DRAWER.analyzeJobId="j1"; JOB_META={status:"running", progress:{done:1,total:2}}; renderSessionActionsFoot("raw", "s1")', sandbox);
console.log(JSON.stringify({ idleRaw, idleGold, busyRaw }));
""",
                encoding="utf-8",
            )
            run = subprocess.run(
                ["node", str(harness), str(js_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(0, run.returncode, run.stderr)
            out = json.loads(run.stdout.strip().splitlines()[-1])
            self.assertIn("sessionDeleteBtn", out["idleRaw"])
            self.assertIn("rawViewGoldBtn", out["idleRaw"])
            self.assertIn("drawerJsonBtn", out["idleGold"])
            self.assertNotIn("sessionDeleteBtn", out["busyRaw"])
            self.assertNotIn("rawAnalyzeBtn", out["busyRaw"])


class P0CursorToPiTests(unittest.TestCase):
    """P0-3: Cursor 会话采集完整保留 thinking、tool_result 与真实模型名。"""

    def test_cursor_to_pi_preserves_thinking_tool_results_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "cursor_chat.jsonl"
            dest = Path(tmp) / "pi_chat.jsonl"
            lines = [
                {
                    "role": "user",
                    "message": {"content": [{"type": "text", "text": "帮我分析 sample.apk 的资源结构"}]},
                },
                {
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "先列出目录并用脚本检查 APK 魔数。"},
                            {"type": "tool_use", "id": "toolu_01", "name": "bash", "input": {"command": "ls -la"}},
                        ]
                    },
                },
                {
                    "role": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_01",
                                "content": [{"type": "text", "text": "assets/ classes.dex"}],
                            }
                        ]
                    },
                },
                {
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "已确认包含 classes.dex，总结结论。"},
                            {"type": "text", "text": "APK 结构核验完毕。"},
                        ]
                    },
                },
            ]
            src.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
            _cursor_to_pi(src, dest, "cur-sid-1", r"D:\crack\sample")

            parsed = pilog.parse_session(dest)
            self.assertIn("claude-3-7-sonnet", parsed["models"])
            assts = [m for m in parsed["messages"] if m["role"] == "assistant"]
            self.assertEqual(2, len(assts))
            self.assertEqual("先列出目录并用脚本检查 APK 魔数。", assts[0]["thinking"])
            self.assertEqual(1, len(assts[0]["tool_results"]))
            self.assertEqual("assets/ classes.dex", assts[0]["tool_results"][0]["content"])
            # 纯 tool_result 的 user 行不得产生多余的空 user 消息打断轨迹
            users = [m for m in parsed["messages"] if m["role"] == "user"]
            self.assertEqual(1, len(users))

            q = analyze_step_quality(
                assts[0]["thinking"],
                assts[0]["text"],
                assts[0]["tool_calls"],
                assts[0]["tool_results"],
            )
            self.assertNotIn("missing_thinking_block", q["pathologies"])


class P0SeamAndDpoJudgeTests(unittest.TestCase):
    """P0-4 & P0-5: Judge 接缝上下文完整性与 DPO 长工具调用截断/防漏判。"""

    def test_purify_by_judge_populates_seams_for_prune_turn_and_silent(self) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["judge"] = {"enabled": True, "seam_min_silent_chars": 20}
        flow = [
            {"id": "sys", "role": "system", "content": "sys"},
            {"id": "u0", "role": "user", "content": "破解 a.apk", "turn_index": 0},
            {
                "id": "a1",
                "role": "assistant",
                "turn_index": 0,
                "thinking": "第一步正常侦察。",
                "text": "这是一段超过二十个字符的中间播报文本，会被机械静默清空。",
                "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
                "tool_results": [{"id": "r1", "tool_call_id": "c1", "name": "bash", "content": "ok", "isError": False}],
            },
            {"id": "tr1", "role": "tool", "tool_call_id": "c1", "name": "bash", "content": "ok"},
            {
                "id": "a2",
                "role": "assistant",
                "turn_index": 0,
                "thinking": "手滑输错命令。",
                "text": "",
                "tool_calls": [{"id": "c2", "name": "bash", "arguments": {"command": "badcmd"}}],
                "tool_results": [{"id": "r2", "tool_call_id": "c2", "name": "bash", "content": "command not found", "isError": True}],
            },
            {"id": "tr2", "role": "tool", "tool_call_id": "c2", "name": "bash", "content": "command not found"},
            {
                "id": "a3",
                "role": "assistant",
                "turn_index": 1,
                "folded_turn_index": 1,
                "folded_turn_type": "continuation",
                "thinking": "Oops, 刚才报错了，现在改用正确脚本。",
                "text": "完成。",
                "tool_calls": [],
                "tool_results": [],
            },
        ]
        turn_marks = [
            {
                "turn_index": 1,
                "kind": "continuation",
                "action": "folded",
                "folded": True,
                "dropped": False,
                "text": "继续",
                "reason": "rule",
                "assistant_steps": 1,
                "popped_await_msg": False,
            }
        ]
        captured_seams = []

        def fake_decide(_msgs, **_kw):
            return {"a2": {"action": "prune", "reason": "手滑命令", "rewritten": ""}}

        def fake_audit_seams(seams, **_kw):
            captured_seams.extend(seams)
            return {
                "prune::a2": {
                    "smooth": False,
                    "issue": "残留 Oops 断头引用",
                    "patch_msg_id": "a3",
                    "patched_thinking": "承接目录侦察结果，直接运行核验脚本完成收尾。",
                }
            }

        with patch("pipeline.judge.decide_steps", side_effect=fake_decide), patch(
            "pipeline.judge.audit_seams", side_effect=fake_audit_seams
        ):
            gold_msgs, _pruned, _stats, audit = stage2_gold.purify_by_judge(
                flow, cfg, r"D:\crack\demo", turn_marks=turn_marks, log=lambda *_: None
            )

        by_sid = {s["seam_id"]: s for s in captured_seams}
        self.assertIn("prune::a2", by_sid)
        self.assertEqual("a1", by_sid["prune::a2"]["before"].get("msg_id"))
        self.assertEqual("a3", by_sid["prune::a2"]["after"].get("msg_id"))

        self.assertIn("turn::1", by_sid)
        self.assertEqual("a1", by_sid["turn::1"]["before"].get("msg_id"))
        self.assertEqual("a3", by_sid["turn::1"]["after"].get("msg_id"))

        self.assertIn("silent::a1", by_sid)
        self.assertEqual("a1", by_sid["silent::a1"]["before"].get("msg_id"))
        self.assertEqual("a3", by_sid["silent::a1"]["after"].get("msg_id"))

        self.assertEqual(1, audit["seams_fixed"])
        a3 = next(m for m in gold_msgs if m["id"] == "a3")
        self.assertIn("seam_repaired", a3["flags"])

    def test_dpo_payload_preserves_tool_call_and_diff_after_long_thinking(self) -> None:
        long_think = "详细分析字节偏移与头部魔数。" * 120  # >1500 字，远超旧 900 字上限
        common_code = "line = 'same'\n" * 180
        chosen_tc = f'<tool_call>\n{{"name": "write", "arguments": {{"path": "dec.py", "content": "{common_code}KEY=0x5A"}}}}\n</tool_call>'
        rejected_tc = f'<tool_call>\n{{"name": "write", "arguments": {{"path": "dec.py", "content": "{common_code}KEY=0x00"}}}}\n</tool_call>'
        pair_a = {
            "sample_id": "dpo::s1::fold1",
            "conversations": [{"from": "human", "value": "解密配置"}],
            "chosen": {"from": "gpt", "value": f"<think>\n{long_think}\n</think>\n{chosen_tc}"},
            "rejected": {"from": "gpt", "value": f"<think>\n{long_think}\n</think>\n{rejected_tc}"},
            "metadata": {"reason": "state_folding_bug_fix"},
        }
        pair_b = copy.deepcopy(pair_a)
        pair_b["chosen"]["value"] = f"<think>\n{long_think}\n</think>\n{rejected_tc}"

        self.assertNotEqual(judge._vkey("m", "dpo", pair_a), judge._vkey("m", "dpo", pair_b))
        payload = judge._dpo_pair_payload(pair_a)
        self.assertIn("<tool_call>", payload["chosen"])
        self.assertIn("KEY=0x5A", payload["chosen"])
        self.assertIn("KEY=0x00", payload["rejected"])

        # 若超长中间段不同导致头尾摘要相同，应自动补充 body_diff
        mid_a = "A" * 1200 + "FIXED_BUG" + "Z" * 1200
        mid_b = "A" * 1200 + "BROKEN_OP" + "Z" * 1200
        pair_mid = {
            "sample_id": "dpo::s1::mid",
            "conversations": [{"from": "human", "value": "q"}],
            "chosen": {"from": "gpt", "value": f"<think>\n{long_think}\n</think>\n<tool_call>{mid_a}</tool_call>"},
            "rejected": {"from": "gpt", "value": f"<think>\n{long_think}\n</think>\n<tool_call>{mid_b}</tool_call>"},
        }
        payload_mid = judge._dpo_pair_payload(pair_mid)
        self.assertIn("body_diff", payload_mid)
        self.assertIn("FIXED_BUG", payload_mid["body_diff"]["chosen_span"])
        self.assertIn("BROKEN_OP", payload_mid["body_diff"]["rejected_span"])


class P1SourceNeutralAndToolsTests(unittest.TestCase):
    """P1-1 ~ P1-5: 来源中立、推理代理、契约自检、格式互转、会话删除与 Build 清单。"""

    def test_judge_enabled_is_source_neutral_and_curation_handles_low_conf_trainable(self) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["judge"] = {"enabled": True}
        for label in ("positive", "negative", "pending"):
            self.assertTrue(judge.enabled(cfg, label))

        ann = {
            "review_key": "1:abc:6",
            "model_report": {"verdict": "trainable", "confidence": 0.4},
            "review_rounds": [],
        }
        eff = curation.resolve(ann, cfg)
        self.assertEqual("auto_gold", eff["state"])

    def test_reasoning_proxy_splits_stream_and_parses_xml_and_json_tools(self) -> None:
        from server.reasoning_proxy import _StreamSplitter, _rebuild_nonstream, make_app

        proxy_app = make_app("http://127.0.0.1:8000/v1")
        self.assertIsNotNone(proxy_app)

        sp = _StreamSplitter(enabled=True)
        r1, c1 = sp.feed("<thi")
        r2, c2 = sp.feed("nk>\n先推理偏移")
        r3, c3 = sp.feed("再验证</th")
        r4, c4 = sp.feed("ink>\n最终正文")
        fr, fc = sp.flush()
        self.assertEqual("先推理偏移再验证", (r1 + r2 + r3 + r4 + fr).strip())
        self.assertEqual("最终正文", c1 + c2 + c3 + c4 + fc)

        xml_resp = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "<think>\n查目录\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
                    },
                }
            ]
        }
        rebuilt = _rebuild_nonstream(xml_resp)
        msg = rebuilt["choices"][0]["message"]
        self.assertEqual("查目录", msg["reasoning_content"])
        self.assertEqual("", msg["content"])
        self.assertEqual("tool_calls", rebuilt["choices"][0]["finish_reason"])
        self.assertEqual("bash", msg["tool_calls"][0]["function"]["name"])
        self.assertEqual({"command": "ls -la"}, json.loads(msg["tool_calls"][0]["function"]["arguments"]))

    def test_check_toolcall_contract_handles_coder_xml_stray_xml_and_dpo(self) -> None:
        path = ROOT / "tools" / "check_toolcall_contract.py"
        spec = importlib.util.spec_from_file_location("check_toolcall_contract", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            valid_xml = Path(tmp) / "valid_xml.jsonl"
            dpo_xml = Path(tmp) / "dpo_xml.jsonl"
            stray_xml = Path(tmp) / "stray_xml.jsonl"
            tools_col = default_tools_json()
            xml_call = "<think>\nok\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"

            write_jsonl_atomic(valid_xml, [{
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": xml_call}],
                "tools": tools_col,
            }])
            st_valid = mod.scan_file(valid_xml, "coder-xml")
            self.assertEqual(0, st_valid["stray_xml"])
            self.assertEqual([], st_valid["bad"])
            self.assertEqual(1, st_valid["ok"])

            # DPO chosen 用合法 coder-xml，rejected 混入 hermes-json -> 应检出 wrong_style
            bad_hermes = '<think>\nbad\n</think>\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
            write_jsonl_atomic(dpo_xml, [{
                "conversations": [{"from": "human", "value": "q"}],
                "chosen": {"from": "gpt", "value": xml_call},
                "rejected": {"from": "gpt", "value": bad_hermes},
                "tools": tools_col,
            }])
            st_dpo = mod.scan_file(dpo_xml, "coder-xml")
            self.assertEqual(1, st_dpo["wrong_style"])

            # <tool_call> 之外出现裸 <function=...> -> 应检出 stray_xml
            write_jsonl_atomic(stray_xml, [{
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "裸露 <function=bash> 调用"}],
                "tools": tools_col,
            }])
            st_stray = mod.scan_file(stray_xml, "coder-xml")
            self.assertEqual(1, st_stray["stray_xml"])

    def test_format_roundtrip_preserves_string_types_nested_closing_tags_and_single_newline_think(self) -> None:
        # 1) </think>\n 单换行规范（无双换行伪影）
        m = {
            "id": "m1",
            "role": "assistant",
            "thinking": "先看头部魔数",
            "text": "",
            "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "123", "timeout": 30}}],
        }
        sgpt_val = to_sharegpt(m)[0]["value"]
        self.assertIn("</think>\n<tool_call>", sgpt_val)
        self.assertNotIn("</think>\n\n", sgpt_val)
        oai_val = to_openai({**m, "text": "正文", "tool_calls": []})["content"]
        self.assertEqual("<think>\n先看头部魔数\n</think>\n正文", oai_val)

        # 2) JSON 字符串参数、纯数字字符串参数与含字面量 </parameter> / </tool_call> 的源码往返不失真
        tricky_code = 'cfg = {"port": 8080}\npat = "</parameter></tool_call>"\n'
        row = {
            "conversations": [
                {"from": "human", "value": "写文件"},
                {
                    "from": "gpt",
                    "value": (
                        "<think>\n写配置与命令\n</think>\n"
                        + '<tool_call>\n'
                        + json.dumps({"name": "write", "arguments": {"path": "a.json", "content": '{"port": 8080}'}}, ensure_ascii=False)
                        + "\n</tool_call>\n"
                        + '<tool_call>\n'
                        + json.dumps({"name": "bash", "arguments": {"command": "123", "timeout": 30}}, ensure_ascii=False)
                        + "\n</tool_call>\n"
                        + '<tool_call>\n'
                        + json.dumps({"name": "write", "arguments": {"path": "b.py", "content": tricky_code}}, ensure_ascii=False)
                        + "\n</tool_call>"
                    ),
                },
            ],
            "tools": default_tools_json(),
        }
        c1, u1 = convert_sharegpt_row(row, "coder-xml")
        self.assertEqual((3, 0), (c1, u1))
        c2, u2 = convert_sharegpt_row(row, "hermes-json")
        self.assertEqual((3, 0), (c2, u2))

        final_text = row["conversations"][1]["value"]
        _, _, unparsed = convert_toolcall_text(final_text, "coder-xml")
        self.assertEqual(0, unparsed)
        from pipeline.common.format import _iter_tc_spans

        spans = _iter_tc_spans(final_text)
        self.assertEqual(3, len(spans))
        parsed_calls = [_parse_tc_block(final_text[s:e]) for s, e in spans]
        self.assertIsInstance(parsed_calls[0]["arguments"]["content"], str)
        self.assertEqual('{"port": 8080}', parsed_calls[0]["arguments"]["content"])
        self.assertIsInstance(parsed_calls[1]["arguments"]["command"], str)
        self.assertEqual("123", parsed_calls[1]["arguments"]["command"])
        self.assertEqual(30, parsed_calls[1]["arguments"]["timeout"])
        self.assertEqual(tricky_code, parsed_calls[2]["arguments"]["content"])

    def test_session_delete_and_build_manifest_handle_scope_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            for sub in ("00_state", "01_raw", "02_gold", "03_train"):
                (data / sub).mkdir(parents=True)
            cfg = copy.deepcopy(DEFAULTS)
            cfg["paths"]["data_dir"] = str(data)

            write_json_atomic(data / "01_raw" / "index.json", {"sessions": {"s1": {"session_id": "s1", "label": "positive"}, "s2": {"session_id": "s2", "label": "positive"}}})
            write_json_atomic(data / "02_gold" / "index.json", {"sessions": {}})
            draft_obj = {
                "version": 1,
                "decisions": {"step::s1::a1": {"action": "exclude"}, "step::s2::a1": {"action": "exclude"}},
                "scope_decisions": {
                    "session:s1": {"action": "exclude"},
                    "session_sft:s1": {"action": "exclude"},
                    "session_dpo:s1": {"action": "exclude"},
                    "session:s2": {"action": "exclude"},
                    "project:game::demo": {"action": "exclude"},
                },
                "split_plan": {
                    "assignments": {"game::demo": "train"},
                    "session_assignments": {"s1": "eval", "s2": "eval"},
                },
            }
            write_json_atomic(data / "00_state" / "dataset_draft.json", draft_obj)
            write_jsonl_atomic(data / "03_train" / "sft.train.jsonl", [{"sample_id": "step::s2::a1", "metadata": {"session_id": "s2"}}])

            _, build_man = dataset_build.create_build(
                data / "03_train", cfg, "f" * 16, draft_obj, {"sft.train.jsonl": 1}
            )
            self.assertIn("scope_decisions", build_man)
            self.assertIn("project:game::demo", build_man["scope_decisions"])

            delete_session("s1", cfg=cfg)
            after_draft = read_json(data / "00_state" / "dataset_draft.json")
            self.assertEqual({"session:s2", "project:game::demo"}, set(after_draft["scope_decisions"]))
            self.assertEqual({"s2": "eval"}, after_draft["split_plan"]["session_assignments"])


class P2AndP3EngineeringTests(unittest.TestCase):
    """P2 & P3: 本地分词器同源、API 安全中间件与互斥锁、仓库卫生与领域 Profile 配置化。"""

    def test_length_gate_loads_local_seqlength_tokenizer_without_transformers_network(self) -> None:
        class DummyFastTok:
            def encode(self, text, add_special_tokens=False):
                class _Out:
                    ids = [ord(c) for c in text]
                return _Out()

            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids)

        length_gate._TOKENIZER_CACHE.clear()
        with patch("pipeline.common.seqlength.find_tokenizer", return_value=(DummyFastTok(), "local:qwen3.8")):
            tok, src = length_gate.load_gate_tokenizer()
        self.assertIsNotNone(tok)
        self.assertEqual("local:qwen3.8", src)
        ids = tok.encode("hello")
        self.assertEqual("hello", tok.decode(ids))
        length_gate._TOKENIZER_CACHE.clear()

    def test_server_security_middleware_and_sync_write_job_lock(self) -> None:
        import server.app as sa
        from fastapi import HTTPException

        class FakeURL:
            path = "/api/config"

        class FakeReq:
            def __init__(self, method: str, headers: dict):
                self.method = method
                self.headers = headers
                self.url = FakeURL()

        async def ok_next(_req):
            return {"passed": True}

        def _drive(coro):
            try:
                coro.send(None)
            except StopIteration as stop:
                return stop.value
            raise AssertionError("expected immediate coroutine completion")

        # 1) DNS Rebinding 非本地 Host 拦截
        r_host = _drive(sa._local_security_guard(FakeReq("GET", {"host": "evil.attacker.com"}), ok_next))
        self.assertEqual(403, r_host.status_code)

        # 2) 跨站 Origin 写请求拦截
        r_origin = _drive(
            sa._local_security_guard(
                FakeReq("PUT", {"host": "127.0.0.1:8765", "origin": "http://evil.attacker.com"}),
                ok_next,
            )
        )
        self.assertEqual(403, r_origin.status_code)

        # 3) 合法本地请求放行
        r_local = _drive(
            sa._local_security_guard(
                FakeReq("PUT", {"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:8765"}),
                ok_next,
            )
        )
        self.assertEqual({"passed": True}, r_local)

        # 4) 未设置 GOLDTHREAD_API_TOKEN 时，禁止通过 HTTP API 篡改 pi_models_json
        with self.assertRaises(HTTPException) as ctx_cfg:
            sa._guard_sensitive_config_change({"paths": {"pi_models_json": "/tmp/evil_models.json"}})
        self.assertEqual(403, ctx_cfg.exception.status_code)

        # 5) 后台写作业运行时，同步触发 Stage 2/3 重建接口应返回 409 互斥保护
        job = sa.Job("stage3", False, {})
        job.status = "running"
        with sa.JOBS_LOCK:
            sa.JOBS[job.id] = job
        try:
            with self.assertRaises(HTTPException) as ctx_lock:
                sa._assert_no_gold_write_running()
            self.assertEqual(409, ctx_lock.exception.status_code)
        finally:
            with sa.JOBS_LOCK:
                sa.JOBS.pop(job.id, None)

    def test_gitignore_utf8_and_ssot_wrappers_and_domain_profile(self) -> None:
        # 1) .gitignore 必须是合法纯 UTF-8 文本，无 NUL 字节或乱码
        raw_gi = (ROOT / ".gitignore").read_bytes()
        self.assertEqual(0, raw_gi.count(0))
        gi_text = raw_gi.decode("utf-8")
        self.assertIn("data/04_release/", gi_text)
        self.assertIn(".scratch/", gi_text)

        # 2) tools/ 与 pipeline/train_template/tools/ 同源委托一致
        for name, attr in (
            ("check_data.py", "alignment_error"),
            ("filter_by_cutoff.py", "resolve_tokenizer_id"),
            ("lowmem_dpo.py", "patch_dpo_trainer"),
            ("train_lowmem.py", "patch_trainer"),
        ):
            p = ROOT / "tools" / name
            spec = importlib.util.spec_from_file_location(f"wrap_{name}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, attr), f"{name} 缺少 {attr}")

        # 3) domain profile 支持自定义 system_prompt / tools 与关闭逆向专属病灶规则
        cfg = copy.deepcopy(DEFAULTS)
        custom_tool = [{"type": "function", "function": {"name": "sql_query", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        cfg["domain"] = {
            "profile": "general_coding",
            "system_prompt": "你是一名全栈后端工程 Agent。",
            "use_session_system_prompt": False,
            "tools": custom_tool,
            "enable_re_pathologies": False,
        }
        session = {
            "session_id": "dom1",
            "cwd": r"D:\crack\app-a",
            "timestamp": "",
            "messages": [
                {"id": "u0", "role": "user", "content": "查询数据库表", "timestamp": ""},
                {
                    "id": "a0",
                    "role": "assistant",
                    "thinking": "0x10 = 16 ✓ 读取 D:\\crack\\app-b\\schema.sql",
                    "text": "ok",
                    "tool_calls": [],
                    "tool_results": [],
                    "timestamp": "",
                },
            ],
        }
        flow, _, _ = pilog.build_flow(session, cfg)
        self.assertIn("全栈后端工程 Agent", flow[0]["content"])
        self.assertIn("sql_query", default_tools_json(cfg))

        q_re_on = analyze_step_quality(
            "0x10 = 16 ✓ 读取 D:\\crack\\app-b\\schema.sql",
            "ok",
            [],
            [],
            r"D:\crack\app-a",
            enable_re_pathologies=True,
        )
        q_re_off = analyze_step_quality(
            "0x10 = 16 ✓ 读取 D:\\crack\\app-b\\schema.sql",
            "ok",
            [],
            [],
            r"D:\crack\app-a",
            enable_re_pathologies=False,
        )
        self.assertIn("mental_hex_checkmark", q_re_on["pathologies"])
        self.assertEqual([], q_re_off["pathologies"])


if __name__ == "__main__":
    unittest.main()
