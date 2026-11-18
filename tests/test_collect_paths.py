"""采集路径校验：允许 pi sessions 下的 Junction，拒绝 .. 逃逸。"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.collect import (
    _pi_header_and_title,
    _under,
    _user_text,
    ingest_selected,
    list_sessions,
    preview_session,
    resolve_catalog,
)


class CollectPathTests(unittest.TestCase):
    def test_user_text_unwraps_file_wrapper_for_title(self):
        raw = (
            '<file name="D:\\crack_test\\teacher\\_retry\\app\\prompt.txt">\n'
            '用户是安全研究员，分析对象为其自有/已授权应用。\n'
            "</file>\n"
        )
        self.assertEqual(
            "用户是安全研究员，分析对象为其自有/已授权应用。",
            _user_text(raw),
        )
        self.assertFalse(_user_text(raw).startswith("<file"))

    def test_user_text_prefers_user_query_over_file_shell(self):
        raw = '<user_query>\n真正的问题\n</user_query>\n<file name="x.txt">附件</file>'
        self.assertEqual("真正的问题", _user_text(raw))

    def test_pi_header_title_skips_file_name_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-11-10T08-27-48-357Z_sid123.jsonl"
            path.write_text(
                "\n".join([
                    '{"type":"session","id":"sid123","cwd":"D:\\\\proj","timestamp":"2026-11-10T08:27:48.357Z"}',
                    json.dumps({
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": (
                                    '<file name="D:\\\\proj\\\\prompt.txt">\n'
                                    "请分析这个已授权应用并整理资源。\n"
                                    "</file>"
                                ),
                            }],
                        },
                    }, ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            info = _pi_header_and_title(path)
            self.assertEqual("sid123", info["session_id"])
            self.assertIn("请分析这个已授权应用", info["title"])
            self.assertNotIn("<file", info["title"])
            self.assertNotIn("prompt.txt", info["title"])

    def test_under_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "路径超出会话源目录"):
                _under(root, root / "proj" / ".." / ".." / "outside.jsonl")

    def test_under_accepts_path_inside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            target = root / "proj" / "a.jsonl"
            target.parent.mkdir(parents=True)
            target.write_text("{}\n", encoding="utf-8")
            self.assertEqual(target.absolute(), _under(root, target))

    @unittest.skipUnless(os.name == "nt", "Windows Junction only")
    def test_under_allows_windows_junction_under_sessions_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "sessions"
            real = tmp_path / "real_project" / "sessions"
            real.mkdir(parents=True)
            session = real / "chat.jsonl"
            session.write_text('{"type":"session","id":"j1"}\n', encoding="utf-8")
            link = root / "--D--fake-project--"
            root.mkdir()
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(real)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.skipTest(f"当前运行环境禁止创建 NTFS Junction: {proc.stderr or proc.stdout}")
            self.assertTrue(link.exists())
            # resolve 会跳到 real，旧逻辑会误判超根
            self.assertNotEqual(session.resolve(), (link / "chat.jsonl").absolute())
            got = _under(root, link / "chat.jsonl")
            self.assertEqual((link / "chat.jsonl").absolute(), got)
            self.assertTrue(got.is_file())

            catalog = "--D--fake-project--/chat.jsonl"
            resolved = resolve_catalog(
                "pi",
                catalog,
                cfg={"paths": {"sessions_dir": str(root), "data_dir": str(tmp_path / "data")}},
            )
            self.assertEqual(got, resolved)

    def test_list_sessions_q_matches_pi_catalog_folder(self):
        """cwd 不含 teacher 时，仍应按会话目录名搜到（例如 teacher-com.dopuz.klotski…）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            folder = root / "--D--crack_test-teacher-com.dopuz.klotski.riddle_6.1001--"
            folder.mkdir(parents=True)
            path = folder / "2026-10-23T07-50-35-752Z_01a167bd.jsonl"
            path.write_text(
                "\n".join([
                    '{"type":"session","id":"01a167bd","cwd":"D:\\\\crack\\\\riddle","timestamp":"2026-10-23T07:50:35.752Z"}',
                    json.dumps({
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "帮我破解游戏"}],
                        },
                    }, ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            data_dir = Path(tmp) / "data"
            (data_dir / "01_raw").mkdir(parents=True)
            (data_dir / "01_raw" / "index.json").write_text(
                json.dumps({"sessions": {}}), encoding="utf-8"
            )
            cfg = {"paths": {"sessions_dir": str(root), "data_dir": str(data_dir)}}
            with patch("pipeline.collect.load_config", return_value=cfg):
                miss = list_sessions("pi", "teacher", cfg=cfg)
            ids = [r["id"] for r in miss["sessions"]]
            self.assertEqual(1, miss["total"])
            self.assertTrue(any("klotski.riddle" in i for i in ids))
            with patch("pipeline.collect.load_config", return_value=cfg):
                by_pkg = list_sessions("pi", "com.dopuz.klotski", cfg=cfg)
            self.assertEqual(1, by_pkg["total"])

    def test_preview_surfaces_empty_minimax_api_failures(self):
        """助手全是 503 空回复时，预览应标明 empty，而不是只剩默认 system 像“空白”。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            folder = root / "--D--crack_test-teacher-minimax-com.vitastudio.mahjong_3.43.0--"
            folder.mkdir(parents=True)
            path = folder / "2026-11-12T15-25-23-017Z_sid.jsonl"
            err = '503: {"message":"No available Coding Plan account in pool"}'
            path.write_text(
                "\n".join([
                    '{"type":"session","id":"sid","cwd":"D:\\\\crack_test\\\\teacher\\\\minimax\\\\com.vitastudio.mahjong_3.43.0","timestamp":"2026-11-12T15:25:23.017Z"}',
                    json.dumps({
                        "type": "message",
                        "id": "u1",
                        "timestamp": "2026-11-12T15:25:23.100Z",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "分析 mahjong 游戏"}],
                        },
                    }, ensure_ascii=False),
                    json.dumps({
                        "type": "message",
                        "id": "a1",
                        "timestamp": "2026-11-12T15:25:23.300Z",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "model": "MiniMax-M3.1-Flash",
                            "provider": "cursor-pluse-minimax",
                            "errorMessage": err,
                            "stopReason": "error",
                        },
                    }, ensure_ascii=False),
                    json.dumps({
                        "type": "context_edit",
                        "id": "e1",
                        "timestamp": "2026-11-12T15:25:23.400Z",
                        "targetId": "a1",
                        "replacement": None,
                    }, ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            data_dir = Path(tmp) / "data"
            (data_dir / "01_raw").mkdir(parents=True)
            (data_dir / "01_raw" / "index.json").write_text(
                json.dumps({"sessions": {}}), encoding="utf-8"
            )
            cfg = {"paths": {"sessions_dir": str(root), "data_dir": str(data_dir)}, "flows": {}}
            with patch("pipeline.collect.load_config", return_value=cfg):
                prev = preview_session("pi", f"{folder.name}/{path.name}", cfg=cfg)
            self.assertTrue(prev["empty"])
            self.assertIn("No available Coding Plan", prev["empty_reason"])
            self.assertEqual(1, len(prev["assistant_errors"]))
            self.assertTrue(prev["assistant_errors"][0]["deleted"])
            self.assertEqual(f"{folder.name}/{path.name}", prev["path"])
            self.assertIn("mahjong", prev["project"])

    @unittest.skipUnless(os.name == "nt", "Windows Junction only")
    def test_ingest_pi_junction_uses_catalog_rel_not_resolve(self):
        """Junction 目标在 D:\\crack_test 时，ingest 仍应用 catalog 相对路径，不能 resolve 后 relative_to。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "sessions"
            real = tmp_path / "crack_test" / "teacher" / "com.pank0.usagishima_1.9.2" / "sessions"
            real.mkdir(parents=True)
            session = real / "2026-11-10T11-22-28-909Z_01a1c531.jsonl"
            session.write_text(
                '{"type":"session","id":"01a1c531","cwd":"D:\\\\crack_test\\\\teacher","timestamp":"2026-11-10T11:22:28.909Z"}\n'
                '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}\n',
                encoding="utf-8",
            )
            link = root / "--D--crack_test-teacher-com.pank0.usagishima_1.9.2--"
            root.mkdir()
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(real)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.skipTest(f"当前运行环境禁止创建 NTFS Junction: {proc.stderr or proc.stdout}")
            catalog = f"{link.name}/{session.name}"
            cfg = {"paths": {"sessions_dir": str(root), "data_dir": str(tmp_path / "data")}}
            captured: dict = {}

            def fake_adopt(src, rel, *, origin, catalog_id, cfg=None, log=print):  # noqa: ANN001
                captured.update({"src": Path(src), "rel": rel, "catalog_id": catalog_id, "origin": origin})
                return "01a1c531"

            with patch("pipeline.collect.adopt_file", side_effect=fake_adopt):
                sids = ingest_selected("pi", [catalog], cfg=cfg, log=lambda *_: None)
            self.assertEqual(["01a1c531"], sids)
            self.assertEqual(catalog, captured["rel"])
            self.assertEqual(catalog, captured["catalog_id"])
            self.assertEqual("pi", captured["origin"])
            # 读到的仍是 junction 下路径；rel 必须是 catalog 相对路径，不能是 resolve 后的盘符绝对路径
            self.assertTrue(str(captured["src"]).endswith(session.name))
            self.assertFalse(Path(captured["rel"]).is_absolute())
            self.assertNotIn(":\\", captured["rel"])
            self.assertNotIn("D:/", captured["rel"])


if __name__ == "__main__":
    unittest.main()
