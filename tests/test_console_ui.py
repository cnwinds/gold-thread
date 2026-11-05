"""控制台 UI 重构的页面结构与前端行为回归测试。"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import app


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
SCRIPT = "\n".join(
    re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", HTML)
)


class ConsoleWorkspaceTests(unittest.TestCase):
    def test_console_route_serves_all_workspaces(self) -> None:
        response = TestClient(app).get("/")
        self.assertEqual(200, response.status_code)
        for page in ("overview", "raw", "gold", "train", "tasks", "runs", "config"):
            self.assertIn(f'id="view-{page}"', response.text)

    def test_data_workspaces_keep_filters_and_primary_actions(self) -> None:
        expected = {
            "raw library": (
                'id="rawProject"', 'id="rawModel"', 'id="rawAdopted"', 'id="rawSearch"', 'id="rawPick"',
                'data-analyze="${esc(s.session_id)}"',
            ),
            "collect": (
                'id="pickSource"', 'id="pickFilter"', 'id="pickIngest"',
                'id="pickGo"', 'id="pickPreview"',
            ),
            "gold": (
                'id="goldProject"', 'id="goldModel"', 'id="goldOutput"', 'id="goldSearch"',
                'data-gold-open="${esc(s.session_id)}"', "goldSetStepGate",
            ),
            "train": (
                'id="dsProject"', 'id="dsSplit"', 'id="dsSearch"',
                "renderDatasetIndex", "setTrainInclude",
            ),
            "tasks": ("taskRowHtml", "pollTaskLogs", "taskOpenAction"),
            "delivery": ("selectExportProfile", "exportPackage", "downloadZip"),
            "config": ('id="cfgSearch"', "cfgSave", "cfgDiscard"),
        }
        for workspace, snippets in expected.items():
            with self.subTest(workspace=workspace):
                for snippet in snippets:
                    self.assertIn(snippet, HTML)

    def test_raw_and_gold_drawers_share_confirmed_delete_action(self) -> None:
        for snippet in (
            "function sessionDeleteButton(sid)",
            'id="sessionDeleteBtn"',
            "function requestSessionDelete(sid)",
            'method: "DELETE"',
            "/api/session/${encodeURIComponent(sid)}",
            "watchJob(job_id)",
            "pi / Cursor 源目录和已冻结的历史交付包不会被修改。",
        ):
            self.assertIn(snippet, HTML)
        self.assertIn('!running ? sessionDeleteButton(sid)', HTML)

    def test_drawer_load_failures_surface_error_instead_of_hanging(self) -> None:
        for snippet in (
            '加载失败：${esc(err.message || err)}',
            "toast(String(err.message || err), \"err\")",
        ):
            self.assertEqual(2, HTML.count(snippet), snippet)

    def test_gold_drawer_browses_sft_and_dpo_content(self) -> None:
        for snippet in (
            'data-gold-view="gold"', 'data-gold-view="sft"', 'data-gold-view="dpo"',
            "function loadGoldDataset(kind)", "function selectGoldSample(kind, index)",
            "function renderGoldSampleDetail(row, kind)",
            'sft_step_windows_sharegpt.jsonl', 'dpo_preference_pairs.jsonl',
            'buildDiff(rejected, chosen', 'convToMsgs(row.conversations || row.messages)',
            'class="gold-sample-item"', 'id="goldSampleDetail"',
        ):
            self.assertIn(snippet, HTML)

    def test_drawer_footer_is_hidden_when_it_has_no_actions(self) -> None:
        for snippet in (
            "function setDrawerFoot(html", "foot.hidden = !html.trim()",
            'setDrawerFoot("")', ".drawer .df[hidden]{display:none!important}",
            'class="session-foot"', 'class="session-actions"',
        ):
            self.assertIn(snippet, HTML)

    def test_session_role_controls_are_removed_from_the_console(self) -> None:
        for snippet in (
            'id="sessionLabelSwitch"', 'id="rawLabel"', 'id="goldLabel"',
            "renderSessionLabelFootPane", "relabelFromDrawer", "labelSeg_ctl",
            "数据集角色", "待设角色", "正例", "反例", "待定",
        ):
            self.assertNotIn(snippet, HTML)
        self.assertIn("d.source_models", HTML)
        self.assertIn('output:f.output||""', HTML)
        self.assertIn('adopted:f.adopted||""', HTML)

    def test_delivery_workspace_only_handles_packages_and_history(self) -> None:
        render_match = re.search(r"function renderRunsView\(rb\) \{([\s\S]*?)\n\}", HTML)
        self.assertIsNotNone(render_match)
        render_source = render_match.group(1)
        for snippet in ("runStage", "rw-shell", "rw-head", "jobConsole", "历史运行记录", "rwRenderStageDetail"):
            self.assertNotIn(snippet, render_source)

        detail_match = re.search(r"function rwRenderDeliveryDetail\(rb\) \{([\s\S]*?)function packHistHtml", HTML)
        self.assertIsNotNone(detail_match)
        detail_source = detail_match.group(1)
        for snippet in (
            "delivery-workspace", "delivery-format-list", "delivery-current-panel",
            "delivery-history", "selectExportProfile", "exportPackage", "downloadZip",
            "packHistHtml()", "delivery-meta",
        ):
            self.assertIn(snippet, detail_source)

        load_match = re.search(r"async function loadRuns\(\) \{([\s\S]*?)function stopPackTimer", HTML)
        self.assertIsNotNone(load_match)
        self.assertNotIn('api("/api/runs")', load_match.group(1))
        self.assertIn('api("/api/export/history")', load_match.group(1))

    def test_reference_conversation_and_overview_views_are_preserved(self) -> None:
        for snippet in (
            "function assayBand(", 'class="workflow-head"',
            "function renderMessages(", 'id="flowMode"', 'id="flowBody"',
            'class="trace ${open ? "open" : ""}"',
        ):
            self.assertIn(snippet, HTML)

    def test_shared_workspace_styles_cover_responsive_and_reduced_motion(self) -> None:
        for snippet in (
            ".metric-strip", ".workspace-panel", ".filter-search",
            ".tasks-shell", ".collect-actions", ".train-plan",
            "@media (max-width:860px)", "@media (prefers-reduced-motion:reduce)",
        ):
            self.assertIn(snippet, HTML)

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查内联脚本")
    def test_inline_script_syntax_and_task_progress_behavior(self) -> None:
        match = re.search(r"(function taskProgressPercent\(t\) \{.*?\n\})", SCRIPT, re.S)
        self.assertIsNotNone(match)
        progress_source = match.group(1)
        node_script = f"""
{progress_source}
const cases = [
  [{{status: 'done'}}, 100],
  [{{status: 'running', progress: {{done: 2, total: 8}}}}, 25],
  [{{status: 'running', progress: {{current: 3, max: 4}}}}, 75],
  [{{status: 'running'}}, null],
  [{{status: 'failed'}}, null],
];
for (const [input, expected] of cases) {{
  const actual = taskProgressPercent(input);
  if (actual !== expected) throw new Error(JSON.stringify({{input, expected, actual}}));
}}
console.log(JSON.stringify({{cases: cases.length}}));
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "console_ui_test.js"
            path.write_text(SCRIPT, encoding="utf-8")
            subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"cases": 5}, json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
