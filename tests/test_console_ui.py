"""控制台极简三栏导航（数据集图谱 / 账本与任务 / 配置）与前端交互回归测试。"""
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
    def test_console_route_serves_streamlined_three_tabs_and_removes_legacy_views(self) -> None:
        response = TestClient(app).get("/")
        self.assertEqual(200, response.status_code)
        for page in ("overview", "tasks", "config"):
            self.assertIn(f'id="view-{page}"', response.text)
            self.assertIn(f'data-tab="{page}"', response.text)

        # 旧工作区底座、交付侧栏与运行全流程按钮均已彻底移除
        for removed_dom in (
            'id="view-raw"',
            'id="view-gold"',
            'id="view-train"',
            'id="view-runs"',
            'data-tab="raw"',
            'data-tab="gold"',
            'data-tab="train"',
            'data-tab="runs"',
            'id="btnRunAll"',
            "工作区底座",
            "function loadRaw(",
            "function loadGold(",
            "function loadTrainDraft(",
            "function loadRuns(",
        ):
            self.assertNotIn(removed_dom, response.text)

    def test_lineage_graph_absorbs_collect_pack_and_session_actions(self) -> None:
        expected = {
            "graph_and_layers": (
                'id="vdsDagPanel"',
                "renderDatasetDagHtml(",
                "renderVdsBreakdownHtml(",
                "loadVirtualDatasetSubView(",
                "openDatasetSessionDrawer(",
                "openVirtualTrainSampleDrawer(",
                "toggleVdsSessionSft(",
            ),
            "collect_drawer": (
                "openCollectDrawer(",
                "loadCollect(",
                'id="pickSource"',
                'id="pickFilter"',
                'id="pickIngest"',
                'id="pickGo"',
                'id="pickPreview"',
                "会话几乎没有有效内容",
                "assistant_errors",
            ),
            "pack_and_diff_drawers": (
                "openPackEval(",
                "openPackDocs(",
                "openPackLengthStats(",
                "openPackDiffDrawer(",
                "buildWorkspaceFromGraph(",
                "exportPackage(",
                "downloadZip(",
                "deletePack(",
            ),
            "tasks_and_config": (
                "taskRowHtml",
                "pollTaskLogs",
                "taskOpenAction",
                'id="cfgSearch"',
                "cfgSave",
                "cfgDiscard",
            ),
        }
        for group, snippets in expected.items():
            with self.subTest(group=group):
                for snippet in snippets:
                    self.assertIn(snippet, HTML)

    def test_job_queue_status_is_visible_in_task_hub(self) -> None:
        for snippet in (
            'queued:"排队中"',
            "JOB_ACTIVE",
            "已加入排队",
            "排队等待中",
            "TASK_HUB.queued",
            'r.status === "queued"',
        ):
            self.assertIn(snippet, HTML)

    def test_task_log_keeps_timestamps_and_preserves_scroll(self) -> None:
        match = re.search(r"function paintTaskLogPanel\(\) \{([\s\S]*?)\n\}", SCRIPT)
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("atBottom", body)
        self.assertIn("prevTop", body)
        self.assertIn("if (atBottom) el.scrollTop = el.scrollHeight", body)
        self.assertIn("else el.scrollTop = prevTop", body)
        self.assertIn('class="ts"', body)
        self.assertNotIn("replace(/^\\d{2}:\\d{2}:\\d{2}\\s*/", body)
        self.assertIn("time.strftime('%H:%M:%S')", (ROOT / "server" / "app.py").read_text(encoding="utf-8"))

    def test_session_drawer_retains_confirmed_delete_action(self) -> None:
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
        self.assertIn('!busy ? sessionDeleteButton(sid)', HTML)

    def test_sample_and_conversation_renderers_are_preserved(self) -> None:
        for snippet in (
            "function renderGoldSampleDetail(row, kind)",
            "splitThinkBody(",
            "thinkDisplayText(",
            "renderDpoPreferenceCard(",
            "Chosen 思考",
            "共同工具动作",
            "convToMsgs(row.conversations || row.messages)",
            "function renderMessages(",
            'id="flowMode"',
            'id="flowBody"',
            'class="trace ${open ? "open" : ""}"',
            "function setDrawerFoot(html",
            "foot.hidden = !html.trim()",
        ):
            self.assertIn(snippet, HTML)

    def test_session_role_and_release_controls_remain_absent(self) -> None:
        for snippet in (
            'id="sessionLabelSwitch"',
            "数据集角色",
            "待设角色",
            "正例",
            "反例",
            "待定",
            "发布当前 Build",
            "回滚到上一版本",
            "迁入 Build",
            "function activateRelease",
            "发布版本",
        ):
            self.assertNotIn(snippet, HTML)

    def test_shared_workspace_styles_cover_responsive_and_reduced_motion(self) -> None:
        for snippet in (
            ".metric-strip",
            ".workspace-panel",
            ".filter-search",
            ".tasks-shell",
            ".collect-actions",
            "@media (max-width:860px)",
            "@media (prefers-reduced-motion:reduce)",
        ):
            self.assertIn(snippet, HTML)

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查打包完成回调")
    def test_fast_package_completion_notifies_watchers(self) -> None:
        match = re.search(r"(function watchJob\(jobId, onDone\) \{[\s\S]*?\n\})", SCRIPT)
        self.assertIsNotNone(match)
        node_script = f"""
const JOB_ACTIVE = st => st === "running" || st === "queued";
let JOB_TIMER = null, JOB_ID = null, JOB_LINES = [], JOB_META = null;
let STATE = {{tab:'overview'}};
let RAW_DRAWER = {{analyzeJobId:null}};
let completed = 0;
function registerJobWatch() {{}}
function onTaskFinished(j) {{ if (j.status === 'done') completed += 1; }}
function clearInterval() {{}}
function setInterval() {{ return 1; }}
function paintRawDrawerAnalyze() {{}}
async function api() {{ return {{status:'done', stage:'package', id:'quick', logs:[]}}; }}
{match.group(1)}
(async () => {{
  watchJob('quick', () => {{}});
  await new Promise(resolve => setTimeout(resolve, 0));
  if (completed !== 1) throw new Error(`terminal callback count=${{completed}}`);
  console.log(JSON.stringify({{completed}}));
}})().catch(err => {{ console.error(err); process.exit(1); }});
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watch_job_test.js"
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"completed": 1}, json.loads(result.stdout))

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查任务登记后的即时查询")
    def test_register_job_watch_reads_a_finished_job_immediately(self) -> None:
        match = re.search(r"(function registerJobWatch\(jobId, onDone\) \{[\s\S]*?\n\})", SCRIPT)
        self.assertIsNotNone(match)
        node_script = f"""
const JOB_ACTIVE = st => st === "running" || st === "queued";
const TASK_HUB = {{ watched: new Map(), pollTimer: 1, notified: new Set() }};
const finished = [];
function taskHubEnsure() {{}}
function onTaskFinished(j) {{
  if (!j || TASK_HUB.notified.has(j.id)) return;
  TASK_HUB.notified.add(j.id);
  TASK_HUB.watched.delete(j.id);
  finished.push(j.id);
}}
let status = 'done';
async function api() {{ return {{status, id:'quick', stage:'package'}}; }}
{match.group(1)}
(async () => {{
  registerJobWatch('quick', () => {{}});
  await new Promise(resolve => setTimeout(resolve, 0));
  status = 'running';
  registerJobWatch('later', () => {{}});
  await new Promise(resolve => setTimeout(resolve, 0));
  if (JSON.stringify(finished) !== JSON.stringify(['quick'])) throw new Error(JSON.stringify(finished));
  console.log(JSON.stringify({{finished}}));
}})().catch(err => {{ console.error(err); process.exit(1); }});
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "register_watch_test.js"
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"finished": ["quick"]}, json.loads(result.stdout))

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查导出格式选择")
    def test_saved_export_profile_does_not_require_runbook(self) -> None:
        list_match = re.search(r"(function exportProfileList\(rb\) \{[\s\S]*?\n\})", SCRIPT)
        current_match = re.search(r"(function currentExportProfile\(rb\) \{[\s\S]*?\n\})", SCRIPT)
        self.assertIsNotNone(list_match)
        self.assertIsNotNone(current_match)
        node_script = f"""
let PACK = {{profile:'Qwen3.8-8B', profileChosen:true}};
{list_match.group(1)}
{current_match.group(1)}
const selected = currentExportProfile({{}});
if (selected !== 'Qwen3.8-8B') throw new Error(`selected=${{selected}}`);
console.log(JSON.stringify({{selected}}));
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export_profile_test.js"
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"selected": "Qwen3.8-8B"}, json.loads(result.stdout))

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
