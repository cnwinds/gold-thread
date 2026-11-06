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
                "loadTrainDraft", "setDraftDecisions", "renderSplitWorkspace",
            ),
            "tasks": ("taskRowHtml", "pollTaskLogs", "taskOpenAction"),
            "delivery": ("selectExportProfile", "exportPackage", "downloadZip"),
            "config": ('id="cfgSearch"', "cfgSave", "cfgDiscard"),
        }
        for workspace, snippets in expected.items():
            with self.subTest(workspace=workspace):
                for snippet in snippets:
                    self.assertIn(snippet, HTML)

    def test_train_draft_uses_server_decisions_not_local_file_switches(self) -> None:
        for snippet in (
            'api("/api/dataset/draft', 'function setDraftDecisions(',
            'function renderSplitWorkspace(', 'function replanSplit(',
            'body.dirty', 'data-draft-decision', 'id="draftBuild"',
            'trainRequestCurrent', 'expected_hash', 'syncDraftSelection',
            'dataset_evaluate', 'pending_evaluations', 'DPO 偏好对',
            'loadTrainBuilds', 'buildWorkspace', 'build_id',
            'onTaskFinished(j)', 'RUNS_REQUEST_SEQ', 'DRAWER_REQUEST_SEQ',
            'e.target.closest("[data-draft-toggle]")',
        ):
            self.assertIn(snippet, HTML)
        self.assertNotIn("gt.trainInclude", HTML)
        self.assertNotIn("function setTrainInclude", HTML)

    def test_drawer_async_loaders_capture_seq_after_open_drawer(self) -> None:
        for fn in ("openDraftSample", "openBuildDetail", "openTrainHistorySample"):
            with self.subTest(fn=fn):
                match = re.search(rf"async function {fn}\([^)]*\) \{{([\s\S]*?)const seq = DRAWER_REQUEST_SEQ", SCRIPT)
                self.assertIsNotNone(match, f"{fn} 应在 openDrawer 之后捕获 DRAWER_REQUEST_SEQ")
                self.assertIn("openDrawer(", match.group(1))

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
        # 原始、黄金、数据草稿、Build 和历史文件五个抽屉都必须结束 loading 并显示错误。
        self.assertEqual(5, HTML.count('加载失败：${esc(err.message || err)}'))
        self.assertEqual(2, HTML.count("toast(String(err.message || err), \"err\")"))

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

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查草稿交互辅助函数")
    def test_train_replan_and_stale_response_helpers(self) -> None:
        replan = re.search(r"(function replanNeedsSave\(preview, current\) \{[\s\S]*?\n\})", SCRIPT)
        current = re.search(r"(function trainRequestCurrent\(request\) \{[\s\S]*?\n\})", SCRIPT)
        self.assertIsNotNone(replan)
        self.assertIsNotNone(current)
        node_script = f"""
let TRAIN_REQUEST_SEQ = 2;
let STATE = {{train: {{workspace: 'draft'}}}};
{current.group(1)}
{replan.group(1)}
if (!trainRequestCurrent({{id:2, workspace:'draft'}})) throw new Error('current response rejected');
if (trainRequestCurrent({{id:1, workspace:'draft'}})) throw new Error('stale response accepted');
if (trainRequestCurrent({{id:2, workspace:'files'}})) throw new Error('wrong workspace accepted');
const currentPlan = {{eval_ratio:0.25, seed:42}};
const cases = [
  [{{changes:{{}}, split_plan:{{eval_ratio:0.25, seed:42}}}}, false],
  [{{changes:{{}}, base_plan:{{eval_ratio:0.5, seed:42}}, split_plan:{{eval_ratio:0.5, seed:42}}}}, false],
  [{{changes:{{}}, split_plan:{{eval_ratio:0.5, seed:42}}}}, true],
  [{{changes:{{}}, split_plan:{{eval_ratio:0.25, seed:7}}}}, true],
  [{{changes:{{'game::a':{{before:'train',after:'eval'}}}}, split_plan:{{eval_ratio:0.25, seed:42}}}}, true],
];
for (const [preview, expected] of cases) {{
  const actual = replanNeedsSave(preview, currentPlan);
  if (actual !== expected) throw new Error(JSON.stringify({{preview, expected, actual}}));
}}
console.log(JSON.stringify({{cases: cases.length + 3}}));
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_draft_helpers_test.js"
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"cases": 8}, json.loads(result.stdout))

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查打包完成回调")
    def test_fast_package_completion_notifies_watchers(self) -> None:
        match = re.search(r"(function watchJob\(jobId, onDone\) \{[\s\S]*?\n\})", SCRIPT)
        self.assertIsNotNone(match)
        node_script = f"""
let JOB_TIMER = null, JOB_ID = null, JOB_LINES = [], JOB_META = null, RB_TIMER = null;
let STATE = {{tab:'train'}};
let RAW_DRAWER = {{analyzeJobId:null}};
let completed = 0;
function registerJobWatch() {{}}
function onTaskFinished(j) {{ if (j.status === 'done') completed += 1; }}
function clearInterval() {{}}
function setInterval() {{ return 1; }}
function paintRawDrawerAnalyze() {{}}
function paintJob() {{}}
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
