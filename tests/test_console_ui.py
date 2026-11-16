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
                'id="pickGo"', 'id="pickPreview"', "会话几乎没有有效内容", "assistant_errors",
                "pick-path", "preview-path", "cwd ${esc(s.project)}",
                "入集：剪枝", "dpo_candidates",
            ),
            "gold": (
                'id="goldProject"', 'id="goldModel"', 'id="goldOutput"', 'id="goldSearch"',
                'data-gold-open="${esc(s.session_id)}"', "goldSetStepGate",
                "goldSetUsage", "goldSetKindUsage", "goldSetSampleDecision",
                "data-gold-split", "session_sft", "session_dpo",
                "停用本会话全部", "SFT 停用", "DPO 停用",
                "usage-badge", "usage-badge sft-off", "usage-badge dpo-off", "col-dest",
            ),
            "train": (
                'id="dsProject"', 'id="dsSplit"', 'id="dsSearch"',
                "loadTrainDraft", "setDraftDecisions", "renderTrainTree",
                "treeGroupHtml", "data-scope-toggle", "data-session-split",
            ),
            "tasks": ("taskRowHtml", "pollTaskLogs", "taskOpenAction"),
            "delivery": ("selectExportProfile", "exportPackage", "downloadZip", "deletePack"),
            "config": ('id="cfgSearch"', "cfgSave", "cfgDiscard"),
        }
        for workspace, snippets in expected.items():
            with self.subTest(workspace=workspace):
                for snippet in snippets:
                    self.assertIn(snippet, HTML)

    def test_train_draft_uses_server_decisions_not_local_file_switches(self) -> None:
        for snippet in (
            'api("/api/dataset/draft', 'function setDraftDecisions(',
            'api("/api/dataset/draft/tree', 'function renderTrainTree(',
            'function treeGroupHtml(', 'function treeSessionHtml(', 'function treeTrajectoryHtml(',
            'function treeKindHtml(', 'function treeSampleHtml(', '黄金会话 → 轨迹 → SFT / DPO',
            'function treeSingletonChildKey(', 'function treeSmartExpand(',
            '.tree-trajectory{', '.tree-kind{',
            'SFT ${fmt(sftTotal)} 条 · 纳入 ${fmt(sftUsed)} · DPO ${fmt(dpoTotal)} 条 · 纳入 ${fmt(dpoUsed)}',
            'function setScopeDecision(', 'data-scope-toggle', 'data-session-split',
            'data-project-split', 'data-sample-toggle', 'function bindTrainTree(',
            'else treeSmartExpand(key)', 'treeSmartExpand("g:" + first.key)',
            'body.dirty', 'id="draftBuild"',
            'trainRequestCurrent', 'syncDraftSelection',
            'pending_evaluations', 'DPO 偏好对',
            'DPO_GATE_ZH', '审核',
            'treeSampleDecisionBits', 'tree-sample-label', 'tree-sample-meta', 'tree-sample-status',
            'tree-action on', 'tree-action off',
            'r.title', 'DPO_SRC_ZH[r.stage]', '内容无效', '坏思考 ', '好思考 ', '同动作',
            '权重 ${esc(String(r.sample_weight))}', '质量 ${esc(String(r.quality_score))}',
            'loadTrainBuilds', 'buildWorkspace', 'build_id',
            'onTaskFinished(j)', 'RUNS_REQUEST_SEQ', 'DRAWER_REQUEST_SEQ',
            'e.target.closest("[data-draft-toggle]")',
            'function goldSetUsage(', 'function goldSetSessionSplit(',
        ):
            self.assertIn(snippet, HTML)
        self.assertNotIn("gt.trainInclude", HTML)
        self.assertNotIn("function setTrainInclude", HTML)
        self.assertNotIn("function renderSplitWorkspace(", HTML)
        self.assertNotIn("预览重切分", HTML)
        self.assertIn(".tree-node{", HTML)
        self.assertIn(".ds-row{display:grid;grid-template-columns:minmax(0,1fr) auto", HTML)
        self.assertIn("function trainTableHead(", HTML)
        self.assertIn("${trainTableHead(kind)}", HTML)

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
            'splitThinkBody(', 'thinkDisplayText(', 'renderDpoPreferenceCard(',
            'Chosen 思考', '共同工具动作', 'pref-block',
            '（无思考）', '双方均无思考内容', '无工具调用', '（无工具）',
            'SFT 窗口要等构建后才有', 'DPO 候选', '评估并构建训练集',
            'function inferDraftSampleKind(', 'inferDraftSampleKind(row, sampleId)',
            'convToMsgs(row.conversations || row.messages)',
            'class="gold-sample-item"', 'id="goldSampleDetail"',
        ):
            self.assertIn(snippet, HTML)
        self.assertNotIn("rParts.think || rejected", HTML)
        self.assertNotIn("cParts.think || chosen", HTML)
        # 思考与动作在同一偏好对照卡片内，不再拆成第二张「动作对照」panel
        self.assertNotIn('<div class="panel"><div class="panel-h"><h3>动作对照</h3>', HTML)
        self.assertNotIn('<div class="panel"><div class="panel-h"><h3>共同工具动作</h3>', HTML)

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
            "delivery-workspace", "delivery-history",
            "packHistHtml(rb)", "新建包", "历史交付包",
        ):
            self.assertIn(snippet, detail_source)
        self.assertNotIn("delivery-format-list", detail_source)
        self.assertNotIn("delivery-current-panel", detail_source)
        self.assertIn("delivery-pack-list", HTML)
        self.assertIn("openPackDocs(", HTML)
        self.assertIn("openPackLengthStats(", HTML)
        self.assertIn("packLenSparkHtml(", HTML)
        self.assertIn("pack-len-plot", HTML)
        self.assertIn("height:72px", HTML)
        self.assertIn("const w = 100, h = 56", HTML)
        self.assertIn("downloadPackWithFormat(", HTML)
        self.assertIn("downloadZip(", HTML)
        self.assertIn("/api/export/download", HTML)
        self.assertIn(">下载</button>", HTML)
        self.assertIn('aria-label="下载格式"', HTML)
        self.assertIn("deletePack(", HTML)
        self.assertIn('method: "DELETE"', HTML)
        self.assertIn("/api/export/pack?name=", HTML)
        self.assertIn(">删除</button>", HTML)
        self.assertIn("/api/export/pack/docs", HTML)
        self.assertIn("/api/export/pack/length-stats", HTML)
        self.assertIn("读取包内长度分布", HTML)
        self.assertIn('openDrawer("长度分布"', HTML)
        self.assertIn("pack-len-chart", HTML)
        self.assertIn("pack-len-ruler", HTML)
        self.assertIn("packLenRulerTicks(", HTML)
        self.assertIn("packLenLinePoints(", HTML)
        self.assertIn("line-sft", HTML)
        self.assertIn("line-dpo", HTML)
        self.assertIn("packTokMark(", HTML)
        self.assertIn("packTok(", HTML)
        self.assertNotIn("pack-len-pair", HTML)
        self.assertNotIn("packLenSeriesChart(", HTML)
        # 主操作不再单独放「长度分布」按钮；卡片用密柱图 + 标尺，点图看明细
        self.assertNotIn(">长度分布</button>", HTML)
        self.assertNotIn("SFT train 长度", HTML)
        self.assertNotIn("全部数据集", HTML)
        self.assertNotIn("packLenBinShort(", HTML)
        self.assertIn("delivery-workspace{display:flex;flex-direction:column", HTML)
        # 构建与打包分离：训练页构建，交付页才出 zip
        self.assertIn('["交付",', HTML)
        self.assertIn("评估并构建", HTML)
        self.assertIn("仅 DPO 评估", HTML)
        self.assertIn("去交付打包", HTML)
        self.assertNotIn("生成训练文件", HTML)
        self.assertNotIn("去交付中心打包", HTML)
        self.assertNotIn("梳理与交付", HTML)
        self.assertIn("可到交付页打包", HTML)

        load_match = re.search(r"async function loadRuns\(\) \{([\s\S]*?)function stopPackTimer", HTML)
        self.assertIsNotNone(load_match)
        self.assertNotIn('api("/api/runs")', load_match.group(1))
        self.assertIn('api("/api/export/history")', load_match.group(1))
        # 交付页只服务 zip 历史包；发布/激活/回滚已从 UI 移除（API 仍保留给脚本）
        self.assertNotIn('api("/api/releases")', load_match.group(1))
        self.assertNotIn("releasePanelHtml()", detail_source)
        for snippet in ("发布当前 Build", "回滚到上一版本", "迁入 Build", "function activateRelease", "发布版本"):
            self.assertNotIn(snippet, HTML)

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

    @unittest.skipUnless(shutil.which("node"), "需要 Node.js 检查训练树智能展开")
    def test_train_tree_smart_expand_follows_singleton_children(self) -> None:
        helpers = re.search(
            r"(function treeSingletonChildKey\(key\) \{[\s\S]*?)(?=\nfunction treeSampleHtml\()",
            SCRIPT,
        )
        self.assertIsNotNone(helpers)
        node_script = f"""
let STATE = {{train: {{
  expanded: {{}},
  treeData: {{
    groups: [{{
      key: 'game::solo',
      sessions: [{{
        session_id: 'sess-1',
        trajectories: [{{
          trajectory_id: 'traj::sess-1',
          kinds: [
            {{kind: 'sft', samples: [{{sample_id: 'sft-1'}}]}},
            {{kind: 'dpo', samples: [{{sample_id: 'dpo-1'}}]}},
          ],
        }}],
      }}],
    }}, {{
      key: 'game::multi',
      sessions: [
        {{session_id: 'a', trajectories: [{{trajectory_id: 'traj::a', kinds: [{{kind: 'sft', samples: []}}]}}]}},
        {{session_id: 'b', trajectories: [{{trajectory_id: 'traj::b', kinds: [{{kind: 'sft', samples: []}}]}}]}},
      ],
    }}],
  }},
}}}};
{helpers.group(1)}
treeSmartExpand('g:game::solo');
const solo = STATE.train.expanded;
if (!solo['g:game::solo'] || !solo['s:sess-1'] || !solo['t:traj::sess-1']) {{
  throw new Error('singleton chain not expanded: ' + JSON.stringify(solo));
}}
if (solo['k:traj::sess-1:sft'] || solo['k:traj::sess-1:dpo']) {{
  throw new Error('should stop when trajectory has two kinds: ' + JSON.stringify(solo));
}}
STATE.train.expanded = {{}};
treeSmartExpand('g:game::multi');
const multi = STATE.train.expanded;
if (!multi['g:game::multi'] || multi['s:a'] || multi['s:b']) {{
  throw new Error('multi-session group should not cascade: ' + JSON.stringify(multi));
}}
STATE.train.treeData.groups[0].sessions[0].trajectories[0].kinds = [{{kind: 'sft', samples: [{{sample_id: 'only'}}]}}];
STATE.train.expanded = {{}};
treeSmartExpand('g:game::solo');
const oneKind = STATE.train.expanded;
if (!oneKind['k:traj::sess-1:sft']) {{
  throw new Error('single kind should auto-expand: ' + JSON.stringify(oneKind));
}}
console.log(JSON.stringify({{ok: true}}));
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_tree_smart_expand_test.js"
            path.write_text(node_script, encoding="utf-8")
            result = subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
        self.assertEqual({"ok": True}, json.loads(result.stdout))

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
const JOB_ACTIVE = st => st === "running" || st === "queued";
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
