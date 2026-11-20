"""会话抽屉内联 DPO 偏好对查看器（只读）回归测试。

用 Node 执行 web/index.html 中真实的前端函数，覆盖：
- 「DPO 偏好对」页签始终可用（CAS dpo_count 为 0 也可点），且不把 CAS 候选数当查看器总数
- 懒加载：只有切到 DPO 页签才请求 /train?kind=dpo&session_id=...&offset=N&limit=1
- 工作区「上次生成」/ 历史包「包内样本」标签
- 一次只展示一对、总数与上一对/下一对、页码夹取
- 加载 / 空 / 失败 + 重试状态
- 过期响应不得覆盖模式切换、另一会话、关闭抽屉、新抽屉、后续页
- chosen/rejected 的 value / content / string 三种形态与 HTML 转义
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

VIEWER_BLOCK = re.compile(
    r"(let _VDS_DPO_VIEW = null;[\s\S]*?)\nasync function openDatasetSessionDrawer"
)
DPO_BUTTON = re.compile(r'<button type="button" data-sm="dpo"[^>]*>DPO 偏好对</button>')


def _extract_function(name: str) -> str:
    pattern = re.compile(
        r"\nfunction %s\([\s\S]*?\n(?=(?:\n)*(?:function |const |let |var |/\*|document\.|// ))"
        % re.escape(name)
    )
    match = pattern.search(HTML)
    assert match is not None, f"未找到前端函数 {name}"
    return match.group(0)


def _require_node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("需要 node 执行前端真实函数")
    return node


def _run_node(source: str, *args: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        js_file = Path(tmp) / "harness.js"
        js_file.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [_require_node(), str(js_file), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])


VIEWER_PRELUDE = r"""
const esc = v => String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmt = v => String(v == null ? 0 : v);
const detailCalls = [];
function renderGoldSampleDetail(row, kind) {
  detailCalls.push({ kind, raw_row: row });
  return `<div id="stub-detail" data-kind="${esc(kind)}">${esc((row && row.sample_id) || "")}</div>`;
}
function renderMessages(msgs) { return `<div class="stub-messages">${(msgs || []).length}</div>`; }
function renderLineageHtml() { return `<div class="stub-lineage"></div>`; }
function bindLineageEvents() {}
function short(s, n) { return String(s || "").slice(0, n); }
let _VDS_SESSION_CACHE = null;
let _VDS_SESSION_MODE = "gold";
let DRAWER_REQUEST_SEQ = 7;
const drawerEl = { open: true, classList: { contains: c => c === "open" && drawerEl.open } };
let hostEl = null;
let bodyEl = null;
const $ = sel => sel === "#vdsDpoViewer" ? hostEl
  : sel === "#drawer" ? drawerEl
  : sel === "#vdsDrawerLineageBody" ? bodyEl
  : null;
const $$ = () => [];
const calls = { urls: [] };
let apiImpl = async () => ({ total: 0, rows: [] });
async function api(url) { calls.urls.push(url); return apiImpl(url); }
const detailSnapshot = () => detailCalls.map(d => ({ kind: d.kind, sample_id: (d.raw_row || {}).sample_id || "" }));
"""

VIEWER_DRIVER = r"""
const results = {};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};
function setup(opts = {}) {
  const o = Object.assign({ dataset_id: "workspace", session_id: "sess-full-1", mode: "dpo", open: true }, opts);
  _VDS_SESSION_CACHE = { dataset_id: o.dataset_id, session_id: o.session_id };
  _VDS_SESSION_MODE = o.mode;
  drawerEl.open = o.open;
  hostEl = { innerHTML: "" };
  bodyEl = { innerHTML: "" };
  resetVdsDpoView();
  calls.urls.length = 0;
  detailCalls.length = 0;
  apiImpl = async () => ({ total: 0, rows: [] });
}
function snapshot(extra) {
  return Object.assign({
    urls: calls.urls.slice(),
    html: hostEl.innerHTML,
    body: bodyEl.innerHTML,
    details: detailSnapshot(),
  }, extra || {});
}
const offsetOf = url => Number(new URL(url, "http://x").searchParams.get("offset"));

async function main() {
  // 1. 工作区首屏：单对 + 总数 + 懒加载请求
  setup();
  apiImpl = async () => ({ total: 3, rows: [{ kind: "dpo", raw_row: { sample_id: "sample-1" } }] });
  const v1 = openVdsDpoView("workspace");
  await v1.pending;
  results.workspace_first = snapshot({ offset: v1.offset, total: v1.total });

  // 2. 历史包：包内样本 + 下一对
  setup({ dataset_id: "pack-abc123", session_id: "sess-pack-9" });
  apiImpl = async url => ({ total: 3, rows: [{ raw_row: { sample_id: `sample-${offsetOf(url)}` } }] });
  const v2 = openVdsDpoView("pack-abc123");
  await v2.pending;
  const beforeNext = snapshot({ offset: v2.offset });
  await vdsDpoNav(1);
  results.pack_next = Object.assign(snapshot({ offset: v2.offset }), { beforeNext });

  // 3. 页码夹取：已知总数时越界 offset 被夹到末页；总数变少时回夹并重取
  setup();
  let total = 3;
  apiImpl = async url => {
    const off = offsetOf(url);
    return { total, rows: off < total ? [{ raw_row: { sample_id: `page-${off}` } }] : [] };
  };
  const v3 = openVdsDpoView("workspace");
  await v3.pending;
  await loadVdsDpoPage(v3, 99);
  const clamped = snapshot({ offset: v3.offset });
  total = 1;
  await loadVdsDpoPage(v3, 2);
  results.clamp = { clamped, shrunk: snapshot({ offset: v3.offset, total: v3.total }) };

  // 3b. 边界：首页上一对 / 末页下一对都不再发请求
  setup();
  apiImpl = async url => ({ total: 3, rows: [{ raw_row: { sample_id: `page-${offsetOf(url)}` } }] });
  const v3b = openVdsDpoView("workspace");
  await v3b.pending;
  const urlsAfterFirst = calls.urls.slice();
  vdsDpoNav(-1);
  const urlsAfterPrev = calls.urls.slice();
  await loadVdsDpoPage(v3b, 2);
  const urlsAtLast = calls.urls.slice();
  vdsDpoNav(1);
  results.nav_edges = { urlsAfterFirst, urlsAfterPrev, urlsAtLast, urlsAfterLastNext: calls.urls.slice() };

  // 4. 空态
  setup();
  apiImpl = async () => ({ total: 0, rows: [] });
  const v4 = openVdsDpoView("workspace");
  await v4.pending;
  results.empty = snapshot({ total: v4.total });

  // 5. 失败 + 重试（错误消息中的标签必须转义）
  setup();
  apiImpl = async () => { throw new Error("boom <script>alert(1)</script>"); };
  const v5 = openVdsDpoView("workspace");
  await v5.pending;
  const failed = snapshot();
  apiImpl = async () => ({ total: 2, rows: [{ raw_row: { sample_id: "retry-ok" } }] });
  await vdsDpoRetry();
  results.error_retry = { failed, retried: snapshot({ offset: v5.offset }) };

  // 6. 同一会话：后发页码先返回，旧页响应不得覆盖
  setup();
  const gates = {};
  apiImpl = url => { const off = offsetOf(url); gates[off] = deferred(); return gates[off].promise; };
  const v6 = openVdsDpoView("workspace");
  const p0 = v6.pending;
  const p1 = loadVdsDpoPage(v6, 1);
  gates["1"].resolve({ total: 2, rows: [{ raw_row: { sample_id: "page-1" } }] });
  await p1;
  gates["0"].resolve({ total: 2, rows: [{ raw_row: { sample_id: "page-0" } }] });
  await p0;
  results.stale_page = snapshot({ offset: v6.offset });

  // 7. 模式切换到血缘后，迟到的 DPO 响应不得写回
  setup();
  let gate = deferred();
  apiImpl = () => gate.promise;
  const v7 = openVdsDpoView("workspace");
  resetVdsDpoView();
  _VDS_SESSION_MODE = "lineage";
  hostEl.innerHTML = "MODE-SWITCHED";
  gate.resolve({ total: 5, rows: [{ raw_row: { sample_id: "late" } }] });
  await v7.pending;
  results.stale_mode = snapshot({ offset: v7.offset });

  // 8. 会话已切换
  setup();
  gate = deferred();
  apiImpl = () => gate.promise;
  const v8 = openVdsDpoView("workspace");
  _VDS_SESSION_CACHE = { dataset_id: "workspace", session_id: "other-session" };
  hostEl.innerHTML = "OTHER-SESSION";
  gate.resolve({ total: 5, rows: [{ raw_row: { sample_id: "late" } }] });
  await v8.pending;
  results.stale_session = snapshot({ offset: v8.offset });

  // 9. 抽屉已关闭
  setup();
  gate = deferred();
  apiImpl = () => gate.promise;
  const v9 = openVdsDpoView("workspace");
  drawerEl.open = false;
  hostEl.innerHTML = "DRAWER-CLOSED";
  gate.resolve({ total: 5, rows: [{ raw_row: { sample_id: "late" } }] });
  await v9.pending;
  results.stale_closed = snapshot();

  // 10. 打开另一个抽屉
  setup();
  gate = deferred();
  apiImpl = () => gate.promise;
  const v10 = openVdsDpoView("workspace");
  DRAWER_REQUEST_SEQ += 1;
  hostEl.innerHTML = "NEW-DRAWER";
  gate.resolve({ total: 5, rows: [{ raw_row: { sample_id: "late" } }] });
  await v10.pending;
  results.stale_drawer = snapshot();

  // 11. 页签懒加载：黄金模式不发 DPO 请求，点 DPO 页签才发
  setup({ session_id: "sess-tab" });
  _VDS_SESSION_CACHE = {
    dataset_id: "workspace",
    session_id: "sess-tab",
    gold_messages: [{ role: "assistant" }],
    raw_messages: [],
    steps: [],
    turns: [],
  };
  apiImpl = async () => ({ total: 2, rows: [{ raw_row: { sample_id: "tab-sample" } }] });
  switchVdsSessionMode("gold");
  const urlsAfterGold = calls.urls.slice();
  switchVdsSessionMode("dpo");
  await _VDS_DPO_VIEW.pending;
  results.lazy_switch = Object.assign(snapshot({ offset: _VDS_DPO_VIEW.offset }), { urlsAfterGold });

  console.log(JSON.stringify(results));
}

main().then(() => {}, err => { console.error((err && err.stack) || err); process.exit(1); });
"""


@pytest.fixture(scope="module")
def viewer() -> dict:
    match = VIEWER_BLOCK.search(HTML)
    assert match is not None, "未找到会话抽屉 DPO 查看器函数"
    functions = match.group(1)
    source = VIEWER_PRELUDE + "\n" + functions + "\n" + VIEWER_DRIVER
    return _run_node(source)


RENDER_PRELUDE = r"""
const esc = v => String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmt = v => String(v == null ? 0 : v);
const PHASE = [["highlight", "高光"], ["opening", "开局"], ["routine", "常规推进"]];
const DPO_SRC_ZH = { stage2_pruned: "弯路", negative_step: "坏步", review_rewrite: "改写" };
const ctxCalls = [];
function renderMessages(msgs) {
  ctxCalls.push(msgs);
  return `<div id="stub-ctx" data-n="${(msgs || []).length}"></div>`;
}
"""

RENDER_DRIVER = r"""
const cases = JSON.parse(process.argv[3] || "{}");
const out = {};
for (const [name, row] of Object.entries(cases)) {
  ctxCalls.length = 0;
  const html = renderGoldSampleDetail(row, "dpo");
  out[name] = {
    html,
    ctx: ctxCalls.map(ms => (ms || []).map(m => ({ role: m.role, content: m.content, text: m.text }))),
    pref: {
      value: goldPrefText({ value: "v" }),
      content: goldPrefText({ content: "c" }),
      string: goldPrefText("s"),
      empty: goldPrefText(null),
      unknown: goldPrefText({ other: "x" }),
      dup: goldPrefText({ value: "v", content: "c" }),
    },
  };
}
console.log(JSON.stringify(out));
"""

RENDER_CASES = {
    "value_forms": {
        "sample_id": "s-value",
        "chosen": {"value": "value-chosen"},
        "rejected": {"value": "value-rejected"},
        "conversations": [
            {"from": "human", "value": "ctx-question"},
            {"from": "gpt", "value": "ctx-answer"},
        ],
    },
    "content_forms": {
        "sample_id": "s-content",
        "chosen": {"content": "content-chosen"},
        "rejected": {"content": "content-rejected"},
        "conversations": [
            {"from": "human", "content": "ctx-content-question"},
            {"from": "gpt", "content": "ctx-content-answer"},
        ],
    },
    "string_forms": {
        "sample_id": "s-string",
        "chosen": "string-chosen",
        "rejected": "string-rejected",
        "messages": [
            {"role": "user", "content": "ctx-messages-question"},
            {"role": "assistant", "content": "ctx-messages-answer"},
        ],
    },
    # value 与 content 同时存在时只能渲染一次，避免重复
    "no_duplicate": {
        "chosen": {"value": "only-once", "content": "only-once"},
        "rejected": {"value": "reject-once", "content": "reject-once"},
    },
    "hostile": {
        # 共同工具动作走单份 <pre> + esc，可用连续子串验证转义
        "chosen": '<think>same</think><tool_call>{"name":"<script>bad()</script><img src=x onerror=alert(1)>"}</tool_call>',
        "rejected": '<think>same</think><tool_call>{"name":"<script>bad()</script><img src=x onerror=alert(1)>"}</tool_call>',
        "conversations": [{"from": "human", "value": "<b>ctx</b>"}],
    },
}


@pytest.fixture(scope="module")
def rendered() -> dict:
    parts = [
        RENDER_PRELUDE,
        _extract_function("lcsDiff"),
        _extract_function("tokenPair"),
        _extract_function("splitThinkBody"),
        _extract_function("thinkDisplayText"),
        _extract_function("renderDpoPreferenceCard"),
        _extract_function("buildDiff"),
        _extract_function("goldPrefText"),
        _extract_function("convToMsgs"),
        _extract_function("renderGoldSampleDetail"),
        RENDER_DRIVER,
    ]
    return _run_node("\n".join(parts), "unused", json.dumps(RENDER_CASES, ensure_ascii=False))


def test_dpo_tab_is_always_available_and_never_labelled_with_cas_count() -> None:
    button = DPO_BUTTON.search(HTML)
    assert button is not None, "会话抽屉缺少 DPO 偏好对页签"
    assert "disabled" not in button.group(0)
    assert "switchVdsSessionMode('dpo')" in button.group(0)
    # CAS 的 dpo_count 是 Stage 2 候选数，不能当作查看器总数
    assert "DPO 候选" in HTML
    assert '· DPO ${fmt(lin.dpo_count || 0)} 对' not in HTML
    match = VIEWER_BLOCK.search(HTML)
    assert match is not None
    block = match.group(1)
    assert "dpo_count" not in block
    assert "res.total" in block
    # 查看器与页签都按数据集类型使用「上次生成 / 包内样本」
    assert '"上次生成" : "包内样本"' in block
    assert "DPO 偏好对" in HTML
    assert "上一对" in HTML and "下一对" in HTML


def test_workspace_and_pack_scope_labels(viewer: dict) -> None:
    assert "上次生成" in viewer["workspace_first"]["html"]
    assert "包内样本" in viewer["pack_next"]["beforeNext"]["html"]
    assert "上次生成" not in viewer["pack_next"]["beforeNext"]["html"]


def test_clicking_dpo_tab_lazily_requests_exact_session_page(viewer: dict) -> None:
    lazy = viewer["lazy_switch"]
    assert lazy["urlsAfterGold"] == []
    assert lazy["urls"] == [
        "/api/datasets/workspace/train?kind=dpo&session_id=sess-tab&offset=0&limit=1"
    ]
    assert '<div id="vdsDpoViewer">' in lazy["body"]

    assert viewer["workspace_first"]["urls"] == [
        "/api/datasets/workspace/train?kind=dpo&session_id=sess-full-1&offset=0&limit=1"
    ]
    pack_urls = viewer["pack_next"]["urls"]
    assert pack_urls[0] == (
        "/api/datasets/pack-abc123/train?kind=dpo&session_id=sess-pack-9&offset=0&limit=1"
    )
    assert pack_urls[1] == (
        "/api/datasets/pack-abc123/train?kind=dpo&session_id=sess-pack-9&offset=1&limit=1"
    )
    # 只带精确会话筛选，不夹带 split / project / q 等其它筛选
    for url in viewer["workspace_first"]["urls"] + pack_urls:
        query = url.split("?", 1)[1]
        keys = [part.split("=", 1)[0] for part in query.split("&")]
        assert keys == ["kind", "session_id", "offset", "limit"]
        assert "session_id=sess-pack-9" in url or "session_id=sess-full-1" in url or "session_id=sess-tab" in url


def test_presents_one_pair_with_total_and_navigation(viewer: dict) -> None:
    first = viewer["workspace_first"]
    assert "第 <b>1</b> / 3 对" in first["html"]
    assert 'id="vdsDpoPrev" disabled' in first["html"]
    assert 'id="vdsDpoNext" disabled' not in first["html"]
    assert first["details"] == [{"kind": "dpo", "sample_id": "sample-1"}]
    assert '<div id="stub-detail" data-kind="dpo">sample-1</div>' in first["html"]

    after_next = viewer["pack_next"]
    assert after_next["beforeNext"]["offset"] == 0
    assert after_next["offset"] == 1
    assert "第 <b>2</b> / 3 对" in after_next["html"]
    assert after_next["details"][-1] == {"kind": "dpo", "sample_id": "sample-1"}
    # 一次只渲染当前一对，不把整页堆在一起
    assert after_next["html"].count('id="stub-detail"') == 1


def test_page_offsets_are_clamped_at_boundaries_and_after_shrink(viewer: dict) -> None:
    clamped = viewer["clamp"]["clamped"]
    assert clamped["offset"] == 2
    assert all("offset=99" not in url for url in clamped["urls"])
    assert clamped["urls"][-1].endswith("offset=2&limit=1")

    shrunk = viewer["clamp"]["shrunk"]
    assert shrunk["offset"] == 0
    assert shrunk["total"] == 1
    assert "第 <b>1</b> / 1 对" in shrunk["html"]
    assert shrunk["urls"][-1].endswith("offset=0&limit=1")
    assert shrunk["details"][-1] == {"kind": "dpo", "sample_id": "page-0"}

    edges = viewer["nav_edges"]
    assert edges["urlsAfterPrev"] == edges["urlsAfterFirst"]
    assert edges["urlsAfterLastNext"] == edges["urlsAtLast"]


def test_empty_and_error_states_offer_retry(viewer: dict) -> None:
    empty = viewer["empty"]
    assert empty["total"] == 0
    assert "没有 DPO 偏好对" in empty["html"]
    assert 'id="vdsDpoPrev" disabled' in empty["html"]
    assert 'id="vdsDpoNext" disabled' in empty["html"]
    assert "重新读取" in empty["html"]
    assert "stub-detail" not in empty["html"]

    failed = viewer["error_retry"]["failed"]
    assert "读取上次生成的 DPO 偏好对失败" in failed["html"]
    assert "&lt;script&gt;" in failed["html"]
    assert "<script>alert(1)</script>" not in failed["html"]
    assert "重试" in failed["html"]
    assert "stub-detail" not in failed["html"]

    retried = viewer["error_retry"]["retried"]
    assert retried["details"][-1] == {"kind": "dpo", "sample_id": "retry-ok"}
    assert len(retried["urls"]) == 2
    assert retried["urls"][0] == retried["urls"][1]


def test_stale_responses_do_not_overwrite_current_view(viewer: dict) -> None:
    stale_page = viewer["stale_page"]
    assert stale_page["offset"] == 1
    assert "page-1" in stale_page["html"]
    assert "page-0" not in stale_page["html"]

    assert viewer["stale_mode"]["html"] == "MODE-SWITCHED"
    assert viewer["stale_session"]["html"] == "OTHER-SESSION"
    assert viewer["stale_closed"]["html"] == "DRAWER-CLOSED"
    assert viewer["stale_drawer"]["html"] == "NEW-DRAWER"
    for key in ("stale_mode", "stale_session", "stale_closed", "stale_drawer"):
        assert "stub-detail" not in viewer[key]["html"]


def test_viewer_is_read_only(viewer: dict) -> None:
    match = VIEWER_BLOCK.search(HTML)
    assert match is not None
    block = match.group(1)
    for forbidden in ("method:", "dataset/draft", "toggleVdsSessionSft", "DELETE", '"PUT"', '"POST"'):
        assert forbidden not in block
    # 查看器只有翻页与重试按钮，没有写入/提交类动作
    for html in (
        viewer["workspace_first"]["html"],
        viewer["empty"]["html"],
        viewer["error_retry"]["failed"]["html"],
    ):
        assert "onclick=\"toggle" not in html
        assert "onclick=\"vdsDpoNav" in html or "onclick=\"vdsDpoRetry" in html


def test_chosen_and_rejected_support_value_content_and_string_forms(rendered: dict) -> None:
    value = rendered["value_forms"]
    assert "value-chosen" in value["html"]
    assert "value-rejected" in value["html"]
    contents = [m.get("content") for m in value["ctx"][0]]
    assert "ctx-question" in contents
    assert any(m.get("text") == "ctx-answer" for m in value["ctx"][0])

    content = rendered["content_forms"]
    assert "content-chosen" in content["html"]
    assert "content-rejected" in content["html"]
    contents = [m.get("content") for m in content["ctx"][0]]
    assert "ctx-content-question" in contents
    assert any(m.get("text") == "ctx-content-answer" for m in content["ctx"][0])

    strings = rendered["string_forms"]
    assert "string-chosen" in strings["html"]
    assert "string-rejected" in strings["html"]
    contents = [m.get("content") for m in strings["ctx"][0]]
    assert "ctx-messages-question" in contents
    assert any(m.get("text") == "ctx-messages-answer" for m in strings["ctx"][0])

    pref = strings["pref"]
    assert pref == {
        "value": "v",
        "content": "c",
        "string": "s",
        "empty": "",
        "unknown": "",
        # value 与 content 同时存在时只取 value，不拼接、不重复
        "dup": "v",
    }


def test_preference_text_is_not_duplicated_and_is_html_escaped(rendered: dict) -> None:
    once = rendered["no_duplicate"]["html"]
    # value 与 content 相同时不会拼接出第二份文本（同一文本可能在思考/动作两块各显示一次）
    assert once.count("only-once") <= 2
    assert "only-onceonly-once" not in once
    assert once.count("reject-once") <= 2
    assert "reject-oncereject-once" not in once

    hostile = rendered["hostile"]["html"]
    assert "&lt;script&gt;bad()&lt;/script&gt;" in hostile
    assert "&lt;img src=x onerror=alert(1)&gt;" in hostile
    assert "<script>bad()" not in hostile
    assert "<img src=x" not in hostile