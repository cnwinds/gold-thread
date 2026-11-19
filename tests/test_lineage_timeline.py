"""数据集演进时间轴（分支泳道 + 横向 journal + 共享刻度轴）前端回归测试。

用 Node 执行 web/index.html 中真实的时间轴函数，覆盖：
- 一个版本一列、按真实 generated_at 时间升序，同时间戳按 id 稳定排序
- 缺失 / 非法时间戳显示「时间未知」并稳定排在有效时间之后；草稿显示「未发布」
- 空图谱 / dirty 工作区；head_pack 有效时即当前版本，失效时回退到最新有效版本
- 只渲染后端给出的真实 incoming edge，分支与孤立节点都保留，时间相邻不自动连边
- 初始挂载 / 回到当前：head 卡片在横向视口内（居中并夹取），不依赖 scrollIntoView
- 用户手动滚动偏移在刷新 / 重渲染后保留；用户选择卡片 / 关系时按需露出
- 聚焦项被删除（含草稿消失、边消失）时确定性回退到当前版本
- 分支泳道：同一父包的两个兄弟分支各占一行、HEAD 祖先链留在主泳道、后代留在自己分支
- 桥（真实 SVG 连线 + 列间桥标签）的像素几何：始终在父卡片与子卡片之间、落在子节点泳道行、
  线路不穿过任何无关卡片；倒挂时间戳 / 损坏环也有确定性渲染
- 手机宽度下选中关系时，桥标签与子卡片同时保持可见
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

TIMELINE_BLOCK = re.compile(
    r"/\* ===== VDS_TIMELINE_START ===== \*/([\s\S]*?)/\* ===== VDS_TIMELINE_END ===== \*/"
)

HARNESS_PRELUDE = r"""
const esc = v => String(v == null ? "" : v)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmt = v => (v === undefined || v === null) ? "–" : String(v);
const jsArg = v => JSON.stringify(v);
const STATE = { vds: { graph: null, focusType: "dataset", selectedId: "", selectedEdge: "", mounted: false, scrollLeft: null } };
const CANVAS_RECT = { left: 40, top: 120, width: 600, height: 400 };
const dagCanvas = {
  innerHTML: "", clientWidth: 600, scrollLeft: 0, scrollWidth: 2000,
  clientLeft: 0, clientTop: 0,
  getBoundingClientRect() { return Object.assign({}, CANVAS_RECT, { right: 640, bottom: 520 }); },
  addEventListener() {},
};
// 真实布局里卡片 offsetLeft 相对的是定位的 .dag-grid（通常为 0），
// 内容坐标必须通过 getBoundingClientRect 与滚动容器的位置/滚动量换算。
function makeJournal(html) {
  const ids = [...html.matchAll(/data-vds-id="([^"]+)"/g)].map(m => m[1]);
  return {
    querySelectorAll() {
      return ids.map((id, i) => {
        const contentX = 20 + i * 250;
        return {
          dataset: { vdsId: id },
          offsetLeft: 0,
          offsetWidth: 228,
          getBoundingClientRect() {
            const left = CANVAS_RECT.left + dagCanvas.clientLeft
              + contentX - (Number(dagCanvas.scrollLeft) || 0);
            return { left, top: CANVAS_RECT.top, width: 228, height: 90, right: left + 228,
              bottom: CANVAS_RECT.top + 90 };
          },
        };
      });
    },
  };
}
const $ = sel => sel === "#vdsDagCanvas" ? dagCanvas
  : sel === "#vdsDagJournal" ? makeJournal(dagCanvas.innerHTML)
  : null;
function renderVdsDetailHost() { return Promise.resolve(); }
function toast() {}
"""

HARNESS_DRIVER = r"""
const fx = JSON.parse(process.argv[2]);
const out = { graphs: {}, scroll: {}, fallback: {} };

const cardsOf = html => [...html.matchAll(/data-vds-id="([^"]+)"/g)].map(m => m[1]);
const decode = s => String(s).replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&amp;/g, "&");
const relsOf = html => [...html.matchAll(/data-edge-id="([^"]+)"[^>]*data-edge-from="([^"]+)"/g)]
  .map(m => ({ id: decode(m[1]), child: decode(m[1]).split("->").pop(), from: decode(m[2]) }));
const datesOf = html => [...html.matchAll(/<(?:time|div) class="(dag-date[^"]*)"[^>]*>([\s\S]*?)<\/(?:time|div)>/g)]
  .map(m => ({ cls: m[1], text: m[2].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() }));
const classMatch = (html, cls) => {
  const re = new RegExp(`class="vds-node[^"]*${cls}[^"]*"[^>]*data-vds-id="([^"]+)"`);
  const m = re.exec(html);
  return m ? m[1] : "";
};

for (const [name, graph] of Object.entries(fx.graphs)) {
  STATE.vds.graph = graph;
  STATE.vds.focusType = "dataset";
  STATE.vds.selectedId = vdsCurrentId(graph);
  const html = renderDatasetDagHtml(graph);
  out.graphs[name] = {
    order: vdsTimelineEntries(graph).map(e => e.id),
    current: vdsCurrentId(graph),
    cards: cardsOf(html),
    relations: relsOf(html),
    dates: datesOf(html),
    currentCard: classMatch(html, "is-current"),
    selectedCard: classMatch(html, "selected"),
    hasUnpublished: html.includes("未发布"),
    hasUnknown: html.includes("时间未知"),
    hasNaN: html.includes("NaN"),
    unicodePackage: html.includes("📦"),
  };
}

const graph = fx.scrollGraph;
const visible = (id) => {
  const card = vdsFindTimelineCard($("#vdsDagJournal"), id);
  if (!card) return false;
  const rect = card.getBoundingClientRect();
  const canvasRect = dagCanvas.getBoundingClientRect();
  const contentX = rect.left - canvasRect.left - dagCanvas.clientLeft
    + (Number(dagCanvas.scrollLeft) || 0);
  return contentX >= dagCanvas.scrollLeft
    && contentX + rect.width <= dagCanvas.scrollLeft + dagCanvas.clientWidth;
};

// 首次挂载：选中当前 head，并横向居中且完整可见。
STATE.vds.graph = graph;
STATE.vds.focusType = "dataset";
STATE.vds.selectedId = vdsCurrentId(graph);
STATE.vds.mounted = false;
STATE.vds.scrollLeft = null;
dagCanvas.scrollLeft = 0;
dagCanvas.innerHTML = renderDatasetDagHtml(graph);
vdsBindTimelineScroll();
const initialScroll = vdsRevealTimeline({ center: true });
out.scroll.initial = {
  id: STATE.vds.selectedId,
  scrollLeft: initialScroll,
  currentId: vdsCurrentId(graph),
  visible: visible("pack-b"),
};

// 刷新 / 重渲染：保留用户手动滚动偏移，不回到 HEAD 或左端。
dagCanvas.scrollLeft = 900;
STATE.vds.scrollLeft = 640;
dagCanvas.innerHTML = renderDatasetDagHtml(graph);
out.scroll.retained = {
  scrollLeft: vdsRevealTimeline({}),
  focused: STATE.vds.selectedId,
  focusType: STATE.vds.focusType,
};

// 用户选择历史版本：按需露出，不做居中回跳。
dagCanvas.scrollLeft = 0;
STATE.vds.scrollLeft = 0;
selectVirtualDataset("pack-d");
out.scroll.selected = {
  id: STATE.vds.selectedId,
  scrollLeft: dagCanvas.scrollLeft,
  visible: visible("pack-d"),
};

// 用户选择关系：露出该边的子版本卡片。
selectLineageEdge("pack-b->pack-c");
out.scroll.edge = {
  edge: STATE.vds.selectedEdge,
  focusType: STATE.vds.focusType,
  scrollLeft: dagCanvas.scrollLeft,
  visible: visible("pack-c"),
};

// 回到当前：重新选中 head 并居中。
dagCanvas.scrollLeft = 1600;
STATE.vds.scrollLeft = 1600;
const backId = focusCurrentDataset();
out.scroll.back = {
  id: backId,
  focusType: STATE.vds.focusType,
  scrollLeft: dagCanvas.scrollLeft,
  visible: visible(backId),
};

// 失效聚焦回退。
STATE.vds.focusType = "dataset";
STATE.vds.selectedId = "gone";
out.fallback.removed = {
  changed: vdsSyncSelection(fx.graphs.branch),
  id: STATE.vds.selectedId,
  current: vdsCurrentId(fx.graphs.branch),
};

const dirty = fx.graphs.dirty_draft;
const clean = Object.assign({}, dirty, {
  workspace: Object.assign({}, dirty.workspace, { dirty: false, head_pack: "head" }),
  edges: (dirty.edges || []).filter(e => e.to !== "workspace"),
});
STATE.vds.graph = dirty;
STATE.vds.focusType = "dataset";
STATE.vds.selectedId = "workspace";
out.fallback.draftGone = {
  changed: vdsSyncSelection(clean),
  id: STATE.vds.selectedId,
  current: vdsCurrentId(clean),
};

STATE.vds.focusType = "edge";
STATE.vds.selectedEdge = "missing->edge";
out.fallback.edgeGone = {
  changed: vdsSyncSelection(fx.graphs.branch),
  focusType: STATE.vds.focusType,
  id: STATE.vds.selectedId,
};

const scroller = (scrollWidth, scrollLeft = 0) => ({
  clientWidth: 600, clientLeft: 0, scrollLeft, scrollWidth,
  getBoundingClientRect() { return Object.assign({}, CANVAS_RECT, { right: 640, bottom: 520 }); },
});
const cardAt = (contentX, width, scrollLeft = 0) => {
  const left = CANVAS_RECT.left + contentX - scrollLeft;
  return {
    offsetLeft: 0,
    offsetWidth: width,
    getBoundingClientRect() {
      return { left, top: CANVAS_RECT.top, width, height: 90, right: left + width,
        bottom: CANVAS_RECT.top + 90 };
    },
  };
};
out.scroll.math = {
  center: vdsScrollCardIntoView(scroller(2000), cardAt(900, 240), { center: true }),
  centerScrolled: vdsScrollCardIntoView(scroller(2000, 400), cardAt(900, 240, 400), { center: true }),
  leftClamp: vdsScrollCardIntoView(scroller(2000), cardAt(20, 228), { center: true }),
  rightClamp: vdsScrollCardIntoView(scroller(2060), cardAt(1800, 240), { center: true }),
  nearest: vdsScrollCardIntoView(scroller(2000), cardAt(770, 228), { center: false }),
  nearestScrolled: vdsScrollCardIntoView(scroller(2000, 200), cardAt(770, 228, 200), { center: false }),
};

console.log(JSON.stringify(out));
"""


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("需要 node 执行前端真实函数")
    return node


def _fixtures() -> dict:
    graph = {
        "nodes": [
            {"id": "b", "generated_at": "2024-01-02T10:00:00", "projects_count": 2, "sessions_count": 4},
            {"id": "a", "generated_at": "2024-01-01T09:00:00", "length_gate_ok": True},
            {"id": "c", "generated_at": "2024-01-03T11:00:00"},
            {"id": "iso", "generated_at": "2024-01-04T12:00:00"},
            {"id": "dup1", "generated_at": "2024-01-02T10:00:00"},
        ],
        "head_pack": "iso",
        "workspace": {"dirty": False},
        "edges": [
            {"id": "a->b", "from": "a", "to": "b", "inferred": False,
             "reports": [{"report_id": "r1"}], "ledger_ops": [{"op_id": "o1"}, {"op_id": "o2"}]},
            {"id": "a->c", "from": "a", "to": "c", "inferred": True,
             "reports": [], "ledger_ops": []},
        ],
    }
    return {
        "graphs": {
            "branch": graph,
            "no_valid_head": {
                "nodes": graph["nodes"],
                "head_pack": "removed-pack",
                "workspace": {"dirty": False},
                "edges": graph["edges"],
            },
            "invalid_ties": {
                "nodes": [
                    {"id": "z", "generated_at": "not-a-date"},
                    {"id": "y", "generated_at": ""},
                    {"id": "m", "generated_at": "2024-05-01T00:00:00"},
                    {"id": "n", "generated_at": "2024-05-02T00:00:00"},
                    {"id": "n2", "generated_at": "2024-05-02T00:00:00"},
                ],
                "head_pack": None,
                "workspace": {"dirty": False},
                "edges": [],
            },
            "dirty_draft": {
                "nodes": [{"id": "head", "generated_at": "2024-06-01T08:00:00"}],
                "head_pack": "head",
                "workspace": {
                    "dirty": True,
                    "reasons": ["unpurified_raw_sessions"],
                    "counts": {"projects_count": 3, "gold_sessions": 7, "sft_train": 12, "dpo_train": 2},
                },
                "edges": [
                    {"id": "head->workspace", "from": "head", "to": "workspace", "kind": "draft",
                     "inferred": False, "reports": [], "ledger_ops": [{"op_id": "o1"}, {"op_id": "o2"}]},
                ],
            },
            "empty_clean": {"nodes": [], "head_pack": None, "workspace": {"dirty": False}, "edges": []},
            "empty_dirty": {
                "nodes": [], "head_pack": None,
                "workspace": {"dirty": True, "reasons": ["no_delivery_pack"], "counts": {}},
                "edges": [],
            },
            "real_date": {
                "nodes": [{"id": "dt", "generated_at": "2024-03-05T12:34:00"}],
                "head_pack": "dt",
                "workspace": {"dirty": False},
                "edges": [],
            },
        },
        "scrollGraph": {
            "nodes": [
                {"id": "pack-a", "generated_at": "2024-01-01T08:00:00"},
                {"id": "pack-b", "generated_at": "2024-01-02T08:00:00"},
                {"id": "pack-c", "generated_at": "2024-01-03T08:00:00"},
                {"id": "pack-d", "generated_at": "2024-01-04T08:00:00"},
            ],
            "head_pack": "pack-b",
            "workspace": {"dirty": False},
            "edges": [
                {"id": "pack-a->pack-b", "from": "pack-a", "to": "pack-b",
                 "reports": [], "ledger_ops": []},
                {"id": "pack-b->pack-c", "from": "pack-b", "to": "pack-c",
                 "reports": [], "ledger_ops": [{"op_id": "o1"}]},
            ],
        },
    }


@pytest.fixture(scope="module")
def rendered() -> dict:
    match = TIMELINE_BLOCK.search(HTML)
    assert match is not None, "未找到 VDS 时间轴函数块"
    source = HARNESS_PRELUDE + match.group(1) + HARNESS_DRIVER
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "timeline.js"
        path.write_text(source, encoding="utf-8")
        proc = subprocess.run(
            [_node(), str(path), json.dumps(_fixtures(), ensure_ascii=False)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])


def test_timeline_order_is_chronological_and_stable(rendered: dict) -> None:
    branch = rendered["graphs"]["branch"]
    assert branch["order"] == ["a", "b", "dup1", "c", "iso"]
    assert branch["cards"] == branch["order"]
    assert branch["current"] == "iso"
    assert branch["currentCard"] == "iso"

    invalid = rendered["graphs"]["invalid_ties"]
    assert invalid["order"] == ["m", "n", "n2", "y", "z"]
    assert invalid["current"] == "n2"
    assert invalid["hasUnknown"] is True
    assert invalid["hasUnpublished"] is False
    assert invalid["hasNaN"] is False


def test_only_real_incoming_edges_are_rendered_and_branches_kept(rendered: dict) -> None:
    branch = rendered["graphs"]["branch"]
    assert branch["relations"] == [
        {"id": "a->b", "child": "b", "from": "a"},
        {"id": "a->c", "child": "c", "from": "a"},
    ]
    # 孤立节点与第二个分支都在，且没有按时间相邻补出的伪关系。
    assert set(branch["cards"]) == {"a", "b", "c", "iso", "dup1"}
    assert all(rel["child"] not in {"iso", "dup1"} for rel in branch["relations"])

    # head_pack 失效时回退到最新有效时间戳的版本。
    fallback = rendered["graphs"]["no_valid_head"]
    assert fallback["current"] == "iso"
    assert fallback["currentCard"] == "iso"


def test_draft_and_missing_timestamps_never_invent_dates(rendered: dict) -> None:
    dirty = rendered["graphs"]["dirty_draft"]
    assert dirty["order"] == ["head", "workspace"]
    assert dirty["current"] == "head"
    assert dirty["hasUnpublished"] is True
    assert dirty["hasNaN"] is False
    assert dirty["relations"] == [{"id": "head->workspace", "child": "workspace", "from": "head"}]
    assert any(d["cls"].find("is-draft") >= 0 and d["text"] == "未发布" for d in dirty["dates"])

    for name in ("empty_clean", "empty_dirty"):
        empty = rendered["graphs"][name]
        assert empty["order"] == ["workspace"]
        assert empty["current"] == "workspace"
        assert empty["hasUnpublished"] is True
        assert empty["hasUnknown"] is False
        assert empty["hasNaN"] is False

    real = rendered["graphs"]["real_date"]
    date_text = " ".join(d["text"] for d in real["dates"])
    assert "2024-03-05" in date_text
    assert "12:34" in date_text
    assert not real["unicodePackage"]


def test_initial_and_return_current_scroll_keep_head_fully_visible(rendered: dict) -> None:
    initial = rendered["scroll"]["initial"]
    assert initial["id"] == initial["currentId"] == "pack-b"
    assert initial["visible"] is True
    assert initial["scrollLeft"] == 84

    back = rendered["scroll"]["back"]
    assert back["id"] == "pack-b"
    assert back["focusType"] == "dataset"
    assert back["visible"] is True
    assert back["scrollLeft"] == 84

    math = rendered["scroll"]["math"]
    assert math["center"] == 720
    # 当前滚动量不同的同一内容坐标算出同一目标，证明没有误用 offsetLeft=0。
    assert math["centerScrolled"] == 720
    assert math["leftClamp"] == 0
    assert math["rightClamp"] == 1460
    assert math["nearest"] == 398
    assert math["nearestScrolled"] == 398


def test_refresh_retains_manual_offset_and_selection_reveals(rendered: dict) -> None:
    retained = rendered["scroll"]["retained"]
    assert retained["scrollLeft"] == 640
    assert retained["focused"] == "pack-b"
    assert retained["focusType"] == "dataset"

    selected = rendered["scroll"]["selected"]
    assert selected["id"] == "pack-d"
    assert selected["visible"] is True
    assert selected["scrollLeft"] == 398

    edge = rendered["scroll"]["edge"]
    assert edge["edge"] == "pack-b->pack-c"
    assert edge["focusType"] == "edge"
    assert edge["visible"] is True


def test_timeline_markup_is_compact_and_avoids_scroll_into_view() -> None:
    assert "<h3>数据集演进</h3>" in HTML
    assert "回到当前" in HTML
    assert 'id="vdsDagCanvas"' in HTML
    match = TIMELINE_BLOCK.search(HTML)
    assert match is not None
    block = match.group(1)
    assert ".scrollIntoView(" not in block
    assert 'class="vds-node' in block
    assert '<button type="button" class="dag-rel vds-edge-card' in block

def test_stale_focus_falls_back_deterministically(rendered: dict) -> None:
    removed = rendered["fallback"]["removed"]
    assert removed["changed"] is True
    assert removed["id"] == removed["current"] == "iso"

    draft_gone = rendered["fallback"]["draftGone"]
    assert draft_gone["changed"] is True
    assert draft_gone["id"] == draft_gone["current"] == "head"

    edge_gone = rendered["fallback"]["edgeGone"]
    assert edge_gone["changed"] is True
    assert edge_gone["focusType"] == "dataset"
    assert edge_gone["id"] == "iso"
# ---------------------------------------------------------------------------
# 分支泳道（BRANCH LANES）：真实父边 → 确定性 lane 分配 + 桥的像素几何
# ---------------------------------------------------------------------------

LANE_PRELUDE = r"""
const esc = v => String(v == null ? "" : v)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmt = v => (v === undefined || v === null) ? "–" : String(v);
// 与生产一致的转义版 jsArg（会被写进行内 onclick / aria-label / title）。
const jsArg = v => esc(JSON.stringify(String(v ?? "")));
const STATE = { vds: { graph: null, focusType: "dataset", selectedId: "", selectedEdge: "", mounted: false, scrollLeft: null } };
const CANVAS = { left: 40, top: 120 };
// 可替换的 $() 宿主：布局测试只需要 journal 与横向 scroller。
const MOCK = { scroller: null, journal: null };
const $ = sel => sel === "#vdsDagCanvas" ? MOCK.scroller
  : sel === "#vdsDagJournal" ? MOCK.journal : null;
function renderVdsDetailHost() { return Promise.resolve(); }
function toast() {}
"""

LANE_DRIVER = r"""
const graphs = JSON.parse(process.argv[2]);
const decode = s => String(s).replace(/&gt;/g, ">").replace(/&lt;/g, "<")
  .replace(/&quot;/g, '"').replace(/&amp;/g, "&");
const attr = (s, name) => {
  const m = new RegExp(name + '="([^"]*)"').exec(s);
  return m ? decode(m[1]) : "";
};
const px = (style, name) => {
  const m = new RegExp(name + ':(-?[\\d.]+)px').exec(style);
  return m ? Number(m[1]) : NaN;
};
const cardsOf = html => [...html.matchAll(/class="vds-node[^"]*"([^>]*?)data-vds-id="([^"]+)"/g)]
  .map(m => {
    const style = attr(m[1], "style");
    const col = /grid-column:(\d+)/.exec(style);
    const row = /grid-row:(\d+)/.exec(style);
    return { id: decode(m[2]), col: col ? Number(col[1]) : 0, row: row ? Number(row[1]) : 0 };
  });
const chipsOf = html => [...html.matchAll(/<button type="button" class="dag-rel vds-edge-card([^"]*)"([^>]*)>/g)]
  .map(m => {
    const style = attr(m[2], "style");
    return {
      id: attr(m[2], "data-edge-id"),
      from: attr(m[2], "data-edge-from"),
      to: attr(m[2], "data-edge-to"),
      cls: m[1].trim(),
      label: attr(m[2], "aria-label"),
      title: attr(m[2], "title"),
      left: px(style, "left"), top: px(style, "top"),
      width: px(style, "width"), height: px(style, "height"),
      selects: /onclick="selectLineageEdge\(/.test(m[2]),
      tabStripped: /tabindex="-1"/.test(m[2]),
    };
  });
const segHits = (a, b, r) => Math.max(a[0], b[0]) > r.x && Math.min(a[0], b[0]) < r.x + r.w
  && Math.max(a[1], b[1]) > r.y && Math.min(a[1], b[1]) < r.y + r.h;

const snapshot = graph => {
  const L = vdsLaneLayout(graph);
  STATE.vds.graph = graph;
  STATE.vds.focusType = "dataset";
  STATE.vds.selectedId = vdsCurrentId(graph);
  const html = renderDatasetDagHtml(graph);
  const violations = [];
  for (const e of L.edges) {
    const others = L.nodes.filter(n => n.id !== e.from.id && n.id !== e.to.id);
    for (let i = 1; i < e.points.length; i++) {
      for (const n of others) {
        if (segHits(e.points[i - 1], e.points[i], { x: n.x, y: n.y, w: n.w, h: n.h })) {
          violations.push({ edge: e.edge.id, node: n.id });
        }
      }
    }
  }
  const shuffled = (() => {
    const g2 = JSON.parse(JSON.stringify(graph));
    g2.nodes = (g2.nodes || []).slice().reverse();
    g2.edges = (g2.edges || []).slice().reverse();
    const acc = {};
    for (const n of vdsLaneLayout(g2).nodes) acc[n.id] = { lane: n.lane, col: n.col };
    return acc;
  })();
  const laneOf = {};
  for (const lane of L.lanes) for (const id of lane.nodes) laneOf[id] = lane.index;
  return {
    head: L.headId,
    current: vdsCurrentId(graph),
    lanes: L.lanes,
    laneOf,
    nodes: L.nodes.map(n => ({ id: n.id, col: n.col, lane: n.lane, x: n.x, y: n.y, w: n.w, h: n.h })),
    edges: L.edges.map(e => ({
      id: e.edge.id, dir: e.dir, chip: e.chip, points: e.points,
      inferred: !!e.edge.inferred, kind: String(e.edge.kind || ""),
      from: { id: e.from.id, x: e.from.x, y: e.from.y, w: e.from.w, h: e.from.h, col: e.from.col, lane: e.from.lane },
      to: { id: e.to.id, x: e.to.x, y: e.to.y, w: e.to.w, h: e.to.h, col: e.to.col, lane: e.to.lane },
    })),
    parentless: L.nodes.filter(n => !L.parentOf.get(n.id)).map(n => n.id),
    violations,
    cycleFree: (() => {
      for (const id of L.parentOf.keys()) {
        let cur = L.parentOf.get(id), guard = 0;
        while (cur && guard++ <= L.nodes.length) {
          if (cur === id) return false;
          cur = L.parentOf.get(cur);
        }
      }
      return true;
    })(),
    width: L.width,
    height: L.height,
    axisTop: L.axisTop,
    minPathY: L.edges.length ? Math.min(...L.edges.flatMap(e => e.points.map(p => p[1]))) : 0,
    cards: cardsOf(html),
    chips: chipsOf(html),
    svg: {
      present: html.includes('class="dag-bridges"'),
      grid: html.includes('class="dag-grid"'),
      viewBox: (/viewBox="0 0 (\d+) (\d+)"/.exec(html) || []).slice(1).map(Number),
    },
    markup: {
      gridAt: html.indexOf('class="dag-grid"'),
      axisAt: html.indexOf('class="dag-axis"'),
      chipAt: html.indexOf('class="dag-rel vds-edge-card'),
    },
    again: vdsLaneLayout(graph).lanes.map(l => l.nodes.join(",")),
    shuffled,
  };
};

// 手机宽度（400px）下的按需露出：桥标签与子卡片必须同时可见。
const rectFor = (x, y, w, h) => ({
  left: CANVAS.left + x - (Number(MOCK.scroller && MOCK.scroller.scrollLeft) || 0),
  top: CANVAS.top + y, width: w, height: h,
  right: CANVAS.left + x - (Number(MOCK.scroller && MOCK.scroller.scrollLeft) || 0) + w,
  bottom: CANVAS.top + y + h,
});
const edgeReveal = (graph, edgeId, opts) => {
  const L = vdsLaneLayout(graph);
  const html = renderDatasetDagHtml(graph);
  const chips = chipsOf(html);
  const cards = L.nodes.map(n => ({
    dataset: { vdsId: n.id },
    getBoundingClientRect: () => rectFor(n.x, n.y, n.w, n.h),
  }));
  MOCK.scroller = {
    clientWidth: opts.clientWidth, clientLeft: 0, scrollLeft: opts.scrollLeft,
    scrollWidth: Math.max(opts.clientWidth, L.width),
    getBoundingClientRect: () => ({ left: CANVAS.left, top: CANVAS.top, width: opts.clientWidth,
      height: 400, right: CANVAS.left + opts.clientWidth, bottom: CANVAS.top + 400 }),
    addEventListener() {},
  };
  MOCK.journal = {
    querySelectorAll: sel => (sel === "[data-vds-id]" ? cards : []),
    querySelector: sel => {
      const m = /\[data-edge-id="(.*)"\]$/.exec(sel);
      if (!m) return null;
      const hit = chips.find(c => c.id === m[1]);
      if (!hit) return null;
      return { getBoundingClientRect: () => rectFor(hit.left, hit.top, hit.width, hit.height) };
    },
  };
  STATE.vds.graph = graph;
  STATE.vds.focusType = "edge";
  STATE.vds.selectedEdge = edgeId;
  STATE.vds.selectedId = vdsCurrentId(graph);
  const target = vdsRevealTimeline({ reveal: true });
  const bridge = L.edges.find(e => e.edge.id === edgeId);
  const child = L.nodes.find(n => n.id === String(bridge.edge.to));
  const chipLeft = bridge.chip.left - target;
  const cardLeft = child.x - target;
  return {
    scrollLeft: target,
    chipVisible: chipLeft >= 0 && chipLeft + bridge.chip.width <= opts.clientWidth,
    cardVisible: cardLeft >= 0 && cardLeft + child.w <= opts.clientWidth,
  };
};

const out = { graphs: {} };
for (const [name, graph] of Object.entries(graphs)) out.graphs[name] = snapshot(graph);
out.reveal = {
  phone: edgeReveal(graphs.fork, "a->c", { clientWidth: 400, scrollLeft: 0 }),
  phoneScrolledRight: edgeReveal(graphs.fork, "a->c", { clientWidth: 400, scrollLeft: 900 }),
  reversed: edgeReveal(graphs.reversed, "p->q", { clientWidth: 400, scrollLeft: 0 }),
};

// 确定性伪随机 DAG：把布局 / 路径 / 标签的全部不变量在大量图形上跑一遗。
const ratio = seed => () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const validate = (graph, name) => {
  const snap = snapshot(graph);
  const problems = [];
  const ids = snap.nodes.map(n => n.id);
  if (new Set(ids).size !== ids.length) problems.push("duplicate-node");
  if (snap.cards.map(c => c.id).join(",") !== ids.join(",")) problems.push("cards-mismatch");
  if (snap.chips.map(c => c.id).join(",") !== snap.edges.map(e => e.id).join(",")) problems.push("chips-mismatch");
  if (snap.violations.length) problems.push("path-crosses-card");
  if (!snap.cycleFree) problems.push("cycle");
  for (const e of snap.edges) {
    const chip = e.chip, f = e.from, t = e.to;
    const gapOk = e.dir === "right"
      ? (f.x + f.w <= chip.left && chip.left + chip.width <= t.x)
      : (t.x + t.w <= chip.left && chip.left + chip.width <= f.x);
    if (!gapOk) problems.push(`chip-gap:${e.id}`);
    if (!(t.y <= chip.top && chip.top + chip.height <= t.y + t.h)) problems.push(`chip-lane:${e.id}`);
    if (f.id === t.id) problems.push(`self-edge:${e.id}`);
  }
  const kids = {};
  for (const e of snap.edges) (kids[e.from.id] = kids[e.from.id] || []).push(e.to.id);
  for (const pid of Object.keys(kids)) {
    const used = kids[pid].map(id => snap.laneOf[id]);
    if (new Set(used).size !== used.length) problems.push(`sibling-lane:${pid}`);
  }
  return { name, problems, lanes: snap.lanes.length, nodes: ids.length, edges: snap.edges.length };
};
out.fuzz = [];
for (let gi = 0; gi < 40; gi++) {
  const rnd = ratio(gi * 7919 + 13);
  const count = 2 + Math.floor(rnd() * 7);
  const nodes = [];
  for (let i = 0; i < count; i++) {
    const blank = rnd() < 0.12;
    const day = 1 + Math.floor(rnd() * 20);
    nodes.push({
      id: "n" + i, kind: "pack",
      generated_at: blank ? "" : `2024-01-${String(day).padStart(2, "0")}T00:00:00`,
    });
  }
  const edges = [];
  for (let i = 1; i < count; i++) {
    if (rnd() < 0.2) continue;   // 独立根
    const p = Math.floor(rnd() * i);
    edges.push({ id: `n${p}->n${i}`, from: "n" + p, to: "n" + i, inferred: rnd() < 0.3 });
  }
  out.fuzz.push(validate({ nodes, head_pack: nodes[count - 1].id, workspace: { dirty: false }, edges }, `fuzz${gi}`));
}
console.log(JSON.stringify(out));
"""


def _lane_fixtures() -> dict:
    def pack(pid: str, ts: str) -> dict:
        return {"id": pid, "generated_at": ts, "kind": "pack"}

    def edge(a: str, b: str, **kw) -> dict:
        base = {"id": f"{a}->{b}", "from": a, "to": b, "inferred": False, "reports": [], "ledger_ops": []}
        base.update(kw)
        return base

    fork_edges = [
        edge("a", "b", reports=[{"report_id": "r1"}]),
        edge("a", "c", inferred=True),
        edge("b", "d", ledger_ops=[{"op_id": "o1"}, {"op_id": "o2"}]),
        edge("c", "e"),
    ]
    fork_nodes = [
        pack("a", "2024-01-01T00:00:00"),
        pack("b", "2024-01-02T00:00:00"),
        pack("c", "2024-01-03T00:00:00"),
        pack("d", "2024-01-04T00:00:00"),
        pack("e", "2024-01-05T00:00:00"),
    ]
    return {
        "chain": {
            "nodes": [
                pack("a", "2024-01-01T00:00:00"),
                pack("b", "2024-01-02T00:00:00"),
                pack("c", "2024-01-03T00:00:00"),
            ],
            "head_pack": "c",
            "workspace": {"dirty": False},
            "edges": [edge("a", "b"), edge("b", "c")],
        },
        "fork": {
            "nodes": fork_nodes,
            "head_pack": "e",
            "workspace": {"dirty": False},
            "edges": fork_edges,
        },
        "fork_later": {
            "nodes": fork_nodes + [pack("g", "2024-01-06T00:00:00")],
            "head_pack": "e",
            "workspace": {"dirty": False},
            "edges": fork_edges + [edge("a", "g")],
        },
        "nested": {
            "nodes": [
                pack("a", "2024-01-01T00:00:00"),
                pack("b", "2024-01-02T00:00:00"),
                pack("c", "2024-01-03T00:00:00"),
                pack("d", "2024-01-04T00:00:00"),
                pack("e", "2024-01-05T00:00:00"),
                pack("f", "2024-01-06T00:00:00"),
            ],
            "head_pack": "f",
            "workspace": {"dirty": False},
            "edges": [
                edge("a", "b"),
                edge("a", "c"),
                edge("c", "d"),
                edge("c", "e"),
                edge("d", "f"),
            ],
        },
        "roots": {
            "nodes": [
                pack("r2", "2024-01-01T00:00:00"),
                pack("r1", "2024-01-02T00:00:00"),
                pack("x", "2024-01-03T00:00:00"),
                pack("y", "2024-01-04T00:00:00"),
            ],
            "head_pack": "y",
            "workspace": {"dirty": False},
            "edges": [edge("r1", "x"), edge("r2", "y")],
        },
        "reversed": {
            "nodes": [
                pack("p", "2024-02-10T00:00:00"),
                pack("q", "2024-01-01T00:00:00"),
            ],
            "head_pack": "q",
            "workspace": {"dirty": False},
            "edges": [edge("p", "q")],
        },
        "unknown": {
            "nodes": [
                {"id": "u", "generated_at": "", "kind": "pack"},
                pack("v", "2024-01-02T00:00:00"),
            ],
            "head_pack": "v",
            "workspace": {"dirty": False},
            "edges": [edge("u", "v")],
        },
        "cycle": {
            "nodes": [
                pack("a", "2024-01-01T00:00:00"),
                pack("b", "2024-01-02T00:00:00"),
                pack("c", "2024-01-03T00:00:00"),
            ],
            "head_pack": "c",
            "workspace": {"dirty": False},
            "edges": [edge("a", "b"), edge("b", "c"), edge("c", "a")],
        },
        "draft": {
            "nodes": [pack("head", "2024-06-01T00:00:00")],
            "head_pack": "head",
            "workspace": {"dirty": True, "reasons": ["unpurified_raw_sessions"], "counts": {}},
            "edges": [
                {"id": "head->workspace", "from": "head", "to": "workspace", "kind": "draft",
                 "inferred": False, "reports": [], "ledger_ops": [{"op_id": "o1"}, {"op_id": "o2"}]},
            ],
        },
        "empty": {
            "nodes": [],
            "head_pack": None,
            "workspace": {"dirty": True, "reasons": ["no_delivery_pack"], "counts": {}},
            "edges": [],
        },
    }


@pytest.fixture(scope="module")
def lanes() -> dict:
    match = TIMELINE_BLOCK.search(HTML)
    assert match is not None, "未找到 VDS 时间轴函数块"
    source = LANE_PRELUDE + match.group(1) + LANE_DRIVER
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lanes.js"
        path.write_text(source, encoding="utf-8")
        proc = subprocess.run(
            [_node(), str(path), json.dumps(_lane_fixtures(), ensure_ascii=False)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])


def test_sibling_branches_get_separate_rows_and_head_keeps_top_lane(lanes: dict) -> None:
    fork = lanes["graphs"]["fork"]
    lane = fork["laneOf"]
    # A -> B 与 A -> C 是两个兄弟分支，必须落在不同行。
    assert lane["b"] != lane["c"]
    # HEAD 祖先链（A -> C -> E）留在主泳道上。
    assert fork["head"] == "e"
    assert fork["lanes"][0]["index"] == 0
    assert fork["lanes"][0]["nodes"] == ["a", "c", "e"]
    assert lane["a"] == lane["c"] == lane["e"] == 0
    # 后代留在自己分支的泳道上。
    assert lane["d"] == lane["b"]
    # 每个节点恰好出现一次，列保持时间轴顺序。
    listed = [i for item in fork["lanes"] for i in item["nodes"]]
    assert sorted(listed) == sorted(n["id"] for n in fork["nodes"])
    assert len(set(listed)) == len(listed)
    assert [n["col"] for n in fork["nodes"]] == [0, 1, 2, 3, 4]

    chain = lanes["graphs"]["chain"]
    assert len(chain["lanes"]) == 1
    assert chain["lanes"][0]["nodes"] == ["a", "b", "c"]


def test_nested_fork_and_independent_roots_have_no_fake_links(lanes: dict) -> None:
    nested = lanes["graphs"]["nested"]
    lane = nested["laneOf"]
    # 嵌套分叉：A -> C -> D -> F 是 HEAD 祖先链；A->B 与 C->E 各占新泳道。
    assert lane["a"] == lane["c"] == lane["d"] == lane["f"] == 0
    assert lane["b"] != 0
    assert lane["e"] != lane["c"]
    assert len({lane["b"], lane["e"]}) == 2
    assert nested["parentless"] == ["a"]
    assert len(nested["edges"]) == 5

    roots = lanes["graphs"]["roots"]
    root_lane = roots["laneOf"]
    # 两个独立根各占一条泳道，彼此之间绝不编造连线。
    assert root_lane["r1"] != root_lane["r2"]
    assert root_lane["x"] == root_lane["r1"]
    assert root_lane["y"] == root_lane["r2"]
    assert sorted(roots["parentless"]) == ["r1", "r2"]
    assert sorted(e["id"] for e in roots["edges"]) == ["r1->x", "r2->y"]
    assert len(roots["chips"]) == 2
    assert roots["violations"] == []


def test_bridges_span_parent_to_child_in_the_actual_child_lane(lanes: dict) -> None:
    gap = 124
    for name in ("chain", "fork", "nested", "roots", "reversed", "unknown", "draft"):
        snap = lanes["graphs"][name]
        for e in snap["edges"]:
            chip, frm, child = e["chip"], e["from"], e["to"]
            # 桥标签物理上落在「父卡片边缘」与「子卡片边缘」之间，而不是卡片上方。
            if e["dir"] == "right":
                assert frm["x"] + frm["w"] <= chip["left"], name
                assert chip["left"] + chip["width"] <= child["x"], name
                assert e["points"][0] == [frm["x"] + frm["w"], frm["y"] + frm["h"] / 2], name
                assert e["points"][-1] == [child["x"], child["y"] + child["h"] / 2], name
            else:
                assert child["x"] + child["w"] <= chip["left"], name
                assert chip["left"] + chip["width"] <= frm["x"], name
                assert e["points"][0] == [frm["x"], frm["y"] + frm["h"] / 2], name
                assert e["points"][-1] == [child["x"] + child["w"], child["y"] + child["h"] / 2], name
            # 桥标签与子卡片同一条泳道行，垂直居中在卡片高度内。
            assert child["y"] <= chip["top"], name
            assert chip["top"] + chip["height"] <= child["y"] + child["h"], name
            # 桥应落在实际连线上，而非只是在两张卡之间悬空。
            x, y = chip["left"] + chip["width"] / 2, chip["top"] + chip["height"] / 2
            assert any(p[1] == q[1] == y and min(p[0], q[0]) <= x <= max(p[0], q[0])
                       for p, q in zip(e["points"], e["points"][1:])), (name, e["id"])
            # 标签宽度必须塞得进列间距，文字不会压到卡片上。
            assert chip["width"] + 2 * 6 <= gap, name
            assert chip["width"] >= 90, name
            # 连线只走列间 / 行间空隙，不穿过任何无关卡片。
            assert snap["violations"] == [], name
            assert snap["minPathY"] >= 0, name


def test_each_node_is_rendered_once_with_its_lane_placement(lanes: dict) -> None:
    for name, snap in lanes["graphs"].items():
        node_ids = [n["id"] for n in snap["nodes"]]
        assert [c["id"] for c in snap["cards"]] == node_ids, name
        assert len(set(node_ids)) == len(node_ids), name
        assert snap["svg"]["present"] is True, name
        assert snap["svg"]["grid"] is True, name
        # SVG viewBox 与 CSS 尺寸同一坐标体系（1 单位 = 1px）。
        assert snap["svg"]["viewBox"] == [snap["width"], snap["height"]], name
        for card in snap["cards"]:
            node = next(n for n in snap["nodes"] if n["id"] == card["id"])
            assert card["col"] == node["col"] + 1, name
            assert card["row"] == node["lane"] + 1, name
        # 桥标签渲染在网格之外（不是节点上方 / 节点内部），且按真实边一一对应。
        assert [c["id"] for c in snap["chips"]] == [e["id"] for e in snap["edges"]], name
        if snap["chips"]:
            assert snap["markup"]["gridAt"] < snap["markup"]["axisAt"] < snap["markup"]["chipAt"], name
        for chip in snap["chips"]:
            assert chip["selects"] is True, name
            assert chip["tabStripped"] is False, name
            for node in snap["nodes"]:
                inside = (chip["left"] > node["x"] and chip["left"] < node["x"] + node["w"]
                          and chip["top"] > node["y"]
                          and chip["top"] + chip["height"] < node["y"] + node["h"])
                assert not inside, f"{name}:{chip['id']} 落入卡片 {node['id']}"
        for chip, e in zip(snap["chips"], snap["edges"]):
            assert chip["left"] == e["chip"]["left"], name
            assert chip["top"] == e["chip"]["top"], name
            assert chip["width"] == e["chip"]["width"], name
            assert chip["height"] == e["chip"]["height"], name
            # 可访问名 / title 同时包含两端。
            assert e["from"]["id"] in chip["label"], name
            assert e["to"]["id"] in chip["label"], name
            assert e["from"]["id"] in chip["title"], name
            assert e["to"]["id"] in chip["title"], name
        # 底部共享时间轴：每条泳道都在轴上方。
        assert snap["height"] > snap["axisTop"] > 0, name
        for node in snap["nodes"]:
            assert node["y"] + node["h"] <= snap["axisTop"], name


def test_inferred_and_draft_branches_are_visually_distinct(lanes: dict) -> None:
    fork = lanes["graphs"]["fork"]
    inferred = next(c for c in fork["chips"] if c["id"] == "a->c")
    explicit = next(c for c in fork["chips"] if c["id"] == "b->d")
    assert "is-inferred" in inferred["cls"].split()
    assert "推断" in inferred["label"]
    assert "is-inferred" not in explicit["cls"].split()

    draft = lanes["graphs"]["draft"]
    assert len(draft["cards"]) == 2
    assert draft["cards"][-1]["id"] == "workspace"
    chip = draft["chips"][0]
    assert chip["id"] == "head->workspace"
    assert "is-draft" in chip["cls"].split()
    assert "草稿" in chip["label"]
    assert chip["to"] == "workspace"
    assert draft["violations"] == []

    empty = lanes["graphs"]["empty"]
    assert [c["id"] for c in empty["cards"]] == ["workspace"]
    assert empty["chips"] == []
    assert empty["svg"]["present"] is True


def test_leftward_and_cyclic_edges_render_deliberate_bridges(lanes: dict) -> None:
    rev = lanes["graphs"]["reversed"]
    assert len(rev["edges"]) == 1
    edge = rev["edges"][0]
    # 时钟倒挂：父包时间晚于子包，仍然绘制一条有意的真实连接。
    assert edge["dir"] == "left"
    assert edge["from"]["id"] == "p" and edge["to"]["id"] == "q"
    assert rev["chips"][0]["to"] == "q"
    assert "is-left" in rev["chips"][0]["cls"].split()
    assert rev["violations"] == []

    unknown = lanes["graphs"]["unknown"]
    assert unknown["nodes"][-1]["id"] == "u"
    assert unknown["edges"][0]["dir"] == "left"
    assert unknown["chips"][0]["id"] == "u->v"

    cyc = lanes["graphs"]["cycle"]
    assert cyc["cycleFree"] is True
    # 损坏的环会被切断成有限的边，绝不无限渲染或自连。
    assert len(cyc["edges"]) < len(_lane_fixtures()["cycle"]["edges"])
    assert all(e["from"]["id"] != e["to"]["id"] for e in cyc["edges"])
    assert len({n["id"] for n in cyc["nodes"]}) == 3
    assert cyc["violations"] == []


def test_lane_assignment_is_deterministic_and_stable(lanes: dict) -> None:
    for name, snap in lanes["graphs"].items():
        assert snap["again"] == [",".join(lane["nodes"]) for lane in snap["lanes"]], name
        # 与 nodes / edges 的数组顺序无关。
        for node in snap["nodes"]:
            assert snap["shuffled"][node["id"]] == {"lane": node["lane"], "col": node["col"]}, name

    # 新增一个更晚的兄弟分支后，已有节点的泳道保持不变（刷新后分支稳定）。
    base = lanes["graphs"]["fork"]["laneOf"]
    later = lanes["graphs"]["fork_later"]["laneOf"]
    for nid, lane in base.items():
        assert later[nid] == lane
    assert later["g"] not in (0, base["b"])


def test_phone_reveal_keeps_bridge_chip_and_child_card_visible(lanes: dict) -> None:
    reveal = lanes["reveal"]
    phone = reveal["phone"]
    assert phone["chipVisible"] is True
    assert phone["cardVisible"] is True
    scrolled = reveal["phoneScrolledRight"]
    assert scrolled["chipVisible"] is True
    assert scrolled["cardVisible"] is True
    rev = reveal["reversed"]
    assert rev["chipVisible"] is True
    assert rev["cardVisible"] is True


def test_branch_lane_styles_and_help_text_are_wired() -> None:
    match = TIMELINE_BLOCK.search(HTML)
    assert match is not None
    block = match.group(1)
    for frag in (
        'class="dag-journal"', 'class="dag-grid"', 'class="dag-bridges"', 'class="dag-axis"',
        "viewBox=", "grid-column:", "grid-row:", "VDS_LANE_GEOM",
    ):
        assert frag in block, frag
    # 卡片上方的旧「源」胶囊槽位与卡片 stem 已移除。
    for gone in ("dag-rel-slot", "dag-card-slot", "dag-stem"):
        assert gone not in HTML, gone
    # 面板帮助文案说明泳道与桥标签。
    assert "泳道" in HTML
    for frag in (".dag-bridges{", ".dag-bridge{", ".dag-bridge.is-inferred{", ".dag-bridge.is-draft{",
                 ".dag-axis{", ".dag-tick{", ".vds-edge-card.is-inferred{", ".vds-edge-card.is-draft{"):
        assert frag in HTML, frag

def test_synthetic_dags_keep_lane_and_bridge_invariants(lanes: dict) -> None:
    fuzz = lanes["fuzz"]
    assert len(fuzz) == 40
    for item in fuzz:
        assert item["problems"] == [], (item["name"], item["problems"])
    # 覆盖分叉 / 独立根 / 倒挂时间戳等情形，而不是只跑一条链。
    assert sum(1 for item in fuzz if item["lanes"] > 1) >= 8
    assert sum(item["edges"] for item in fuzz) >= 60
