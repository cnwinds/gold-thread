"""虚拟数据集黄金列表与抽屉的工作区实时范围徽标回归测试。

用 Node 执行 web/index.html 中真实的前端函数，覆盖：
- 历史包冻结计数只显示纯数字（含 0 / 15 的仅 DPO 行与正数行），不再出现「仅DPO」说明
- 表头仍区分冻结与工作区
- 历史包冻结计数 + 工作区停用共存
- 工作区仅 SFT / 仅 DPO / 全停用 / 正常
- 抽屉按钮可见文字只有「恢复 SFT」/「停用 SFT」，状态、作用域与 DPO 影响在 title / aria-label
- 继承自项目/会话范围的停用只显示状态、不提供无效按钮
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

NODE_SCRIPT_EXTRACT = re.compile(
    r"(function vdsScopeZh\(scope\) \{[\s\S]*?)\nfunction renderVdsBreakdownHtml"
)
INLINE_SCRIPT_EXTRACT = re.compile(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>")
BUTTON_EXTRACT = re.compile(r"<button\b[^>]*>(.*?)</button>", re.S)
TAG_EXTRACT = re.compile(r"<[^>]+>")

HARNESS_PRELUDE = r"""
const esc = v => String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmt = v => String(v == null ? 0 : v);
const jsArg = v => JSON.stringify(v);
"""

HARNESS_DRIVER = r"""
const rows = JSON.parse(process.argv[3] || "{}");
const out = {};
for (const [name, fixture] of Object.entries(rows)) {
  const usage = vdsWorkspaceUsage(fixture.row, { workspaceRow: !!fixture.workspaceRow });
  out[name] = {
    badge: vdsWorkspaceBadgeHtml(usage),
    workspaceCell: renderVdsGoldWorkspaceCellHtml(fixture.row, fixture.datasetId || "pack-old"),
    countCell: renderVdsGoldCountCellHtml(fixture.row),
    controls: renderVdsSessionUsageControlsHtml(fixture.datasetId || "pack-old", fixture.row, !!fixture.isOff),
  };
}
console.log(JSON.stringify(out));
"""


def visible_text(html: str) -> str:
    return TAG_EXTRACT.sub("", html).strip()


def only_button(html: str) -> str:
    matches = BUTTON_EXTRACT.findall(html)
    assert len(matches) == 1, f"期望恰好一个按钮，实际 {len(matches)}：{html}"
    return BUTTON_EXTRACT.search(html).group(0)


@pytest.fixture(scope="module")
def rendered() -> dict:
    if not shutil.which("node"):
        pytest.skip("需要 node 执行前端真实函数")
    match = NODE_SCRIPT_EXTRACT.search(HTML)
    assert match is not None, "未找到工作区范围渲染函数"
    functions = match.group(1)

    fixtures = {
        "history_positive_live_full_off": {
            "row": {
                "session_id": "s-history",
                "sft_windows": 4,
                "dpo_count": 2,
                "sft_excluded": False,
                "workspace_usage": {
                    "in_workspace": True,
                    "session_excluded": False,
                    "session_exclusion_scope": "",
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
            "frozen": True,
        },
        "history_inherited_project_off": {
            "row": {
                "session_id": "s-proj",
                "sft_windows": 3,
                "dpo_count": 1,
                "workspace_usage": {
                    "in_workspace": True,
                    "session_excluded": True,
                    "session_exclusion_scope": "project",
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
            "frozen": True,
        },
        "frozen_dpo_only_zero_sft": {
            "row": {
                "session_id": "s-only-dpo",
                "sft_windows": 0,
                "dpo_count": 15,
                "sft_excluded": True,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
            "frozen": True,
        },
        "frozen_dpo_only_live_dpo_off": {
            "row": {
                "session_id": "s-only-dpo-off",
                "sft_windows": 0,
                "dpo_count": 3,
                "sft_excluded": True,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": True,
                },
            },
            "frozen": True,
        },
        "frozen_positive": {
            "row": {
                "session_id": "s-active-frozen",
                "sft_windows": 2,
                "dpo_count": 1,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
            "frozen": True,
        },
        "workspace_dpo_only_zero_sft": {
            "row": {
                "session_id": "s-ws-only-dpo",
                "sft_windows": 0,
                "dpo_count": 15,
                "sft_excluded": True,
                "dpo_excluded": False,
            },
            "datasetId": "workspace",
            "workspaceRow": True,
        },
        "partial_sft_off": {
            "row": {
                "session_id": "s-sft",
                "sft_windows": 0,
                "dpo_count": 2,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": True,
                    "dpo_excluded": False,
                },
            },
            "frozen": True,
        },
        "partial_dpo_off": {
            "row": {
                "session_id": "s-dpo",
                "sft_windows": 2,
                "dpo_count": 2,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": True,
                },
            },
            "frozen": True,
        },
        "active": {
            "row": {
                "session_id": "s-active",
                "sft_windows": 2,
                "dpo_count": 1,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
            "frozen": True,
        },
        "active_zero_zero": {
            "row": {
                "session_id": "s-zero",
                "sft_windows": 0,
                "dpo_count": 0,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
            "frozen": True,
        },
        "zero_zero_full_off": {
            "row": {
                "session_id": "s-zero-off",
                "sft_windows": 0,
                "dpo_count": 0,
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
            "frozen": True,
        },
        "missing_workspace_usage": {
            "row": {"session_id": "s-missing", "sft_windows": 1, "dpo_count": 1},
            "frozen": True,
        },
        "drawer_inherited_project_off": {
            "row": {
                "session_id": "s-drawer-proj",
                "workspace_usage": {
                    "in_workspace": True,
                    "session_excluded": True,
                    "session_exclusion_scope": "project",
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
        },
        "drawer_inherited_session_off": {
            "row": {
                "session_id": "s-drawer-sess",
                "workspace_usage": {
                    "in_workspace": True,
                    "session_excluded": True,
                    "session_exclusion_scope": "session",
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
        },
        "drawer_both_off": {
            "row": {
                "session_id": "s-drawer-both",
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": True,
                    "dpo_excluded": True,
                },
            },
            "isOff": True,
        },
        "drawer_sft_off": {
            "row": {
                "session_id": "s-drawer-sft",
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": True,
                    "dpo_excluded": False,
                },
            },
            "isOff": True,
        },
        "drawer_dpo_off": {
            "row": {
                "session_id": "s-drawer-dpo",
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": True,
                },
            },
        },
        "drawer_active": {
            "row": {
                "session_id": "s-drawer-active",
                "workspace_usage": {
                    "in_workspace": True,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
        },
        "drawer_not_in_workspace": {
            "row": {
                "session_id": "s-drawer-absent",
                "workspace_usage": {
                    "in_workspace": False,
                    "sft_excluded": False,
                    "dpo_excluded": False,
                },
            },
        },
    }

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        js_file = Path(tmp) / "usage_badges.js"
        js_file.write_text(
            HARNESS_PRELUDE + functions + "\n" + HARNESS_DRIVER,
            encoding="utf-8",
        )
        result = subprocess.run(
            ["node", str(js_file), "unused", json.dumps(fixtures, ensure_ascii=False)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])


def test_count_cells_show_only_numbers(rendered: dict) -> None:
    assert rendered["history_positive_live_full_off"]["countCell"] == "4 / 2"
    assert rendered["frozen_positive"]["countCell"] == "2 / 1"
    assert rendered["frozen_dpo_only_zero_sft"]["countCell"] == "0 / 15"
    assert rendered["workspace_dpo_only_zero_sft"]["countCell"] == "0 / 15"
    assert rendered["active_zero_zero"]["countCell"] == "0 / 0"
    for key in (
        "history_positive_live_full_off",
        "frozen_dpo_only_zero_sft",
        "frozen_dpo_only_live_dpo_off",
        "workspace_dpo_only_zero_sft",
    ):
        cell = rendered[key]["countCell"]
        assert "仅DPO" not in cell
        assert "<" not in cell and "chip" not in cell


def test_count_header_still_marks_frozen_vs_workspace() -> None:
    assert "工作区实时范围" in HTML
    assert 'SFT / DPO${frozen ? "（冻结）" : ""}' in HTML
    assert 'const frozen = !!res.readonly;' in HTML


def test_history_positive_counts_keep_frozen_numbers_and_live_full_off_badge(rendered: dict) -> None:
    item = rendered["history_positive_live_full_off"]
    assert visible_text(item["badge"]) == "工作区已停用"
    assert visible_text(item["workspaceCell"]) == "工作区已停用"
    assert item["countCell"] == "4 / 2"


def test_inherited_project_exclusion_marks_workspace_off_with_scope(rendered: dict) -> None:
    item = rendered["history_inherited_project_off"]
    assert visible_text(item["badge"]) == "工作区已停用"
    assert "项目级" in item["badge"]


def test_partial_workspace_scope_badges(rendered: dict) -> None:
    sft = rendered["partial_sft_off"]
    assert visible_text(sft["badge"]) == "工作区 SFT 已停用"
    assert visible_text(sft["badge"]) != "工作区已停用"
    dpo = rendered["partial_dpo_off"]
    assert visible_text(dpo["badge"]) == "工作区 DPO 已停用"
    assert visible_text(dpo["badge"]) != "工作区已停用"


def test_active_and_missing_usage_show_no_disabled_badge(rendered: dict) -> None:
    for key in ("active", "missing_workspace_usage"):
        item = rendered[key]
        assert visible_text(item["badge"]) == ""
        assert visible_text(item["workspaceCell"]) == "–"
    assert rendered["active"]["countCell"] == "2 / 1"


def test_zero_zero_active_is_distinguishable_from_disabled(rendered: dict) -> None:
    active = rendered["active_zero_zero"]
    assert active["countCell"] == "0 / 0"
    assert visible_text(active["badge"]) == ""
    disabled = rendered["zero_zero_full_off"]
    assert disabled["countCell"] == "0 / 0"
    assert visible_text(disabled["badge"]) == "工作区已停用"


def test_frozen_only_dpo_rows_have_no_chip_even_when_live_dpo_off(rendered: dict) -> None:
    frozen = rendered["frozen_dpo_only_zero_sft"]
    assert frozen["countCell"] == "0 / 15"
    live_off = rendered["frozen_dpo_only_live_dpo_off"]
    assert live_off["countCell"] == "0 / 3"
    assert visible_text(live_off["badge"]) == "工作区 DPO 已停用"


def test_drawer_inherited_off_has_no_ineffective_sft_restore(rendered: dict) -> None:
    for key, scope in (
        ("drawer_inherited_project_off", "项目级"),
        ("drawer_inherited_session_off", "会话级"),
    ):
        item = rendered[key]["controls"]
        assert "toggleVdsSessionSft" not in item
        assert "<button" not in item
        assert visible_text(item) == "工作区已停用"
        assert scope in item
        assert "恢复需解除" + scope in item


def test_drawer_independently_disabled_channels_can_restore_only_sft(rendered: dict) -> None:
    item = rendered["drawer_both_off"]["controls"]
    assert visible_text(item) == "工作区已停用恢复 SFT"
    assert visible_text(only_button(item)) == "恢复 SFT"
    assert "is-off" in item
    assert "toggleVdsSessionSft" in item
    assert "DPO 仍停用" in item
    assert "仅DPO" not in item


def test_drawer_sft_off_offers_restore_without_stale_only_dpo_claim(rendered: dict) -> None:
    item = rendered["drawer_sft_off"]["controls"]
    assert visible_text(item) == "恢复 SFT"
    assert visible_text(only_button(item)) == "恢复 SFT"
    assert "is-off" in item
    assert "toggleVdsSessionSft" in item
    assert "DPO 不受影响" in item
    assert "仅DPO" not in item


def test_drawer_partial_dpo_off_keeps_toggle_without_misleading_only_dpo(rendered: dict) -> None:
    item = rendered["drawer_dpo_off"]["controls"]
    assert visible_text(item) == "工作区 DPO 已停用停用 SFT"
    button = only_button(item)
    assert visible_text(button) == "停用 SFT"
    assert "toggleVdsSessionSft" in item
    assert "DPO 仍停用" in button
    assert "历史包不变" in button
    assert "仅留 DPO" not in item


def test_drawer_active_and_absent_states(rendered: dict) -> None:
    active = rendered["drawer_active"]["controls"]
    button = only_button(active)
    assert visible_text(active) == "停用 SFT"
    assert visible_text(button) == "停用 SFT"
    assert "is-off" not in active
    assert "DPO 不受影响" in button
    assert "已停用" not in TAG_EXTRACT.sub("", active)
    assert rendered["drawer_not_in_workspace"]["controls"] == ""


def test_drawer_button_title_and_aria_describe_state_and_dpo(rendered: dict) -> None:
    restore = rendered["drawer_both_off"]["controls"]
    button = only_button(restore)
    assert "aria-label=" in button
    assert "title=" in button
    assert "恢复 SFT" in button
    assert "DPO" in button
    stop = rendered["drawer_dpo_off"]["controls"]
    assert "aria-label=" in only_button(stop)
    assert "工作区将无训练数据" not in stop


def test_inline_script_syntax_ok() -> None:
    if not shutil.which("node"):
        pytest.skip("需要 node 检查内联脚本")
    script = "\n".join(INLINE_SCRIPT_EXTRACT.findall(HTML))
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.inline.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True, encoding="utf-8"
        )
        assert result.returncode == 0, result.stderr