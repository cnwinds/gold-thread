"""配置 HTTP 契约：GET 只读有效值、原文 PUT、结构化 POST 与校验端点共用同一套语义校验。

所有用例都在临时目录里跑，不读取真实 config/pipeline.yaml，
不发起任何模型/网络请求。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from fastapi.testclient import TestClient

from goldthread import cli
from pipeline.common import config as config_module
from pipeline.common.config import DEFAULTS, parse_config_text
from pipeline.common.config_info import describe_config
import server.app as web


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
PRODUCTION_CONFIG_SHA256 = "b6fc7dbf5f4b51cc3a62aad7b528ea3b6d8c69a966b1eac43ed2a8a1bee90d82"


@pytest.fixture(autouse=True)
def _guard_production_config():
    """HTTP 契约测试必须完全隔离，真实 config/pipeline.yaml 不得被改写。"""
    before = PRODUCTION_CONFIG.read_bytes()
    assert hashlib.sha256(before).hexdigest() == PRODUCTION_CONFIG_SHA256
    yield
    assert PRODUCTION_CONFIG.read_bytes() == before, "测试不得修改生产配置文件"


DEFAULT_TEXT = """# 保留注释标记
paths:
  sessions_dir: "~/.pi/agent/sessions"
  data_dir: "data"

llm:
  model: "glm-test"
  temperature: 0.2

gold:
  min_gold_steps: 5

judge:
  enabled: false

synth:
  window_turns: 8
  eval_ratio: 0.25
"""

CUSTOM_MODELS_TEXT = DEFAULT_TEXT.replace(
    '  data_dir: "data"', '  data_dir: "data"\n  pi_models_json: "~/custom/models.json"'
)


def _dump(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


BAD_TEXTS = {
    "root_list": "- 只有一个\n- 顶层列表\n",
    "unknown_key": _dump({"synthx": {"window_turns": 3}}),
    "typo_nested": _dump({"synth": {"window_turn": 3}}),
    "string_bool": _dump({"judge": {"enabled": "true"}}),
    "string_number": _dump({"synth": {"window_turns": "8"}}),
    "out_of_range": _dump({"synth": {"eval_ratio": 1.7}}),
    "bad_regex": _dump({"turns": {"root_task_patterns": ["[unclosed"]}}),
    "wrong_container": _dump({"paths": []}),
}


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def editor(tmp_path, monkeypatch):
    monkeypatch.delenv("GOLDTHREAD_API_TOKEN", raising=False)
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(DEFAULT_TEXT, encoding="utf-8")
    # 读取与写入都指向同一个临时文件；load_config 读的是 config 模块里的路径。
    monkeypatch.setattr(web, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    return TestClient(web.app), cfg_path


def test_get_config_preserves_shape_and_lists_effective_values(editor):
    client, cfg_path = editor
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()

    # 旧契约字段保持不变
    assert data["path"] == str(cfg_path)
    assert data["yaml"] == DEFAULT_TEXT
    assert isinstance(data["parsed"], dict)
    assert data["parsed"]["llm"]["model"] == "glm-test"
    assert "_config_path" not in data["parsed"]

    # 新增只读有效值：path / value / source(file|default) / stage
    rows = {row["path"]: row for row in data["effective"]}
    assert rows["llm.model"]["value"] == "glm-test"
    assert rows["llm.model"]["source"] == "file"
    assert rows["domain.profile"]["source"] == "default"
    assert rows["domain.profile"]["value"] == "reverse_engineering"
    for row in data["effective"]:
        assert {"path", "value", "source", "stage"} <= set(row)
        assert row["source"] in {"file", "default"}
    assert data["config_valid"] is True


def test_get_config_does_not_resolve_or_leak_credential_files(editor, tmp_path):
    client, cfg_path = editor
    secret_file = tmp_path / "models-with-secret.json"
    secret_file.write_text('{"apiKey": "SECRET-MARKER-123"}', encoding="utf-8")
    cfg_path.write_text(
        DEFAULT_TEXT.replace('  data_dir: "data"', f'  data_dir: "data"\n  pi_models_json: "{secret_file.as_posix()}"'),
        encoding="utf-8",
    )

    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert "SECRET-MARKER-123" not in resp.text
    rows = {row["path"]: row for row in resp.json()["effective"]}
    # 只展示配置里的路径字符串，不解析 pi models.json 内容
    assert rows["paths.pi_models_json"]["value"] == secret_file.as_posix()


def test_effective_rows_include_token_budget_and_match_cli_shared_description(editor):
    client, cfg_path = editor
    body = client.get("/api/config").json()
    assert body["config_valid"] is True
    rows = body["effective"]
    by_path = {row["path"]: row for row in rows}

    # 公开配置里没有解析后的凭据；合法名字（含 max_tokens）不能被 secret 正则误过滤。
    budget = by_path["llm.max_tokens"]
    assert budget["value"] == DEFAULTS["llm"]["max_tokens"]
    assert budget["source"] == "default"
    assert "单次输出预算" in budget["stage"]

    raw = parse_config_text(cfg_path.read_text(encoding="utf-8"))
    assert rows == describe_config(raw)

    code, out, err = _run_cli(["inspect", "config", "--config", str(cfg_path), "--json"])
    assert code == 0, err
    assert json.loads(out) == rows


def test_get_repair_mode_does_not_return_unvalidated_parsed_or_effective(editor):
    client, cfg_path = editor
    broken = DEFAULT_TEXT + "export_profiles:\n  - id: qwen-xml\n    created: 2024-01-02\n"
    cfg_path.write_text(broken, encoding="utf-8")

    resp = client.get("/api/config")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["yaml"] == broken
    assert body["config_valid"] is False
    assert "export_profiles[0].created" in body["config_error"]
    # 未通过共享校验的解析结果不得伪装成有效值，也不能因 date 序列化 500。
    assert "parsed" not in body
    assert "effective" not in body

    # 语义非法但源码可读时，不动敏感路径的源码修复应放行。
    fixed = client.put("/api/config", json={"yaml": DEFAULT_TEXT})
    assert fixed.status_code == 200, fixed.text
    assert cfg_path.read_text(encoding="utf-8") == DEFAULT_TEXT


def test_raw_put_fails_closed_when_existing_sensitive_path_cannot_be_read(editor):
    client, cfg_path = editor
    cfg_path.write_text("paths: [unclosed\n", encoding="utf-8")
    before = cfg_path.read_bytes()

    resp = client.put("/api/config", json={"yaml": DEFAULT_TEXT})
    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert "paths.pi_models_json" in detail
    assert "本机编辑" in detail
    assert cfg_path.read_bytes() == before


def test_invalid_existing_config_can_be_repaired_via_raw_editor(editor):
    client, cfg_path = editor
    broken = DEFAULT_TEXT + _dump({"synthx": {"window_turns": 3}})
    cfg_path.write_text(broken, encoding="utf-8")

    got = client.get("/api/config")
    assert got.status_code == 200
    body = got.json()
    assert body["yaml"] == broken  # 源码模式必须先能看到原文件才能修
    assert body["config_valid"] is False
    assert "synthx" in body["config_error"]

    fixed = DEFAULT_TEXT.replace("window_turns: 8", "window_turns: 12")
    saved = client.put("/api/config", json={"yaml": fixed})
    assert saved.status_code == 200, saved.text
    assert cfg_path.read_text(encoding="utf-8") == fixed
    assert client.get("/api/config").json()["config_valid"] is True


@pytest.mark.parametrize("case", sorted(BAD_TEXTS))
def test_raw_put_rejects_invalid_config_atomically(editor, case):
    client, cfg_path = editor
    before = cfg_path.read_bytes()

    resp = client.put("/api/config", json={"yaml": BAD_TEXTS[case]})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["ok"] is False and body["error"]
    assert cfg_path.read_bytes() == before, "校验失败不得改写配置文件"


@pytest.mark.parametrize(
    "case,token",
    [
        ("unknown_key", "synthx"),
        ("typo_nested", "synth.window_turn"),
        ("string_bool", "judge.enabled"),
        ("out_of_range", "synth.eval_ratio"),
        ("bad_regex", "turns.root_task_patterns"),
    ],
)
def test_raw_put_error_reports_offending_field(editor, case, token):
    client, _cfg_path = editor
    resp = client.put("/api/config", json={"yaml": BAD_TEXTS[case]})
    assert resp.status_code == 400
    body = resp.json()
    assert token in body["error"]
    assert body.get("field") == token


def test_raw_put_sensitive_path_guard_covers_omission_and_parent_replacement(editor):
    client, cfg_path = editor
    cfg_path.write_text(CUSTOM_MODELS_TEXT, encoding="utf-8")
    before = cfg_path.read_bytes()

    # 1) 显式改到其它文件 -> 403
    changed = CUSTOM_MODELS_TEXT.replace("~/custom/models.json", "/tmp/evil-models.json")
    blocked = client.put("/api/config", json={"yaml": changed})
    assert blocked.status_code == 403
    assert cfg_path.read_bytes() == before

    # 2) 省略该键 = 合并后恢复内建默认，同样属于修改 -> 403
    omitted = _dump(yaml.safe_load(DEFAULT_TEXT))
    blocked_omitted = client.put("/api/config", json={"yaml": omitted})
    assert blocked_omitted.status_code == 403
    assert cfg_path.read_bytes() == before

    # 3) 不动敏感项的合法编辑照常放行
    allowed = CUSTOM_MODELS_TEXT.replace("min_gold_steps: 5", "min_gold_steps: 6")
    assert client.put("/api/config", json={"yaml": allowed}).status_code == 200
    assert cfg_path.read_text(encoding="utf-8") == allowed


def test_raw_put_sensitive_path_guard_allows_change_with_token(editor, monkeypatch):
    client, cfg_path = editor
    cfg_path.write_text(CUSTOM_MODELS_TEXT, encoding="utf-8")
    monkeypatch.setenv("GOLDTHREAD_API_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    changed = CUSTOM_MODELS_TEXT.replace("~/custom/models.json", "/tmp/other-models.json")
    resp = client.put("/api/config", json={"yaml": changed}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert cfg_path.read_text(encoding="utf-8") == changed


def test_structured_save_applies_typed_values_and_keeps_comments(editor):
    client, cfg_path = editor
    resp = client.post(
        "/api/config/structured",
        json={
            "values": [
                {"path": "gold.min_gold_steps", "value": 7},
                {"path": "judge.enabled", "value": True},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True and body["applied"] == 2
    assert body["config_hash"]

    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert saved["gold"]["min_gold_steps"] == 7
    assert saved["judge"]["enabled"] is True
    if body["engine"] == "ruamel":
        assert "# 保留注释标记" in cfg_path.read_text(encoding="utf-8")


def test_structured_defers_type_and_bounds_to_shared_parser(editor):
    client, cfg_path = editor
    resp = client.post(
        "/api/config/structured",
        json={
            "values": [
                # 共享 schema 允许负 seed；服务端不得自行加数值下限。
                {"path": "synth.seed", "value": -7},
                # 这两项可能不在稀疏 YAML 里，但属于共享 schema 的合法字段。
                {"path": "judge.task_models.dpo", "value": "dpo-model"},
                {"path": "paths.cursor_sessions_dir", "value": ""},
                {"path": "judge.labels", "value": ["positive"]},
                {"path": "synth.sft_labels", "value": ["positive"]},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert saved["synth"]["seed"] == -7
    assert saved["judge"]["task_models"]["dpo"] == "dpo-model"
    assert saved["paths"]["cursor_sessions_dir"] == ""
    assert saved["judge"]["labels"] == ["positive"]
    assert saved["synth"]["sft_labels"] == ["positive"]

    rows = {row["path"]: row for row in client.get("/api/config").json()["effective"]}
    assert rows["synth.seed"]["value"] == -7
    assert rows["synth.seed"]["source"] == "file"
    assert rows["judge.task_models.dpo"]["source"] == "file"


def test_structured_zero_thinking_target_rejected_by_shared_parser(editor):
    client, cfg_path = editor
    before = cfg_path.read_bytes()
    resp = client.post(
        "/api/config/structured",
        json={"values": [{"path": "gold.target_thinking_chars", "value": 0}]},
    )
    assert resp.status_code == 400, resp.text
    assert "gold.target_thinking_chars" in resp.json()["error"]
    assert cfg_path.read_bytes() == before


@pytest.mark.parametrize(
    "path,value",
    [
        ("judge.enabled", "true"),  # 字符串布尔不得被当成 True
        ("synth.window_turns", "8"),  # 字符串数字不得被当成 8
        ("synth.eval_ratio", 1.7),  # 类型正确但超出范围
        ("synth.window_turn", 3),  # 拼错路径
        ("llm.model.extra", "x"),  # 越过叶子继续下钻
        ("_config_path", "/tmp/x.yaml"),  # 内部键
    ],
)
def test_structured_rejects_bad_paths_and_types_atomically(editor, path, value):
    client, cfg_path = editor
    before = cfg_path.read_bytes()
    resp = client.post("/api/config/structured", json={"values": [{"path": path, "value": value}]})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["ok"] is False and body["error"]
    assert cfg_path.read_bytes() == before, "校验失败不得改写配置文件"


def test_structured_parent_replacement_cannot_bypass_sensitive_guard(editor):
    client, cfg_path = editor
    cfg_path.write_text(CUSTOM_MODELS_TEXT, encoding="utf-8")
    before = cfg_path.read_bytes()

    resp = client.post(
        "/api/config/structured",
        json={"values": [{"path": "paths", "value": {"data_dir": "data"}}]},
    )
    assert resp.status_code == 403, resp.text
    assert cfg_path.read_bytes() == before


def test_structured_refuses_semantically_invalid_base_config(editor):
    client, cfg_path = editor
    cfg_path.write_text(_dump({"synthx": {"window_turns": 3}}), encoding="utf-8")
    before = cfg_path.read_bytes()

    resp = client.post(
        "/api/config/structured",
        json={"values": [{"path": "gold.min_gold_steps", "value": 6}]},
    )
    assert resp.status_code == 400, resp.text
    assert cfg_path.read_bytes() == before


def test_validate_endpoint_uses_same_semantics_without_writing(editor):
    client, cfg_path = editor
    before = cfg_path.read_bytes()

    ok = client.post("/api/config/validate", json={"yaml": DEFAULT_TEXT})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    assert "gold" in ok.json()["keys"] and "synth" in ok.json()["keys"]

    for case in ("root_list", "typo_nested", "string_bool", "out_of_range"):
        resp = client.post("/api/config/validate", json={"yaml": BAD_TEXTS[case]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False and body["error"], case

    assert cfg_path.read_bytes() == before

WEB_HTML = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")


def test_web_config_puts_gold_first_and_model_tuning_last(tmp_path):
    gold = WEB_HTML.index('{id:"gold"')
    admission = WEB_HTML.index('{id:"admission"')
    synth = WEB_HTML.index('{id:"synth"')
    model = WEB_HTML.index('{id:"model"')
    judge = WEB_HTML.index('{id:"judge"')
    assert gold < admission < synth < model < judge

    for path in (
        "gold.state_folding",
        "gold.detour_pruning",
        "gold.fix_missing_thinking",
        "gold.silent_intermediate_steps",
        "gold.max_thinking_chars",
        "gold.target_thinking_chars",
        "gold.min_gold_steps",
        "gold.max_fold_edits",
    ):
        assert f'p:"{path}"' in WEB_HTML, f"Web 黄金控件缺少已核实的配置名: {path}"

    # 有效值只读清单、来源/阶段、离线构建强制说明、YAML 源码编辑器；
    # 公开清单来自共享 parser，不再按字段名二次过滤（避免误伤 llm.max_tokens）。
    assert "cfgEffectiveHtml()" in WEB_HTML
    assert 'CFG_OFFLINE_FORCED = {"judge.enabled": false, "turns.use_model": false}' in WEB_HTML
    assert "离线构建强制" in WEB_HTML
    assert "CFG_SECRET_RE" not in WEB_HTML
    assert 'title="${esc(r.stage)}"' in WEB_HTML
    assert 'id="cfgEditor"' in WEB_HTML
    assert "config_valid" not in WEB_HTML or "d.config_valid" in WEB_HTML

    # 提示词不得把 DPO 候选说成已批准偏好，也不得把短轨迹标记说成 SFT 硬门槛。
    assert "仅生成 DPO 候选" in WEB_HTML
    assert "不是 SFT 硬门槛" in WEB_HTML
    # 思考压缩目标不能由滑条发出 0。
    assert '{p:"gold.target_thinking_chars", k:"range", min:1,' in WEB_HTML


def _js_declaration(name: str) -> str:
    function_start = WEB_HTML.find(f"function {name}(")
    if function_start >= 0:
        start = function_start
    else:
        start = WEB_HTML.index(f"const {name} =")
    brace = WEB_HTML.index("{", start)
    depth = 0
    for idx in range(brace, len(WEB_HTML)):
        char = WEB_HTML[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return WEB_HTML[start : idx + 1] + ";"
    raise AssertionError(f"unclosed JS declaration: {name}")


def test_web_effective_rows_display_token_budget_and_full_stage(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node 不可用")
    budget_stage = "auxiliary · 单次输出预算，不是输入上下文"
    domain_stage = "stage2/stage3 · 领域 system 契约"
    rows = [
        {"path": "llm.max_tokens", "value": 8192, "source": "file", "stage": budget_stage},
        {"path": "domain.system_prompt", "value": "prompt", "source": "default", "stage": domain_stage},
    ]
    script = "\n".join(
        [
            'const CFG_STAGE_LABELS = {"stage1":"① 准入/回合","stage2":"② 黄金提纯","stage3":"③ 训练合成"};',
            "const CFG_OFFLINE_FORCED = {};",
            "const CFG = {q:\"\", effOpen:false, effective:" + json.dumps(rows, ensure_ascii=False) + "};",
            'const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");',
            _js_declaration("cfgFmtValue"),
            _js_declaration("cfgEffectiveSource"),
            _js_declaration("cfgStageLabel"),
            _js_declaration("cfgEffectiveRows"),
            _js_declaration("cfgEffectiveHtml"),
            "const html = cfgEffectiveHtml();",
            'if (!html.includes("llm.max_tokens")) throw new Error("token budget row missing");',
            'if (!html.includes("8192")) throw new Error("token budget value missing");',
            f'const budgetStage = {json.dumps(budget_stage, ensure_ascii=False)};',
            f'const domainStage = {json.dumps(domain_stage, ensure_ascii=False)};',
            'if (!html.includes(`title="${budgetStage}"`)) throw new Error("budget stage title missing");',
            'if (!html.includes(`title="${domainStage}"`)) throw new Error("multi-stage title missing");',
        ]
    )
    js_file = tmp_path / "config_effective_rows.js"
    js_file.write_text(script, encoding="utf-8")
    subprocess.run(["node", str(js_file)], check=True, capture_output=True)


def test_web_inline_script_syntax(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node 不可用")
    script = "\n".join(re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", WEB_HTML))
    js_file = tmp_path / "config_inline_check.js"
    js_file.write_text(script, encoding="utf-8")
    subprocess.run(["node", "--check", str(js_file)], check=True, capture_output=True)


def test_get_config_with_root_list_still_returns_source(editor):
    client, cfg_path = editor
    cfg_path.write_text("- 只有一个\n- 顶层列表\n", encoding="utf-8")
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["yaml"] == "- 只有一个\n- 顶层列表\n"
    assert body["config_valid"] is False
    assert body["config_error"]


def test_structured_on_root_list_config_returns_400(editor):
    client, cfg_path = editor
    cfg_path.write_text("- 只有一个\n- 顶层列表\n", encoding="utf-8")
    before = cfg_path.read_bytes()
    resp = client.post(
        "/api/config/structured",
        json={"values": [{"path": "gold.min_gold_steps", "value": 6}]},
    )
    assert resp.status_code == 400, resp.text
    assert cfg_path.read_bytes() == before
