"""配置读取失败关闭：不可读原文 == 不写文件，且旧敏感路径基线必须来自同一份快照。

覆盖点（全部使用临时目录，不触碰真实 ``config/pipeline.yaml``）：

1. 配置文件不可读（PermissionError 等，用假路径对象模拟，不改真实文件权限）时，
   GET 明确报错而不是伪装成空配置；PUT 失败关闭且原文一个字节都不变。
2. 旧文件里的 ``paths.pi_models_json`` 若因重复键/YAML 语法错误/类型非法而无法
   确定，不得建立基线，PUT 一律 403。
3. 与敏感路径无关的语义错误仍允许通过原文 PUT 修复（保留修复通道）。
4. 配置缺失（只有 FileNotFoundError）依旧等价于空配置，可直接写入。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.common import config as config_module
import server.app as web


DEFAULT_TEXT = """paths:
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

SENSITIVE_TEXT = DEFAULT_TEXT.replace(
    '  data_dir: "data"', '  data_dir: "data"\n  pi_models_json: "~/custom/models.json"'
)


def _error_text(resp) -> str:
    """兼容两种错误载荷：配置端点用 ``error``，``HTTPException`` 用 ``detail``。"""
    body = resp.json()
    return str(body.get("error") or body.get("detail") or "")


def _unreadable(path: Path, exc: BaseException) -> object:
    """包装真实临时路径：读原文时抛出给定异常，避免改动真实文件权限。"""

    class _UnreadablePath:
        def __init__(self, inner: Path) -> None:
            self._inner = inner

        def __str__(self) -> str:
            return str(self._inner)

        def __fspath__(self) -> str:
            return str(self._inner)

        def read_text(self, encoding: str = "utf-8", errors: str | None = None) -> str:
            raise exc

        @property
        def parent(self) -> Path:
            return self._inner.parent

        @property
        def name(self) -> str:
            return self._inner.name

    return _UnreadablePath(path)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    monkeypatch.delenv("GOLDTHREAD_API_TOKEN", raising=False)
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(SENSITIVE_TEXT, encoding="utf-8")
    monkeypatch.setattr(web, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    return TestClient(web.app, raise_server_exceptions=False), cfg_path


def test_put_fails_closed_when_config_content_unreadable(editor, monkeypatch):
    client, cfg_path = editor
    before = cfg_path.read_bytes()
    monkeypatch.setattr(web, "CONFIG_PATH", _unreadable(cfg_path, PermissionError(13, "Permission denied")))

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT})

    assert resp.status_code == 500, resp.text
    assert "本机" in _error_text(resp), resp.json()
    assert cfg_path.read_bytes() == before, "读不到旧配置时绝不能落盘"


def test_get_config_reports_unreadable_content_instead_of_empty_yaml(editor, monkeypatch):
    client, cfg_path = editor
    monkeypatch.setattr(web, "CONFIG_PATH", _unreadable(cfg_path, OSError(5, "Input/output error")))

    resp = client.get("/api/config")

    assert resp.status_code == 500, resp.text
    assert "本机" in _error_text(resp), resp.json()


def test_non_utf8_content_is_not_treated_as_empty(editor, monkeypatch):
    client, cfg_path = editor
    before = cfg_path.read_bytes()
    monkeypatch.setattr(
        web, "CONFIG_PATH", _unreadable(cfg_path, UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))
    )

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT})

    assert resp.status_code == 500, resp.text
    assert cfg_path.read_bytes() == before


def test_duplicate_sensitive_key_file_cannot_establish_baseline(editor):
    client, cfg_path = editor
    cfg_path.write_text(
        'paths:\n  data_dir: "data"\n  pi_models_json: "~/a.json"\n  pi_models_json: "~/b.json"\n',
        encoding="utf-8",
    )
    before = cfg_path.read_bytes()

    # 候选值等于「后出现的那个」旧值，若按宽松解析就会误判为未改动。
    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT.replace("~/custom/models.json", "~/b.json")})

    assert resp.status_code == 403, resp.text
    assert cfg_path.read_bytes() == before


def test_malformed_yaml_file_cannot_establish_baseline(editor):
    client, cfg_path = editor
    cfg_path.write_text('paths:\n  pi_models_json: "~/custom/models.json"\nllm: [unclosed\n', encoding="utf-8")
    before = cfg_path.read_bytes()

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT})

    assert resp.status_code == 403, resp.text
    assert cfg_path.read_bytes() == before


@pytest.mark.parametrize("bad_value", ["123", "~", "[]", '""'])
def test_non_string_old_models_path_cannot_establish_baseline(editor, bad_value):
    client, cfg_path = editor
    original = f'paths:\n  data_dir: "data"\n  pi_models_json: {bad_value}\n'
    cfg_path.write_text(original, encoding="utf-8")
    before = cfg_path.read_bytes()

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT})

    assert resp.status_code == 403, resp.text
    assert cfg_path.read_bytes() == before


def test_unrelated_semantic_error_still_allows_repair_put(editor):
    client, cfg_path = editor
    # 只有与敏感路径无关的未知字段错误；旧敏感路径可确定 -> 允许修复。
    cfg_path.write_text(SENSITIVE_TEXT + "\nsynthx:\n  window_turns: 3\n", encoding="utf-8")

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT})

    assert resp.status_code == 200, resp.text
    assert cfg_path.read_text(encoding="utf-8") == SENSITIVE_TEXT


def test_existing_sensitive_path_cannot_change_without_token(editor):
    client, cfg_path = editor
    before = cfg_path.read_bytes()

    resp = client.put("/api/config", json={"yaml": SENSITIVE_TEXT.replace("~/custom/models.json", "~/evil.json")})

    assert resp.status_code == 403, resp.text
    assert cfg_path.read_bytes() == before


def test_missing_config_file_still_counts_as_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("GOLDTHREAD_API_TOKEN", raising=False)
    cfg_path = tmp_path / "absent.yaml"
    monkeypatch.setattr(web, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    client = TestClient(web.app, raise_server_exceptions=False)

    resp = client.put("/api/config", json={"yaml": DEFAULT_TEXT})

    assert resp.status_code == 200, resp.text
    assert cfg_path.read_text(encoding="utf-8") == DEFAULT_TEXT