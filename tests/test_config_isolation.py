"""Effective-config isolation for auxiliary LLM selection, cache and run records.

Every test stays inside ``tmp_path``: model files are local fixtures, transport is
a captured fake, and credentials are synthetic. Nothing may touch the real
``data/`` workspace or ``config/pipeline.yaml``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

import pipeline.run as run_mod
import pipeline.stage1_extract as stage1_extract
from pipeline import judge, turnclassify
from pipeline.common import llm
from pipeline.common.config import DEFAULTS


class _FakeResponse:
    status_code = 200
    text = "ok"

    def __init__(self, content: str) -> None:
        self._content = content

    def json(self) -> Dict[str, Any]:
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {"total_tokens": 3},
        }


def _write_models(
    path: Path,
    provider: str,
    base_url: str,
    api_key: str,
    model_id: str = "shared-model",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "providers": {
                    provider: {
                        "api": "openai-completions",
                        "baseUrl": base_url,
                        "apiKey": api_key,
                        "models": [{"id": model_id}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _cfg(tmp_path: Path, models_path: Path, **llm_overrides: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["paths"]["data_dir"] = str(tmp_path / "workspace")
    cfg["paths"]["pi_models_json"] = str(models_path)
    cfg["llm"].update(llm_overrides)
    cfg["_config_path"] = str(tmp_path / "pipeline.yaml")
    return cfg


def _install_post(monkeypatch, calls: List[Dict[str, Any]], reply: str = "fake-reply") -> None:
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "payload": json, "timeout": timeout})
        return _FakeResponse(reply)

    monkeypatch.setattr(llm.requests, "post", fake_post)


def test_chat_uses_effective_cfg_for_models_defaults_timeout_and_cache(tmp_path, monkeypatch):
    models = _write_models(
        tmp_path / "models.json", "alpha", "https://alpha.example/v1", "sk-alpha-secret"
    )
    cfg = _cfg(
        tmp_path,
        models,
        model="shared-model",
        temperature=0.77,
        max_tokens=1234,
        timeout_seconds=7,
        cache=True,
    )
    calls: List[Dict[str, Any]] = []
    _install_post(monkeypatch, calls, reply="alpha-out")

    out = llm.chat([{"role": "user", "content": "hi"}], cfg=cfg)

    assert out["content"] == "alpha-out"
    assert out["provider"] == "alpha"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://alpha.example/v1/chat/completions"
    assert call["payload"]["model"] == "shared-model"
    assert call["payload"]["temperature"] == 0.77
    assert call["payload"]["max_tokens"] == 1234
    assert call["timeout"] == 7
    assert call["headers"]["Authorization"] == "Bearer sk-alpha-secret"

    cache_dir = Path(cfg["paths"]["data_dir"]) / "00_state" / "llm_cache"
    files = sorted(cache_dir.glob("*.json"))
    assert len(files) == 1
    artifact = files[0].read_text(encoding="utf-8")
    assert "sk-alpha-secret" not in files[0].name
    assert "sk-alpha-secret" not in artifact

    cached = llm.chat([{"role": "user", "content": "hi"}], cfg=cfg)
    assert cached["cached"] is True
    assert len(calls) == 1, "same effective cfg must reuse the cache"


def test_cache_identity_separates_providers_and_endpoints_without_secrets(tmp_path, monkeypatch):
    models_a = _write_models(
        tmp_path / "a.json", "alpha", "https://alpha.example/v1", "sk-secret-alpha"
    )
    models_b = _write_models(
        tmp_path / "b.json", "beta", "https://beta.example/v1", "sk-secret-beta"
    )
    cfg_a = _cfg(tmp_path, models_a, model="shared-model", cache=True)
    cfg_b = _cfg(tmp_path, models_b, model="shared-model", cache=True)
    assert cfg_a["paths"]["data_dir"] == cfg_b["paths"]["data_dir"]

    calls: List[Dict[str, Any]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "payload": json, "timeout": timeout})
        return _FakeResponse("from-alpha" if "alpha.example" in url else "from-beta")

    monkeypatch.setattr(llm.requests, "post", fake_post)

    first_a = llm.chat([{"role": "user", "content": "same"}], cfg=cfg_a)
    first_b = llm.chat([{"role": "user", "content": "same"}], cfg=cfg_b)
    assert first_a["content"] == "from-alpha"
    assert first_b["content"] == "from-beta"
    assert len(calls) == 2, "provider/endpoint must be part of the cache identity"

    cached_a = llm.chat([{"role": "user", "content": "same"}], cfg=cfg_a)
    cached_b = llm.chat([{"role": "user", "content": "same"}], cfg=cfg_b)
    assert cached_a["content"] == "from-alpha" and cached_a["cached"] is True
    assert cached_b["content"] == "from-beta" and cached_b["cached"] is True
    assert len(calls) == 2

    cache_dir = Path(cfg_a["paths"]["data_dir"]) / "00_state" / "llm_cache"
    files = sorted(cache_dir.glob("*.json"))
    assert len(files) == 2
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "sk-secret-alpha" not in path.name + text
        assert "sk-secret-beta" not in path.name + text


def test_chat_honours_effective_cache_setting(tmp_path, monkeypatch):
    models = _write_models(
        tmp_path / "models.json", "alpha", "https://alpha.example/v1", "sk-alpha-secret"
    )
    cfg = _cfg(tmp_path, models, model="shared-model", cache=False)
    calls: List[Dict[str, Any]] = []
    _install_post(monkeypatch, calls)

    llm.chat([{"role": "user", "content": "hi"}], cfg=cfg)
    llm.chat([{"role": "user", "content": "hi"}], cfg=cfg)

    assert len(calls) == 2
    assert not (Path(cfg["paths"]["data_dir"]) / "00_state" / "llm_cache").exists()


def test_test_model_uses_effective_cfg(tmp_path, monkeypatch):
    models = _write_models(
        tmp_path / "models.json", "alpha", "https://alpha.example/v1", "sk-alpha-secret"
    )
    cfg = _cfg(tmp_path, models, model="shared-model", cache=True)
    calls: List[Dict[str, Any]] = []
    _install_post(monkeypatch, calls, reply="可用")

    out = llm.test_model(cfg=cfg)

    assert out["model"] == "shared-model"
    assert out["base_url"] == "https://alpha.example/v1"
    assert out["reply"] == "可用"
    assert calls[0]["payload"]["max_tokens"] == 512
    assert calls[0]["timeout"] == cfg["llm"]["timeout_seconds"]
    assert not (Path(cfg["paths"]["data_dir"]) / "00_state" / "llm_cache").exists()


def test_turnclassify_forwards_effective_cfg_to_model(tmp_path, monkeypatch):
    models = _write_models(
        tmp_path / "models.json", "alpha", "https://alpha.example/v1", "sk-alpha-secret"
    )
    cfg = _cfg(tmp_path, models, model="shared-model")
    seen: Dict[str, Any] = {}

    def fake_chat_json(system, user, **kwargs):
        seen.update(kwargs)
        seen["system"] = system
        return {
            "root_turn": 0,
            "turns": [{"index": 0, "kind": "root", "drop": False, "why": "目标"}],
        }

    monkeypatch.setattr(turnclassify, "chat_json", fake_chat_json)
    session = {
        "cwd": "D:/crack/example",
        "messages": [
            {"role": "user", "content": "破解 example.apk"},
            {"role": "assistant", "content": "开始", "tool_calls": [], "tool_results": []},
        ],
    }

    result = turnclassify.classify_session(session, cfg=cfg, log=lambda *_: None)

    assert seen["cfg"] is cfg
    assert result is not None
    assert result["root_turn"] == 0
    assert result["model"] == "shared-model"


def test_judge_call_forwards_effective_cfg_to_model(tmp_path, monkeypatch):
    models = _write_models(
        tmp_path / "models.json", "alpha", "https://alpha.example/v1", "sk-alpha-secret"
    )
    cfg = _cfg(tmp_path, models, model="shared-model")
    cfg["judge"]["enabled"] = True
    seen: Dict[str, Any] = {}

    def fake_chat_json(system, user, **kwargs):
        seen.update(kwargs)
        return {"decisions": []}

    monkeypatch.setattr(judge, "chat_json", fake_chat_json)
    j = judge._jcfg(cfg)

    out = judge._call("system", "schema", {"x": 1}, cfg, j, "steps", log=lambda *_: None)

    assert out == {"decisions": []}
    assert seen["cfg"] is cfg
    assert seen["model"] == "shared-model"


def _write_pipeline_config(path: Path, data_dir: Path) -> Path:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["pi_models_json"] = str(path.parent / "models.json")
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_pipeline_run_records_effective_workspace_config_and_force(tmp_path, monkeypatch):
    data_dir = tmp_path / "isolated-workspace"
    cfg_path = _write_pipeline_config(tmp_path / "pipeline.yaml", data_dir)

    real_load_config = run_mod.load_config

    def guarded_load_config(path=None):
        if path is None:
            raise AssertionError("pipeline.run must not reload the default config")
        return real_load_config(path)

    monkeypatch.setattr(run_mod, "load_config", guarded_load_config)
    monkeypatch.setattr(stage1_extract, "run", lambda **kwargs: {"stage": "stage1", "ok": True})

    assert run_mod.main(["stage1", "--config", str(cfg_path), "--force"]) == 0

    runs_path = data_dir / "00_state" / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    assert runs[0]["stage"] == "stage1"
    assert runs[0]["status"] == "done"
    assert runs[0]["force"] is True
    assert runs[0]["config"] == str(cfg_path)
    assert runs[0]["result"] == {"stage": "stage1", "ok": True}

    def failing_run(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stage1_extract, "run", failing_run)
    with pytest.raises(RuntimeError, match="boom"):
        run_mod.main(["stage1", "--config", str(cfg_path), "--force"])

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    assert runs[0]["stage"] == "stage1"
    assert runs[0]["status"] == "failed"
    assert runs[0]["force"] is True
    assert runs[0]["config"] == str(cfg_path)
    assert runs[0]["error"] == "boom"