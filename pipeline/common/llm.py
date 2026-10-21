"""通过 pi 的 models.json 配置调用模型（OpenAI 兼容接口）+ 结果缓存。"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import data_root, load_config
from .store import read_json, sha256_text, write_json_atomic

PROMPT_VERSION = "v1"


class LLMError(RuntimeError):
    pass


def _resolve_api_key(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    value = raw.strip()
    if value.startswith("!") and len(value) > 1:
        try:
            return subprocess.run(
                value[1:], shell=True, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:
            return ""
    m = re.fullmatch(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)", value)
    if m:
        return os.environ.get(m.group(1) or m.group(2), "")
    return value


def load_pi_models(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    p = Path(path or cfg["paths"]["pi_models_json"])
    data = read_json(p, default={}) or {}
    return data.get("providers", {}) or {}


def resolve_model(model_id: str, models_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    providers = load_pi_models(models_path)
    for provider_name, provider in providers.items():
        for entry in provider.get("models", []) or []:
            if entry.get("id") == model_id:
                api = provider.get("api", "openai-completions")
                if api != "openai-completions":
                    raise LLMError(f"模型 {model_id} 使用非 OpenAI 兼容协议: {api}")
                return {
                    "provider": provider_name,
                    "model": model_id,
                    "base_url": (provider.get("baseUrl") or "").rstrip("/"),
                    "api_key": _resolve_api_key(provider.get("apiKey")),
                    "context_window": entry.get("contextWindow"),
                    "max_tokens": entry.get("maxTokens"),
                    "reasoning": bool(entry.get("reasoning")),
                    "source": str(models_path or cfg["paths"]["pi_models_json"]),
                }
    raise LLMError(f"在 pi models.json 中找不到模型: {model_id}")


def _cache_dir() -> Path:
    return data_root(load_config()) / "00_state" / "llm_cache"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    p = _cache_dir() / f"{key}.json"
    if p.exists():
        return read_json(p)
    return None


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    write_json_atomic(_cache_dir() / f"{key}.json", value)


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_cache: Optional[bool] = None,
    prompt_version: str = PROMPT_VERSION,
) -> Dict[str, Any]:
    cfg = load_config()
    lcfg = cfg.get("llm", {})
    model_id = model or lcfg.get("model", "deepseek-v4.1-flash-1017")
    temperature = lcfg.get("temperature", 0.2) if temperature is None else temperature
    max_tokens = lcfg.get("max_tokens", 4096) if max_tokens is None else max_tokens
    use_cache = lcfg.get("cache", True) if use_cache is None else use_cache

    resolved = resolve_model(model_id)
    key = sha256_text(
        json.dumps(
            {
                "v": prompt_version,
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )[:32]

    if use_cache:
        cached = _cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if resolved["api_key"]:
        headers["Authorization"] = f"Bearer {resolved['api_key']}"

    url = f"{resolved['base_url']}/chat/completions"
    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=cfg.get("llm", {}).get("timeout_seconds", 180),
        )
    except requests.RequestException as exc:
        raise LLMError(f"请求失败 {url}: {exc}") from exc

    if resp.status_code >= 400:
        raise LLMError(f"HTTP {resp.status_code} {url}: {resp.text[:500]}")

    data = resp.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"响应格式异常: {json.dumps(data, ensure_ascii=False)[:500]}") from exc

    out = {
        "model": model_id,
        "provider": resolved["provider"],
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "usage": data.get("usage", {}),
        "cached": False,
    }
    if use_cache:
        _cache_put(key, out)
    return out


def chat_json(
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    prompt_version: str = PROMPT_VERSION,
) -> Any:
    """要求模型输出 JSON；自动剥离 ``` 围栏并解析。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(2):
        resp = chat(
            messages,
            model=model,
            temperature=temperature,
            prompt_version=f"{prompt_version}.try{attempt}",
        )
        text = (resp.get("content") or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_err = exc
            messages = messages + [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": "上面的输出不是合法 JSON。请只输出合法 JSON，不要任何解释。"},
            ]
    raise LLMError(f"模型未返回可解析 JSON: {last_err}")


def test_model(model: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    model_id = model or cfg["llm"]["model"]
    resolved = resolve_model(model_id)
    resp = chat(
        [{"role": "user", "content": "只回复两个字：可用"}],
        model=model_id,
        max_tokens=512,
        use_cache=False,
    )
    return {
        "model": model_id,
        "provider": resolved["provider"],
        "base_url": resolved["base_url"],
        "reply": (resp.get("content") or "").strip(),
        "usage": resp.get("usage", {}),
    }