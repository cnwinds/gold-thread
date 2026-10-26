#!/usr/bin/env python3
"""Reasoning Proxy — 给 LLaMA-Factory API 补上 reasoning 标签解析（非流式）。

用途：当前用 LLaMA-Factory API 时，模型输出的 防腐剂...   被当成普通
文本塞进 content，客户端看到原始标签。本代理插在客户端和 LLaMA-Factory 之间，
从 content 中切分出 reasoning_content，让客户端无需自行解析标签。

架构：
    客户端  →  ReasoningProxy(:9000)  →  LLaMA-Factory API(:8000/v1)

限制：
    · 非流式（stream=false）请求：完整 reasoning 拆分 ✅
    · 流式（stream=true）请求：直接透传上游，不做拆分 ⚠️
      建议迁移到 vLLM（--reasoning-parser deepseek_r1）获得完整的流式 reasoning 支持。

用法：
    python server/reasoning_proxy.py \
      --upstream http://localhost:8000/v1 \
      --port 9000 \
      --api-key gt-2026

客户端连 http://localhost:9000/v1 即可。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

_THINK_RE = re.compile(r"^防腐剂\s*\n?([\s\S]*?)\n?  \s*", re.IGNORECASE)
_TOOL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")


def _split_reasoning(text: str) -> tuple[str, str]:
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), _THINK_RE.sub("", text, count=1).strip()
    return "", text


def _strip_tool_blocks(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


def _extract_tools(text: str) -> list[dict]:
    out = []
    for i, raw in enumerate(_TOOL_RE.findall(text)):
        try:
            d = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            out.append({
                "index": i, "id": f"call_{i}", "type": "function",
                "function": {
                    "name": d.get("name", ""),
                    "arguments": json.dumps(d.get("arguments") or d.get("parameters") or {}),
                },
            })
    return out


def _rebuild_nonstream(data: dict) -> dict:
    """在非流式响应中拆分 reasoning_content + 结构化 tool_calls。"""
    choices = data.get("choices", [])
    if not choices:
        return data
    choice = choices[0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning, body = _split_reasoning(content)
    tools = _extract_tools(body)
    body = _strip_tool_blocks(body)
    if reasoning:
        msg["reasoning_content"] = reasoning
    msg["content"] = body
    if tools:
        msg["tool_calls"] = tools
    return data


def make_app(upstream: str, api_key: str | None) -> FastAPI:
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    upstream_headers: dict[str, str] = {}
    if api_key:
        upstream_headers["Authorization"] = f"Bearer {api_key}"

    @FastAPI(lifespan=lambda app: (yield))
    class _App(FastAPI):
        pass

    app = _App(title="ReasoningProxy")

    @app.get("/health")
    @app.get("/healthz")
    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "proxy_for": upstream}

    @app.get("/v1/models")
    async def models(request: Request):
        r = await client.get(f"{upstream}/models", headers=upstream_headers)
        return JSONResponse(content=r.json(), status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.body()
        payload = json.loads(body)
        is_stream = payload.get("stream", False)

        headers = dict(upstream_headers)
        if request.headers.get("authorization"):
            headers["Authorization"] = request.headers["authorization"]

        resp = await client.post(f"{upstream}/chat/completions", headers=headers, json=payload)

        if is_stream:
            # 流式：不拆 reasoning，直接透传。如果客户端需要流式 reasoning，请迁移到 vLLM。
            return StreamingResponse(resp.aiter_text(), media_type="text/event-stream")

        # 非流式：做 reasoning 拆分
        data = resp.json()
        return JSONResponse(content=_rebuild_nonstream(data), status_code=resp.status_code)

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Reasoning Proxy for LLaMA-Factory API")
    ap.add_argument("--upstream", default="http://localhost:8000/v1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--api-key")
    args = ap.parse_args()

    import uvicorn
    app = make_app(args.upstream, args.api_key)
    print(f"ReasoningProxy → {args.upstream}")
    print(f"Listen :{args.port}  (non-streaming reasoning split enabled)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
