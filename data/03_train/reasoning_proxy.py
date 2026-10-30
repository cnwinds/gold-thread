#!/usr/bin/env python3
"""Reasoning Proxy — 给 LLaMA-Factory API 补上 reasoning 标签解析。

用途：LLaMA-Factory API 把模型输出的 <think>…</think> 当普通文本塞进 content，
客户端看到原始标签。本代理插在客户端和 LLaMA-Factory 之间，把思考切到
reasoning_content，把 <tool_call> 块转成结构化 tool_calls。

架构：
    客户端  →  ReasoningProxy(:9000)  →  LLaMA-Factory API(:8000/v1)

行为：
    · 非流式：拆分 reasoning_content + 结构化 tool_calls（有工具调用时 finish_reason="tool_calls"）
    · 流式：逐块透传，</think> 之前的增量改写为 delta.reasoning_content；
      请求里 chat_template_kwargs.enable_thinking == false 时不拆分。
      流式不解析 <tool_call>（原样留在 content 中）。
    · Qwen3 等模板会把 <think> 放进生成前缀，输出里只有 </think>：两种形式都支持。

用法：
    python server/reasoning_proxy.py --upstream http://localhost:8000/v1 --port 9000 --api-key gt-2026

客户端连 http://127.0.0.1:9000/v1 即可。默认只监听 127.0.0.1；需要对外提供时显式传 --host 0.0.0.0。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

_THINK_RE = re.compile(r"^\s*(?:<think>)?\s*([\s\S]*?)\s*</think>\s*", re.IGNORECASE)
_TOOL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_OPEN = "<think>"
_CLOSE = "</think>"


def _split_reasoning(text: str) -> Tuple[str, str]:
    m = _THINK_RE.match(text or "")
    if m:
        return m.group(1).strip(), (text or "")[m.end():].strip()
    return "", text or ""


def _strip_tool_blocks(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


def _extract_tools(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in _TOOL_RE.findall(text):
        try:
            d = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        args = d.get("arguments", d.get("parameters", {}))
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False)
        i = len(out)
        out.append({
            "index": i, "id": f"call_{i}", "type": "function",
            "function": {"name": d.get("name", ""), "arguments": args},
        })
    return out


def _rebuild_nonstream(data: Dict[str, Any]) -> Dict[str, Any]:
    """在非流式响应中拆分 reasoning_content + 结构化 tool_calls。"""
    for choice in data.get("choices") or []:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning, body = _split_reasoning(content)
        tools = _extract_tools(body)
        if reasoning:
            msg["reasoning_content"] = reasoning
        msg["content"] = _strip_tool_blocks(body) if tools else body
        if tools:
            msg["tool_calls"] = tools
            choice["finish_reason"] = "tool_calls"
        choice["message"] = msg
    return data


class _StreamSplitter:
    """把增量文本切成 (reasoning, content)：</think> 之前算思考，之后算正文。"""

    def __init__(self, enabled: bool) -> None:
        self.mode = "think" if enabled else "content"
        self.buf = ""
        self.started = False

    def feed(self, text: str) -> Tuple[str, str]:
        if self.mode == "content":
            return "", text
        self.buf += text
        if not self.started:
            head = self.buf.lstrip()
            if len(head) < len(_OPEN) and _OPEN.startswith(head):
                return "", ""
            if head.startswith(_OPEN):
                self.buf = head[len(_OPEN):]
            self.started = True
        idx = self.buf.find(_CLOSE)
        if idx >= 0:
            reasoning, rest = self.buf[:idx], self.buf[idx + len(_CLOSE):].lstrip()
            self.buf, self.mode = "", "content"
            return reasoning, rest
        keep = len(_CLOSE) - 1
        if len(self.buf) <= keep:
            return "", ""
        out, self.buf = self.buf[:-keep], self.buf[-keep:]
        return out, ""

    def flush(self) -> Tuple[str, str]:
        if self.mode == "content" or not self.buf:
            return "", ""
        out, self.buf = self.buf, ""
        return out, ""


async def _stream_split(resp: httpx.Response, enabled: bool) -> AsyncIterator[bytes]:
    splitters: Dict[int, _StreamSplitter] = {}
    try:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                yield (line + "\n").encode("utf-8")
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                yield b"data: [DONE]\n\n"
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                yield (line + "\n").encode("utf-8")
                continue
            for choice in chunk.get("choices") or []:
                sp = splitters.setdefault(int(choice.get("index", 0)), _StreamSplitter(enabled))
                delta = choice.get("delta") or {}
                text = delta.get("content")
                r, c = sp.feed(text) if isinstance(text, str) else ("", "")
                if choice.get("finish_reason"):
                    fr, fc = sp.flush()
                    r, c = r + fr, c + fc
                if isinstance(text, str):
                    if r:
                        delta["reasoning_content"] = (delta.get("reasoning_content") or "") + r
                    delta["content"] = c
                    choice["delta"] = delta
            yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
    finally:
        await resp.aclose()


def make_app(upstream: str, api_key: str | None = None) -> FastAPI:
    upstream = upstream.rstrip("/")
    upstream_headers: Dict[str, str] = {}
    if api_key:
        upstream_headers["Authorization"] = f"Bearer {api_key}"
    state: Dict[str, httpx.AsyncClient] = {}

    def client() -> httpx.AsyncClient:
        if "client" not in state:
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return state["client"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        c = state.pop("client", None)
        if c is not None:
            await c.aclose()

    app = FastAPI(title="ReasoningProxy", lifespan=lifespan)

    @app.get("/health")
    @app.get("/healthz")
    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "proxy_for": upstream}

    @app.get("/v1/models")
    async def models(request: Request):
        headers = dict(upstream_headers)
        if request.headers.get("authorization"):
            headers["Authorization"] = request.headers["authorization"]
        r = await client().get(f"{upstream}/models", headers=headers)
        return JSONResponse(content=r.json(), status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        headers = dict(upstream_headers)
        if request.headers.get("authorization"):
            headers["Authorization"] = request.headers["authorization"]

        if payload.get("stream"):
            kwargs = payload.get("chat_template_kwargs") or {}
            enabled = kwargs.get("enable_thinking", True) is not False
            req = client().build_request("POST", f"{upstream}/chat/completions", headers=headers, json=payload)
            resp = await client().send(req, stream=True)
            if resp.status_code >= 400:
                body = await resp.aread()
                await resp.aclose()
                return JSONResponse(content=_safe_json(body), status_code=resp.status_code)
            return StreamingResponse(_stream_split(resp, enabled), media_type="text/event-stream")

        resp = await client().post(f"{upstream}/chat/completions", headers=headers, json=payload)
        data = _safe_json(resp.content)
        if resp.status_code >= 400 or not isinstance(data, dict):
            return JSONResponse(content=data, status_code=resp.status_code)
        return JSONResponse(content=_rebuild_nonstream(data), status_code=resp.status_code)

    return app


def _safe_json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": body.decode("utf-8", "replace")[:2000]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Reasoning Proxy for LLaMA-Factory API")
    ap.add_argument("--upstream", default="http://localhost:8000/v1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key")
    args = ap.parse_args()

    import uvicorn

    app = make_app(args.upstream, args.api_key)
    print(f"ReasoningProxy → {args.upstream}")
    print(f"Listen {args.host}:{args.port}  (reasoning split: non-stream + stream)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
