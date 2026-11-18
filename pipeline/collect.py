"""采集入口：从 pi 会话目录或 Cursor 会话目录里预览、勾选，再整段收进 01_raw。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common.config import load_config, resolve_path, stage_dirs
from .common import pilog
from .common.store import load_index, now_iso
from .stage1_extract import adopt_file

_USER_QUERY = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)
_FILE_BLOCK = re.compile(
    r"<file\b[^>]*>\s*([\s\S]*?)\s*</file>",
    re.IGNORECASE,
)
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{12,}|pkcp_[A-Za-z0-9_\-]{12,})\b")


def _redact(text: str) -> str:
    return _SECRET.sub("…", text or "")


def _clip(text: str, n: int = 700) -> str:
    text = _redact(text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "…"


def _user_text(text: str) -> str:
    """取用户首问正文：优先 <user_query>，否则展开 <file> 内文，去掉附件标签外壳。"""
    raw = text or ""
    found = _USER_QUERY.search(raw)
    body = found.group(1) if found else raw
    file_parts = [m.group(1).strip() for m in _FILE_BLOCK.finditer(body) if m.group(1).strip()]
    if file_parts:
        body = "\n".join(file_parts)
    body = re.sub(r"<image_files>[\s\S]*?</image_files>", "", body)
    body = re.sub(r"<timestamp>[\s\S]*?</timestamp>", "", body)
    return body.strip()


def _pi_root(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["sessions_dir"]).resolve()


def _cursor_root(cfg: Dict[str, Any]) -> Path:
    custom = (cfg.get("paths") or {}).get("cursor_sessions_dir")
    if custom:
        return resolve_path(custom).resolve()
    return (Path.home() / ".cursor" / "projects").resolve()


def _under(root: Path, path: Path) -> Path:
    """确保 path 逻辑上落在 root 下。

    用 normpath 折叠 ``..``，但不 follow Windows Junction / symlink 目标。
    pi 常把 sessions 子目录联到项目 run 目录；resolve() 会误判为超根。
    """
    root_abs = Path(os.path.normpath(str(root.expanduser().absolute())))
    path_abs = Path(os.path.normpath(str(path.expanduser().absolute())))
    try:
        path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError("路径超出会话源目录") from exc
    return path_abs


def _adopted_ids(cfg: Dict[str, Any]) -> set[str]:
    idx = load_index(stage_dirs(cfg)["raw"] / "index.json")
    return set((idx.get("sessions") or {}).keys())


def _read_json_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _pi_header_and_title(path: Path) -> Dict[str, str]:
    sid = path.stem.split("_")[-1]
    cwd = ""
    ts = ""
    title = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"role":"user"' not in line and '"role": "user"' not in line:
                if line.startswith('{"type":"session"'):
                    obj = _read_json_line(line)
                    if obj:
                        sid = str(obj.get("id") or sid)
                        cwd = str(obj.get("cwd") or "")
                        ts = str(obj.get("timestamp") or "")
                continue
            obj = _read_json_line(line)
            if not obj or obj.get("type") != "message":
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                title = content
            elif isinstance(content, list):
                title = "\n".join(
                    str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            title = _user_text(title)
            break
    return {"session_id": sid, "cwd": cwd, "timestamp": ts, "title": _clip(title, 160)}


def _cursor_title(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"role":"user"' not in line and '"role": "user"' not in line:
                continue
            obj = _read_json_line(line)
            if not obj or obj.get("role") != "user":
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            content = msg.get("content")
            text = ""
            if isinstance(content, list):
                text = "\n".join(
                    str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            return _clip(_user_text(text), 160)
    return ""


def list_sessions(source: str, q: str = "", cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    source = _source(source)
    adopted = _adopted_ids(cfg)
    rows: List[Dict[str, Any]] = []
    if source == "pi":
        root = _pi_root(cfg)
        if root.exists():
            for path in sorted(root.rglob("*.jsonl")):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                info = _pi_header_and_title(path)
                st = path.stat()
                rows.append({
                    "id": rel,
                    "session_id": info["session_id"],
                    "title": info["title"] or info["session_id"],
                    "project": info["cwd"] or rel.split("/")[0],
                    "timestamp": info["timestamp"] or datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                    "bytes": st.st_size,
                    "adopted": info["session_id"] in adopted,
                    "origin": "pi",
                })
    else:
        root = _cursor_root(cfg)
        if root.exists():
            for path in sorted(root.glob("*/agent-transcripts/*/*.jsonl")):
                if "subagents" in path.parts:
                    continue
                uuid = path.parent.name
                if path.stem != uuid:
                    continue
                slug = path.parents[2].name
                st = path.stat()
                sid = uuid
                rows.append({
                    "id": f"{slug}/{uuid}",
                    "session_id": sid,
                    "title": _cursor_title(path) or uuid,
                    "project": slug,
                    "timestamp": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                    "bytes": st.st_size,
                    "adopted": sid in adopted,
                    "origin": "cursor",
                })
    needle = (q or "").strip().lower()
    if needle:
        # 一并匹配 catalog 相对路径：pi 会话目录名常含 teacher/包名，而 cwd 可能仍是旧工程路径
        rows = [
            r for r in rows
            if needle in f"{r['title']} {r['project']} {r['session_id']} {r['id']}".lower()
        ]
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    rows.sort(key=lambda r: r["adopted"])
    return {
        "source": source,
        "root": str(root),
        "total": len(rows),
        "adopted": sum(1 for r in rows if r["adopted"]),
        "sessions": rows[:400],
    }


def _source(source: str) -> str:
    if source not in ("pi", "cursor"):
        raise ValueError("source 必须是 pi 或 cursor")
    return source


def resolve_catalog(source: str, catalog_id: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    source = _source(source)
    catalog_id = (catalog_id or "").replace("\\", "/").strip()
    if not catalog_id or catalog_id.startswith("/") or ".." in catalog_id.split("/"):
        raise ValueError("非法会话编号")
    if source == "pi":
        root = _pi_root(cfg)
        path = _under(root, root / catalog_id)
        if path.suffix != ".jsonl" or not path.is_file():
            raise FileNotFoundError("找不到这份 pi 会话")
        return path
    parts = catalog_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("非法 Cursor 会话编号")
    slug, uuid = parts
    root = _cursor_root(cfg)
    path = _under(root, root / slug / "agent-transcripts" / uuid / f"{uuid}.jsonl")
    if not path.is_file():
        raise FileNotFoundError("找不到这份 Cursor 会话")
    return path


def _redact_flow_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(msg)
    for key in ("content", "text", "thinking"):
        if isinstance(out.get(key), str):
            out[key] = _redact(out[key])
    if isinstance(out.get("tool_calls"), list):
        out["tool_calls"] = [
            {
                **tc,
                "arguments": _redact(json.dumps(tc.get("arguments"), ensure_ascii=False))
                if tc.get("arguments") is not None
                else tc.get("arguments"),
            }
            if isinstance(tc, dict)
            else tc
            for tc in out["tool_calls"]
        ]
    if isinstance(out.get("tool_results"), list):
        out["tool_results"] = [
            {**tr, "content": _redact(str(tr.get("content") or ""))} if isinstance(tr, dict) else tr
            for tr in out["tool_results"]
        ]
    return out


def _parse_catalog_session(source: str, path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if source == "cursor":
        uuid = path.parent.name
        slug = path.parents[2].name
        staged = stage_dirs(cfg)["state"] / "inbox" / "preview" / f"{uuid}.jsonl"
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            _cursor_to_pi(path, staged, uuid, slug)
            return pilog.parse_session(staged, max_tool_output_chars=8000)
        finally:
            staged.unlink(missing_ok=True)
    return pilog.parse_session(path, max_tool_output_chars=8000)


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(p for p in parts if p)


def preview_session(source: str, catalog_id: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    source = _source(source)
    path = resolve_catalog(source, catalog_id, cfg)
    if source == "pi":
        info = _pi_header_and_title(path)
        session_id = info["session_id"]
        title = info["title"] or catalog_id
    else:
        session_id = path.parent.name
        title = _cursor_title(path) or session_id
    adopted = session_id in _adopted_ids(cfg)
    session = _parse_catalog_session(source, path, cfg)
    if source == "pi":
        project = str(info.get("cwd") or session.get("cwd") or catalog_id.split("/")[0])
    else:
        project = path.parents[2].name
    flow, turns, _stats = pilog.build_flow(session, cfg, None, {})
    flow = [_redact_flow_message(m) for m in flow]
    assistant_errors = list(session.get("assistant_errors") or [])
    useful = [
        m for m in flow
        if m.get("role") not in ("system",) and (
            (m.get("content") or m.get("text") or "").strip() or m.get("tool_calls")
        )
    ]
    empty = len(useful) == 0
    empty_reason = ""
    if empty and assistant_errors:
        empty_reason = str(assistant_errors[-1].get("error") or "")
    elif empty:
        empty_reason = "没有有效的助手回复"
    return {
        "id": catalog_id,
        "source": source,
        "session_id": session_id,
        "title": title,
        "project": project,
        "path": catalog_id,
        "adopted": adopted,
        "messages": flow,
        "turns": turns,
        "truncated": False,
        "assistant_errors": assistant_errors,
        "empty": empty,
        "empty_reason": empty_reason,
    }


def _tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item.get("text")))
                elif item.get("type") == "image":
                    parts.append("[Image Output]")
                elif "content" in item:
                    sub = _tool_result_text(item.get("content"))
                    if sub:
                        parts.append(sub)
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        if "text" in value:
            return str(value.get("text") or "")
        if "content" in value:
            return _tool_result_text(value.get("content"))
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _parse_tool_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except Exception:  # noqa: BLE001
            return {"raw": raw}
    return {}


def _cursor_to_pi(src: Path, dest: Path, session_id: str, cwd: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: List[str] = []
    out.append(json.dumps({
        "type": "session", "version": 3, "id": session_id, "timestamp": ts, "cwd": cwd,
    }, ensure_ascii=False))
    active_model = "cursor"
    active_provider = "cursor"
    out.append(json.dumps({
        "type": "model_change", "id": "cursor-model", "timestamp": ts,
        "provider": active_provider, "modelId": active_model,
    }, ensure_ascii=False))
    n = 0
    call_names: Dict[str, str] = {}

    def _emit_tool_result(call_id: str, tool_name: str, is_err: bool, text_val: str, ev_ts: str) -> None:
        nonlocal n
        n += 1
        out.append(json.dumps({
            "type": "message",
            "id": f"tr{n}",
            "timestamp": ev_ts,
            "message": {
                "role": "toolResult",
                "toolCallId": call_id,
                "toolName": tool_name or call_names.get(call_id) or "tool",
                "isError": bool(is_err),
                "content": [{"type": "text", "text": text_val}],
            },
        }, ensure_ascii=False))

    with src.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            obj = _read_json_line(line)
            if not obj:
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            role = obj.get("role") or msg.get("role")
            content = msg.get("content") if "content" in msg else obj.get("content")
            ev_ts = str(obj.get("timestamp") or msg.get("timestamp") or ts)
            raw_model = str(msg.get("model") or obj.get("model") or "").strip()
            raw_provider = str(msg.get("provider") or obj.get("provider") or active_provider).strip() or "cursor"
            if raw_model and raw_model != active_model:
                active_model = raw_model
                active_provider = raw_provider
                n += 1
                out.append(json.dumps({
                    "type": "model_change", "id": f"mc{n}", "timestamp": ev_ts,
                    "provider": active_provider, "modelId": active_model,
                }, ensure_ascii=False))

            if role == "user":
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") in ("tool_result", "toolResult"):
                            cid = str(item.get("tool_use_id") or item.get("toolCallId") or item.get("id") or "")
                            tname = str(item.get("name") or item.get("toolName") or call_names.get(cid) or "tool")
                            err = bool(item.get("is_error") or item.get("isError"))
                            raw_res = item.get("content") if "content" in item else item.get("output", item.get("text", ""))
                            _emit_tool_result(cid, tname, err, _tool_result_text(raw_res), ev_ts)
                text = _user_text(_blocks_text(content))
                if not text:
                    continue
                n += 1
                out.append(json.dumps({
                    "type": "message",
                    "id": f"u{n}",
                    "timestamp": ev_ts,
                    "message": {"role": "user", "content": [{"type": "text", "text": text}]},
                }, ensure_ascii=False))
            elif role in ("tool", "toolResult", "tool_result"):
                cid = str(
                    msg.get("toolCallId") or msg.get("tool_call_id") or msg.get("tool_use_id")
                    or obj.get("toolCallId") or obj.get("tool_call_id") or obj.get("tool_use_id") or ""
                )
                tname = str(
                    msg.get("toolName") or msg.get("name")
                    or obj.get("toolName") or obj.get("name")
                    or call_names.get(cid) or "tool"
                )
                err = bool(msg.get("isError") or msg.get("is_error") or obj.get("isError") or obj.get("is_error"))
                raw_res = content if content is not None else msg.get("output", obj.get("output", ""))
                _emit_tool_result(cid, tname, err, _tool_result_text(raw_res), ev_ts)
            elif role == "assistant":
                blocks: List[Dict[str, Any]] = []
                top_think = str(
                    msg.get("thinking") or msg.get("reasoning_content")
                    or obj.get("thinking") or obj.get("reasoning_content") or ""
                ).strip()
                has_block_think = False
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        itype = item.get("type")
                        if itype in ("thinking", "reasoning"):
                            th = str(item.get("thinking") or item.get("text") or "").strip()
                            if th:
                                blocks.append({"type": "thinking", "thinking": th})
                                has_block_think = True
                        elif itype == "text" and item.get("text"):
                            blocks.append({"type": "text", "text": str(item.get("text"))})
                        elif itype in ("tool_use", "toolCall"):
                            n += 1
                            cid = str(item.get("id") or item.get("toolUseId") or f"c{n}")
                            tname = str(item.get("name") or "tool")
                            raw_args = item.get("input") if "input" in item else item.get("arguments")
                            call_names[cid] = tname
                            blocks.append({
                                "type": "toolCall",
                                "id": cid,
                                "name": tname,
                                "arguments": _parse_tool_args(raw_args),
                            })
                elif isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                if top_think and not has_block_think:
                    blocks.insert(0, {"type": "thinking", "thinking": top_think})
                for tc in (msg.get("tool_calls") or obj.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    n += 1
                    cid = str(tc.get("id") or f"c{n}")
                    tname = str(fn.get("name") or tc.get("name") or "tool")
                    raw_args = fn.get("arguments") if "arguments" in fn else tc.get("arguments")
                    call_names[cid] = tname
                    blocks.append({
                        "type": "toolCall",
                        "id": cid,
                        "name": tname,
                        "arguments": _parse_tool_args(raw_args),
                    })
                if not blocks:
                    continue
                n += 1
                out.append(json.dumps({
                    "type": "message",
                    "id": f"a{n}",
                    "timestamp": ev_ts,
                    "message": {
                        "role": "assistant",
                        "model": active_model,
                        "provider": active_provider,
                        "content": blocks,
                    },
                }, ensure_ascii=False))
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")


def ingest_selected(source: str, ids: Iterable[str], cfg: Optional[Dict[str, Any]] = None, log=print) -> List[str]:
    cfg = cfg or load_config()
    source = _source(source)
    chosen = []
    for raw in ids:
        item = str(raw or "").strip()
        if item and item not in chosen:
            chosen.append(item)
    if not chosen:
        raise RuntimeError("没有选择会话")
    if len(chosen) > 12:
        raise RuntimeError("一次最多采集 12 条，避免分析时间过长")
    sids: List[str] = []
    inbox = stage_dirs(cfg)["state"] / "inbox" / "cursor"
    for catalog_id in chosen:
        path = resolve_catalog(source, catalog_id, cfg)
        log(f"[collect] 读取 {source} {catalog_id}")
        if source == "cursor":
            uuid = path.parent.name
            slug = path.parents[2].name
            staged = inbox / f"{uuid}.jsonl"
            _cursor_to_pi(path, staged, uuid, slug)
            rel = f"cursor/{slug}/{uuid}.jsonl"
            sid = adopt_file(staged, rel, origin="cursor", catalog_id=catalog_id, cfg=cfg, log=log)
        else:
            # catalog_id 已是 pi sessions 下的相对路径；勿用 resolve()，
            # Windows Junction 会跳到 D:\crack_test\... 导致 relative_to 失败。
            rel = catalog_id.replace("\\", "/")
            sid = adopt_file(path, rel, origin="pi", catalog_id=catalog_id, cfg=cfg, log=log)
        sids.append(sid)
        log(f"[collect] {now_iso()} 已写入原始库 {sid[:8]}")
    return sids
