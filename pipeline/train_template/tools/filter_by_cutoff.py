#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 cutoff_len 过滤超长样本：从 data/ 读原始全集，写到本次运行的数据目录。

data/ 下的原始文件永远只读。一般由 autotrain.py 调用；手动用法：
    python tools/filter_by_cutoff.py --src data --dst /tmp/run_data --cutoff 8192 \
        --keys gt_dpo_train_lf,gt_dpo_eval_lf

规则：
- 长度用 settings.yaml 的 length_tokenizer 计数；加载失败直接退出，不用字符数估算。
- 超长时先软压缩 observation/tool/function（最后一条 assistant 不改），仍超长则整条剔除。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

OBS_ROLES = frozenset({"observation", "tool", "function"})
ASSISTANT_ROLES = frozenset({"gpt", "assistant"})
SOFT_LIMIT_NORMAL = 500
SOFT_LIMIT_DUMP = 200
DUMP_HEAD_LINES = 20
SOFT_LIMIT_NORMAL_OLD = 200
SOFT_LIMIT_DUMP_OLD = 80
DUMP_HEAD_LINES_OLD = 8
RECENCY_RECENT_COUNT = 2
JSON_STRING_TRIM = 120
DUPLICATE_OBS_MARKER = "[duplicate tool result omitted, same as later turn]"


ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "observation": "tool",
    "tool": "tool",
    "gpt": "assistant",
    "assistant": "assistant",
}

_HEX_TOKEN_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}|[0-9a-fA-F]{8,}|0x[0-9a-fA-F]+)")
_ADDR_LINE_RE = re.compile(r"^\s*(?:0x)?[0-9a-fA-F]{6,}\b")


def _log(msg: str) -> None:
    if os.environ.get("RANK", "0") == "0":
        print(f"[filter_by_cutoff] {msg}", flush=True)


def load_gate_tokenizer(model_id: str) -> Tuple[Any, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[filter_by_cutoff] 无法 import transformers（{type(exc).__name__}: {exc}）。"
        ) from exc
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[filter_by_cutoff] 无法加载 tokenizer {model_id}（{type(exc).__name__}: {exc}）。"
        ) from exc
    return tok, model_id


def _encode_ids(tok: Any, text: str) -> List[Any]:
    out = tok.encode(text, add_special_tokens=False)
    if hasattr(out, "ids"):
        return list(out.ids)
    return list(out)


def count_text_tokens(tok: Any, text: str) -> int:
    return len(_encode_ids(tok, text))


def _decode_ids(tok: Any, ids: Sequence[Any]) -> str:
    if not ids:
        return ""
    if isinstance(ids[0], str):
        return "".join(ids)
    return tok.decode(ids, skip_special_tokens=True)


def render(prefix: str, turns: Sequence[Dict[str, Any]], open_assistant: bool = False) -> str:
    out: List[str] = []
    for t in turns:
        role = ROLE_MAP.get(str(t.get("from", "")), "user")
        v = t.get("value") or ""
        if role == "system":
            out.append(f"<|im_start|>system\n{prefix}{v}<|im_end|>\n")
        elif role == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{v}\n</tool_response><|im_end|>\n")
        elif role == "assistant":
            out.append(f"<|im_start|>assistant\n{v}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>user\n{v}<|im_end|>\n")
    text = "".join(out)
    return text + ("<|im_start|>assistant\n" if open_assistant else "")


def row_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if row.get("conversations"):
        return [{"from": c.get("from"), "value": c.get("value") or ""} for c in row["conversations"]]
    return []


def _side_turns(side: Any) -> List[Dict[str, Any]]:
    if isinstance(side, str):
        return [{"from": "gpt", "value": side}]
    if isinstance(side, dict):
        if "value" in side:
            return [{"from": side.get("from") or "gpt", "value": side.get("value") or ""}]
        side = side.get("conversations") or side.get("messages") or []
    if isinstance(side, list):
        return [
            {"from": c.get("from"), "value": c.get("value") or ""}
            if "from" in c
            else {
                "from": "gpt" if c.get("role") == "assistant" else "human",
                "value": c.get("content") or "",
            }
            for c in side
            if isinstance(c, dict)
        ]
    return []


def tools_prefix(row: Dict[str, Any]) -> str:
    """近似模板把工具定义渲染进 system 的那段文字（只用于计长度）。"""
    tools = row.get("tools")
    if not tools:
        return ""
    if not isinstance(tools, str):
        tools = json.dumps(tools, ensure_ascii=False)
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{tools}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format "
        "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>"
        "\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n"
    )


def measure_row_tokens(tok: Any, row: Dict[str, Any], *, ranking: bool, prefix: str = "") -> int:
    prefix = tools_prefix(row) + prefix
    turns = row_turns(row)
    if not ranking:
        return count_text_tokens(tok, render(prefix, turns))
    base = render(prefix, turns, open_assistant=True)
    nc = count_text_tokens(tok, base + render("", _side_turns(row.get("chosen"))))
    nr = count_text_tokens(tok, base + render("", _side_turns(row.get("rejected"))))
    return max(nc, nr)


def is_high_entropy_dump(text: str) -> bool:
    if not text or len(text) < 600:
        return False
    sample = text if len(text) <= 200_000 else text[:100_000] + text[-50_000:]
    hex_hits = len(_HEX_TOKEN_RE.findall(sample))
    lines = sample.splitlines() or [sample]
    addr_lines = sum(1 for ln in lines if _ADDR_LINE_RE.match(ln))
    hex_chars = sum(1 for ch in sample if ch in "0123456789abcdefABCDEF\\x")
    hex_char_ratio = hex_chars / max(1, len(sample))
    if hex_hits >= 80:
        return True
    if addr_lines >= 25 and addr_lines / max(1, len(lines)) >= 0.35:
        return True
    if len(text) >= 4000 and hex_char_ratio >= 0.45:
        return True
    if len(lines) >= 40:
        from collections import Counter

        norm = [re.sub(r"[0-9a-fA-F]{4,}", "H", ln.strip()[:80]) for ln in lines if ln.strip()]
        if norm:
            top = Counter(norm).most_common(1)[0][1]
            if top >= 20 and top / len(norm) >= 0.4:
                return True
    return False


def light_json_trim(text: str, max_str: int = JSON_STRING_TRIM) -> Tuple[str, bool]:
    """若正文是 JSON object/array：保留键与短字符串值（长串截到 max_str）再交 token 软截。"""
    if not text:
        return text, False
    s = text.strip()
    if not s or s[0] not in "{[":
        return text, False
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text, False

    changed = False

    def _trim(o: Any) -> Any:
        nonlocal changed
        if isinstance(o, dict):
            return {str(k): _trim(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_trim(v) for v in o]
        if isinstance(o, str) and len(o) > max_str:
            changed = True
            return o[:max_str] + "…"
        return o

    new_obj = _trim(obj)
    if not changed:
        return text, False
    return json.dumps(new_obj, ensure_ascii=False), True


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_near_duplicate_obs(older: str, newer: str) -> bool:
    """连续 obs 近似重复：完全相同，或一方包含另一方且长度重叠 >90%。"""
    a = _normalize_ws(older)
    b = _normalize_ws(newer)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b:
        return (len(a) / max(1, len(b))) > 0.9
    if b in a:
        return (len(b) / max(1, len(a))) > 0.9
    return False


def soft_truncate_tokens(
    tok: Any,
    text: str,
    max_tokens: int,
    *,
    head_ratio: float = 0.7,
) -> Tuple[str, bool]:
    """头 head_ratio + 尾 (1-head_ratio)，中间插入 …[truncated N chars]…。"""
    if max_tokens <= 8 or not text:
        return text, False
    ids = _encode_ids(tok, text)
    if len(ids) <= max_tokens:
        return text, False

    marker_template = "…[truncated {n} chars]…"
    head_n = max(1, int(max_tokens * head_ratio))
    tail_n = max(1, max_tokens - head_n)
    head = tail = ""
    for _ in range(3):
        probe = marker_template.format(n=max(0, len(text) - 1))
        marker_cost = max(1, count_text_tokens(tok, probe))
        budget = max_tokens - marker_cost
        if budget < 4:
            budget = max(4, max_tokens // 2)
            marker_cost = max_tokens - budget
        head_n = max(1, int(budget * head_ratio))
        tail_n = max(1, budget - head_n)
        head = _decode_ids(tok, ids[:head_n])
        tail = _decode_ids(tok, ids[-tail_n:]) if tail_n else ""
        omitted = max(0, len(text) - len(head) - len(tail))
        marker = marker_template.format(n=omitted)
        out = head + marker + tail
        if count_text_tokens(tok, out) <= max_tokens:
            return out, True
        max_tokens = max(8, max_tokens - 8)
    omitted = max(0, len(text) - len(head) - len(tail))
    return head + marker_template.format(n=omitted) + tail, True


def compress_observation(
    tok: Any,
    text: str,
    *,
    soft_limit_normal: int = SOFT_LIMIT_NORMAL,
    soft_limit_dump: int = SOFT_LIMIT_DUMP,
    dump_head_lines: int = DUMP_HEAD_LINES,
) -> Tuple[str, bool]:
    """对单条 observation/tool/function 做软压缩。返回 (新文本, 是否改动)。"""
    if not text:
        return text, False
    changed_any = False
    trimmed, did_json = light_json_trim(text)
    if did_json:
        text = trimmed
        changed_any = True
    if is_high_entropy_dump(text):
        lines = text.splitlines()
        summary = (
            f"[dump summary: {len(lines)} lines, {len(text)} chars, "
            f"approx_tokens={count_text_tokens(tok, text)}]"
        )
        if len(lines) > dump_head_lines:
            head = "\n".join(lines[:dump_head_lines])
            candidate = head + "\n" + summary
        else:
            candidate = text
        out, changed = soft_truncate_tokens(tok, candidate, soft_limit_dump, head_ratio=0.7)
        if not changed and candidate != text:
            return candidate, True
        return out, changed or candidate != text or changed_any
    n = count_text_tokens(tok, text)
    if n <= soft_limit_normal:
        return text, changed_any
    out, changed = soft_truncate_tokens(tok, text, soft_limit_normal, head_ratio=0.7)
    return out, changed or changed_any


def soft_compress_row(tok: Any, row: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """软压缩可压角色；永不改最后一条 assistant/gpt。

    Recency：按出现顺序枚举 obs/tool/function；最后 RECENCY_RECENT_COUNT 条用常规上限，
    更早的用 OLD_* 更紧上限。连续近似重复的旧条替换为 DUPLICATE_OBS_MARKER。
    """
    conv = list(row.get("conversations") or [])
    if not conv:
        return row, False
    last_assistant_idx = -1
    for i in range(len(conv) - 1, -1, -1):
        if str(conv[i].get("from") or "") in ASSISTANT_ROLES:
            last_assistant_idx = i
            break

    obs_indices: List[int] = []
    for i, msg in enumerate(conv):
        role = str(msg.get("from") or "")
        value = msg.get("value") if isinstance(msg.get("value"), str) else ""
        if role in OBS_ROLES and value:
            obs_indices.append(i)

    recent_set = set(obs_indices[-RECENCY_RECENT_COUNT:]) if obs_indices else set()

    dedupe_replace: Dict[int, str] = {}
    for j in range(len(obs_indices) - 1):
        older_i = obs_indices[j]
        newer_i = obs_indices[j + 1]
        older_v = conv[older_i].get("value") if isinstance(conv[older_i].get("value"), str) else ""
        newer_v = conv[newer_i].get("value") if isinstance(conv[newer_i].get("value"), str) else ""
        if is_near_duplicate_obs(older_v, newer_v):
            dedupe_replace[older_i] = DUPLICATE_OBS_MARKER

    changed = False
    new_conv: List[Dict[str, Any]] = []
    for i, msg in enumerate(conv):
        role = str(msg.get("from") or "")
        value = msg.get("value") if isinstance(msg.get("value"), str) else ""
        if i == last_assistant_idx and role in ASSISTANT_ROLES:
            new_conv.append(dict(msg))
            continue
        if role in OBS_ROLES and value:
            if i in dedupe_replace:
                changed = True
                m = dict(msg)
                m["value"] = dedupe_replace[i]
                new_conv.append(m)
                continue
            if i in recent_set:
                lim_n, lim_d, lim_h = SOFT_LIMIT_NORMAL, SOFT_LIMIT_DUMP, DUMP_HEAD_LINES
            else:
                lim_n, lim_d, lim_h = SOFT_LIMIT_NORMAL_OLD, SOFT_LIMIT_DUMP_OLD, DUMP_HEAD_LINES_OLD
            new_val, did = compress_observation(
                tok,
                value,
                soft_limit_normal=lim_n,
                soft_limit_dump=lim_d,
                dump_head_lines=lim_h,
            )
            if did:
                changed = True
                m = dict(msg)
                m["value"] = new_val
                new_conv.append(m)
                continue
        new_conv.append(dict(msg) if changed else msg)

    if not changed:
        return row, False
    out = dict(row)
    out["conversations"] = new_conv
    return out, True



def _sample_id(row: Dict[str, Any], index: int) -> str:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("sample_id"):
        return str(meta["sample_id"])
    return f"#{index}"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    tok: Any,
    cutoff: int,
    ranking: bool,
    soft_compress: bool,
    prefix: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    salvaged = 0
    dropped = 0
    for i, row in enumerate(rows):
        cur = row
        before = measure_row_tokens(tok, cur, ranking=ranking, prefix=prefix)
        did = False
        if soft_compress and before > cutoff:
            cur, did = soft_compress_row(tok, row)
            if did:
                salvaged += 1
        after = (
            measure_row_tokens(tok, cur, ranking=ranking, prefix=prefix)
            if did or before > cutoff
            else before
        )
        if after > cutoff:
            dropped += 1
            continue
        kept.append(cur)
    return kept, {
        "input": len(rows),
        "kept": len(kept),
        "dropped": dropped,
        "salvaged": salvaged,
        "cutoff": cutoff,
    }


def filter_into_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    keys: Sequence[str],
    cutoff: int,
    tokenizer_path: str,
    *,
    soft_compress: bool = True,
    fill_tools: Optional[str] = None,
) -> Dict[str, Any]:
    """把 src_dir 里 keys 对应的 jsonl 过滤后写到 dst_dir，并生成只含这些键的 dataset_info.json。

    dataset_info 声明了 tools 列、但行里没有 tools 时：给了 fill_tools 就补进副本；
    否则去掉该列映射（LLaMA-Factory 遇到缺列会 KeyError）。原始数据不改。
    """
    src_dir, dst_dir = Path(src_dir).resolve(), Path(dst_dir).resolve()
    if src_dir == dst_dir:
        raise SystemExit("[filter_by_cutoff] 目标目录不能是原始数据目录")
    info = json.loads((src_dir / "dataset_info.json").read_text(encoding="utf-8"))
    tok, tok_label = load_gate_tokenizer(tokenizer_path)
    dst_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"cutoff_len": cutoff, "tokenizer": tok_label, "files": {}}
    out_info: Dict[str, Any] = {}
    for key in keys:
        entry = info.get(key)
        if not isinstance(entry, dict) or not entry.get("file_name"):
            raise SystemExit(f"[filter_by_cutoff] dataset_info.json 里没有 {key}")
        src = src_dir / entry["file_name"]
        if not src.is_file():
            raise SystemExit(f"[filter_by_cutoff] 数据文件不存在: {src}")
        rows = load_jsonl(src)
        entry = dict(entry)
        tools_col = (entry.get("columns") or {}).get("tools")
        missing = sum(1 for r in rows if tools_col and not r.get(tools_col)) if tools_col else 0
        filled = 0
        if missing and fill_tools:
            for r in rows:
                if not r.get(tools_col):
                    r[tools_col] = fill_tools
                    filled += 1
            _log(f"[WARN] {key}: {missing}/{len(rows)} 行缺 {tools_col}，已在副本里补上共享工具定义")
        elif missing == len(rows) and rows:
            entry["columns"] = {k: v for k, v in entry["columns"].items() if k != "tools"}
            _log(f"[WARN] {key}: 所有行都没有 {tools_col}，已去掉 tools 列映射（训练时无工具定义）")
        elif missing:
            _log(f"[WARN] {key}: {missing}/{len(rows)} 行缺 {tools_col}，按空工具定义训练")
            for r in rows:
                r.setdefault(tools_col, "")
        kept, stats = filter_rows(
            rows,
            tok=tok,
            cutoff=cutoff,
            ranking=bool(entry.get("ranking")),
            soft_compress=soft_compress,
        )
        if stats["kept"] == 0:
            raise SystemExit(f"[filter_by_cutoff] {key} 在 cutoff_len={cutoff} 下无样本可训")
        dump_jsonl(dst_dir / entry["file_name"], kept)
        out_info[key] = entry
        if filled:
            stats["tools_filled"] = filled
        summary["files"][key] = stats
        _log(
            f"{key}: cutoff={cutoff} 输入={stats['input']} 保留={stats['kept']} "
            f"丢弃={stats['dropped']} 软压缩={stats['salvaged']}"
        )
    (dst_dir / "dataset_info.json").write_text(json.dumps(out_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def sort_jsonl_by_tokens(
    path: str | Path,
    tokenizer_path: str,
    *,
    ranking: bool = False,
    reverse: bool = True,
    top_n_log: int = 5,
) -> List[int]:
    """按真实 token 长度排序 jsonl（不是字符数）。返回最长的若干条 token 数供日志。"""
    path = Path(path)
    tok, _ = load_gate_tokenizer(tokenizer_path)
    rows = load_jsonl(path)
    scored = [
        (measure_row_tokens(tok, row, ranking=ranking), row) for row in rows
    ]
    scored.sort(key=lambda x: x[0], reverse=reverse)
    dump_jsonl(path, [row for _, row in scored])
    tops = [n for n, _ in scored[:top_n_log]]
    _log(f"按 token 排序 {path.name}: 共 {len(scored)} 条；最长 {tops}")
    return tops


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="按 cutoff_len 过滤数据集到新目录（原始数据只读）")
    p.add_argument("--src", default="data")
    p.add_argument("--dst", required=True)
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--keys", required=True, help="逗号分隔的 dataset_info 键")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--no-soft-compress", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    filter_into_dir(
        args.src, args.dst, [k for k in args.keys.split(",") if k],
        args.cutoff, args.tokenizer, soft_compress=not args.no_soft_compress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
