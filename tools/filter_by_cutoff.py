#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练前按 YAML cutoff_len 过滤超长样本（交付 zip 自包含，不依赖 pipeline.*）。

用法（一般由 train_lowmem.py 自动调用）：
    python filter_by_cutoff.py train_sft.yaml
    python filter_by_cutoff.py train_dpo.yaml --no-soft-compress

环境变量：
    GT_SKIP_LENGTH_FILTER=1  跳过过滤（train_lowmem 也认）
    GT_LENGTH_FILTER_SOFT=0  关闭软压缩（默认开启：先压 observation 再量）

口径与 pipeline/length_gate.py 对齐：
- tokenizer：Qwen/Qwen2.5-0.5B；加载失败则明确退出，绝不静默用字符估算。
- 软压缩只动 observation/tool/function；最后一条 assistant/gpt 不改。
- 仍超 cutoff → 从本次训练用的 jsonl 中移除（包内原文件可被就地改写；
  会先写 .prefilter.bak 备份，同一次运行只备份一次）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
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


def read_cutoff_len(yaml_path: Path, default: Optional[int] = None) -> int:
    text = Path(yaml_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith("cutoff_len:"):
            raw = code.split(":", 1)[1].strip()
            try:
                return int(raw)
            except ValueError as exc:
                raise SystemExit(f"无法解析 cutoff_len: {raw!r} ({yaml_path})") from exc
    if default is not None:
        return default
    raise SystemExit(f"{yaml_path} 缺少 cutoff_len，拒绝在未知截断长度下开训")


def _yaml_simple_value(text: str, key: str) -> Optional[str]:
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        stripped = code.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def load_gate_tokenizer(model_id: str = TOKENIZER_ID) -> Tuple[Any, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[filter_by_cutoff] 无法 import transformers（{type(exc).__name__}: {exc}）。"
            f"拒绝静默估算长度；请安装 transformers 或设置 GT_SKIP_LENGTH_FILTER=1（不推荐）。"
        ) from exc
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[filter_by_cutoff] 无法加载 tokenizer {model_id}（{type(exc).__name__}: {exc}）。"
            f"拒绝静默估算；请确保可访问该模型或设置 GT_SKIP_LENGTH_FILTER=1。"
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


def measure_row_tokens(tok: Any, row: Dict[str, Any], *, ranking: bool, prefix: str = "") -> int:
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


def resolve_dataset_files(yaml_path: Path) -> List[Tuple[Path, bool, str]]:
    """返回 [(jsonl_path, ranking, logical_name), ...]。"""
    text = yaml_path.read_text(encoding="utf-8")
    dataset_dir = _yaml_simple_value(text, "dataset_dir") or "./data"
    train_key = _yaml_simple_value(text, "dataset")
    eval_key = _yaml_simple_value(text, "eval_dataset")
    stage = (_yaml_simple_value(text, "stage") or "").lower()
    ranking_default = "dpo" in stage or "dpo" in yaml_path.name.lower()

    root = yaml_path.parent
    data_root = (root / dataset_dir).resolve()
    info_path = data_root / "dataset_info.json"
    if not info_path.is_file():
        raise SystemExit(f"找不到 {info_path}，无法定位数据集文件")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    out: List[Tuple[Path, bool, str]] = []
    for key in (train_key, eval_key):
        if not key:
            continue
        entry = info.get(key)
        if not isinstance(entry, dict):
            _log(f"WARN: dataset_info 无键 {key}，跳过")
            continue
        fname = entry.get("file_name")
        if not fname:
            continue
        path = data_root / fname
        ranking = bool(entry.get("ranking")) or ranking_default
        out.append((path, ranking, key))
    return out


def filter_datasets_for_yaml(
    yaml_path: str | Path,
    *,
    soft_compress: bool = True,
    prefix: str = "",
) -> Dict[str, Any]:
    """供 train_lowmem 调用：按 yaml cutoff 过滤对应 jsonl（就地改写，带 .prefilter.bak）。"""
    if os.environ.get("GT_SKIP_LENGTH_FILTER", "").strip() in ("1", "true", "True", "yes"):
        _log("GT_SKIP_LENGTH_FILTER=1，跳过长度过滤")
        return {"skipped": True, "reason": "GT_SKIP_LENGTH_FILTER"}

    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"YAML 不存在: {yaml_path}")

    soft_env = os.environ.get("GT_LENGTH_FILTER_SOFT", "").strip()
    if soft_env in ("0", "false", "False", "no"):
        soft_compress = False

    cutoff = read_cutoff_len(yaml_path)
    tok, tok_label = load_gate_tokenizer()
    files = resolve_dataset_files(yaml_path)
    if not files:
        raise SystemExit(f"{yaml_path.name} 未解析到任何数据集文件")

    summary: Dict[str, Any] = {
        "yaml": str(yaml_path),
        "cutoff_len": cutoff,
        "tokenizer": tok_label,
        "soft_compress": soft_compress,
        "files": {},
    }
    for path, ranking, key in files:
        if not path.is_file():
            raise SystemExit(f"数据集文件不存在: {path} ({key})")
        rows = load_jsonl(path)
        kept, stats = filter_rows(
            rows,
            tok=tok,
            cutoff=cutoff,
            ranking=ranking,
            soft_compress=soft_compress,
            prefix=prefix,
        )
        bak = path.with_suffix(path.suffix + ".prefilter.bak")
        if not bak.is_file():
            bak.write_bytes(path.read_bytes())
        dump_jsonl(path, kept)
        summary["files"][key] = {**stats, "path": str(path), "backup": str(bak)}
        _log(
            f"{key}: cutoff={cutoff} 输入={stats['input']} 保留={stats['kept']} "
            f"丢弃={stats['dropped']} 软压缩={stats['salvaged']} → {path.name}"
        )
        if stats["kept"] == 0:
            raise SystemExit(
                f"[filter_by_cutoff] {key} 在 cutoff_len={cutoff} 下无样本可训，中止。"
                f"请提高 cutoff_len 或检查数据。"
            )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="按 YAML cutoff_len 过滤交付包数据集")
    p.add_argument("yaml", help="train_*.yaml / eval_*.yaml")
    p.add_argument("--no-soft-compress", action="store_true", help="不做 observation 软压缩，直接按长度丢弃")
    p.add_argument("--prefix", default="", help="ChatML 模板前缀（一般留空）")
    args = p.parse_args(list(argv) if argv is not None else None)
    filter_datasets_for_yaml(
        args.yaml,
        soft_compress=not args.no_soft_compress,
        prefix=args.prefix or "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
