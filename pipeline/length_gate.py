"""交付打包长度处理：observation 软压缩抢救；打包阶段永不因超长丢弃样本。

口径：
- tokenize 优先 Qwen/Qwen2.5-0.5B（与训练同族）；加载失败则跳过处理并写入明确原因，
  绝不静默改用字符估算或其他模型。
- 软压缩只动 role/from ∈ {observation, tool, function}；最后一条 assistant/gpt
  （含 think / tool_call）禁止改写。
- 压缩后再按 ChatML 渲染量整条；SFT / DPO 默认 cutoff 从 train_*.yaml 的 cutoff_len 读，
  仅用于决定是否软压缩与统计 would-be-overlength（`overlength_at_default_cutoff`）。
- 硬过滤留给训练入口（filter_by_cutoff / train_lowmem），按当次实际 cutoff_len 执行。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.common import seqlength as sl

logger = logging.getLogger(__name__)

OBS_ROLES = frozenset({"observation", "tool", "function"})
ASSISTANT_ROLES = frozenset({"gpt", "assistant"})

DEFAULT_SFT_CUTOFF = 12288
DEFAULT_DPO_CUTOFF = 5120
SOFT_LIMIT_NORMAL = 500
SOFT_LIMIT_DUMP = 200
DUMP_HEAD_LINES = 20
# Recency bias: last RECENCY_RECENT_COUNT obs/tool/function msgs use caps above;
# older ones use tighter OLD_* caps.
SOFT_LIMIT_NORMAL_OLD = 200
SOFT_LIMIT_DUMP_OLD = 80
DUMP_HEAD_LINES_OLD = 8
RECENCY_RECENT_COUNT = 2
JSON_STRING_TRIM = 120
DUPLICATE_OBS_MARKER = "[duplicate tool result omitted, same as later turn]"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"

# Bump when soft-compress policy / aggressiveness changes so delivery zip
# package_hash renames even if post-gate bytes happen to match.
LENGTH_GATE_VERSION = "obs-soft-v2"


def length_gate_fingerprint() -> str:
    """Opaque string mixed into delivery package_hash."""
    return (
        f"{LENGTH_GATE_VERSION}"
        f"|n={SOFT_LIMIT_NORMAL}|d={SOFT_LIMIT_DUMP}"
        f"|n_old={SOFT_LIMIT_NORMAL_OLD}|d_old={SOFT_LIMIT_DUMP_OLD}"
        f"|head={DUMP_HEAD_LINES}|head_old={DUMP_HEAD_LINES_OLD}"
        f"|recency={RECENCY_RECENT_COUNT}"
    )


_HEX_TOKEN_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}|[0-9a-fA-F]{8,}|0x[0-9a-fA-F]+)")
_ADDR_LINE_RE = re.compile(r"^\s*(?:0x)?[0-9a-fA-F]{6,}\b")


@dataclass
class GateRecord:
    sample_id: str
    tokens_before: int
    tokens_after: int
    reason: str
    salvaged: bool = False


@dataclass
class GateStats:
    cutoff: int
    tokenizer: str
    kept: int = 0
    salvaged_by_obs_truncate: int = 0
    overlength_at_default_cutoff: int = 0
    records_salvaged: List[GateRecord] = field(default_factory=list)
    records_overlength: List[GateRecord] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    # 兼容旧字段名（打包不再丢弃，恒为 0 / 空）
    @property
    def dropped_overlength(self) -> int:
        return 0

    @property
    def records_dropped(self) -> List[GateRecord]:
        return []

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "cutoff": self.cutoff,
            "tokenizer": self.tokenizer,
            "kept": self.kept,
            "salvaged_by_obs_truncate": self.salvaged_by_obs_truncate,
            "overlength_at_default_cutoff": self.overlength_at_default_cutoff,
            "dropped_overlength": 0,
        }
        if self.skipped:
            out["skipped"] = True
            out["skip_reason"] = self.skip_reason
        if self.records_overlength:
            out["overlength_sample_ids"] = [r.sample_id for r in self.records_overlength]
            out["overlength_details"] = [
                {
                    "sample_id": r.sample_id,
                    "tokens_before": r.tokens_before,
                    "tokens_after": r.tokens_after,
                    "reason": r.reason,
                    "salvaged_attempted": r.salvaged,
                    "status": "overlength",
                }
                for r in self.records_overlength
            ]
        if self.records_salvaged:
            out["salvaged_sample_ids"] = [r.sample_id for r in self.records_salvaged]
            out["salvaged_details"] = [
                {
                    "sample_id": r.sample_id,
                    "tokens_before": r.tokens_before,
                    "tokens_after": r.tokens_after,
                    "reason": r.reason,
                    "status": "salvaged",
                }
                for r in self.records_salvaged
            ]
        return out


def read_cutoff_len(yaml_path: Path, default: int) -> int:
    """从 train_*.yaml 读 cutoff_len；读不到用 default。"""
    path = Path(yaml_path)
    if not path.is_file():
        return default
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith("cutoff_len:"):
            raw = code.split(":", 1)[1].strip()
            try:
                return int(raw)
            except ValueError:
                return default
    return default


def cutoffs_for_root(root: Path) -> Tuple[int, int]:
    root = Path(root)
    sft = read_cutoff_len(root / "train_sft.yaml", DEFAULT_SFT_CUTOFF)
    dpo = read_cutoff_len(root / "train_dpo.yaml", DEFAULT_DPO_CUTOFF)
    return sft, dpo


_TOKENIZER_CACHE: Dict[str, Tuple[Optional[Any], str]] = {}


def load_gate_tokenizer(
    model_id: str = TOKENIZER_ID,
) -> Tuple[Optional[Any], str]:
    """返回 (tokenizer, 来源说明)。失败时 tokenizer=None，来源里写清原因。"""
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        result = (None, f"skip:transformers_import_failed:{type(exc).__name__}:{exc}")
        _TOKENIZER_CACHE[model_id] = result
        return result
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        result = (None, f"skip:tokenizer_load_failed:{model_id}:{type(exc).__name__}:{exc}")
        _TOKENIZER_CACHE[model_id] = result
        return result
    result = (tok, model_id)
    _TOKENIZER_CACHE[model_id] = result
    return result


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
    # 测试 FakeTok：encode 返回单字符 str 列表
    if isinstance(ids[0], str):
        return "".join(ids)
    return tok.decode(ids, skip_special_tokens=True)


def is_high_entropy_dump(text: str) -> bool:
    """大量 hex / \\x / 重复地址行 → 高熵 dump，软压缩更狠。"""
    if not text or len(text) < 600:
        return False
    sample = text if len(text) <= 200_000 else text[:100_000] + text[-50_000:]
    hex_hits = len(_HEX_TOKEN_RE.findall(sample))
    lines = sample.splitlines() or [sample]
    addr_lines = sum(1 for ln in lines if _ADDR_LINE_RE.match(ln))
    # 字符密度：可打印十六进制占比
    hex_chars = sum(1 for ch in sample if ch in "0123456789abcdefABCDEF\\x")
    hex_char_ratio = hex_chars / max(1, len(sample))
    if hex_hits >= 80:
        return True
    if addr_lines >= 25 and addr_lines / max(1, len(lines)) >= 0.35:
        return True
    if len(text) >= 4000 and hex_char_ratio >= 0.45:
        return True
    # 大量几乎相同的短行（地址/偏移 dump）
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



def _sample_id_of(row: Dict[str, Any], index: int) -> str:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("sample_id"):
        return str(meta["sample_id"])
    return f"#{index}"


def measure_row_tokens(
    tok: Any,
    row: Dict[str, Any],
    *,
    ranking: bool,
    prefix: str = "",
) -> int:
    """按 LLaMA-Factory ChatML 渲染量长度；DPO 取 chosen/rejected 路径较大者。"""

    class _TokAdapter:
        """让 transformers / FakeTok 走 seqlength.count_tokens 的 encode().ids 路径。"""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def encode(self, text: str, add_special_tokens: bool = False) -> Any:
            ids = _encode_ids(self._inner, text)

            class _Ids:
                def __init__(self, ids_: List[Any]) -> None:
                    self.ids = ids_

            return _Ids(ids)

    adapted = _TokAdapter(tok)
    turns = sl.row_turns(row)
    if not ranking:
        return sl.count_tokens(adapted, sl.render(prefix, turns))
    base = sl.render(prefix, turns, open_assistant=True)
    nc = sl.count_tokens(adapted, base + sl.render("", sl._side_turns(row.get("chosen"))))
    nr = sl.count_tokens(adapted, base + sl.render("", sl._side_turns(row.get("rejected"))))
    return max(nc, nr)


def gate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    ranking: bool,
    cutoff: int,
    tok: Any,
    tokenizer_label: str,
    prefix: str = "",
) -> Tuple[List[Dict[str, Any]], GateStats]:
    """软压缩抢救；永不因超长丢弃。返回 (全部行, 统计)。

    after > cutoff 时仍保留（优先保留已软压缩版本），写入 overlength_at_default_cutoff。
    """
    stats = GateStats(cutoff=cutoff, tokenizer=tokenizer_label)
    kept: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        sid = _sample_id_of(row, i)
        before = measure_row_tokens(tok, row, ranking=ranking, prefix=prefix)
        cur = row
        salvaged = False
        if before > cutoff:
            cur, did = soft_compress_row(tok, row)
            salvaged = did
        after = (
            measure_row_tokens(tok, cur, ranking=ranking, prefix=prefix)
            if salvaged or before > cutoff
            else before
        )
        if after > cutoff and not salvaged:
            cur, salvaged = soft_compress_row(tok, row)
            after = measure_row_tokens(tok, cur, ranking=ranking, prefix=prefix)

        if salvaged and before > cutoff and after <= cutoff:
            stats.salvaged_by_obs_truncate += 1
            stats.records_salvaged.append(
                GateRecord(
                    sample_id=sid,
                    tokens_before=before,
                    tokens_after=after,
                    reason="obs_soft_compress",
                    salvaged=True,
                )
            )
        elif after > cutoff:
            stats.overlength_at_default_cutoff += 1
            stats.records_overlength.append(
                GateRecord(
                    sample_id=sid,
                    tokens_before=before,
                    tokens_after=after,
                    reason="over_cutoff_after_obs_soft_compress",
                    salvaged=salvaged,
                )
            )
        kept.append(cur)
        stats.kept += 1
    return kept, stats


def resolve_prefix(root: Path) -> Tuple[str, str]:
    """尽量从 train 日志抠模板前缀；没有则空串（长度偏小，门禁更松）。"""
    root = Path(root)
    for cand in (root / "logs", root.parent / "03_train" / "logs"):
        if cand.is_dir():
            prefix, src = sl.prefix_from_logs(cand)
            if prefix:
                return prefix, src
    return "", "none"


def gate_prepared_files(
    files: Dict[str, List[Dict[str, Any]]],
    *,
    root: Path,
    tok: Optional[Any] = None,
    tokenizer_label: str = "",
    prefix: Optional[str] = None,
    sft_cutoff: Optional[int] = None,
    dpo_cutoff: Optional[int] = None,
    log=None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, GateStats]]:
    """对 prepared.files 里的 *.lf.jsonl 做软压缩；保留全部行并统计 would-be-overlength。"""
    root = Path(root)

    def _emit(msg: str) -> None:
        if log:
            log(msg)

    if tok is None:
        cached = TOKENIZER_ID in _TOKENIZER_CACHE
        if cached:
            _emit("[length_gate] 分词器已就绪（缓存）")
        else:
            _emit("[length_gate] 正在加载分词器…")
        tok, tokenizer_label = load_gate_tokenizer()
        if tok is None:
            _emit(f"[length_gate] 分词器不可用，跳过软压缩: {tokenizer_label}")
        elif cached:
            pass
        else:
            _emit("[length_gate] 分词器加载完成")
    if prefix is None:
        prefix, _pfx_src = resolve_prefix(root)
    if sft_cutoff is None or dpo_cutoff is None:
        s_default, d_default = cutoffs_for_root(root)
        sft_cutoff = s_default if sft_cutoff is None else sft_cutoff
        dpo_cutoff = d_default if dpo_cutoff is None else dpo_cutoff

    out_files = dict(files)
    all_stats: Dict[str, GateStats] = {}

    if tok is None:
        skip = GateStats(
            cutoff=sft_cutoff or DEFAULT_SFT_CUTOFF,
            tokenizer=tokenizer_label or TOKENIZER_ID,
            skipped=True,
            skip_reason=tokenizer_label or "tokenizer_unavailable",
        )
        all_stats["_skip"] = skip
        logger.warning("length_gate skipped: %s", skip.skip_reason)
        return out_files, all_stats

    lf_specs = (
        ("sft.train.lf.jsonl", False, sft_cutoff),
        ("sft.eval.lf.jsonl", False, sft_cutoff),
        ("dpo.train.lf.jsonl", True, dpo_cutoff),
        ("dpo.eval.lf.jsonl", True, dpo_cutoff),
    )
    for name, ranking, cutoff in lf_specs:
        rows = out_files.get(name)
        if not rows:
            continue
        _emit(f"[length_gate] 正在软压缩并计量长度：{name}（{len(rows)} 条，cutoff={cutoff}）…")
        kept, stats = gate_rows(
            rows,
            ranking=ranking,
            cutoff=int(cutoff),
            tok=tok,
            tokenizer_label=tokenizer_label or TOKENIZER_ID,
            prefix=prefix or "",
        )
        out_files[name] = kept
        logical = name.replace(".lf.jsonl", "")
        all_stats[logical] = stats
        salv = int(stats.summary().get("salvaged_by_obs_truncate") or 0)
        over = int(stats.summary().get("overlength_at_default_cutoff") or 0)
        _emit(
            f"[length_gate] 完成 {name}：salvaged={salv} overlength_at_default={over}"
        )
    return out_files, all_stats


def stats_to_manifest(stats_map: Dict[str, GateStats]) -> Dict[str, Any]:
    return {k: v.summary() for k, v in stats_map.items()}


def overlength_report_jsonl(stats_map: Dict[str, GateStats]) -> str:
    """写出 overlength / salvaged 审计行（样本仍在包内）。"""
    lines: List[str] = []
    for logical, st in stats_map.items():
        for rec in st.records_overlength:
            lines.append(
                json.dumps(
                    {
                        "dataset": logical,
                        "sample_id": rec.sample_id,
                        "tokens_before": rec.tokens_before,
                        "tokens_after": rec.tokens_after,
                        "reason": rec.reason,
                        "salvaged_attempted": rec.salvaged,
                        "status": "overlength",
                    },
                    ensure_ascii=False,
                )
            )
        for rec in st.records_salvaged:
            lines.append(
                json.dumps(
                    {
                        "dataset": logical,
                        "sample_id": rec.sample_id,
                        "tokens_before": rec.tokens_before,
                        "tokens_after": rec.tokens_after,
                        "reason": rec.reason,
                        "status": "salvaged",
                    },
                    ensure_ascii=False,
                )
            )
    return ("\n".join(lines) + ("\n" if lines else ""))


def overlength_report_jsonl_from_summary(stats_map: Dict[str, Any]) -> str:
    """从 GateStats.summary() 字典写出审计 jsonl（供打进 zip）。"""
    lines: List[str] = []
    for logical, st in (stats_map or {}).items():
        if not isinstance(st, dict):
            continue
        for rec in st.get("overlength_details") or []:
            row = dict(rec)
            row["dataset"] = logical
            if "status" not in row:
                row["status"] = "overlength"
            lines.append(json.dumps(row, ensure_ascii=False))
        for rec in st.get("salvaged_details") or []:
            row = dict(rec)
            row["dataset"] = logical
            lines.append(json.dumps(row, ensure_ascii=False))
    return ("\n".join(lines) + ("\n" if lines else ""))


# 旧名兼容
def dropped_records_jsonl(stats_map: Dict[str, GateStats]) -> str:
    return overlength_report_jsonl(stats_map)


def dropped_records_jsonl_from_summary(stats_map: Dict[str, Any]) -> str:
    return overlength_report_jsonl_from_summary(stats_map)
