"""把 ShareGPT 样本规整成 LLaMA-Factory 能直接消费的格式。

对照过已成功训练 27B 的包（goldthread-ab56c1d1-lowmem）：

- ``sft.train.lf.jsonl`` / ``sft.eval.lf.jsonl`` / ``dpo.*.lf.jsonl`` 与 v2 ShareGPT
  逐行等价，只要做下面两步：
  1. 把 v2 用来粘连多条 tool 结果的字面量
     ``\\n</tool_response>\\n<tool_response>\\n`` 换成 ``\\n\\n``
     （模板会再包一层 tool_response，沿用旧拼接会套两层标签）；
  2. 把连续同角色消息收成一条（用 ``\\n\\n`` 连接）。
     LLaMA-Factory 的 SharegptDatasetConverter 要求严格交替：
     奇数位 human|observation，偶数位 gpt|function。
     连续 observation / 连续 gpt 会被静默丢弃。
- DPO 的 chosen/rejected 保持 ``{"from","value"}`` 字典（成功训练用的就是字典，不是字符串）。
- 来源优先级：已有且结构合法的 ``*.lf.jsonl`` > v2（与成功训练同一套条数）
  > 规范名 ``sft.train.jsonl`` > stage3 未重切的原始切分。
  因此交付包不会再把未重切的 stage3 窗口当成 YAML 指向的训练集。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 不用字面量尖括号，避免源文件在某些写入路径里被吃掉标签。
_TR_CLOSE = "</" + "tool_response>"
_TR_OPEN = "<" + "tool_response>"
TOOL_RESPONSE_JOIN = "\n" + _TR_CLOSE + "\n" + _TR_OPEN + "\n"
_COLLAPSE_SEP = "\n\n"

_ODD_ROLES = ("human", "observation")
_EVEN_ROLES = ("gpt", "function")

_SHAREGPT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system",
    "observation_tag": "observation",
}

# (逻辑名, lf 文件名, 规范 sharegpt 文件名, 是否 DPO, 候选源，按优先级)
SPECS: Tuple[Tuple[str, str, str, bool, Tuple[str, ...]], ...] = (
    (
        "sft.train",
        "sft.train.lf.jsonl",
        "sft.train.jsonl",
        False,
        (
            "sft_step_windows_sharegpt.train.v2.jsonl",
            "sft.train.jsonl",
            "sft_step_windows_sharegpt.train.jsonl",
        ),
    ),
    (
        "sft.eval",
        "sft.eval.lf.jsonl",
        "sft.eval.jsonl",
        False,
        (
            "sft_step_windows_sharegpt.eval.v2.jsonl",
            "sft.eval.jsonl",
            "sft_step_windows_sharegpt.eval.jsonl",
        ),
    ),
    (
        "dpo.train",
        "dpo.train.lf.jsonl",
        "dpo.train.jsonl",
        True,
        (
            "dpo_preference_pairs.train.v2.jsonl",
            "dpo.train.jsonl",
            "dpo_preference_pairs.train.jsonl",
        ),
    ),
    (
        "dpo.eval",
        "dpo.eval.lf.jsonl",
        "dpo.eval.jsonl",
        True,
        (
            "dpo_preference_pairs.eval.v2.jsonl",
            "dpo.eval.jsonl",
            "dpo_preference_pairs.eval.jsonl",
        ),
    ),
)

LF_KEYS = {
    "sft.train.lf.jsonl": "gt_sft_train_lf",
    "sft.eval.lf.jsonl": "gt_sft_eval_lf",
    "dpo.train.lf.jsonl": "gt_dpo_train_lf",
    "dpo.eval.lf.jsonl": "gt_dpo_eval_lf",
}
SHARE_KEYS = {
    "sft.train.jsonl": "gt_sft_train",
    "sft.eval.jsonl": "gt_sft_eval",
    "dpo.train.jsonl": "gt_dpo_train",
    "dpo.eval.jsonl": "gt_dpo_eval",
}


@dataclass
class PreparedDatasets:
    """打包用的规范文件。``files`` 的键是 zip 内 ``data/`` 下的文件名。"""

    files: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    dropped: Dict[str, List[str]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    length_gate: Dict[str, Any] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return bool(self.files.get(name))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dumps_jsonl(rows: Iterable[Dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def alignment_error(messages: Sequence[Dict[str, Any]], *, ranking: bool) -> Optional[str]:
    """返回 None 表示符合 LLaMA-Factory ShareGPT 交替规则。"""
    roles = [str(m.get("from") or "") for m in messages]
    if roles and roles[0] == "system":
        roles = roles[1:]
    if not roles:
        return "empty"
    for i, role in enumerate(roles):
        allowed = _ODD_ROLES if i % 2 == 0 else _EVEN_ROLES
        if role not in allowed:
            return f"role[{i}]={role}"
    if ranking:
        if len(roles) % 2 == 0:
            return "prompt ends with assistant"
    elif len(roles) % 2 != 0:
        return "ends with user/observation"
    return None


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for raw in messages:
        role = str(raw.get("from") or "")
        value = raw.get("value") if isinstance(raw.get("value"), str) else ""
        if TOOL_RESPONSE_JOIN in value:
            value = value.replace(TOOL_RESPONSE_JOIN, _COLLAPSE_SEP)
        if cleaned and cleaned[-1]["from"] == role:
            cleaned[-1] = {"from": role, "value": cleaned[-1]["value"] + _COLLAPSE_SEP + value}
        else:
            cleaned.append({"from": role, "value": value})
    return cleaned


def _normalize_response(value: Any) -> Any:
    if isinstance(value, dict):
        out = dict(value)
        if isinstance(out.get("value"), str) and TOOL_RESPONSE_JOIN in out["value"]:
            out["value"] = out["value"].replace(TOOL_RESPONSE_JOIN, _COLLAPSE_SEP)
        return out
    if isinstance(value, str) and TOOL_RESPONSE_JOIN in value:
        return value.replace(TOOL_RESPONSE_JOIN, _COLLAPSE_SEP)
    return value


def normalize_sharegpt_row(row: Dict[str, Any], *, ranking: bool) -> Optional[Dict[str, Any]]:
    """规整一条样本。结构仍不合法时返回 None（调用方应丢弃并记录）。"""
    out = dict(row)
    out["conversations"] = normalize_messages(row.get("conversations") or [])
    if ranking:
        if "chosen" in out:
            out["chosen"] = _normalize_response(out.get("chosen"))
        if "rejected" in out:
            out["rejected"] = _normalize_response(out.get("rejected"))
    if alignment_error(out["conversations"], ranking=ranking):
        return None
    return out


def rows_are_aligned(rows: Sequence[Dict[str, Any]], *, ranking: bool) -> bool:
    if not rows:
        return False
    return all(alignment_error(r.get("conversations") or [], ranking=ranking) is None for r in rows)


def _row_sample_id(row: Dict[str, Any]) -> Optional[str]:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("sample_id"):
        return str(meta["sample_id"])
    return None


def _should_keep_existing_lf(
    existing: Sequence[Dict[str, Any]],
    source_rows: Optional[Sequence[Dict[str, Any]]],
    *,
    ranking: bool,
) -> bool:
    """只有结构合法，且相对候选源不是缩水/无 id 的旧子集时，才沿用盘上的 lf。"""
    if not rows_are_aligned(existing, ranking=ranking):
        return False
    if source_rows is None:
        return True
    # 源有 sample_id 而 lf 大量缺失 → 旧 release 子集，必须重生成
    src_ids = {_row_sample_id(r) for r in source_rows}
    src_ids.discard(None)
    lf_ids = {_row_sample_id(r) for r in existing}
    lf_ids.discard(None)
    if src_ids:
        missing = src_ids - lf_ids
        # 相对 v2/候选源丢了超过 5%（或超过 3 条）→ 旧子集，重生成
        if len(missing) > max(3, int(0.05 * len(src_ids))):
            return False
    if len(existing) < int(0.95 * len(source_rows)):
        return False
    return True


def _first_existing(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _sample_label(row: Dict[str, Any], index: int) -> str:
    sid = row.get("sample_id") or (row.get("metadata") or {}).get("sample_id")
    return str(sid or f"#{index}")


def _normalize_all(
    rows: Sequence[Dict[str, Any]], *, ranking: bool
) -> Tuple[List[Dict[str, Any]], List[str]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for i, row in enumerate(rows):
        normalized = normalize_sharegpt_row(row, ranking=ranking)
        if normalized is None:
            reason = alignment_error(
                normalize_messages(row.get("conversations") or []), ranking=ranking
            )
            dropped.append(f"{_sample_label(row, i)}: {reason or 'invalid'}")
            continue
        kept.append(normalized)
    return kept, dropped


def dataset_info_for(prepared: PreparedDatasets) -> Dict[str, Any]:
    """file_name 相对 dataset_dir（交付 zip 里是 ``./data``），不再额外加 ``data/`` 前缀。"""
    info: Dict[str, Any] = {}
    order = [
        "sft.train.jsonl",
        "sft.eval.jsonl",
        "dpo.train.jsonl",
        "dpo.eval.jsonl",
        "sft.train.lf.jsonl",
        "sft.eval.lf.jsonl",
        "dpo.train.lf.jsonl",
        "dpo.eval.lf.jsonl",
    ]
    for name in order:
        if not prepared.has(name):
            continue
        key = SHARE_KEYS.get(name) or LF_KEYS[name]
        ranking = name.startswith("dpo.")
        entry: Dict[str, Any] = {
            "file_name": name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
            "tags": dict(_SHAREGPT_TAGS),
        }
        if ranking:
            entry["columns"] = {
                "messages": "conversations",
                "tools": "tools",
                "chosen": "chosen",
                "rejected": "rejected",
            }
            entry["ranking"] = True
        info[key] = entry
    return info


def prepare_datasets(
    root: Path, *, apply_length_gate: bool = True, log=None
) -> PreparedDatasets:
    """从 ``03_train`` 目录选出并规整出交付用的 sft/dpo（含 lf）。

    默认在规整后跑 observation 软压缩（按 yaml 默认 cutoff），**不因超长丢弃**。
    would-be-overlength 写入 ``length_gate.overlength_at_default_cutoff`` 供 MANIFEST；
    训练入口再按当次 ``cutoff_len`` 硬过滤。
    """
    root = Path(root)
    prepared = PreparedDatasets()
    for logical, lf_name, share_name, ranking, candidates in SPECS:
        source = _first_existing(root, candidates)
        existing_lf = root / lf_name
        lf_rows: Optional[List[Dict[str, Any]]] = None
        origin = ""
        dropped: List[str] = []
        source_rows = load_jsonl(source) if source is not None else None
        if existing_lf.is_file():
            loaded = load_jsonl(existing_lf)
            if _should_keep_existing_lf(loaded, source_rows, ranking=ranking):
                lf_rows = loaded
                origin = f"existing:{lf_name}"
        if lf_rows is None and source_rows is not None:
            lf_rows, dropped = _normalize_all(source_rows, ranking=ranking)
            origin = source.name
        if not lf_rows:
            prepared.dropped[logical] = dropped
            continue
        prepared.files[lf_name] = lf_rows
        prepared.sources[logical] = origin
        prepared.dropped[logical] = dropped
        prepared.counts[lf_name] = len(lf_rows)
        if source is not None:
            prepared.files[share_name] = load_jsonl(source)
            prepared.counts[share_name] = len(prepared.files[share_name])
        elif (root / share_name).is_file():
            prepared.files[share_name] = load_jsonl(root / share_name)
            prepared.counts[share_name] = len(prepared.files[share_name])

    if apply_length_gate and prepared.files:
        from pipeline import length_gate as lg

        gated_files, gate_stats = lg.gate_prepared_files(
            prepared.files, root=root, log=log
        )
        prepared.files = gated_files
        for name, rows in list(prepared.files.items()):
            if name.endswith(".lf.jsonl"):
                prepared.counts[name] = len(rows)
        prepared.length_gate = lg.stats_to_manifest(gate_stats)
        # 超长样本仍在包内；不再写入 prepared.dropped（那只记录结构非法丢弃）
    return prepared


def require_sft(prepared: PreparedDatasets) -> None:
    missing = [
        name
        for name in ("sft.train.lf.jsonl", "sft.eval.lf.jsonl")
        if not prepared.has(name)
    ]
    if not missing:
        return
    raise FileNotFoundError(
        "无法生成可训练的 SFT 文件（"
        + ", ".join(missing)
        + "）。需要已有 *.lf.jsonl，或 v2 / sft.train.jsonl / "
        "sft_step_windows_sharegpt.{train,eval}.jsonl 之一。"
        "已丢弃："
        + json.dumps(prepared.dropped, ensure_ascii=False)
    )
