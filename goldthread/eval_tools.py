"""训练包全身体检、包间 Diff 与学生实测归因工具。"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from pipeline import dataset_build, stage1_extract
from pipeline.common import format as fmt
from pipeline.common import pilog
from pipeline.common import seqlength as sl
from pipeline.common.config import ROOT, load_config, resolve_path, stage_dirs
from pipeline.common.store import read_json, read_jsonl

_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"
_XML_STRAY_RE = re.compile(r"<function=|<parameter=")
_PROMPT_TAG_RE = re.compile(
    r"<(?:file\b[^>]*|/file|user_query|/user_query|image_files|/image_files|timestamp|/timestamp)>",
    re.IGNORECASE,
)
_PREMATURE_TEXT_RE = re.compile(
    r"(是否继续|请确认|请指示|请告知|请选择|等待.*指令|下一步.*方案|如何处理)",
    re.IGNORECASE,
)
_CTX_OVERFLOW_RE = re.compile(
    r"(context[_\s-]*length|maximum context|max[_\s-]*tokens|too many tokens|131072|token limit|aborted)",
    re.IGNORECASE,
)
_OFFLINE_EMU_RE = re.compile(r"\b(unicorn|capstone)\b", re.IGNORECASE)

_TEMPLATE_STYLE = {
    "qwen3_8": "coder-xml",
    "qwen3": "hermes-json",
}

_ZIP_HASH_RE = re.compile(r"^(?:goldthread-train-)?([a-f0-9]{8})(?:\.zip)?$", re.IGNORECASE)
_BUILD_HEX_RE = re.compile(r"^[a-f0-9]{16,64}$", re.IGNORECASE)


def resolve_pack_or_build(
    target: str | Path, cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """解析目标为 zip 包、解压/训练目录或不可变 Build 目录。"""
    cfg = cfg or load_config()
    raw_str = str(target or "").strip()
    if not raw_str:
        raise ValueError("评估目标不能为空")

    p = Path(raw_str)
    if p.exists():
        resolved = p.resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".zip":
            m = _ZIP_HASH_RE.match(resolved.name)
            return {
                "kind": "zip",
                "path": resolved,
                "label": resolved.name,
                "pack_hash": m.group(1).lower() if m else None,
                "build_id": None,
            }
        if resolved.is_dir():
            is_build = (resolved / "BUILD.json").is_file()
            return {
                "kind": "build" if is_build else "dir",
                "path": resolved,
                "label": resolved.name,
                "pack_hash": None,
                "build_id": resolved.name if is_build else None,
            }

    dirs = stage_dirs(cfg)
    m = _ZIP_HASH_RE.match(raw_str)
    if m:
        h = m.group(1).lower()
        cand = dirs["root"] / "04_release" / "delivery" / f"goldthread-train-{h}.zip"
        if cand.is_file():
            return {
                "kind": "zip",
                "path": cand.resolve(),
                "label": cand.name,
                "pack_hash": h,
                "build_id": None,
            }

    if raw_str.lower() == "latest" or _BUILD_HEX_RE.match(raw_str):
        bid = None if raw_str.lower() == "latest" else raw_str.lower()
        build_dir, manifest = dataset_build.resolve_build(cfg, bid)
        return {
            "kind": "build",
            "path": build_dir.resolve(),
            "label": str(manifest.get("build_id") or build_dir.name),
            "pack_hash": None,
            "build_id": str(manifest.get("build_id") or build_dir.name),
        }

    raise FileNotFoundError(f"无法解析训练包或 Build 目标: {target}")


# ---------------------------------------------------------------------------
# 包/目录读取器（支持 zip 不落盘读取与 dir/build 读取）
# ---------------------------------------------------------------------------

def _parse_jsonl_bytes(raw: bytes) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _pick_zip_member(names: set[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    for c in candidates:
        suffix = "/" + c.lstrip("/")
        for n in sorted(names):
            if n.endswith(suffix):
                return n
    return None


def _load_pack_contents(resolved: Dict[str, Any]) -> Dict[str, Any]:
    kind = resolved["kind"]
    path = Path(resolved["path"])
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, str] = {}
    manifest: Dict[str, Any] = {}
    settings_cfg: Dict[str, Any] = {}
    length_stats: Optional[Dict[str, Any]] = None
    build_meta: Dict[str, Any] = {}

    file_candidates = {
        "sft.train": ("sft.train.lf.jsonl", "sft.train.jsonl", "sft_step_windows_sharegpt.train.jsonl"),
        "sft.eval": ("sft.eval.lf.jsonl", "sft.eval.jsonl", "sft_step_windows_sharegpt.eval.jsonl"),
        "dpo.train": ("dpo.train.lf.jsonl", "dpo.train.jsonl", "dpo_preference_pairs.train.jsonl"),
        "dpo.eval": ("dpo.eval.lf.jsonl", "dpo.eval.jsonl", "dpo_preference_pairs.eval.jsonl"),
    }

    if kind == "zip":
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            m_arc = _pick_zip_member(
                names, ("goldthread-train/data/MANIFEST.json", "data/MANIFEST.json", "MANIFEST.json")
            )
            if m_arc:
                try:
                    manifest = json.loads(zf.read(m_arc).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    manifest = {}

            s_arc = _pick_zip_member(
                names, ("goldthread-train/settings.yaml", "settings.yaml")
            )
            if s_arc:
                try:
                    settings_cfg = yaml.safe_load(zf.read(s_arc).decode("utf-8")) or {}
                except Exception:  # noqa: BLE001
                    settings_cfg = {}

            ls_arc = _pick_zip_member(
                names, ("goldthread-train/data/length_stats.json", "data/length_stats.json", "length_stats.json")
            )
            if ls_arc:
                try:
                    cand_ls = json.loads(zf.read(ls_arc).decode("utf-8"))
                    if isinstance(cand_ls, dict) and cand_ls.get("ok"):
                        length_stats = cand_ls
                except (json.JSONDecodeError, UnicodeDecodeError):
                    length_stats = None

            for key, cands in file_candidates.items():
                expanded_cands = []
                for fn in cands:
                    expanded_cands.extend([f"goldthread-train/data/{fn}", f"data/{fn}", fn])
                arc = _pick_zip_member(names, expanded_cands)
                if arc:
                    datasets[key] = _parse_jsonl_bytes(zf.read(arc))
                    sources[key] = Path(arc).name
    else:
        data_dir = path / "data" if (path / "data").is_dir() else path
        m_path = data_dir / "MANIFEST.json"
        if m_path.is_file():
            manifest = read_json(m_path, default={}) or {}
        b_path = path / "BUILD.json"
        if b_path.is_file():
            build_meta = read_json(b_path, default={}) or {}
        s_path = path / "settings.yaml"
        if s_path.is_file():
            try:
                settings_cfg = yaml.safe_load(s_path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                settings_cfg = {}
        ls_path = data_dir / "length_stats.json"
        if ls_path.is_file():
            cand_ls = read_json(ls_path, default=None)
            if isinstance(cand_ls, dict) and cand_ls.get("ok"):
                length_stats = cand_ls

        for key, cands in file_candidates.items():
            for fn in cands:
                fp = data_dir / fn
                if fp.is_file():
                    datasets[key] = read_jsonl(fp)
                    sources[key] = fn
                    break

    return {
        "datasets": datasets,
        "sources": sources,
        "manifest": manifest,
        "settings": settings_cfg,
        "length_stats": length_stats,
        "build_meta": build_meta,
    }


def _enrich_metadata_from_build(
    datasets: Dict[str, List[Dict[str, Any]]],
    manifest: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    """若包内样本缺少 metadata 且本地存在关联 Build，则尝试补读 metadata。"""
    build_id = str(manifest.get("build_id") or "").strip()
    if not build_id:
        return
    needs_enrich = any(
        not isinstance(r.get("metadata"), dict) or not r["metadata"].get("project_group")
        for rows in datasets.values()
        for r in rows
    )
    if not needs_enrich:
        return
    try:
        build_dir, _bm = dataset_build.resolve_build(cfg, build_id)
    except FileNotFoundError:
        return

    meta_by_sid: Dict[str, Dict[str, Any]] = {}
    for fn in (
        "sft_step_windows_sharegpt.train.jsonl",
        "sft_step_windows_sharegpt.eval.jsonl",
        "dpo_preference_pairs.train.jsonl",
        "dpo_preference_pairs.eval.jsonl",
    ):
        fp = build_dir / fn
        if not fp.is_file():
            continue
        for row in read_jsonl(fp):
            sid = str(row.get("sample_id") or "")
            md = row.get("metadata")
            if sid and isinstance(md, dict):
                meta_by_sid[sid] = md
    if not meta_by_sid:
        return

    for rows in datasets.values():
        for row in rows:
            if isinstance(row.get("metadata"), dict) and row["metadata"].get("project_group"):
                continue
            sid = str(row.get("sample_id") or "").split("#w", 1)[0]
            if sid in meta_by_sid:
                merged = dict(meta_by_sid[sid])
                if isinstance(row.get("metadata"), dict):
                    merged.update(row["metadata"])
                row["metadata"] = merged


def _infer_expected_style(
    manifest: Dict[str, Any],
    settings_cfg: Dict[str, Any],
    datasets: Dict[str, List[Dict[str, Any]]],
    explicit_style: Optional[str],
) -> str:
    if explicit_style:
        return explicit_style
    fmt_sec = manifest.get("format") or {}
    fc_sec = manifest.get("format_contract") or {}
    for cand in (fmt_sec.get("tool_call"), fc_sec.get("tool_style")):
        if cand in fmt.TOOL_CALL_STYLES:
            return str(cand)
    tpl = str(((settings_cfg.get("project") or {}).get("template")) or fmt_sec.get("train_template") or "").strip()
    if tpl in _TEMPLATE_STYLE:
        return _TEMPLATE_STYLE[tpl]

    for key in ("sft.train", "sft.eval", "dpo.train", "dpo.eval"):
        for row in (datasets.get(key) or [])[:30]:
            for conv in row.get("conversations") or []:
                if conv.get("from") == "gpt":
                    st = fmt.detect_tool_style(conv.get("value") or "")
                    if st in fmt.TOOL_CALL_STYLES:
                        return st
    return "coder-xml"


def _origin_sample_id(row: Dict[str, Any], idx: int) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    origin = meta.get("origin_sample_id") or row.get("sample_id") or meta.get("sample_id")
    if origin:
        return str(origin).split("#w", 1)[0]
    return f"__row_{idx}"


def _assistant_texts_from_row(row: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for c in row.get("conversations") or []:
        if c.get("from") == "gpt":
            yield "gpt", str(c.get("value") or "")
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            yield "openai-tool_calls", ""
            continue
        yield "assistant", str(m.get("content") or "")
    for key in ("chosen", "rejected"):
        val = row.get(key)
        if isinstance(val, dict):
            if val.get("from") in (None, "gpt", "assistant"):
                yield key, str(val.get("value") or "")
        elif isinstance(val, str):
            yield key, val


def _tool_names_from_field(tools_field: Any) -> set[str]:
    if not tools_field:
        return set()
    if isinstance(tools_field, str):
        try:
            tools_field = json.loads(tools_field)
        except json.JSONDecodeError:
            return set()
    out: set[str] = set()
    for item in tools_field or []:
        fn = (item or {}).get("function") or {}
        if fn.get("name"):
            out.add(str(fn["name"]))
    return out


def _extract_tool_blocks(text: str) -> Tuple[List[Tuple[str, str, Optional[Dict[str, Any]]]], str]:
    spans = fmt._iter_tc_spans(text)  # noqa: SLF001
    if not spans:
        return [], text
    blocks: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    outside_parts: List[str] = []
    cursor = 0
    for s, e in spans:
        outside_parts.append(text[cursor:s])
        cursor = e
        raw_b = text[s:e]
        stripped = raw_b.strip()
        kind = (
            "hermes-json"
            if fmt._HERMES_OPEN_RE.match(stripped)  # noqa: SLF001
            else ("coder-xml" if fmt._XML_OPEN_RE.match(stripped) else "unknown")  # noqa: SLF001
        )
        blocks.append((kind, raw_b, fmt._parse_tc_block(raw_b)))  # noqa: SLF001
    outside_parts.append(text[cursor:])
    return blocks, "".join(outside_parts)


def _sft_assistant_turns(row: Dict[str, Any]) -> List[Tuple[bool, str, bool, str]]:
    """返回 [(has_think, think_body, has_tool_call, full_text)]。"""
    turns: List[Tuple[bool, str, bool, str]] = []
    convs = row.get("conversations")
    if isinstance(convs, list) and convs:
        for c in convs:
            if c.get("from") != "gpt":
                continue
            val = str(c.get("value") or "")
            has = _OPEN_THINK in val and _CLOSE_THINK in val
            body = val.split(_OPEN_THINK, 1)[1].split(_CLOSE_THINK, 1)[0] if has else ""
            has_tc = "<tool_call>" in val
            turns.append((has, body, has_tc, val))
        return turns
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        rc = str(m.get("reasoning_content") or "")
        content = str(m.get("content") or "")
        has_inline = _OPEN_THINK in content and _CLOSE_THINK in content
        has = bool(rc.strip()) or has_inline
        body = rc if rc.strip() else (
            content.split(_OPEN_THINK, 1)[1].split(_CLOSE_THINK, 1)[0] if has_inline else ""
        )
        has_tc = bool(m.get("tool_calls")) or "<tool_call>" in content
        turns.append((has, body, has_tc, content))
    return turns


def _row_calls_per_step(row: Dict[str, Any]) -> List[int]:
    if row.get("conversations"):
        return [
            str(c.get("value") or "").count("<tool_call>")
            for c in row["conversations"]
            if c.get("from") == "gpt"
        ]
    out: List[int] = []
    for m in row.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        out.append(len(m.get("tool_calls") or []) or str(m.get("content") or "").count("<tool_call>"))
    return out


def _is_premature_stop(row: Dict[str, Any], target_text: str) -> bool:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if meta:
        if meta.get("admit_reason") == "premature_stop_without_tool":
            return True
        if "step_index" in meta and "total_steps" in meta:
            try:
                if int(meta["step_index"]) < int(meta["total_steps"]) - 1:
                    return True
            except (TypeError, ValueError):
                pass
        elif "step_rank" in meta and "total_steps" in meta:
            try:
                if int(meta["step_rank"]) < int(meta["total_steps"]):
                    return True
            except (TypeError, ValueError):
                pass
        if meta.get("stage") and meta.get("stage") != "closing":
            return True
        return False
    return bool(_PREMATURE_TEXT_RE.search(target_text))


def _estimate_p99_from_bins(bins: List[Dict[str, Any]], count: int, fallback_max: int) -> int:
    if not bins or count <= 0:
        return fallback_max
    target_idx = max(1, int(count * 0.99))
    cum = 0
    for b in bins:
        c = int(b.get("count") or 0)
        cum += c
        if cum >= target_idx:
            hi = int(b.get("hi") or fallback_max)
            return min(hi, fallback_max) if hi < 10**8 else fallback_max
    return fallback_max


def _compact_length_gate(raw_gate: Any) -> Dict[str, Any]:
    if not isinstance(raw_gate, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in raw_gate.items():
        if isinstance(v, dict):
            out[k] = {
                kk: vv
                for kk, vv in v.items()
                if not isinstance(vv, list)
            }
        else:
            out[k] = v
    return out


def _find_eval_tokenizer() -> Tuple[Any, str]:
    tok, src = sl.find_tokenizer()
    if tok is not None:
        return tok, src
    extra = ROOT / ".tokenizer" / "qwen3_tok.json"
    if extra.is_file():
        tok2, src2 = sl.find_tokenizer(explicit=str(extra))
        if tok2 is not None:
            return tok2, src2
    return None, src


def _compute_token_lengths(
    datasets: Dict[str, List[Dict[str, Any]]],
    length_stats: Optional[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if isinstance(length_stats, dict) and length_stats.get("ok"):
        ds_map = length_stats.get("datasets") or {}
        summary: Dict[str, Any] = {}
        for key in ("sft.train", "sft.eval", "dpo.train", "dpo.eval"):
            raw_ds = ds_map.get(key)
            if not isinstance(raw_ds, dict):
                continue
            bins = raw_ds.get("spark") or raw_ds.get("bins") or []
            over_12k = sum(
                int(b.get("count") or 0)
                for b in (raw_ds.get("bins") or [])
                if int(b.get("lo") or 0) >= 12288
            )
            if not over_12k and raw_ds.get("spark"):
                over_12k = sum(
                    int(b.get("count") or 0)
                    for b in raw_ds["spark"]
                    if int(b.get("lo") or 0) >= 12288
                )
            p99 = raw_ds.get("p99")
            if p99 is None:
                p99 = _estimate_p99_from_bins(
                    bins, int(raw_ds.get("count") or 0), int(raw_ds.get("max") or 0)
                )
            summary[key] = {
                "count": int(raw_ds.get("count") or 0),
                "p50": int(raw_ds.get("p50") or 0),
                "p90": int(raw_ds.get("p90") or 0),
                "p99": int(p99),
                "max": int(raw_ds.get("max") or 0),
                "mean": int(raw_ds.get("mean") or 0),
                "over_10k": int(raw_ds.get("over_10k") or 0),
                "over_12k": int(over_12k),
            }
        raw_gate = length_stats.get("length_gate") or manifest.get("length_gate") or {}
        return summary, _compact_length_gate(raw_gate), "pack"

    tok, tok_src = _find_eval_tokenizer()
    summary = {}
    for key in ("sft.train", "sft.eval", "dpo.train", "dpo.eval"):
        rows = datasets.get(key) or []
        if not rows:
            continue
        if key.startswith("dpo"):
            lens = [n for n, _ in sl.dpo_lengths(rows, tok, "")]
        else:
            lens = [n for n, _ in sl.sft_lengths(rows, tok, "")]
        q = sl.quantiles(lens)
        summary[key] = {
            "count": int(q.get("n") or 0),
            "p50": int(q.get("p50") or 0),
            "p90": int(q.get("p90") or 0),
            "p99": int(q.get("p99") or 0),
            "max": int(q.get("max") or 0),
            "mean": int(q.get("mean") or 0),
            "over_10k": sum(1 for x in lens if x >= 10240),
            "over_12k": int(q.get("over_12k") or 0),
        }
    gate = _compact_length_gate(manifest.get("length_gate") or {})
    if not gate:
        gate = {"tokenizer": tok_src, "computed": True}
    return summary, gate, f"computed:{tok_src}"


def eval_pack(
    target: str | Path,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    expected_style: Optional[str] = None,
) -> Dict[str, Any]:
    """对交付 zip 包、解压目录或 Build 执行全维度体检。"""
    cfg = cfg or load_config()
    resolved = resolve_pack_or_build(target, cfg)
    loaded = _load_pack_contents(resolved)
    datasets = loaded["datasets"]
    manifest = loaded["manifest"]
    settings_cfg = loaded["settings"]
    length_stats = loaded["length_stats"]
    build_meta = loaded["build_meta"]

    _enrich_metadata_from_build(datasets, manifest, cfg)
    style = _infer_expected_style(manifest, settings_cfg, datasets, expected_style)

    sft_train = datasets.get("sft.train") or []
    sft_eval = datasets.get("sft.eval") or []
    dpo_train = datasets.get("dpo.train") or []
    dpo_eval = datasets.get("dpo.eval") or []

    # 1) 基础规模与权重展开
    sft_train_unique_ids = {_origin_sample_id(r, i) for i, r in enumerate(sft_train)}
    sft_eval_unique_ids = {_origin_sample_id(r, i) for i, r in enumerate(sft_eval)}
    if (
        len(sft_train_unique_ids) == len(sft_train)
        and isinstance(manifest.get("weighted_expansion"), dict)
    ):
        we = (manifest["weighted_expansion"].get("sft.train") or {})
        sft_train_unique = int(we.get("unique_rows") or len(sft_train_unique_ids))
    else:
        sft_train_unique = len(sft_train_unique_ids)
    sft_eval_unique = len(sft_eval_unique_ids)

    # 2) 项目覆盖与归组规范性（缺失 project_group 的样本严格标为 unknown 并计入非规范归组）
    train_projects_set: set[str] = set()
    eval_projects_set: set[str] = set()
    for rows, bucket in (
        (sft_train, train_projects_set),
        (dpo_train, train_projects_set),
        (sft_eval, eval_projects_set),
        (dpo_eval, eval_projects_set),
    ):
        for r in rows:
            md = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            pg = str(md.get("project_group") or "").strip() or "unknown"
            bucket.add(pg)
    all_projects_set = train_projects_set | eval_projects_set
    projects = sorted(all_projects_set)
    non_canonical_projects = [p for p in projects if not p.startswith("game::")]

    # 3) 工具调用契约 & 6) Prompt 标签污染率
    ok_blocks = wrong_style = stray_xml = bad_blocks = 0
    bad_examples: List[Dict[str, Any]] = []
    prompt_tag_leaks = 0
    sft_prompt_tag_leaks = 0
    dpo_prompt_tag_leaks = 0
    total_checked_rows = 0

    for ds_key in ("sft.train", "sft.eval", "dpo.train", "dpo.eval"):
        for idx, row in enumerate(datasets.get(ds_key) or []):
            total_checked_rows += 1
            names = _tool_names_from_field(row.get("tools"))
            leaked = False
            for turn in (row.get("conversations") or row.get("messages") or []):
                role = str(turn.get("from") or turn.get("role") or "")
                if role in ("human", "user"):
                    val = str(turn.get("value") or turn.get("content") or "")
                    if _PROMPT_TAG_RE.search(val):
                        leaked = True
                        break
            if leaked:
                prompt_tag_leaks += 1
                if ds_key.startswith("sft"):
                    sft_prompt_tag_leaks += 1
                else:
                    dpo_prompt_tag_leaks += 1

            for where, text in _assistant_texts_from_row(row):
                if not text.strip():
                    continue
                blocks, outside = _extract_tool_blocks(text)
                if _XML_STRAY_RE.search(outside):
                    stray_xml += 1
                    if len(bad_examples) < 5:
                        bad_examples.append({
                            "dataset": ds_key,
                            "row_index": idx,
                            "where": where,
                            "issue": "stray_xml",
                            "sample_id": row.get("sample_id"),
                        })
                for kind, _raw_b, parsed in blocks:
                    if kind == "unknown" or parsed is None:
                        bad_blocks += 1
                        if len(bad_examples) < 5:
                            bad_examples.append({
                                "dataset": ds_key,
                                "row_index": idx,
                                "where": where,
                                "issue": "unparseable_block",
                                "sample_id": row.get("sample_id"),
                            })
                        continue
                    if kind != style:
                        wrong_style += 1
                        if len(bad_examples) < 5:
                            bad_examples.append({
                                "dataset": ds_key,
                                "row_index": idx,
                                "where": where,
                                "issue": f"wrong_style:{kind}!=expected:{style}",
                                "sample_id": row.get("sample_id"),
                            })
                        continue
                    fn_name = parsed.get("name")
                    if not isinstance(fn_name, str) or not fn_name:
                        bad_blocks += 1
                        continue
                    if names and fn_name not in names:
                        bad_blocks += 1
                        if len(bad_examples) < 5:
                            bad_examples.append({
                                "dataset": ds_key,
                                "row_index": idx,
                                "where": where,
                                "issue": f"unknown_tool:{fn_name}",
                                "sample_id": row.get("sample_id"),
                            })
                        continue
                    if not isinstance(parsed.get("arguments"), dict):
                        bad_blocks += 1
                        continue
                    ok_blocks += 1

    prompt_tag_leak_rate = (
        round(prompt_tag_leaks / total_checked_rows, 4) if total_checked_rows else 0.0
    )

    # 4) 思考形态一致性 & 5) 中途假收尾 stop 步检测 & 7) 并行工具调用率
    hist_total = hist_with_think = 0
    missing_assistant_targets = 0
    missing_target_thinking = 0
    invalid_target_think_tags = 0
    invalid_think_examples: List[Dict[str, Any]] = []
    short_target_thinking = 0
    stop_targets = 0
    premature_stop_targets = 0
    premature_stop_examples: List[Dict[str, Any]] = []
    file_shapes: set[str] = set()

    parallel_stats: Dict[str, List[int]] = {
        "sft.train": [0, 0],  # [total_steps, parallel_steps]
        "sft.eval": [0, 0],
        "target": [0, 0],
    }

    for ds_key in ("sft.train", "sft.eval"):
        rows = datasets.get(ds_key) or []
        f_hist = f_hist_th = 0
        for row_number, row in enumerate(rows, start=1):
            conversations = row.get("conversations")
            sharegpt = isinstance(conversations, list) and bool(conversations)
            convs = conversations if sharegpt else row.get("messages") or []
            role_key, expected_role = ("from", "gpt") if sharegpt else ("role", "assistant")
            if not convs or not isinstance(convs[-1], dict) or convs[-1].get(role_key) != expected_role:
                missing_assistant_targets += 1
                missing_target_thinking += 1
                continue
            turns = _sft_assistant_turns(row)
            for has, _body, _has_tc, _full in turns[:-1]:
                f_hist += 1
                hist_total += 1
                if has:
                    f_hist_th += 1
                    hist_with_think += 1
            t_has, t_body, t_has_tc, t_full = turns[-1]
            if not t_has:
                missing_target_thinking += 1
            elif len(t_body.strip()) < 20:
                short_target_thinking += 1
            opens = t_full.count(_OPEN_THINK)
            closes = t_full.count(_CLOSE_THINK)
            # ShareGPT/LF requires inline tags; OpenAI may carry reasoning_content separately.
            if (sharegpt or opens or closes) and not (
                opens == closes == 1 and t_full.index(_OPEN_THINK) < t_full.index(_CLOSE_THINK)
            ):
                invalid_target_think_tags += 1
                if len(invalid_think_examples) < 5:
                    invalid_think_examples.append({
                        "dataset": ds_key, "row_number": row_number, "sample_id": row.get("sample_id"),
                        "opens": opens, "closes": closes,
                    })
            if not t_has_tc:
                stop_targets += 1
                if _is_premature_stop(row, t_full):
                    premature_stop_targets += 1
                    if len(premature_stop_examples) < 5:
                        md = row.get("metadata") or {}
                        premature_stop_examples.append({
                            "dataset": ds_key,
                            "sample_id": row.get("sample_id"),
                            "session_id": md.get("session_id"),
                            "step_rank": md.get("step_rank"),
                            "total_steps": md.get("total_steps"),
                            "stage": md.get("stage"),
                        })

            calls_list = _row_calls_per_step(row)
            for n_calls in calls_list:
                parallel_stats[ds_key][0] += 1
                if n_calls >= 2:
                    parallel_stats[ds_key][1] += 1
            if calls_list:
                parallel_stats["target"][0] += 1
                if calls_list[-1] >= 2:
                    parallel_stats["target"][1] += 1

        if f_hist > 0:
            r_ratio = f_hist_th / f_hist
            file_shapes.add("keep" if r_ratio > 0.9 else ("strip" if r_ratio < 0.1 else "mixed"))

    hist_think_ratio = round(hist_with_think / hist_total, 4) if hist_total else 0.0
    if not file_shapes:
        thinking_shape = "none"
    elif len(file_shapes) > 1 or "mixed" in file_shapes:
        thinking_shape = "mixed"
    else:
        thinking_shape = next(iter(file_shapes))
    empty_target_thinking = missing_target_thinking + short_target_thinking

    def _rate(pair: List[int]) -> float:
        return round(pair[1] / pair[0], 4) if pair[0] else 0.0

    # 8) DPO 质量与来源分布
    dpo_invalid_pairs = 0
    dpo_style_mismatch_pairs = 0
    dpo_unexpected_style_mismatch_pairs = 0
    dpo_by_source: Counter = Counter()
    dpo_by_reason: Counter = Counter()

    for row in (*dpo_train, *dpo_eval):
        md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        src = str(md.get("source") or "unknown")
        reason = str(md.get("reason") or "unknown")
        dpo_by_source[src] += 1
        dpo_by_reason[reason] += 1
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        c_val = chosen.get("value", "") if isinstance(chosen, dict) else str(chosen or "")
        r_val = rejected.get("value", "") if isinstance(rejected, dict) else str(rejected or "")
        if not c_val.strip() or not r_val.strip() or c_val.strip() == r_val.strip():
            dpo_invalid_pairs += 1
        if ("<tool_call>" in c_val) != ("<tool_call>" in r_val):
            dpo_style_mismatch_pairs += 1
            if reason != "premature_stop_folded":
                dpo_unexpected_style_mismatch_pairs += 1

    # 9) Token 长度分布
    token_lengths, length_gate, length_source = _compute_token_lengths(
        datasets, length_stats, manifest
    )

    # 10) 总体健康结论
    violations: List[Dict[str, Any]] = []
    if len(sft_train) == 0 or len(sft_eval) == 0:
        violations.append({
            "severity": "error",
            "code": "missing_sft",
            "message": f"SFT 数据缺失 (train={len(sft_train)}, eval={len(sft_eval)})",
        })
    if wrong_style or stray_xml or bad_blocks:
        violations.append({
            "severity": "error",
            "code": "toolcall_contract",
            "message": (
                f"工具调用契约异常 (wrong_style={wrong_style}, stray_xml={stray_xml}, bad_blocks={bad_blocks})"
            ),
        })
    if thinking_shape == "mixed":
        violations.append({
            "severity": "error",
            "code": "thinking_shape_mixed",
            "message": f"SFT 历史思考形态不一致 (ratio={hist_think_ratio:.2%})",
        })
    if missing_assistant_targets > 0:
        violations.append({
            "severity": "error",
            "code": "missing_assistant_target",
            "message": f"SFT 样本缺少有效的末尾 assistant 监督目标 ({missing_assistant_targets} 条)",
        })
    if missing_target_thinking > 0:
        violations.append({
            "severity": "error",
            "code": "missing_target_thinking",
            "message": f"目标步缺少 <think> 块 ({missing_target_thinking} 条)",
        })
    if invalid_target_think_tags > 0:
        violations.append({
            "severity": "error",
            "code": "invalid_target_think_tags",
            "message": f"SFT 目标步须恰好一对顺序正确的 <think> 标签 ({invalid_target_think_tags} 条)",
            "examples": invalid_think_examples,
        })
    if short_target_thinking > 0:
        violations.append({
            "severity": "warning",
            "code": "short_target_thinking",
            "message": f"目标步思考过短 (<20 字符, {short_target_thinking} 条)",
        })
    if premature_stop_targets > 0:
        violations.append({
            "severity": "error",
            "code": "premature_stop_targets",
            "message": f"SFT 混入中途假收尾 stop 步 ({premature_stop_targets} 条)",
        })
    if prompt_tag_leaks > 0:
        violations.append({
            "severity": "error",
            "code": "prompt_tag_leaks",
            "message": f"用户 Prompt 残留采集外壳标签 ({prompt_tag_leaks} 条, {prompt_tag_leak_rate:.2%})",
        })
    if non_canonical_projects:
        violations.append({
            "severity": "error",
            "code": "non_canonical_project_group",
            "message": f"存在非规范项目组命名: {', '.join(non_canonical_projects[:5])}",
        })
    if dpo_invalid_pairs > 0:
        violations.append({
            "severity": "error",
            "code": "dpo_invalid_pairs",
            "message": f"存在无效 DPO 偏好对 ({dpo_invalid_pairs} 对)",
        })
    if dpo_unexpected_style_mismatch_pairs > 0:
        violations.append({
            "severity": "warning",
            "code": "dpo_style_mismatch_pairs",
            "message": f"DPO 存在非预期的单侧 tool_call 不一致 ({dpo_unexpected_style_mismatch_pairs} 对)",
        })
    if isinstance(length_gate, dict) and (
        "_skip" in length_gate
        or any(isinstance(v, dict) and v.get("skipped") for v in length_gate.values())
    ):
        violations.append({
            "severity": "error",
            "code": "length_gate_skipped",
            "message": "长度闸门被跳过（未生效）",
        })

    healthy = not any(v.get("severity") == "error" for v in violations)

    return {
        "target": str(target),
        "resolved_kind": resolved["kind"],
        "resolved_path": str(resolved["path"]),
        "package_hash": manifest.get("package_hash") or resolved.get("pack_hash"),
        "build_id": manifest.get("build_id") or build_meta.get("build_id") or resolved.get("build_id"),
        "sources": loaded["sources"],
        # 1) 基础规模
        "sft_train_rows": len(sft_train),
        "sft_eval_rows": len(sft_eval),
        "dpo_train_rows": len(dpo_train),
        "dpo_eval_rows": len(dpo_eval),
        "sft_train_unique": sft_train_unique,
        "sft_eval_unique": sft_eval_unique,
        "sft_unique_windows": sft_train_unique + sft_eval_unique,
        # 2) 项目覆盖
        "projects": projects,
        "project_count": len(projects),
        "train_projects": sorted(train_projects_set),
        "eval_projects": sorted(eval_projects_set),
        "non_canonical_projects": non_canonical_projects,
        "non_canonical_project_count": len(non_canonical_projects),
        # 3) 工具调用契约
        "expected_style": style,
        "ok_blocks": ok_blocks,
        "wrong_style": wrong_style,
        "stray_xml": stray_xml,
        "bad_blocks": bad_blocks,
        "bad_examples": bad_examples,
        # 4) 思考形态
        "hist_think_total": hist_total,
        "hist_think_count": hist_with_think,
        "hist_think_ratio": hist_think_ratio,
        "thinking_shape": thinking_shape,
        "empty_target_thinking": empty_target_thinking,
        "missing_assistant_targets": missing_assistant_targets,
        "missing_target_thinking": missing_target_thinking,
        "invalid_target_think_tags": invalid_target_think_tags,
        "short_target_thinking": short_target_thinking,
        # 5) 中途假收尾
        "stop_targets": stop_targets,
        "premature_stop_targets": premature_stop_targets,
        "premature_stop_examples": premature_stop_examples,
        # 6) 标签污染
        "prompt_tag_leaks": prompt_tag_leaks,
        "sft_prompt_tag_leaks": sft_prompt_tag_leaks,
        "dpo_prompt_tag_leaks": dpo_prompt_tag_leaks,
        "prompt_tag_leak_rate": prompt_tag_leak_rate,
        # 7) 并行工具调用率
        "sft_train_parallel_rate": _rate(parallel_stats["sft.train"]),
        "sft_eval_parallel_rate": _rate(parallel_stats["sft.eval"]),
        "target_parallel_rate": _rate(parallel_stats["target"]),
        # 8) DPO 质量
        "dpo_invalid_pairs": dpo_invalid_pairs,
        "dpo_style_mismatch_pairs": dpo_style_mismatch_pairs,
        "dpo_unexpected_style_mismatch_pairs": dpo_unexpected_style_mismatch_pairs,
        "dpo_by_source": dict(dpo_by_source),
        "dpo_by_reason": dict(dpo_by_reason),
        # 9) 长度分布
        "length_source": length_source,
        "length_gate": length_gate,
        "token_lengths": token_lengths,
        # 10) 健康结论
        "healthy": healthy,
        "violations": violations,
    }


def _compact_eval_summary(ev: Dict[str, Any]) -> Dict[str, Any]:
    sft_len = (ev.get("token_lengths") or {}).get("sft.train") or {}
    dpo_len = (ev.get("token_lengths") or {}).get("dpo.train") or {}
    return {
        "label": ev.get("package_hash") or ev.get("build_id") or Path(ev["resolved_path"]).name,
        "resolved_path": ev.get("resolved_path"),
        "build_id": ev.get("build_id"),
        "project_count": ev.get("project_count", 0),
        "non_canonical_project_count": ev.get("non_canonical_project_count", 0),
        "sft_unique_windows": ev.get("sft_unique_windows", 0),
        "sft_train_rows": ev.get("sft_train_rows", 0),
        "sft_eval_rows": ev.get("sft_eval_rows", 0),
        "dpo_total_rows": (ev.get("dpo_train_rows", 0) + ev.get("dpo_eval_rows", 0)),
        "dpo_train_rows": ev.get("dpo_train_rows", 0),
        "dpo_eval_rows": ev.get("dpo_eval_rows", 0),
        "premature_stop_targets": ev.get("premature_stop_targets", 0),
        "prompt_tag_leaks": ev.get("prompt_tag_leaks", 0),
        "prompt_tag_leak_rate": ev.get("prompt_tag_leak_rate", 0.0),
        "sft_train_parallel_rate": ev.get("sft_train_parallel_rate", 0.0),
        "sft_eval_parallel_rate": ev.get("sft_eval_parallel_rate", 0.0),
        "thinking_shape": ev.get("thinking_shape"),
        "empty_target_thinking": ev.get("empty_target_thinking", 0),
        "wrong_style": ev.get("wrong_style", 0),
        "stray_xml": ev.get("stray_xml", 0),
        "bad_blocks": ev.get("bad_blocks", 0),
        "dpo_by_source": ev.get("dpo_by_source", {}),
        "sft_train_p90_tokens": sft_len.get("p90", 0),
        "sft_train_max_tokens": sft_len.get("max", 0),
        "sft_train_over_12k": sft_len.get("over_12k", 0),
        "dpo_train_max_tokens": dpo_len.get("max", 0),
        "healthy": bool(ev.get("healthy")),
        "violation_count": len(ev.get("violations") or []),
    }


def eval_diff(
    before_target: str | Path,
    after_target: str | Path,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    before_eval: Optional[Dict[str, Any]] = None,
    after_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对比两个训练包/Build 的体检指标，产出差值与 Markdown 对照表。"""
    cfg = cfg or load_config()
    before_full = before_eval if before_eval is not None else eval_pack(before_target, cfg=cfg)
    after_full = after_eval if after_eval is not None else eval_pack(after_target, cfg=cfg)
    before = _compact_eval_summary(before_full)
    after = _compact_eval_summary(after_full)

    numeric_keys = (
        "project_count",
        "non_canonical_project_count",
        "sft_unique_windows",
        "sft_train_rows",
        "sft_eval_rows",
        "dpo_total_rows",
        "dpo_train_rows",
        "dpo_eval_rows",
        "premature_stop_targets",
        "prompt_tag_leaks",
        "prompt_tag_leak_rate",
        "sft_train_parallel_rate",
        "sft_eval_parallel_rate",
        "empty_target_thinking",
        "wrong_style",
        "stray_xml",
        "bad_blocks",
        "sft_train_p90_tokens",
        "sft_train_max_tokens",
        "sft_train_over_12k",
        "dpo_train_max_tokens",
    )
    deltas: Dict[str, Dict[str, Any]] = {}
    for k in numeric_keys:
        b = before.get(k, 0)
        a = after.get(k, 0)
        diff = round(a - b, 4) if isinstance(a, float) or isinstance(b, float) else (a - b)
        pct: Optional[float] = None
        if isinstance(b, (int, float)) and b != 0:
            pct = round((a - b) / b * 100.0, 2)
        deltas[k] = {"before": b, "after": a, "delta": diff, "pct": pct}

    rows_spec = [
        ("覆盖项目数", "project_count", "int"),
        ("非规范项目组数", "non_canonical_project_count", "int"),
        ("SFT 唯一窗口数", "sft_unique_windows", "int"),
        ("SFT Train 行数", "sft_train_rows", "int"),
        ("SFT Eval 行数", "sft_eval_rows", "int"),
        ("DPO 总对数", "dpo_total_rows", "int"),
        ("SFT 中途假收尾步数", "premature_stop_targets", "int"),
        ("Prompt 标签污染数", "prompt_tag_leaks", "int"),
        ("SFT Train 并行调用率", "sft_train_parallel_rate", "pct"),
        ("SFT Train P90 Tokens", "sft_train_p90_tokens", "int"),
        ("SFT Train Max Tokens", "sft_train_max_tokens", "int"),
        ("SFT Train >12K 条数", "sft_train_over_12k", "int"),
    ]
    md_lines = [
        f"| 指标 | Before (`{before['label']}`) | After (`{after['label']}`) | 变化 |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for title, key, kind in rows_spec:
        d = deltas[key]
        b, a, delta = d["before"], d["after"], d["delta"]
        if kind == "pct":
            b_s = f"{b * 100:.1f}%"
            a_s = f"{a * 100:.1f}%"
            d_s = f"{delta * 100:+.1f}%"
        else:
            b_s = str(b)
            a_s = str(a)
            d_s = f"{delta:+d}" if isinstance(delta, int) else f"{delta:+.2f}"
            if d["pct"] is not None:
                d_s += f" ({d['pct']:+.1f}%)"
        md_lines.append(f"| {title} | {b_s} | {a_s} | {d_s} |")

    return {
        "before": before,
        "after": after,
        "deltas": deltas,
        "markdown_table": "\n".join(md_lines),
    }


# ---------------------------------------------------------------------------
# 学生实测归因 eval_student
# ---------------------------------------------------------------------------

def _collect_shallow_jsonl(root: Path, max_depth: int = 2) -> List[Path]:
    """浅层扫描目录中的 .jsonl（避免陷入解包产物资产树）。"""
    found: List[Path] = []
    root = Path(root)
    if not root.is_dir():
        return found
    for p in sorted(root.glob("*.jsonl")):
        if p.is_file():
            found.append(p)
    if max_depth >= 2:
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            for p in sorted(sub.glob("*.jsonl")):
                if p.is_file():
                    found.append(p)
            sess_sub = sub / "sessions"
            if sess_sub.is_dir():
                for p in sorted(sess_sub.glob("*.jsonl")):
                    if p.is_file():
                        found.append(p)
    return found


def _analyze_pi_session(path: Path, run_name_hint: str, max_tool_chars: int) -> Optional[Dict[str, Any]]:
    try:
        parsed = pilog.parse_session(path, max_tool_output_chars=max_tool_chars)
    except Exception:  # noqa: BLE001
        return None
    msgs = parsed.get("messages") or []
    if not msgs:
        return None

    cwd = str(parsed.get("cwd") or "")
    rel_hint = run_name_hint or path.parent.name
    project_group = stage1_extract.infer_project_group(cwd, rel_hint)
    blocks = pilog.iter_blocks(msgs)
    asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assistant_steps = len(asst_msgs)
    tool_calls = sum(len(m.get("tool_calls") or []) for m in asst_msgs)
    parallel_steps = sum(1 for m in asst_msgs if len(m.get("tool_calls") or []) >= 2)
    parallel_call_rate = round(parallel_steps / assistant_steps, 4) if assistant_steps else 0.0
    empty_think_steps = sum(1 for m in asst_msgs if not (m.get("thinking") or "").strip())

    # Turn 0 早停判定：后续仍有用户催促轮，或 Turn 0 无工具 stop 且步数过短
    turn0_premature_stop = False
    if blocks:
        b0_asst = [m for m in blocks[0][1:] if m.get("role") == "assistant"]
        b0_last = b0_asst[-1] if b0_asst else None
        b0_ended_no_tool = bool(b0_last) and not (b0_last.get("tool_calls") or [])
        if len(blocks) > 1:
            turn0_premature_stop = True
        elif b0_ended_no_tool and b0_last.get("stop_reason") != "aborted" and len(b0_asst) < 35:
            turn0_premature_stop = True

    # 上下文超限/中断检测
    err_blob = " ".join(str(e.get("error") or "") for e in (parsed.get("assistant_errors") or []))
    last_stop = str(asst_msgs[-1].get("stop_reason") or "") if asst_msgs else ""
    tail_text = " ".join(
        str(m.get("text") or "") for m in asst_msgs[-3:]
    )
    max_context_overflow = bool(
        _CTX_OVERFLOW_RE.search(err_blob)
        or last_stop in ("length", "aborted")
        or _CTX_OVERFLOW_RE.search(tail_text)
    )

    # 是否使用 unicorn / capstone 离线模拟
    offline_emu_used = False
    for m in asst_msgs:
        blob = " ".join(
            [
                str(m.get("thinking") or ""),
                str(m.get("text") or ""),
                json.dumps(m.get("tool_calls") or [], ensure_ascii=False),
            ]
        )
        if _OFFLINE_EMU_RE.search(blob):
            offline_emu_used = True
            break

    return {
        "run_name": run_name_hint,
        "session_id": str(parsed.get("session_id") or path.stem),
        "session_file": str(path),
        "project_group": project_group,
        "cwd": cwd,
        "turns": len(blocks),
        "assistant_steps": assistant_steps,
        "tool_calls": tool_calls,
        "parallel_steps": parallel_steps,
        "parallel_call_rate": parallel_call_rate,
        "empty_think_steps": empty_think_steps,
        "turn0_premature_stop": turn0_premature_stop,
        "max_context_overflow": max_context_overflow,
        "offline_emu_used": offline_emu_used,
        "line_count": int(parsed.get("line_count") or 0),
    }


def _derive_run_name(path: Path, student_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(student_root.resolve())
        if len(rel.parts) >= 2:
            return rel.parts[0]
        return path.stem
    except ValueError:
        parent = path.parent.name
        marker = f"{student_root.name}-"
        if marker in parent:
            tail = parent.split(marker, 1)[1].rstrip("-")
            if tail:
                return tail
        return parent.strip("-") or path.stem


def _project_slug(proj: str) -> str:
    return (proj or "").split("::")[-1].split("-")[0].lower()


def eval_student(
    student_dir: Path | str,
    *,
    sessions_dir: Optional[Path | str] = None,
    teacher_dir: Optional[Path | str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """扫描学生实测目录与 pi 会话日志，计算关键行为指标并与教师基准对比。"""
    cfg = cfg or load_config()
    s_root = Path(student_dir)
    if not s_root.exists() or not s_root.is_dir():
        raise FileNotFoundError(f"学生实测目录不存在: {s_root}")

    max_tool_chars = int((cfg.get("synth") or {}).get("max_tool_output_chars", 12000))
    cand_files: List[Tuple[Path, str]] = [
        (p, _derive_run_name(p, s_root)) for p in _collect_shallow_jsonl(s_root)
    ]

    sess_root: Optional[Path] = None
    if sessions_dir:
        sess_root = Path(sessions_dir)
    elif (cfg.get("paths") or {}).get("sessions_dir"):
        default_sess = resolve_path(cfg["paths"]["sessions_dir"])
        if default_sess.is_dir():
            sess_root = default_sess

    if sess_root and sess_root.is_dir():
        tag = s_root.name.lower()
        for sub in sorted(sess_root.iterdir()):
            if sub.is_dir() and tag in sub.name.lower():
                for p in sorted(sub.glob("*.jsonl")):
                    cand_files.append((p, _derive_run_name(p, s_root)))
            elif sub.is_file() and sub.suffix == ".jsonl" and tag in sub.name.lower():
                cand_files.append((sub, sub.stem))

    by_sid: Dict[str, Dict[str, Any]] = {}
    for jf, run_hint in cand_files:
        rec = _analyze_pi_session(jf, run_hint, max_tool_chars)
        if not rec:
            continue
        sid = rec["session_id"]
        prev = by_sid.get(sid)
        if prev is None or rec["line_count"] > prev["line_count"]:
            by_sid[sid] = rec

    runs = sorted(by_sid.values(), key=lambda x: (x["run_name"], x["session_id"]))

    # Teacher benchmarks
    teacher_by_proj: Dict[str, List[Dict[str, Any]]] = {}
    teacher_by_slug: Dict[str, List[Dict[str, Any]]] = {}
    if teacher_dir:
        t_root = Path(teacher_dir)
        t_files = _collect_shallow_jsonl(t_root)
        if not t_files and sess_root and sess_root.is_dir():
            t_tag = t_root.name.lower()
            for sub in sorted(sess_root.iterdir()):
                if sub.is_dir() and t_tag in sub.name.lower():
                    t_files.extend(sorted(sub.glob("*.jsonl")))
        for jf in t_files:
            t_rec = _analyze_pi_session(jf, jf.parent.name, max_tool_chars)
            if not t_rec:
                continue
            teacher_by_proj.setdefault(t_rec["project_group"], []).append(t_rec)
            teacher_by_slug.setdefault(_project_slug(t_rec["project_group"]), []).append(t_rec)

    for r in runs:
        t_cands = teacher_by_proj.get(r["project_group"]) or teacher_by_slug.get(
            _project_slug(r["project_group"])
        )
        if t_cands:
            best = max(t_cands, key=lambda x: x["assistant_steps"])
            r["teacher_benchmark"] = {
                "session_id": best["session_id"],
                "project_group": best["project_group"],
                "assistant_steps": best["assistant_steps"],
                "tool_calls": best["tool_calls"],
                "parallel_call_rate": best["parallel_call_rate"],
                "step_ratio": round(r["assistant_steps"] / max(1, best["assistant_steps"]), 2),
            }
        else:
            r["teacher_benchmark"] = None

    total_steps = sum(r["assistant_steps"] for r in runs)
    total_parallel = sum(r["parallel_steps"] for r in runs)
    for r in runs:
        r.pop("line_count", None)
        r.pop("parallel_steps", None)

    return {
        "student_dir": str(s_root),
        "sessions_dir": str(sess_root) if sess_root else None,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "summary": {
            "total_runs": len(runs),
            "total_turns": sum(r["turns"] for r in runs),
            "total_steps": total_steps,
            "total_tool_calls": sum(r["tool_calls"] for r in runs),
            "parallel_call_rate": round(total_parallel / total_steps, 4) if total_steps else 0.0,
            "empty_think_steps": sum(r["empty_think_steps"] for r in runs),
            "turn0_premature_stops": sum(1 for r in runs if r["turn0_premature_stop"]),
            "max_context_overflows": sum(1 for r in runs if r["max_context_overflow"]),
            "offline_emu_runs": sum(1 for r in runs if r["offline_emu_used"]),
        },
        "runs": runs,
    }
