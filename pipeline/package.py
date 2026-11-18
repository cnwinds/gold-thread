"""训练交付包：按 goldthread-train 模板把当前数据打成可直接 autotrain 的目录。

格式在打包时选定（默认 Qwen XML）。assistant 侧的工具调用、settings.yaml 的
template、推理用的 --tool-call-parser 一起锁定；03_train 里的原文件不动。

产出（默认在 data/04_release/delivery/）：
    goldthread-train-<package_hash>.zip
        goldthread-train/          与训练模板同布局
            autotrain.py  settings.yaml  configs/  tools/  docs/
            data/*.lf.jsonl  data/dataset_info.json  data/MANIFEST.json
            data/length_stats.json   # 分词长度分布（控制台只读，不再现场算）
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.config import config_hash, load_config, stage_dirs
from pipeline.common.format import convert_sharegpt_row, resolve_export_profile
from pipeline.common.length_stats import (
    LENGTH_STATS_REL,
    build_length_stats,
    dumps_length_stats,
)
from pipeline.common.store import now_iso, read_json
from pipeline.lf_normalize import dumps_jsonl, dataset_info_for, prepare_datasets, require_sft
from pipeline import length_gate as length_gate_mod

_TEMPLATE_ROOT = Path(__file__).resolve().parent / "train_template"
_PACK_ROOT = "goldthread-train"
_ZIP_PREFIX = "goldthread-train-"


# 记在 delivery/ 旁边，不进 zip。stage3 再次打包时沿用上次在界面上选的格式。
_PROFILE_CHOICE = "export_profile.json"


def _remembered_profile(delivery_dir: Path) -> Optional[str]:
    path = delivery_dir / _PROFILE_CHOICE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("profile") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return resolve_export_profile(name)["id"]
    except ValueError:
        return None


def _apply_tool_style(files: Dict[str, List[Dict[str, Any]]], style: str) -> Dict[str, int]:
    """就地改写 assistant 侧工具调用。无法识别的块直接让这次打包失败。"""
    converted = unparsed = 0
    for rows in files.values():
        for row in rows:
            c, u = convert_sharegpt_row(row, style)
            converted += c
            unparsed += u
    if unparsed:
        raise ValueError(
            f"有 {unparsed} 处 assistant 工具调用无法识别，已停止打包。"
            "只接受 hermes JSON 或 Qwen XML 的 <tool_call> 块。"
        )
    return {"blocks_converted": converted, "blocks_unparsed": unparsed}


def _template_static_files() -> Dict[str, bytes]:
    root = _TEMPLATE_ROOT
    if not (root / "autotrain.py").is_file():
        raise FileNotFoundError(f"训练打包模板缺失: {root}")
    skip = {"README.md", "settings.yaml"}
    found: Dict[str, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in skip or rel.startswith("data/"):
            continue
        found[rel] = path.read_bytes()
    if "autotrain.py" not in found or "tools/check_data.py" not in found:
        raise FileNotFoundError("训练打包模板不完整")
    return found


def _patch_settings(text: str, template: str, has_dpo: bool) -> str:
    patched, replaced = re.subn(
        r"(?m)^(  template:\s*)\S+", rf"\g<1>{template}", text, count=1,
    )
    if replaced != 1:
        raise RuntimeError("训练模板 settings.yaml 缺少 project.template")
    if not has_dpo:
        patched = re.sub(r"(?m)^    dpo:.*\n", "", patched, count=1)
    return patched if patched.endswith("\n") else patched + "\n"


def _patch_readme(text: str, counts: Dict[str, int], spec: Dict[str, Any], has_dpo: bool) -> str:
    rows = [
        f"| `gt_sft_train_lf` | `sft.train.lf.jsonl` | {counts.get('sft.train.lf.jsonl', 0)} |",
        f"| `gt_sft_eval_lf` | `sft.eval.lf.jsonl` | {counts.get('sft.eval.lf.jsonl', 0)} |",
    ]
    if has_dpo:
        rows.append(f"| `gt_dpo_train_lf` | `dpo.train.lf.jsonl` | {counts.get('dpo.train.lf.jsonl', 0)} 对 |")
        rows.append(f"| `gt_dpo_eval_lf` | `dpo.eval.lf.jsonl` | {counts.get('dpo.eval.lf.jsonl', 0)} 对 |")
    table = "\n".join(rows) + "\n"
    patched, n = re.subn(
        r"\| `gt_sft_train_lf` \|.*\n\| `gt_sft_eval_lf` \|.*\n"
        r"(?:\| `gt_dpo_train_lf` \|.*\n)?(?:\| `gt_dpo_eval_lf` \|.*\n)?",
        table,
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("训练模板 README 的数据表无法更新")
    xml = (
        "数据是 Qwen XML（`<function=...><parameter=...>`），训练模板是 `qwen3_8`"
        "（`settings.yaml` 的 `project.template`），vLLM 推理用 `--tool-call-parser qwen3_coder`"
    )
    if spec["style"] == "hermes-json":
        repl = (
            "数据是 Hermes JSON（`<tool_call>` 里的 JSON），训练模板是 `qwen3`"
            "（`settings.yaml` 的 `project.template`），vLLM 推理用 `--tool-call-parser hermes`"
        )
    else:
        repl = xml
    if xml not in patched:
        raise RuntimeError("训练模板 README 缺少工具调用格式约定")
    return patched.replace(xml, repl, 1)


def _data_manifest(
    info: Dict[str, Any],
    files: Dict[str, str],
    counts: Dict[str, int],
    spec: Dict[str, Any],
    format_contract: Dict[str, Any],
    pkg_h: str,
    cfg_h: str,
    build_id: str,
    generated_at: str,
    lg_fp: str,
    prepared_sources: Dict[str, str],
    prepared_dropped: Dict[str, Any],
    length_gate: Any,
) -> Dict[str, Any]:
    datasets = {}
    for key, entry in info.items():
        name = str(entry.get("file_name") or "")
        body = files.get(name, "")
        datasets[key] = {
            "file": name,
            "rows": int(counts.get(name) or 0),
            "ranking": bool(entry.get("ranking")),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
    return {
        "_doc": "原始数据清单。data/ 只读：autotrain 每次按 cutoff 过滤到工作目录，不改这里。",
        "source_package": f"{_ZIP_PREFIX}{pkg_h}",
        "format": {
            "style": "ShareGPT 单步窗口（system/human/gpt/observation），只有最后一轮 gpt 有 <think>",
            "tool_call": spec["style"],
            "train_template": spec["template"],
            "serve_tool_parser": spec["parser"],
        },
        "split": "按项目整包留出",
        "datasets": datasets,
        "package_hash": pkg_h,
        "build_id": build_id or None,
        "config_hash": cfg_h,
        "generated_at": generated_at,
        "counts": counts,
        "format_contract": format_contract,
        "length_gate_version": lg_fp,
        "length_gate": length_gate,
        "sources": prepared_sources,
        "dropped": prepared_dropped,
    }


def compute_package_hash(
    cfg_h: str,
    prepared_files: Dict[str, List[Dict[str, Any]]],
    length_gate_fp: str,
    profile: str = "",
    extras: Optional[Dict[str, str | bytes]] = None,
) -> str:
    """8-hex fingerprint of delivery payload (post-gate lf bytes + config + gate version).

    Soft-compress / data rebuilds that change packed jsonl bytes, or a
    LENGTH_GATE_VERSION bump, produce a new zip filename even when
    pipeline.yaml config_hash is unchanged. `extras` covers the packed YAML /
    scripts, so editing a config or script also yields a new filename instead of
    silently reusing it under the old name.
    """
    parts: List[str] = [cfg_h]
    if profile:
        parts.append(f"profile:{profile}")
    for name, rows in sorted(prepared_files.items()):
        content = dumps_jsonl(rows)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        parts.append(f"{name}:{digest}")
    for name, content in sorted((extras or {}).items()):
        raw = content if isinstance(content, bytes) else content.encode("utf-8")
        parts.append(f"{name}:{hashlib.sha256(raw).hexdigest()}")
    parts.append(length_gate_fp)
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def build_delivery_zip(
    train_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log=None,
    profile: Optional[str] = None,
    delivery_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """按 goldthread-train 模板打包当前数据集，返回 zip 路径与元信息。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    root = Path(train_root or dirs["train"])
    build_manifest = read_json(root / "BUILD.json", default={}) or {}
    build_id = str(build_manifest.get("build_id") or "")
    # 发布产物统一落在 data/04_release/，与 03_train 工作区分离
    release_root = root.parent / "04_release"
    if delivery_dir:
        delivery_dir = Path(delivery_dir)
    elif build_id:
        delivery_dir = stage_dirs(cfg)["root"] / "04_release" / "delivery"
    else:
        delivery_dir = release_root / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_export_profile(profile or _remembered_profile(delivery_dir))

    cfg_h = config_hash(cfg)

    def _emit(msg: str) -> None:
        if log:
            log(msg)

    _emit("[package] 正在准备数据集并做长度闸门（软压缩）…")
    prepared = prepare_datasets(root, log=log)
    require_sft(prepared)
    info = dataset_info_for(prepared)
    available = set(info)
    has_dpo = "gt_dpo_train_lf" in available and "gt_dpo_eval_lf" in available

    _emit(
        f"[package] 导出格式 {spec['label']}："
        f"{spec['style']} / template {spec['template']} / parser {spec['parser']}"
    )
    style_stats = _apply_tool_style(prepared.files, spec["style"])
    _emit(f"[package] 已按该格式改写 {style_stats['blocks_converted']} 处工具调用")

    static_files = _template_static_files()
    settings_text = _patch_settings(
        (_TEMPLATE_ROOT / "settings.yaml").read_text(encoding="utf-8"),
        spec["template"],
        has_dpo,
    )
    readme_text = _patch_readme(
        (_TEMPLATE_ROOT / "README.md").read_text(encoding="utf-8"),
        prepared.counts,
        spec,
        has_dpo,
    )
    file_texts = {name: dumps_jsonl(rows) for name, rows in prepared.files.items()}
    info_json = json.dumps(info, ensure_ascii=False, indent=2) + "\n"

    lg_fp = length_gate_mod.length_gate_fingerprint()
    pkg_h = compute_package_hash(
        cfg_h,
        prepared.files,
        lg_fp,
        spec["id"],
        extras={
            **static_files,
            "settings.yaml": settings_text,
            "README.md": readme_text,
            "data/dataset_info.json": info_json,
            **({"build_id": build_id} if build_id else {}),
        },
    )
    zip_name = f"{_ZIP_PREFIX}{pkg_h}.zip"
    zip_path = delivery_dir / zip_name
    generated_at = now_iso()
    format_contract = {
        "profile": spec["id"],
        "label": spec["label"],
        "habit": spec["habit"],
        "tool_style": spec["style"],
        "train_template": spec["template"],
        "serve_tool_parser": spec["parser"],
        "blocks_converted": style_stats["blocks_converted"],
        "blocks_unparsed": style_stats["blocks_unparsed"],
        "pack_8b_yaml": False,
        "note": "数据 tool_style、settings.yaml template、推理 --tool-call-parser 是一套。",
    }
    manifest = _data_manifest(
        info,
        file_texts,
        prepared.counts,
        spec,
        format_contract,
        pkg_h,
        cfg_h,
        build_id,
        generated_at,
        lg_fp,
        prepared.sources,
        prepared.dropped,
        prepared.length_gate,
    )
    manifest["package"] = zip_name
    manifest["layout"] = _PACK_ROOT
    if prepared.weighted_expansion:
        manifest["weighted_expansion"] = prepared.weighted_expansion
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _emit("[package] 正在统计 token 长度分布…")
    length_stats = build_length_stats(
        prepared.files,
        length_gate=prepared.length_gate if isinstance(prepared.length_gate, dict) else {},
        source="pack",
    )
    length_stats_json = dumps_length_stats(length_stats)
    manifest_files: List[Dict[str, Any]] = []

    def _add(zf: zipfile.ZipFile, rel: str, content: str | bytes) -> None:
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        zf.writestr(f"{_PACK_ROOT}/{rel}", data)
        manifest_files.append({"path": f"{_PACK_ROOT}/{rel}", "size": len(data)})

    _emit(f"[package] 正在写入 zip：{zip_name}…")
    partial = delivery_dir / f".{zip_name}.partial"
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for rel, blob in sorted(static_files.items()):
                _add(zf, rel, blob)
            _add(zf, "settings.yaml", settings_text)
            _add(zf, "README.md", readme_text)
            for name, text in sorted(file_texts.items()):
                _add(zf, f"data/{name}", text)
            _add(zf, "data/dataset_info.json", info_json)
            _add(zf, "data/MANIFEST.json", manifest_json)
            _add(zf, LENGTH_STATS_REL, length_stats_json)
        partial.replace(zip_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    (delivery_dir / _PROFILE_CHOICE).write_text(
        json.dumps(
            {
                "profile": spec["id"],
                "label": spec["label"],
                "updated_at": manifest["generated_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest_files),
        "generated_at": manifest["generated_at"],
        "config_hash": cfg_h,
        "package_hash": pkg_h,
        "build_id": build_id or None,
        "counts": prepared.counts,
        "sources": prepared.sources,
        "datasets": manifest["datasets"],
        "length_gate": prepared.length_gate,
        "format": format_contract,
    }
    if log:
        size_mb = result["size_bytes"] / (1024 * 1024)
        sft_n = prepared.counts.get("sft.train.lf.jsonl", 0)
        dpo_n = prepared.counts.get("dpo.train.lf.jsonl", 0)
        gate = prepared.length_gate or {}
        salv = sum(int(v.get("salvaged_by_obs_truncate") or 0) for v in gate.values() if isinstance(v, dict))
        over = sum(int(v.get("overlength_at_default_cutoff") or 0) for v in gate.values() if isinstance(v, dict))
        log(
            f"[package] 交付包已生成: {zip_path.name} ({size_mb:.1f} MB, "
            f"{spec['label']}, SFT lf {sft_n} / DPO lf {dpo_n}, {result['file_count']} 个文件, "
            f"length_gate salvaged={salv} overlength_at_default={over})"
        )
    return result


def build_delivery_zip_from_build(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    build_id: Optional[str] = None,
    profile: Optional[str] = None,
    log=None,
    delivery_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """从不可变 Build 快照打包交付 zip，严格使用 Build 冻结的配置快照。"""
    from pipeline import dataset_build  # noqa: PLC0415

    cfg = cfg or load_config()
    build_dir, manifest = dataset_build.resolve_build(cfg, build_id)
    build_cfg = dataset_build.load_build_config(build_dir, cfg)
    out_dir = (
        Path(delivery_dir)
        if delivery_dir
        else (stage_dirs(cfg)["root"] / "04_release" / "delivery")
    )
    result = build_delivery_zip(
        build_dir,
        build_cfg,
        log=log,
        profile=profile,
        delivery_dir=out_dir,
    )
    result["build_id"] = manifest["build_id"]
    return result
