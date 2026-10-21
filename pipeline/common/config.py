"""管线配置加载：config/pipeline.yaml + 代码内默认值合并。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "pipeline.yaml"

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "sessions_dir": r"C:\Users\lvan\.pi\agent\sessions",
        "data_dir": str(ROOT / "data"),
        "pi_models_json": r"C:\Users\lvan\.pi\agent\models.json",
    },
    "llm": {
        "model": "deepseek-v4.1-flash-1017",
        "temperature": 0.2,
        "max_tokens": 8192,
        "timeout_seconds": 180,
        "cache": True,
    },
    "admission": {
        "workspace_prefixes": ["--d--crack-"],
        "exclude_folders": ["--d--crack--"],
        "dedupe_duplicate_prefix": True,
    },
    "labeling": {
        "positive_model_patterns": ["^deepseek", "^glm-"],
        "negative_model_patterns": ["qwen"],
        "dominant_ratio_min": 0.6,
        "unknown_label": "pending",
    },
    "turns": {
        "use_model": True,
        "model_max_turn_chars": 800,
        "root_task_patterns": [r"\.xapk", r"\.apk", "破解"],
        "fallback_first_turn_as_root": True,
        "irrelevant_patterns": [],
        "irrelevant_whitelist_patterns": [],
        "continuation_patterns": [r"^继续$", r"^可以$"],
        "correction_patterns": ["不要参考其他的目录"],
        "fold_await_user_messages": True,
    },
    "gold": {
        "state_folding": True,
        "max_fold_edits": 8,
        "detour_pruning": True,
        "fix_missing_thinking": True,
        "silent_intermediate_steps": True,
        "max_thinking_chars": 2400,
        "target_thinking_chars": 1600,
        "min_gold_steps": 10,
    },
    "review": {
        "enabled": False,
        "labels": ["positive", "pending"],
        "compress_thinking_over": 1200,
        "max_steps_per_session": 40,
        "judge_max_think_chars": 3000,
        "temperature": 0.0,
        "rewrite_batch_size": 4,
        "max_rewrite_batches": 6,
        "judge_gold": True,
        "judge_min_score": 60,
        "judge_confidence_min": 0.7,
    },
    "synth": {
        "window_turns": 8,
        "max_tool_output_chars": 12000,
        "weights": {"opening": 3, "key_decision": 2, "closing": 3, "routine": 1},
        "sft_labels": ["positive"],
        "sft_allow_subsegments": [],
        "shuffle": True,
        "seed": 42,
        "eval_projects": [],
        "eval_ratio": 0.25,
        "formats": {"sharegpt": True, "openai": True},
        "dpo": {
            "from_pruned_steps": True,
            "from_negative_steps": True,
            "min_thinking_chars": 80,
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = _deep_merge(DEFAULTS, raw)
    else:
        cfg = copy.deepcopy(DEFAULTS)
    cfg["_config_path"] = str(p)
    return cfg


def save_config(cfg: Dict[str, Any], path: Path | str | None = None) -> None:
    p = Path(path) if path else CONFIG_PATH
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(clean, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def save_config_text(text: str, path: Path | str | None = None) -> None:
    """按原文写入（保留注释与排版），仅用于控制台编辑。"""
    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def config_hash(cfg: Dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def section_hash(cfg: Dict[str, Any], keys: list[str]) -> str:
    """只对影响某阶段的配置段做哈希，避免改 synth 参数就重建 stage1/2。"""
    picked: Dict[str, Any] = {}
    for k in keys:
        cur: Any = cfg
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        picked[k] = cur
    blob = json.dumps(picked, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def data_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["data_dir"]).expanduser()


def stage_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = data_root(cfg)
    return {
        "root": root,
        "state": root / "00_state",
        "raw": root / "01_raw",
        "gold": root / "02_gold",
        "train": root / "03_train",
    }