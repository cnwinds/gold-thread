"""管线配置加载：config/pipeline.yaml + 代码内默认值合并。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .config_schema import (
    ConfigError,
    load_yaml_mapping,
    validate_config_mapping,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "pipeline.yaml"


def resolve_path(value: str | Path) -> Path:
    """解析配置里的路径：展开 ``~``，相对路径一律按项目根目录解析。"""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


DEFAULTS: Dict[str, Any] = {
    "paths": {
        "sessions_dir": "~/.pi/agent/sessions",
        "data_dir": "data",
        "pi_models_json": "~/.pi/agent/models.json",
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
        "use_model": False,
        "model_max_turn_chars": 800,
        "root_task_patterns": [r"\.xapk", r"\.apk", "破解"],
        "fallback_first_turn_as_root": True,
        "normalize_root_prompt": True,
        "root_normalizations": {
            "jigsawcard": "帮我破解 E:\\Download\\com.oakever.jigsawcard@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有解密文件保存在本文件夹",
            "meowdoku": "帮我破解 E:\\Download\\com.oakever.meowdoku@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有文件保存在本文件夹",
        },
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
    "judge": {
        "enabled": False,
        "model": "",
        "decide_model": "",
        "audit_model": "",
        "task_models": {},
        "seam_min_silent_chars": 80,
        "step_batch": 4,
        "seam_batch": 3,
        "dpo_batch": 2,
        "concurrency": 2,
        "session_workers": 4,
        "max_steps_per_audit": 30,
        "quality_batch": 8,
        "item_think_chars": 1100,
        "fallback_to_rules": True,
        "decisions": {
            "prune": True,
            "fold": True,
            "rewrite": True,
            "clear_text": True,
            "exclude": True,
        },
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
        # 并行工具调用（一步 >=2 个 tool_call）的权重倍数；1.0 = 关闭
        "parallel_weight": 1.5,
        # 历史步是否剥离  thinking。True=剥离（历史只有 tool_calls + 工具结果）；
        # False=保留（与 pi/vLLM 的真实推理上下文一致，样本会长 ~18%）。
        "strip_history_thinking": True,
        # strip_history_thinking=false 时，单步历史思考压缩到多少字符（0=不压缩）
        "history_thinking_max_chars": 600,
        "sft_allow_subsegments": [
            {"session_id": "01a15c80", "start_step": 50, "end_step": 106},
        ],
        "shuffle": True,
        "seed": 42,
        "eval_projects": [],
        "eval_ratio": 0.25,
        "formats": {"sharegpt": True, "openai": True},
        "dpo": {
            "from_pruned_steps": True,
            "from_negative_steps": True,
            "from_review_rewrites": True,
            "min_thinking_chars": 80,
        },
    },
    "domain": {
        "profile": "reverse_engineering",
        "system_prompt": "",
        "use_session_system_prompt": False,
        "tools": [],
        "enable_re_pathologies": True,
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


def merge_config(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    """按 ``load_config`` 的语义补齐默认值：字典深合并，列表整体替换。"""
    return _deep_merge(DEFAULTS, raw if isinstance(raw, Mapping) else {})


def parse_config_text(text: str) -> Dict[str, Any]:
    """解析并校验配置原文，返回文件中**显式写入**的原始映射。

    校验对象是 ``DEFAULTS`` 与原文深度合并后的完整配置：省略的简单字段按默认值
    处理，未知字段/类型/范围/正则/跨字段问题都会抛出 :class:`ConfigError`。
    """
    raw = load_yaml_mapping(text)
    validate_config_mapping(_deep_merge(DEFAULTS, raw))
    return raw


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    explicit = path is not None
    p = Path(path) if explicit else CONFIG_PATH
    if explicit and not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    if p.exists():
        raw = parse_config_text(p.read_text(encoding="utf-8"))
        cfg = _deep_merge(DEFAULTS, raw)
    else:
        # 不传路径时沿用历史行为：默认配置缺失就使用内建默认值。
        cfg = copy.deepcopy(DEFAULTS)
    cfg["_config_path"] = str(p)
    return cfg


def _atomic_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def save_config(cfg: Dict[str, Any], path: Path | str | None = None) -> None:
    p = Path(path) if path else CONFIG_PATH
    if not isinstance(cfg, Mapping):
        raise ConfigError("<root>", f"config must be an object, got {type(cfg).__name__}")
    # 与加载共享同一份语义：保存前先按“默认值 + 本次覆盖”验证完整配置；
    # 内部 _config_path 等来源信息不写入文件，也不进入 config_hash。
    clean = {k: v for k, v in cfg.items() if not (isinstance(k, str) and k.startswith("_"))}
    validate_config_mapping(_deep_merge(DEFAULTS, clean))
    text = yaml.safe_dump(clean, allow_unicode=True, sort_keys=False, width=120)
    _atomic_write_text(p, text)


def save_config_text(text: str, path: Path | str | None = None) -> None:
    """按原文原子写入（保留注释与排版），校验失败时不改动旧文件。"""
    p = Path(path) if path else CONFIG_PATH
    parse_config_text(text)
    _atomic_write_text(p, text)


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
    return resolve_path(cfg["paths"]["data_dir"])


def stage_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = data_root(cfg)
    return {
        "root": root,
        "state": root / "00_state",
        "raw": root / "01_raw",
        "gold": root / "02_gold",
        "train": root / "03_train",
    }