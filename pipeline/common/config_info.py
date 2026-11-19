"""只读描述有效管线配置：合并后的叶值、来源（file/default）与作用阶段。

该模块不写任何文件、不读取 ``models.json``/环境中的凭据，也不隐式调用模型。
``describe_config`` 接收已经解析并校验过的显式配置映射（见
:func:`pipeline.common.config.parse_config_text`），按与
:func:`pipeline.common.config.load_config` 相同的深合并语义补全内建默认值。

解析入口直接复用 :mod:`pipeline.common.config`，不做任何 YAML 兜底：非法配置
必须在这里失败，而不是被宽松解析成默认值。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from .config import CONFIG_PATH, merge_config, parse_config_text

# 精确阶段覆盖优先于按 section 前缀的默认映射。
_EXACT_STAGES: dict[str, str] = {
    "paths.pi_models_json": "auxiliary · 辅助模型路径；不读取凭据文件",
    "paths.cursor_sessions_dir": "shared · Cursor 采集（代码支持，非内建默认）",
    "llm.temperature": "auxiliary · 默认调用温度；judge 显式传 0.0",
    "llm.max_tokens": "auxiliary · 单次输出预算，不是输入上下文",
    "turns.use_model": (
        "ops · 离线主链（build/curate/assist）强制 False；"
        "仅显式 assist turns 或兼容 stage1 直跑读取"
    ),
    "turns.normalize_root_prompt": "stage1 · 回合清洗（项目特例应迁入策展）",
    "judge.enabled": (
        "ops/auxiliary · 离线主链强制 False；"
        "仅显式 assist judge 或兼容 stage2 直跑读取"
    ),
    "judge.model": "auxiliary · 中间兜底模型（空=继承 llm.model）",
    "judge.task_models.steps": "auxiliary · 步级模型覆盖（代码支持）",
    "judge.labels": "auxiliary · 旧 judge 来源过滤（代码支持）",
    "judge.decisions.clear_text": (
        "auxiliary · 字段被读取但 judge 不产生该动作；显式 clear_text 另走 curate/gold"
    ),
    "review.judge_confidence_min": "legacy→curation · resolve() 仍读取；旧键名仅兼容",
    "synth.max_tool_output_chars": "stage3 · 采集/解析/回放也读取该截断",
    "synth.sft_allow_subsegments": "stage3 · 历史例外子段，非通用默认",
    "synth.eval_projects": "stage3 · 仅初始草稿；已存 Split Plan 不自动换组",
    "synth.sft_labels": "legacy-only · 旧注释字段，不参与准入",
    "domain.profile": "domain · 元信息；未见实际分发消费",
    "domain.system_prompt": "stage2/stage3 · 领域 system 契约",
    "domain.use_session_system_prompt": "stage2/stage3 · 优先沿用源会话 system",
    "domain.tools": "stage2/stage3 · 静态工具 Schema（与动态 properties 分开）",
    "domain.enable_re_pathologies": "stage2/stage3 · 领域病态检查",
}

# 按顺序匹配；越具体的 section 越靠前。
_PREFIX_STAGES: tuple[tuple[str, str], ...] = (
    ("paths.", "shared · 输入工作区/数据根"),
    ("llm.", "auxiliary · 辅助模型调用"),
    ("admission.", "stage1 · 会话准入/扫描"),
    ("labeling.", "stage1 · 来源识别/血缘；negative 影响 DPO 候选"),
    ("turns.", "stage1 · 回合清洗"),
    ("gold.", "stage2 · 黄金提纯"),
    ("judge.", "auxiliary · 模型判断（仅显式入口）"),
    ("review.", "legacy-only · 旧评审兼容（当前执行链未消费）"),
    ("synth.", "stage3 · 训练合成/切分"),
    ("domain.", "domain · 领域覆盖"),
)

_UNKNOWN_STAGE = "unmapped · 未登记字段（按原值保留）"
# 工具 Schema 内部的 properties 是动态结构，不作为配置叶值展开。
_ATOMIC_PATHS: frozenset[tuple[str, ...]] = frozenset({("domain", "tools")})


def load_explicit_config(
    path: Path | str | None = None,
) -> tuple[dict[str, Any], Path, bool]:
    """读取并校验显式配置文件；显式路径不存在时抛错，默认路径缺失则回退内建默认值。

    解析/校验直接走 :func:`pipeline.common.config.parse_config_text`：非法 YAML、
    未知字段或类型错误都会抛 :class:`pipeline.common.config_schema.ConfigError`，
    不会退化成 ``yaml.safe_load`` 的宽松结果。
    """
    explicit = path is not None and str(path) != ""
    target = Path(path).expanduser() if explicit else Path(CONFIG_PATH)
    if not target.exists():
        if explicit:
            raise FileNotFoundError(f"配置文件不存在: {target}")
        return {}, target, False
    return parse_config_text(target.read_text(encoding="utf-8")), target, True


def _has_key_path(data: Mapping[str, Any], key_path: tuple[str, ...]) -> bool:
    """按原始键元组判断显式来源，动态键里的 ``.`` 不当作层级分隔符。"""
    current: Any = data
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return True


def _is_atomic_path(key_path: tuple[str, ...]) -> bool:
    for atomic in _ATOMIC_PATHS:
        if key_path[: len(atomic)] == atomic:
            return True
    return False


def _iter_public_leaves(
    value: Any, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if _is_atomic_path(prefix):
        if prefix:
            yield prefix, value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                continue
            # 顶层 ``_config_path`` 之类的来源标记不是配置叶值；嵌套键全部保留，
            # ``turns.root_normalizations`` 允许以下划线开头的动态键。
            if not prefix and key.startswith("_"):
                continue
            child = (*prefix, key)
            item = value[key]
            if _is_atomic_path(child):
                yield child, item
            elif isinstance(item, Mapping):
                # 空映射不产生叶值，与 load_config 的深合并语义一致。
                if item:
                    yield from _iter_public_leaves(item, child)
            else:
                yield child, item
        return
    if prefix:
        yield prefix, value


def _stage_for(path: str) -> str:
    exact = _EXACT_STAGES.get(path)
    if exact:
        return exact
    for prefix, stage in _PREFIX_STAGES:
        if path.startswith(prefix):
            return stage
    return _UNKNOWN_STAGE


def describe_config(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """返回有效配置叶值列表，按 path 排序，每项含 path/value/source/stage。"""
    if not isinstance(raw, Mapping):
        raise TypeError("raw 必须是配置映射")
    merged = merge_config(raw)
    entries: list[dict[str, Any]] = []
    for key_path, value in _iter_public_leaves(merged):
        # path 只用于展示与阶段匹配；来源判定始终基于原始键元组。
        path = ".".join(key_path)
        entries.append(
            {
                "path": path,
                "value": value,
                "source": "file" if _has_key_path(raw, key_path) else "default",
                "stage": _stage_for(path),
            }
        )
    entries.sort(key=lambda item: item["path"])
    return entries