"""管线配置的共享 YAML 解析与语义校验。

`config.py` 的加载、原文保存、结构化保存与 ``parse_config_text`` 都走这里，
保证同一份输入在 CLI / Web / 测试里得到同一组错误。这里只做只读校验，不修改
传入的配置，也不把任何运行时来源信息写回配置。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import yaml


class ConfigError(ValueError):
    """配置错误：``path`` 可直接定位到出错字段。"""

    def __init__(self, path: str, message: str) -> None:
        self.path = str(path or "")
        self.field = self.path
        text = f"{self.path}: {message}" if self.path else message
        super().__init__(text)


class _Spec:
    pass


@dataclass(frozen=True)
class Bool(_Spec):
    pass


@dataclass(frozen=True)
class Str(_Spec):
    allow_empty: bool = False


@dataclass(frozen=True)
class Int(_Spec):
    minimum: Optional[int] = None
    maximum: Optional[int] = None


@dataclass(frozen=True)
class Number(_Spec):
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass(frozen=True)
class RegexList(_Spec):
    pass


@dataclass(frozen=True)
class ListOf(_Spec):
    item: Any


@dataclass(frozen=True)
class MapOf(_Spec):
    fields: Dict[str, Any]
    # 必填字段：缺省时下游会静默套用兜底值（例如子段放宽成整会话），必须显式报错。
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class DynamicMap(_Spec):
    value: Any


@dataclass(frozen=True)
class OneOf(_Spec):
    """固定枚举字符串：下游把它直接当目录名/标签用时，不能任由用户自造值。"""

    values: tuple[str, ...]


@dataclass(frozen=True)
class Nullable(_Spec):
    inner: Any


@dataclass(frozen=True)
class AnyMap(_Spec):
    """只要求是对象；用于已知的兼容顶层键（如 export_profiles 的条目）。"""


@dataclass(frozen=True)
class Tools(_Spec):
    """domain.tools：列表内允许灵活嵌套的 OpenAI/工具 schema，但根必须是列表。"""


def _join(path: str, key: Any) -> str:
    if not path or path == "<root>":
        return str(key)
    return f"{path}.{key}"


def _typename(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _fail(path: str, message: str) -> None:
    raise ConfigError(path, message)


def _validate(value: Any, spec: Any, path: str) -> None:
    if isinstance(spec, Bool):
        if type(value) is not bool:
            _fail(path, f"expected boolean, got {_typename(value)}")
        return

    if isinstance(spec, Str):
        if not isinstance(value, str):
            _fail(path, f"expected string, got {_typename(value)}")
        if not spec.allow_empty and not value.strip():
            _fail(path, "must not be empty")
        return

    if isinstance(spec, Int):
        if type(value) is not int:
            _fail(path, f"expected integer, got {_typename(value)}")
        if spec.minimum is not None and value < spec.minimum:
            _fail(path, f"must be >= {spec.minimum}, got {value}")
        if spec.maximum is not None and value > spec.maximum:
            _fail(path, f"must be <= {spec.maximum}, got {value}")
        return

    if isinstance(spec, Number):
        if type(value) not in (int, float):
            _fail(path, f"expected number, got {_typename(value)}")
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            _fail(path, "must be a finite number")
        if spec.minimum is not None and value < spec.minimum:
            _fail(path, f"must be >= {spec.minimum}, got {value}")
        if spec.maximum is not None and value > spec.maximum:
            _fail(path, f"must be <= {spec.maximum}, got {value}")
        return

    if isinstance(spec, RegexList):
        if not isinstance(value, list):
            _fail(path, f"expected list of regex strings, got {_typename(value)}")
        for idx, pattern in enumerate(value):
            item_path = f"{path}[{idx}]"
            if not isinstance(pattern, str):
                _fail(item_path, f"expected regex string, got {_typename(pattern)}")
            try:
                re.compile(pattern)
            except re.error as exc:
                _fail(item_path, f"invalid regular expression: {exc}")
        return

    if isinstance(spec, ListOf):
        if not isinstance(value, list):
            _fail(path, f"expected list, got {_typename(value)}")
        for idx, item in enumerate(value):
            _validate(item, spec.item, f"{path}[{idx}]")
        return

    if isinstance(spec, MapOf):
        if not isinstance(value, Mapping):
            _fail(path, f"expected object, got {_typename(value)}")
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(_join(path, key), f"field name must be a string, got {_typename(key)}")
            child_path = _join(path, key)
            if key not in spec.fields:
                _fail(child_path, "unknown field")
            _validate(item, spec.fields[key], child_path)
        missing = [name for name in spec.required if name not in value]
        if missing:
            _fail(path, f"missing required field(s): {', '.join(missing)}")
        return

    if isinstance(spec, OneOf):
        if not isinstance(value, str):
            _fail(path, f"expected string, got {_typename(value)}")
        if value not in spec.values:
            _fail(path, f"must be one of {', '.join(spec.values)}, got {value!r}")
        return

    if isinstance(spec, DynamicMap):
        if not isinstance(value, Mapping):
            _fail(path, f"expected object, got {_typename(value)}")
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                _fail(path, "dynamic keys must be non-empty strings")
            _validate(item, spec.value, _join(path, key))
        return

    if isinstance(spec, Nullable):
        if value is None:
            return
        _validate(value, spec.inner, path)
        return

    if isinstance(spec, AnyMap):
        if not isinstance(value, Mapping):
            _fail(path, f"expected object, got {_typename(value)}")
        return

    if isinstance(spec, Tools):
        _validate_tools(value, path)
        return

    raise AssertionError(f"unknown schema spec: {spec!r}")


def _validate_tools(value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(path, f"expected list of tool schemas, got {_typename(value)}")
    for idx, tool in enumerate(value):
        item_path = f"{path}[{idx}]"
        if not isinstance(tool, Mapping):
            _fail(item_path, f"expected tool object, got {_typename(tool)}")
        if not tool:
            _fail(item_path, "tool object must not be empty")
        _validate_tool_fields(tool, item_path)


def _validate_tool_fields(tool: Mapping, path: str) -> None:
    """校验常见的工具 schema 外壳，同时保留 parameters/properties 的嵌套自由度。"""
    if "type" in tool and (not isinstance(tool["type"], str) or not tool["type"].strip()):
        _fail(_join(path, "type"), "must be a non-empty string")
    if "name" in tool and (not isinstance(tool["name"], str) or not tool["name"].strip()):
        _fail(_join(path, "name"), "must be a non-empty string")
    if "description" in tool and not isinstance(tool["description"], str):
        _fail(_join(path, "description"), f"expected string, got {_typename(tool['description'])}")

    fn = tool.get("function")
    if fn is not None:
        fn_path = _join(path, "function")
        if not isinstance(fn, Mapping):
            _fail(fn_path, f"expected object, got {_typename(fn)}")
        if "name" in fn and (not isinstance(fn["name"], str) or not fn["name"].strip()):
            _fail(_join(fn_path, "name"), "must be a non-empty string")
        if "description" in fn and not isinstance(fn["description"], str):
            _fail(_join(fn_path, "description"), f"expected string, got {_typename(fn['description'])}")
        if "strict" in fn and type(fn["strict"]) is not bool:
            _fail(_join(fn_path, "strict"), f"expected boolean, got {_typename(fn['strict'])}")
        if "parameters" in fn:
            _validate_json_schema_object(fn["parameters"], _join(fn_path, "parameters"))

    if "parameters" in tool:
        _validate_json_schema_object(tool["parameters"], _join(path, "parameters"))


def ensure_json_tree(value: Any, path: str = "<root>") -> None:
    """确认整棵配置是有限、可互操作的 JSON 树。

    保存后的 UI 会计算哈希并序列化配置，所以即使处在 ``domain.tools`` /
    ``export_profiles`` 这类宽松结构里，也不能出现 YAML-only 标量（date/set/binary…）、
    非字符串键、NaN/Inf 或循环引用；否则失败会推迟到 ``json.dumps`` 才以 TypeError 暴露。
    """
    _check_json_tree(value, path, set())


def _check_json_tree(value: Any, path: str, active: set[int]) -> None:
    if value is None or type(value) is bool or type(value) is int or isinstance(value, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "must be a finite number")
        return
    if isinstance(value, (list, tuple)):
        token = id(value)
        if token in active:
            _fail(path, "recursive structure is not valid JSON")
        active.add(token)
        try:
            for idx, item in enumerate(value):
                _check_json_tree(item, f"{path}[{idx}]", active)
        finally:
            active.discard(token)
        return
    if isinstance(value, Mapping):
        token = id(value)
        if token in active:
            _fail(path, "recursive structure is not valid JSON")
        active.add(token)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    _fail(path, f"object key must be a string, got {_typename(key)}")
                _check_json_tree(item, _join(path, key), active)
        finally:
            active.discard(token)
        return
    _fail(path, f"{_typename(value)} is not a JSON value")


def _validate_json_schema_object(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {_typename(value)}")
    # JSON Schema 的 properties/items 允许任意嵌套形状；只保证容器类型正确，
    # 避免把合法的自定义工具 schema 误判为未知字段。
    if "properties" in value and not isinstance(value["properties"], Mapping):
        _fail(_join(path, "properties"), f"expected object, got {_typename(value['properties'])}")
    if "items" in value and not isinstance(value["items"], (Mapping, list)):
        _fail(_join(path, "items"), f"expected object or list, got {_typename(value['items'])}")
    if "required" in value:
        req = value["required"]
        if not isinstance(req, list) or any(not isinstance(item, str) for item in req):
            _fail(_join(path, "required"), "expected list of strings")


_BOOL = Bool()
_STR = Str()
_EMPTY_OK_STR = Str(allow_empty=True)


def _int(minimum: Optional[int] = None, maximum: Optional[int] = None) -> Int:
    return Int(minimum=minimum, maximum=maximum)


def _number(minimum: Optional[float] = None, maximum: Optional[float] = None) -> Number:
    return Number(minimum=minimum, maximum=maximum)


def _list(item: Any) -> ListOf:
    return ListOf(item)


def _map(**fields: Any) -> MapOf:
    return MapOf(fields)


def _required_map(required: tuple[str, ...], **fields: Any) -> MapOf:
    return MapOf(fields=fields, required=required)


def _one_of(*values: str) -> OneOf:
    return OneOf(values=tuple(values))


SCHEMA = _map(
    paths=_map(
        sessions_dir=_STR,
        data_dir=_STR,
        pi_models_json=_STR,
        cursor_sessions_dir=_EMPTY_OK_STR,
    ),
    llm=_map(
        model=_STR,
        temperature=_number(0),
        max_tokens=_int(1),
        timeout_seconds=_number(1),
        cache=_BOOL,
    ),
    admission=_map(
        workspace_prefixes=_list(_STR),
        exclude_folders=_list(_STR),
        dedupe_duplicate_prefix=_BOOL,
    ),
    labeling=_map(
        positive_model_patterns=RegexList(),
        negative_model_patterns=RegexList(),
        dominant_ratio_min=_number(0, 1),
        # Stage1 直接把该值当输出目录名使用，只允许仓库已支持的三种标签。
        unknown_label=_one_of("positive", "negative", "pending"),
    ),
    turns=_map(
        use_model=_BOOL,
        model_max_turn_chars=_int(1),
        root_task_patterns=RegexList(),
        fallback_first_turn_as_root=_BOOL,
        normalize_root_prompt=_BOOL,
        root_normalizations=DynamicMap(_STR),
        irrelevant_patterns=RegexList(),
        irrelevant_whitelist_patterns=RegexList(),
        continuation_patterns=RegexList(),
        correction_patterns=RegexList(),
        fold_await_user_messages=_BOOL,
    ),
    gold=_map(
        state_folding=_BOOL,
        max_fold_edits=_int(1),
        detour_pruning=_BOOL,
        fix_missing_thinking=_BOOL,
        silent_intermediate_steps=_BOOL,
        max_thinking_chars=_int(1),
        target_thinking_chars=_int(1),
        min_gold_steps=_int(1),
    ),
    judge=_map(
        enabled=_BOOL,
        model=_EMPTY_OK_STR,
        decide_model=_EMPTY_OK_STR,
        audit_model=_EMPTY_OK_STR,
        task_models=_map(
            steps=_EMPTY_OK_STR,
            seams=_EMPTY_OK_STR,
            quality=_EMPTY_OK_STR,
            dpo=_EMPTY_OK_STR,
        ),
        labels=Nullable(_list(_STR)),
        seam_min_silent_chars=_int(0),
        step_batch=_int(1),
        seam_batch=_int(1),
        dpo_batch=_int(1),
        concurrency=_int(1),
        session_workers=_int(1),
        max_steps_per_audit=_int(1),
        quality_batch=_int(1),
        item_think_chars=_int(1),
        fallback_to_rules=_BOOL,
        decisions=_map(
            prune=_BOOL,
            fold=_BOOL,
            rewrite=_BOOL,
            clear_text=_BOOL,
            exclude=_BOOL,
        ),
    ),
    review=_map(
        enabled=_BOOL,
        labels=_list(_STR),
        compress_thinking_over=_int(1),
        max_steps_per_session=_int(1),
        judge_max_think_chars=_int(1),
        temperature=_number(0),
        rewrite_batch_size=_int(1),
        max_rewrite_batches=_int(1),
        judge_gold=_BOOL,
        judge_min_score=_number(0, 100),
        judge_confidence_min=_number(0, 1),
    ),
    synth=_map(
        window_turns=_int(1),
        max_tool_output_chars=_int(1),
        weights=_map(
            opening=_int(1),
            key_decision=_int(1),
            closing=_int(1),
            routine=_int(1),
        ),
        parallel_weight=_number(0),
        strip_history_thinking=_BOOL,
        history_thinking_max_chars=_int(0),
        sft_allow_subsegments=_list(
            # 三个字段缺一不可：缺 session_id 会静默不生效，缺 start/end 会把子段
            # 悄悄放宽到整个会话（stage3 的兜底范围）。
            _required_map(
                ("session_id", "start_step", "end_step"),
                session_id=_STR,
                start_step=_int(1),
                end_step=_int(1),
            )
        ),
        # 旧配置兼容字段：当前执行链不再读取，但历史快照仍应能加载。
        sft_labels=_list(_STR),
        shuffle=_BOOL,
        seed=_int(),
        eval_projects=_list(_STR),
        eval_ratio=_number(0, 1),
        formats=_map(
            sharegpt=_BOOL,
            openai=_BOOL,
        ),
        dpo=_map(
            from_pruned_steps=_BOOL,
            from_negative_steps=_BOOL,
            from_review_rewrites=_BOOL,
            min_thinking_chars=_int(0),
        ),
    ),
    domain=_map(
        profile=_EMPTY_OK_STR,
        system_prompt=_EMPTY_OK_STR,
        use_session_system_prompt=_BOOL,
        tools=Tools(),
        enable_re_pathologies=_BOOL,
    ),
    # 兼容测试/历史工作区里用 safe_dump(DEFAULTS + 覆盖) 生成的配置。
    # 该键不是现行导出逻辑的输入，不扩展成任意未知顶层字段的入口。
    export_profiles=_list(AnyMap()),
    _config_path=_EMPTY_OK_STR,
)


def _strip_blank_yaml_lines(text: str) -> str:
    without_comments = re.sub(r"(?m)#.*$", "", text)
    without_markers = re.sub(r"(?m)^\s*(?:---|\.\.\.)\s*$", "", without_comments)
    return without_markers


def _is_blank_yaml(text: str) -> bool:
    return not _strip_blank_yaml_lines(text.lstrip("\ufeff")).strip()


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML 默认静默覆盖重复键；配置错误必须能定位，因此直接拒绝。"""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._config_path_stack: list[str] = []

    def _config_key_path(self, key: Any) -> str:
        return ".".join([*self._config_path_stack, str(key)]) or str(key)

    def construct_yaml_map(self, node: yaml.Node) -> Dict[Any, Any]:
        # 默认实现是生成器，子映射在父循环结束后才构造，导致重复键只能看到局部路径。
        # 配置树不需要递归 alias，直接同步构造即可保留完整路径。
        return self.construct_mapping(node)

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> Dict[Any, Any]:  # type: ignore[override]
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        self.flatten_mapping(node)
        mapping: Dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                _fail("<yaml>", f"unhashable mapping key {key!r}")
            key_path = self._config_key_path(key)
            if key in mapping:
                _fail(key_path, f"duplicate YAML key {key!r}")
            self._config_path_stack.append(str(key))
            try:
                mapping[key] = self.construct_object(value_node, deep=deep)
            finally:
                self._config_path_stack.pop()
        return mapping


# SafeConstructor 在类创建时把生成器版 construct_yaml_map 固定进 yaml_constructors，
# 子类 override 不会自动生效，需要显式注册。
_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:map", _UniqueKeyLoader.construct_yaml_map)


def load_yaml_mapping(text: str) -> Dict[str, Any]:
    """解析配置原文；空白/纯注释返回 ``{}``，重复键/语法错误/根形状错误明确报错。"""
    if not isinstance(text, str):
        _fail("<root>", f"config text must be a string, got {_typename(text)}")
    if _is_blank_yaml(text):
        return {}
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        _fail("<yaml>", f"invalid YAML: {exc}")
    if data is None:
        _fail("<root>", "config root must be an object, got null")
    if not isinstance(data, Mapping):
        _fail("<root>", f"config root must be an object, got {_typename(data)}")
    return dict(data)


def validate_config_mapping(cfg: Any) -> None:
    """校验一份完整/已合并的配置对象（只读）。"""
    _validate(cfg, SCHEMA, "<root>")
    ensure_json_tree(cfg)

    gold = cfg.get("gold") if isinstance(cfg, Mapping) else None
    if isinstance(gold, Mapping):
        target = gold.get("target_thinking_chars")
        maximum = gold.get("max_thinking_chars")
        if type(target) is int and type(maximum) is int and target > maximum:
            _fail(
                "gold.target_thinking_chars",
                f"must be <= gold.max_thinking_chars ({maximum}), got {target}",
            )

    synth = cfg.get("synth") if isinstance(cfg, Mapping) else None
    if isinstance(synth, Mapping):
        for idx, rule in enumerate(synth.get("sft_allow_subsegments") or []):
            if not isinstance(rule, Mapping):
                continue
            start = rule.get("start_step")
            end = rule.get("end_step")
            if type(start) is int and type(end) is int and start > end:
                _fail(
                    f"synth.sft_allow_subsegments[{idx}].end_step",
                    f"must be >= start_step ({start}), got {end}",
                )

    # 兜底：任何序列化/编译层面的失败都必须在写入前变成 ConfigError/ValueError。
    try:
        json.dumps(cfg, sort_keys=True)
    except (TypeError, ValueError) as exc:
        _fail("<root>", f"config must be JSON-serializable: {exc}")