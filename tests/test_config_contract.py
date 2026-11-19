"""配置契约：共享解析/校验语义、兼容键与原子保存。"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from pipeline.common import config as config_mod
from pipeline.common.config import (
    CONFIG_PATH,
    DEFAULTS,
    ConfigError,
    config_hash,
    load_config,
    parse_config_text,
    save_config,
    save_config_text,
)
from pipeline.common.config_schema import validate_config_mapping


def _assert_error(text: str, *needles: str) -> ConfigError:
    with pytest.raises(ConfigError) as exc:
        parse_config_text(text)
    message = str(exc.value)
    for needle in needles:
        assert needle in message, message
    return exc.value


def test_blank_or_comment_only_text_is_empty_defaults() -> None:
    assert parse_config_text("") == {}
    assert parse_config_text("   \n\n") == {}
    assert parse_config_text("# only a comment\n---\n...\n# tail\n") == {}


def test_partial_override_is_validated_against_defaults_and_returns_explicit_raw() -> None:
    raw = parse_config_text(
        """
synth:
  window_turns: 3
gold:
  target_thinking_chars: 100
"""
    )
    assert raw == {"synth": {"window_turns": 3}, "gold": {"target_thinking_chars": 100}}
    # 省略的 max_thinking_chars 用默认值参与跨字段校验。
    assert DEFAULTS["gold"]["max_thinking_chars"] > 100


def test_current_repository_config_still_loads_without_rewriting_values() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    raw = parse_config_text(text)
    assert raw == (yaml.safe_load(text) or {})
    cfg = load_config()
    # 校验层只读：文件里显式写的值原样进入合并结果。
    assert cfg["_config_path"] == str(CONFIG_PATH)
    for key, value in raw.items():
        if not isinstance(value, dict):
            assert cfg[key] == value


def test_explicit_missing_path_fails_fast_but_default_missing_uses_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)

    monkeypatch.setattr(config_mod, "CONFIG_PATH", missing)
    cfg = load_config()
    assert cfg["synth"]["window_turns"] == DEFAULTS["synth"]["window_turns"]
    assert cfg["_config_path"] == str(missing)


def test_unknown_fields_are_rejected_with_readable_path() -> None:
    _assert_error("synth:\n  window_turn: 3\n", "synth.window_turn", "unknown")
    _assert_error("unknown_root: 1\n", "unknown_root", "unknown")
    _assert_error("judge:\n  task_models:\n    mystery: glm\n", "judge.task_models.mystery")
    _assert_error("paths:\n  data_dir: data\n  extra: x\n", "paths.extra")


def test_root_shape_and_yaml_syntax_are_rejected_without_masquerading_as_empty() -> None:
    _assert_error("- one\n- two\n", "<root>", "object")
    _assert_error("null\n", "<root>", "null")
    _assert_error('"just a scalar"\n', "<root>", "string")
    _assert_error("a: [\n", "<yaml>", "invalid YAML")


def test_duplicate_yaml_keys_are_rejected() -> None:
    _assert_error("synth:\n  window_turns: 3\n  window_turns: 4\n", "synth.window_turns", "duplicate")
    _assert_error("llm: {}\nllm: {}\n", "llm", "duplicate")


@pytest.mark.parametrize("text,path", [
    ('llm:\n  cache: "true"\n', "llm.cache"),
    ('gold:\n  state_folding: 1\n', "gold.state_folding"),
    ('judge:\n  enabled: "false"\n', "judge.enabled"),
])
def test_quoted_or_numeric_booleans_are_rejected(text: str, path: str) -> None:
    _assert_error(text, path, "boolean")


@pytest.mark.parametrize("text,path", [
    ("llm:\n  max_tokens: true\n", "llm.max_tokens"),
    ("judge:\n  step_batch: 4.0\n", "judge.step_batch"),
    ("synth:\n  window_turns: false\n", "synth.window_turns"),
])
def test_booleans_and_floats_are_not_accepted_as_integers(text: str, path: str) -> None:
    _assert_error(text, path, "integer")


@pytest.mark.parametrize("text,path", [
    ("synth:\n  eval_ratio: 1.7\n", "synth.eval_ratio"),
    ("labeling:\n  dominant_ratio_min: -0.1\n", "labeling.dominant_ratio_min"),
    ("review:\n  judge_confidence_min: 1.2\n", "review.judge_confidence_min"),
    ("synth:\n  eval_ratio: .nan\n", "synth.eval_ratio"),
    ("synth:\n  parallel_weight: .inf\n", "synth.parallel_weight"),
])
def test_ratios_and_numbers_reject_nonfinite_or_out_of_range(text: str, path: str) -> None:
    _assert_error(text, path)


@pytest.mark.parametrize("text,path", [
    ("llm:\n  max_tokens: 0\n", "llm.max_tokens"),
    ("llm:\n  timeout_seconds: 0\n", "llm.timeout_seconds"),
    ("turns:\n  model_max_turn_chars: -1\n", "turns.model_max_turn_chars"),
    ("gold:\n  min_gold_steps: 0\n", "gold.min_gold_steps"),
    ("judge:\n  concurrency: 0\n", "judge.concurrency"),
    ("judge:\n  session_workers: 0\n", "judge.session_workers"),
    ("judge:\n  quality_batch: -3\n", "judge.quality_batch"),
    ("synth:\n  window_turns: 0\n", "synth.window_turns"),
    ("synth:\n  weights:\n    opening: 0\n", "synth.weights.opening"),
])
def test_nonpositive_budgets_are_rejected(text: str, path: str) -> None:
    _assert_error(text, path)


def test_history_thinking_cap_allows_zero_but_not_negative() -> None:
    assert parse_config_text("synth:\n  history_thinking_max_chars: 0\n")["synth"][
        "history_thinking_max_chars"
    ] == 0
    _assert_error("synth:\n  history_thinking_max_chars: -1\n", "synth.history_thinking_max_chars")


def test_regex_patterns_are_compiled_and_errors_name_the_item() -> None:
    raw = parse_config_text("turns:\n  root_task_patterns:\n    - '\\.apk'\n")
    assert raw["turns"]["root_task_patterns"] == [r"\.apk"]
    _assert_error(
        "turns:\n  root_task_patterns:\n    - 'ok'\n    - '[unclosed'\n",
        "turns.root_task_patterns[1]",
        "regular expression",
    )
    _assert_error("turns:\n  irrelevant_patterns: 3\n", "turns.irrelevant_patterns", "list")


def test_root_normalizations_are_dynamic_string_to_string() -> None:
    raw = parse_config_text("turns:\n  root_normalizations:\n    custom_game: do it\n")
    assert raw["turns"]["root_normalizations"] == {"custom_game": "do it"}
    _assert_error("turns:\n  root_normalizations:\n    custom_game: 7\n", "turns.root_normalizations.custom_game")
    _assert_error("turns:\n  root_normalizations:\n    123: value\n", "turns.root_normalizations")


def test_cross_field_thinking_target_cannot_exceed_maximum() -> None:
    _assert_error(
        "gold:\n  max_thinking_chars: 100\n  target_thinking_chars: 101\n",
        "gold.target_thinking_chars",
        "max_thinking_chars",
    )
    assert parse_config_text(
        "gold:\n  max_thinking_chars: 100\n  target_thinking_chars: 100\n"
    )


def test_subsegments_validate_sid_and_ordered_positive_steps() -> None:
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: ''\n      start_step: 1\n      end_step: 2\n",
        "session_id",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: s1\n      start_step: 0\n      end_step: 2\n",
        "start_step",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: s1\n      start_step: 9\n      end_step: 2\n",
        "end_step",
        "start_step",
    )


def test_unknown_label_is_limited_to_repository_supported_labels() -> None:
    # Stage1 把该值当作标签名直接拼成输出目录，任意字符串会造出下游不认识的目录。
    for label in ("positive", "negative", "pending"):
        raw = parse_config_text(f"labeling:\n  unknown_label: {label}\n")
        assert raw["labeling"]["unknown_label"] == label
    _assert_error(
        "labeling:\n  unknown_label: quarantined\n",
        "labeling.unknown_label",
        "positive, negative, pending",
    )
    _assert_error("labeling:\n  unknown_label: ''\n", "labeling.unknown_label")
    _assert_error("labeling:\n  unknown_label: 3\n", "labeling.unknown_label", "string")


def test_subsegments_require_all_fields_and_ordered_positive_bounds() -> None:
    # 只给 session_id 会静默不生效；只给区间会把子段悄悄放宽到整个会话。
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: s1\n",
        "synth.sft_allow_subsegments[0]",
        "missing required field(s): start_step, end_step",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - start_step: 5\n      end_step: 9\n",
        "synth.sft_allow_subsegments[0]",
        "missing required field(s): session_id",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: s1\n      start_step: 3\n",
        "missing required field(s): end_step",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - {}\n",
        "missing required field(s): session_id, start_step, end_step",
    )
    _assert_error(
        "synth:\n  sft_allow_subsegments:\n    - session_id: s1\n      start_step: 1\n      end_step: 2\n      note: widened\n",
        "synth.sft_allow_subsegments[0].note",
        "unknown field",
    )
    raw = parse_config_text(
        "synth:\n  sft_allow_subsegments:\n"
        "    - {session_id: s1, start_step: 7, end_step: 7}\n"
    )
    assert raw["synth"]["sft_allow_subsegments"] == [
        {"session_id": "s1", "start_step": 7, "end_step": 7}
    ]


def test_domain_tool_metadata_must_stay_a_finite_json_tree() -> None:
    # 宽松的工具元数据里也不能藏 date / set / NaN / 非字符串键，否则保存后哈希会炸。
    _assert_error(
        "domain:\n  tools:\n    - name: t\n      metadata:\n        created: 2024-01-02\n",
        "domain.tools[0].metadata.created",
        "date is not a JSON value",
    )
    _assert_error(
        "domain:\n  tools:\n    - name: t\n      metadata:\n        nested:\n          - score: .nan\n",
        "domain.tools[0].metadata.nested[0].score",
        "finite",
    )
    _assert_error(
        "domain:\n  tools:\n    - name: t\n      metadata:\n        2024-01-02: tag\n",
        "domain.tools[0].metadata",
        "object key must be a string",
    )
    _assert_error(
        "domain:\n  tools:\n    - name: t\n      metadata: !!set\n        a: null\n",
        "domain.tools[0].metadata",
        "not a JSON value",
    )
    _assert_error(
        "export_profiles:\n  - id: qwen-xml\n    created: 2024-01-02\n",
        "export_profiles[0].created",
        "not a JSON value",
    )


def test_valid_opaque_extensions_still_load() -> None:
    text = """
domain:
  tools:
    - name: t
      metadata:
        when: "2024-01-02"
        labels: [a, b]
        flags: {ok: true, none: null, ratio: 0.5}
export_profiles:
  - id: qwen-xml
    meta:
      nested: [1, 2.5, {deep: ok}]
"""
    raw = parse_config_text(text)
    assert raw["domain"]["tools"][0]["metadata"]["when"] == "2024-01-02"
    assert raw["export_profiles"][0]["meta"]["nested"][0] == 1


def test_cyclic_metadata_is_reported_instead_of_recursing() -> None:
    tool: dict = {"name": "t"}
    tool["metadata"] = {"self": tool}
    cfg = copy.deepcopy(DEFAULTS)
    cfg["domain"]["tools"] = [tool]
    with pytest.raises(ConfigError, match="recursive structure"):
        validate_config_mapping(cfg)


def test_save_rejects_non_json_metadata_without_touching_file(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text("synth:\n  window_turns: 3\n", encoding="utf-8")
    old = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config_text(
            "domain:\n  tools:\n    - name: t\n      metadata:\n        created: 2024-01-02\n",
            path,
        )
    assert path.read_bytes() == old


def test_supported_optional_and_legacy_keys_are_preserved(tmp_path: Path) -> None:
    text = """
paths:
  cursor_sessions_dir: ""
judge:
  labels: ["positive", "pending"]
  task_models:
    steps: decide-model
    seams: ""
    quality: audit-model
    dpo: dpo-model
synth:
  sft_labels: ["positive"]
review:
  enabled: false
  judge_confidence_min: 0.8
export_profiles:
  - id: qwen-xml
    label: Qwen XML
    default: true
    tool_call_style: coder-xml
    template: qwen3_8
    tool_format: qwen3_coder
"""
    raw = parse_config_text(text)
    assert raw["paths"]["cursor_sessions_dir"] == ""
    assert raw["judge"]["task_models"]["steps"] == "decide-model"
    assert raw["synth"]["sft_labels"] == ["positive"]
    assert raw["export_profiles"][0]["id"] == "qwen-xml"
    path = tmp_path / "pipeline.yaml"
    path.write_text(text, encoding="utf-8")
    assert load_config(path)["judge"]["labels"] == ["positive", "pending"]


def test_domain_tools_allow_flexible_nested_schema_but_not_opaque_root() -> None:
    text = """
domain:
  tools:
    - type: function
      function:
        name: bash
        description: Run a command
        parameters:
          type: object
          properties:
            command:
              type: string
              description: shell command
            nested:
              type: object
              properties:
                anything: {type: array, items: {type: string}}
          required: [command]
    - name: custom_tool
      input_schema:
        any: {deeply: {nested: [1, 2, 3]}}
"""
    tools = parse_config_text(text)["domain"]["tools"]
    assert tools[0]["function"]["parameters"]["properties"]["nested"]["properties"]["anything"]
    _assert_error("domain:\n  tools: {name: bash}\n", "domain.tools", "list")
    _assert_error("domain:\n  tools:\n    - just-a-string\n", "domain.tools[0]", "object")
    _assert_error(
        "domain:\n  tools:\n    - type: function\n      function:\n        parameters: []\n",
        "domain.tools[0].function.parameters",
        "object",
    )


def test_load_config_accepts_dumped_defaults_with_legacy_top_level_keys(tmp_path: Path) -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["export_profiles"] = [{"id": "qwen-xml", "default": True}]
    cfg["_config_path"] = "should-be-ignored"
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["export_profiles"][0]["id"] == "qwen-xml"
    assert loaded["_config_path"] == str(path)
    assert loaded["synth"]["window_turns"] == DEFAULTS["synth"]["window_turns"]


def test_save_config_text_is_checked_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text("synth:\n  window_turns: 3\n", encoding="utf-8")
    old = path.read_bytes()
    with pytest.raises(ConfigError):
        save_config_text("synth:\n  window_turns: 0\n", path)
    assert path.read_bytes() == old
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []

    new_text = "# kept comment\nsynth:\n  window_turns: 5\n"
    save_config_text(new_text, path)
    assert path.read_text(encoding="utf-8") == new_text
    assert load_config(path)["synth"]["window_turns"] == 5
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_config_is_checked_atomic_and_does_not_mutate_input(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text("synth:\n  window_turns: 3\n", encoding="utf-8")
    old = path.read_bytes()
    bad = load_config(path)
    bad["synth"]["window_turns"] = 0
    with pytest.raises(ConfigError):
        save_config(bad, path)
    assert path.read_bytes() == old
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []

    cfg = load_config(path)
    before = copy.deepcopy(cfg)
    save_config(cfg, path)
    assert cfg == before
    assert "_config_path" not in yaml.safe_load(path.read_text(encoding="utf-8"))


def test_config_hash_ignores_provenance_and_saved_config_roundtrips(tmp_path: Path) -> None:
    cfg = copy.deepcopy(DEFAULTS)
    bare = config_hash(cfg)
    cfg["_config_path"] = str(tmp_path / "elsewhere.yaml")
    assert config_hash(cfg) == bare

    path = tmp_path / "pipeline.yaml"
    save_config(cfg, path)
    assert config_hash(load_config(path)) == bare