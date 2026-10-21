"""单步质量检测：病态行为识别 + 启发式思维链压缩。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

DOUBT_REGEX = re.compile(
    r"\b(Wait|Actually|Hold on|Let me reconsider|Let's reconsider|Let me double check|"
    r"Wait a second|Wait a minute|No, wait|Hmm, wait)\b|不对|等等|重新考虑|等一下",
    re.IGNORECASE,
)
MENTAL_HEX_CHECK_REGEX = re.compile(
    r"(0x[0-9a-fA-F]{2,}\s*=\s*\d+[^\n]{0,40}✓|\d+\s*=\s*0x[0-9a-fA-F]{2,}[^\n]{0,40}✓|"
    r"IEEE\s*754[^\n]{0,60}✓)",
    re.IGNORECASE,
)
HALLUCINATED_SEARCH_REGEX = re.compile(
    r"(sparse_packed_container\.cpp|packed_data_container_sparse)", re.IGNORECASE
)
LOW_LEVEL_MISTAKE_OUTPUT_REGEX = re.compile(
    r"(command not found|Not a Godot directory|No valid paths provided|"
    r'Parse Error: Static function "has_custom_feature\(\)" not found|'
    r'Parse Error: Cannot find member "APPEND" in base "FileAccess"|'
    r"unrecognized option|Unknown option|No such file or directory)",
    re.IGNORECASE,
)
NEXT_STEP_OOPS_REGEX = re.compile(
    r"^(Oops|It didn't recognize because|has_custom_feature not exposed|`zip` not available|"
    r"There's a bug in _skip usage)",
    re.IGNORECASE,
)

CROSS_DIR_REGEX = re.compile(
    r"(?:[a-zA-Z]:[/\\]+crack[/\\]+|[/\\]d[/\\]+crack[/\\]+)([a-zA-Z0-9_.\-]+)", re.IGNORECASE
)


def analyze_step_quality(
    thinking: str,
    text: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    session_cwd: str = "",
    max_think_chars: int = 4500,
) -> Dict[str, Any]:
    pathologies: List[str] = []
    strengths: List[str] = []
    think_len = len(thinking or "")
    doubt_count = len(DOUBT_REGEX.findall(thinking or ""))

    if think_len == 0 and tool_calls:
        pathologies.append("missing_thinking_block")
    if think_len > max_think_chars:
        pathologies.append("over_rumination_length")
    if doubt_count >= 8:
        pathologies.append("over_rumination_doubt_loop")
    if thinking and MENTAL_HEX_CHECK_REGEX.search(thinking):
        pathologies.append("mental_hex_checkmark")

    combined_cmd = " ".join(json.dumps(tc.get("arguments", {}), ensure_ascii=False) for tc in tool_calls)
    if HALLUCINATED_SEARCH_REGEX.search(thinking or "") or HALLUCINATED_SEARCH_REGEX.search(combined_cmd):
        pathologies.append("hallucinated_source_search")

    if session_cwd and "crack" in session_cwd.lower():
        cwd_norm = session_cwd.replace("\\", "/").rstrip("/").lower()
        cwd_name = cwd_norm.split("/")[-1]
        base = cwd_name.split("-")[0]
        for sib in CROSS_DIR_REGEX.findall(combined_cmd + " " + (thinking or "")):
            sib_low = sib.lower().rstrip("/")
            if sib_low != cwd_name and sib_low.startswith(base):
                pathologies.append(f"cross_dir_violation:{sib}")
                break

    error_count = sum(1 for tr in tool_results if tr.get("isError"))
    combined_out = "\n".join(tr.get("content", "") for tr in tool_results)
    if error_count > 0:
        pathologies.append("tool_execution_error")
    elif LOW_LEVEL_MISTAKE_OUTPUT_REGEX.search(combined_out):
        pathologies.append("tool_logical_failure")

    if 80 <= think_len <= 2200 and doubt_count <= 2:
        strengths.append("concise_decisive_thinking")
    if len(tool_calls) >= 2:
        strengths.append(f"parallel_tool_calls:{len(tool_calls)}")
    if any(
        "python -c" in json.dumps(tc.get("arguments", {}))
        or "python - <<" in json.dumps(tc.get("arguments", {}))
        for tc in tool_calls
    ):
        strengths.append("script_verification")
    if tool_calls and not (text or "").strip():
        strengths.append("silent_tool_execution")

    score = 100.0
    if "missing_thinking_block" in pathologies:
        score -= 25.0
    if "over_rumination_length" in pathologies:
        score -= min(45.0, (think_len - max_think_chars) / 150.0)
    if "over_rumination_doubt_loop" in pathologies:
        score -= min(35.0, (doubt_count - 7) * 5.0)
    if "mental_hex_checkmark" in pathologies:
        score -= 30.0
    if "hallucinated_source_search" in pathologies:
        score -= 25.0
    if any(p.startswith("cross_dir_violation") for p in pathologies):
        score -= 50.0
    if "tool_execution_error" in pathologies or "tool_logical_failure" in pathologies:
        score -= 15.0 * max(1, error_count)
    if any(s.startswith("parallel_tool_calls") for s in strengths):
        score += 10.0
    if "concise_decisive_thinking" in strengths:
        score += 10.0

    is_clean = not pathologies and (think_len > 0 or (not tool_calls and len((text or "").strip()) > 0))
    return {
        "is_clean": is_clean,
        "score": round(max(0.0, min(120.0, score)), 2),
        "pathologies": pathologies,
        "strengths": strengths,
        "metrics": {
            "think_len": think_len,
            "text_len": len(text or ""),
            "doubt_count": doubt_count,
            "tool_call_count": len(tool_calls),
            "tool_error_count": error_count,
        },
    }


def compact_thinking(thinking: str, target_max_chars: int = 1600) -> str:
    """启发式压缩：保留首段（观测）+ 中段无自我推翻内容 + 末两段（收敛结论）。"""
    if not thinking or len(thinking) <= target_max_chars:
        return thinking
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", thinking) if p.strip()]
    if len(paragraphs) <= 2:
        return thinking[: int(target_max_chars * 0.55)] + "\n\n" + thinking[-int(target_max_chars * 0.45) :]

    kept: List[str] = [paragraphs[0]]
    for p in paragraphs[1:-2]:
        if DOUBT_REGEX.search(p) or MENTAL_HEX_CHECK_REGEX.search(p):
            continue
        if sum(len(x) for x in kept) + len(p) < target_max_chars * 0.55:
            kept.append(p)
    for p in paragraphs[-2:]:
        cleaned = re.sub(r"^(Wait|Actually|Hold on|Hmm)[,\s]+", "", p, flags=re.IGNORECASE)
        if cleaned not in kept:
            kept.append(cleaned)

    result = "\n\n".join(kept)
    if len(result) > target_max_chars:
        result = result[: int(target_max_chars * 0.55)] + "\n\n" + result[-int(target_max_chars * 0.45) :]
    return result