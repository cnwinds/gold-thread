#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pi 工程分析会话日志增量提取、黄金轨迹提纯与训练数据集同步工具 (sync_pi_datasets.py)
================================================================================
核心功能特性：
1. 三层噪音过滤与工程分析领域准入网关 (Three-Layer Domain & Turn Filtering)：
   - [Layer 1 会话级准入]：仅提取工程分析相关的会话（`--D--crack-*--` 工作目录且含真实 APK/XAPK 工程分析任务），
     100% 剔除日常闲聊、模型配置 (`zhipu`/`sub2api`/`openwebui`)、`gpt-image-2` 生图、股票分析、其他 Web 项目等无关会话，
     并自动排除早期重复前缀副本（如 `--C--Users-lvan-.pi-agent--\01a1523a`）。
   - [Layer 2 回合级除噪]：即使在破解会话内部，自动剥离前置与中途的 `models.json` 配置、`hi` 连通性测试、
     API 报错排查 (`400/502/503`)、`ollama/openwebui` 调试等无关回合；仅从首个“根破解任务指令 (Root Crack Prompt)”开始提取。
   - [Layer 3 无头追问锚定与纠偏内化]：
     * 将“不要参考其他的目录，自己研究”等因跨目录违规触发的纠偏指令内化为根任务约束，并将违规步转化为 DPO 负例，杜绝产生孤立的无头 Prompt；
     * 将同一会话内的多阶段追问（如 `继续`、`可以`、`帮我看看他的千人千面是怎么实现的`）串联在同一个根任务上下文下。

2. 幂等增量状态追踪 (Incremental Upsert Sync)：
   - 维护 `sync_state.json`，记录每个会话 `.jsonl` 的 `size`、`mtime_ns` 与尾部 4KB `tail_sha256` 指纹。
   - 自动识别 [NEW] 新增、[UPDATED] 追加更新、[UNCHANGED] 未变更（毫秒级跳过）、[DELETED] 已删除会话。

3. 四阶段黄金轨迹提纯流水线 (Gold Trajectory Distillation Pipeline)：
   - [Stage 1] 轨迹剪枝与状态折叠 (`write` + `edit` 合并、手滑命令与空思考步剪枝并转 DPO)；
   - [Stage 2] 思维链去噪（压缩 >2400 字反刍段）与中间步静默执行（清空冗余 `SAY`）；
   - [Stage 3] 单步滑动窗口展开 + 历史回合 `<think>` 剥离 (`strip_thinking=True`) + 根任务指令永久锚定；
   - [Stage 4] 阶段均衡采样权重 (`opening`/`closing` 3x, `key_decision` 2x) 与按项目 (`project_group`) 防泄漏切分。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 默认配置与常量
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS_DIR = Path.home() / ".pi" / "agent" / "sessions"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 严格限定 Teacher 模型范围（仅 DeepSeek V4 与 GLM-5 系列作为 SFT 正样本主干；Qwen3.8-27B 作为被诊断/对齐的 Student 模型）
DEFAULT_TEACHER_PATTERNS = [
    r"^deepseek-v4",
    r"^glm-5\.",
]

# 识别“根任务”开启的关键词正则
CRACK_TASK_START_REGEX = re.compile(
    r"(\.xapk|\.apk|我要破解|帮我破解|自己研究破解|继续破解|安全研究员.*分析对象|还原成明文)",
    re.IGNORECASE,
)

# 识别会话内部与任务无关的“配置/测试/API报错排查/硬件闲聊/磁盘清理”杂音 User Turn 正则
IRRELEVANT_USER_TURN_REGEX = re.compile(
    r"^("
    r"hi|hello|greet|/modle|/model|/clear|/effect|/webui.*|\.mod|fdsaf|批、|[0-9a-zA-Z]"
    r")$"
    r"|models\.json|sub2api|openwebui|ollama|trycloudflare\.com|cursor-pluse|zhipu|gpt-image-2"
    r"|增加.*模型|加入.*模型|添加.*model|上下文.*(512K|1M)|怎么换模型推理能力"
    r"|隔壁的会话报错了|选这个模型报错|这个端口用curl调用|要下载\s*fd\s*linux版"
    r"|pi要下载这2个文件|怎么设置环境变量.*gh_proxy"
    r"|安装\s*rea\s*工具|现在装\s*Ghidra"
    r"|qwen3\.8\s*拒绝回答的问题"
    r"|小米平板1代|华为mate50pro|frida-server\s*是干什么用的|目前界面没有uac弹窗|^先清理$|^b\s*我还需要装什么$|你帮我下载mumu",
    re.IGNORECASE | re.DOTALL,
)

# 识别应当折叠并入主线（不作为独立 User 提问打断自主执行流）的“过程催促/确认/纪律纠偏”指令
CONTINUATION_OR_CORRECTION_TURN_REGEX = re.compile(
    r"^("
    r"不要参考其他的目录.*|你不要看其他的目录.*|"
    r"继续|继续A|可以|可以，\s*并且记录下来|补全结构|"
    r"如果不需要人能看懂.*|如果是这条路.*|这样你在D:\\crack\\jigsawcard.*|"
    r"现在所有的配置表文件都能拿到吗|config\.bin\s*收尾.*|ConfSample\s*这个是什么东西|"
    r"我要你拿到所有的资源.*|类数量灌水.*|所有的文件都要还原成明文|"
    r"你的表现很好，我就是oakever工作室的安全员.*"
    r")$",
    re.IGNORECASE | re.DOTALL,
)

# 反思/自我怀疑高频词（用于检测 Over-rumination 思维链反刍）
DOUBT_REGEX = re.compile(
    r"\b(Wait|Actually|Hold on|Let me reconsider|Let's reconsider|Let me double check|"
    r"Wait a second|Wait a minute|No, wait|Hmm, wait)\b|不对|等等|重新考虑|等一下",
    re.IGNORECASE,
)

# 心算十六进制/浮点数并自打勾 ✓ 的正则
MENTAL_HEX_CHECK_REGEX = re.compile(
    r"(0x[0-9a-fA-F]{2,}\s*=\s*\d+[^\n]{0,40}✓|\d+\s*=\s*0x[0-9a-fA-F]{2,}[^\n]{0,40}✓|"
    r"IEEE\s*754[^\n]{0,60}✓)",
    re.IGNORECASE,
)

# 脑补源码路径搜索特征
HALLUCINATED_SEARCH_REGEX = re.compile(
    r"(sparse_packed_container\.cpp|packed_data_container_sparse)",
    re.IGNORECASE,
)

# 明显的手滑/低级命令错误输出特征（用于轨迹剪枝）
LOW_LEVEL_MISTAKE_OUTPUT_REGEX = re.compile(
    r"(command not found|Not a Godot directory|No valid paths provided|"
    r"Parse Error: Static function \"has_custom_feature\(\)\" not found|"
    r"Parse Error: Cannot find member \"APPEND\" in base \"FileAccess\"|"
    r"unrecognized option|Unknown option)",
    re.IGNORECASE,
)

# 下一步思考中承认上一步手滑覆写/路径传错的特征
NEXT_STEP_OOPS_REGEX = re.compile(
    r"^(Oops|It didn't recognize because|has_custom_feature not exposed|`zip` not available|"
    r"There's a bug in _skip usage)",
    re.IGNORECASE,
)

# 默认工程分析 Agent 系统提示词（注入纪律红线，内化跨目录隔离与科学工具验证规范）
DEFAULT_CRACK_SYSTEM_PROMPT = (
    "你是一名顶尖的移动应用与游戏工程分析专家 Agent。在执行解包、反编译、资产恢复、协议与配置表解密任务时，必须遵守以下工程纪律：\n"
    "1. 严格在当前工作目录内独立完成研究与产出，严禁读取、参考或拷贝同级其他实验目录的文件；\n"
    "2. 禁止在思维链中心算十六进制偏移、XOR 或 IEEE-754 浮点数并自行打勾确认，一切字节偏移与结构假设必须通过 `python -c` 脚本实测验证；\n"
    "3. 优先并发调用工具与复用成熟开源分析工具链（如 jadx、gdre_tools、Il2CppDumper、radare2 等），遇到连续 2 次搜索 404 立即止损切换方案；\n"
    "4. 阶段性关键发现（魔数、密钥、结构体偏移）及时落盘记录，收尾编写报告前必须先完成全量文件计数与 SHA256/明文抽样核验。"
)

# 默认 pi 工具 Schema 定义
DEFAULT_PI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the workspace shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Optional timeout in milliseconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Start line number"},
                    "limit": {"type": "integer", "description": "Number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "oldText": {"type": "string", "description": "Exact text to replace"},
                    "newText": {"type": "string", "description": "New text"},
                },
                "required": ["path", "oldText", "newText"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 工具函数：文件指纹、领域准入过滤、回合清洗、项目分组
# ---------------------------------------------------------------------------

def is_crack_session_path(rel_path: str) -> bool:
    """
    [Layer 1 会话级准入网关]
    仅允许 `--D--crack-<project>--` 下的目标项目会话进入流水线。
    排除：
    - 非 crack 目录（如 `--C--Users-lvan--`, `--D--ai_projects*--`, `--E--Download*--`, `--Y--stock-projects*--`）
    - `--C--Users-lvan-.pi-agent--` 下的早期前缀重复副本
    - `--D--crack--` 根目录下的非项目问答会话
    """
    folder = rel_path.replace("\\", "/").split("/")[0].lower()
    return folder.startswith("--d--crack-") and folder != "--d--crack--"


def is_irrelevant_user_turn(user_text: str) -> bool:
    """
    [Layer 2 回合级除噪网关]
    判断某条 User 消息是否属于与任务无关的模型配置、连通性测试 (`hi`)、API 报错粘贴或短字符误触。
    """
    txt = user_text.strip()
    if len(txt) <= 1:
        return True
    if CRACK_TASK_START_REGEX.search(txt):
        return False
    if IRRELEVANT_USER_TURN_REGEX.search(txt):
        return True
    # 若整段话是在粘贴 Model Error 400/502/503 且未包含破解指令
    if ("Error: 400" in txt or "Error: 502" in txt or "Error: 503" in txt) and "继续" in txt and len(txt) > 60:
        return True
    return False


def compute_file_fingerprint(path: Path) -> Dict[str, Any]:
    """计算会话文件的快速增量指纹（size + mtime_ns + 尾部 4KB SHA256）。"""
    stat = path.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    tail_bytes = b""
    with path.open("rb") as f:
        if size > 4096:
            f.seek(size - 4096)
        tail_bytes = f.read(4096)
    tail_sha256 = hashlib.sha256(tail_bytes).hexdigest()
    return {
        "size": size,
        "mtime_ns": mtime_ns,
        "tail_sha256": tail_sha256,
    }


def is_fingerprint_equal(fp1: Optional[Dict[str, Any]], fp2: Dict[str, Any]) -> bool:
    if not fp1:
        return False
    return (
        fp1.get("size") == fp2.get("size")
        and fp1.get("tail_sha256") == fp2.get("tail_sha256")
    )


def truncate_middle(text: str, max_chars: int) -> str:
    """对超长工具输出进行首尾保留式截断，避免撑爆训练上下文。"""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    head_len = int(max_chars * 0.65)
    tail_len = max_chars - head_len
    omitted = len(text) - max_chars
    return (
        text[:head_len]
        + f"\n\n... [已省略中间 {omitted} 字符以适配训练上下文] ...\n\n"
        + text[-tail_len:]
    )


def is_teacher_model(model_id: str, patterns: List[str]) -> bool:
    if not model_id:
        return False
    for pat in patterns:
        if re.search(pat, model_id, re.IGNORECASE):
            return True
    return False


def infer_project_group(cwd: str, rel_path: str) -> str:
    """推断会话所属的目标项目组（用于按完整游戏维度切分 Train/Eval，防止跨步数据泄漏）。"""
    if cwd:
        norm = cwd.replace("\\", "/").rstrip("/")
        leaf = norm.split("/")[-1].lower()
        if "crack/" in norm.lower():
            return "game::" + leaf.split("-")[0]
        return f"workspace::{leaf}"
    folder = rel_path.split("/")[0].lower()
    if folder.startswith("--d--crack-"):
        sub = folder[len("--d--crack-"):].rstrip("-")
        return "game::" + sub.split("-")[0]
    return f"session_group::{folder}"


def normalize_polling_command(cmd: str) -> str:
    """将后台 `nohup ... & sleep N; tail` 盲等模式规范化为进程存活轮询模式。"""
    pattern = re.compile(
        r"(nohup\s+[^\n&]+>\s*(/tmp/[^\s;&]+)\s+2>&1\s*&)\s*\n?\s*echo\s+started;\s*sleep\s+(\d+);\s*tail\s+(-\d+)\s+\2"
    )
    def _repl(m: re.Match) -> str:
        bg_part = m.group(1)
        log_file = m.group(2)
        wait_sec = int(m.group(3))
        tail_arg = m.group(4)
        loops = max(5, wait_sec // 2)
        return (
            f"{bg_part} BG_PID=$!; echo started pid=$BG_PID; "
            f"for _i in $(seq 1 {loops}); do kill -0 $BG_PID 2>/dev/null || break; sleep 2; done; "
            f"tail {tail_arg} {log_file}"
        )
    return pattern.sub(_repl, cmd)


# ---------------------------------------------------------------------------
# 病态行为与质量检测器 (Pathology & Quality Detector)
# ---------------------------------------------------------------------------

def analyze_step_quality(
    thinking: str,
    text: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    session_cwd: str,
    max_clean_think_chars: int = 4500,
) -> Dict[str, Any]:
    """分析单个 Assistant 步骤的思维链与工具调用质量。"""
    pathologies: List[str] = []
    strengths: List[str] = []

    think_len = len(thinking)
    doubt_matches = DOUBT_REGEX.findall(thinking) if thinking else []
    doubt_count = len(doubt_matches)

    # 0. 检测缺失思考块（如 GLM 偶发的空 thinking 直接调工具）
    if think_len == 0 and len(tool_calls) > 0:
        pathologies.append("missing_thinking_block")

    # 1. 检测过度反刍 (Over-rumination)
    if think_len > max_clean_think_chars:
        pathologies.append("over_rumination_length")
    if doubt_count >= 8:
        pathologies.append("over_rumination_doubt_loop")

    # 2. 检测心算十六进制/浮点数打勾确认偏误
    if thinking and MENTAL_HEX_CHECK_REGEX.search(thinking):
        pathologies.append("mental_hex_checkmark")

    # 3. 检测脑补源码搜索
    combined_cmd = " ".join(
        json.dumps(tc.get("arguments", {}), ensure_ascii=False) for tc in tool_calls
    )
    if HALLUCINATED_SEARCH_REGEX.search(thinking or "") or HALLUCINATED_SEARCH_REGEX.search(combined_cmd):
        pathologies.append("hallucinated_source_search")

    # 4. 检测跨实验目录偷看/复制 (Cross-directory isolation violation)
    if session_cwd and ("crack" in session_cwd.lower()):
        cwd_norm = session_cwd.replace("\\", "/").rstrip("/").lower()
        cwd_name = cwd_norm.split("/")[-1]
        base_proj = cwd_name.split("-")[0]
        sibling_matches = re.findall(
            r"(?:[dD]:[/\\]+crack[/\\]+|[/\\]d[/\\]+crack[/\\]+)([a-zA-Z0-9_.-]+)",
            combined_cmd + " " + (thinking or ""),
        )
        for sib in sibling_matches:
            sib_low = sib.lower()
            if sib_low != cwd_name and sib_low.startswith(base_proj):
                pathologies.append(f"cross_dir_violation:{sib}")
                break

    # 5. 检测工具调用错误或低级失误输出
    error_count = sum(1 for tr in tool_results if tr.get("isError"))
    combined_out = "\n".join(tr.get("content", "") for tr in tool_results)
    if error_count > 0:
        pathologies.append("tool_execution_error")
    elif LOW_LEVEL_MISTAKE_OUTPUT_REGEX.search(combined_out):
        pathologies.append("tool_logical_failure")

    # 正向特征识别 (Strengths)
    if 80 <= think_len <= 2200 and doubt_count <= 2:
        strengths.append("concise_decisive_thinking")
    if len(tool_calls) >= 2:
        strengths.append(f"parallel_tool_calls:{len(tool_calls)}")
    if any("python -c" in json.dumps(tc.get("arguments", {})) or "python - <<" in json.dumps(tc.get("arguments", {})) for tc in tool_calls):
        strengths.append("script_verification")
    if tool_calls and not text.strip():
        strengths.append("silent_tool_execution")

    # 综合评分
    score = 100.0
    if "missing_thinking_block" in pathologies:
        score -= 25.0
    if "over_rumination_length" in pathologies:
        score -= min(45.0, (think_len - max_clean_think_chars) / 150.0)
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

    if "parallel_tool_calls" in " ".join(strengths):
        score += 10.0
    if "concise_decisive_thinking" in strengths:
        score += 10.0

    is_clean = len(pathologies) == 0 and (think_len > 0 or (not tool_calls and len(text.strip()) > 0))

    return {
        "is_clean": is_clean,
        "score": round(max(0.0, min(120.0, score)), 2),
        "pathologies": pathologies,
        "strengths": strengths,
        "metrics": {
            "think_len": think_len,
            "text_len": len(text),
            "doubt_count": doubt_count,
            "tool_call_count": len(tool_calls),
            "tool_error_count": error_count,
        },
    }


def compact_ruminating_thinking(thinking: str, target_max_chars: int = 1600) -> str:
    """
    对冗长/过度反刍的 `<think>` 进行去噪提纯：
    保留首段（观察与问题定义）和末段（最终收敛的推理与行动计划），剔除中间反复自我推翻的段落。
    """
    if not thinking or len(thinking) <= target_max_chars:
        return thinking
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", thinking) if p.strip()]
    if len(paragraphs) <= 2:
        return thinking[: int(target_max_chars * 0.55)] + "\n\n" + thinking[-int(target_max_chars * 0.45) :]

    head_paras = paragraphs[:1]
    tail_paras = paragraphs[-2:]
    middle_paras = paragraphs[1:-2]

    kept: List[str] = list(head_paras)
    for p in middle_paras:
        if len(DOUBT_REGEX.findall(p)) == 0 and not MENTAL_HEX_CHECK_REGEX.search(p):
            if sum(len(x) for x in kept) + len(p) < target_max_chars * 0.55:
                kept.append(p)

    for p in tail_paras:
        cleaned_p = re.sub(r"^(Wait|Actually|Hold on|Hmm)[,\s]+", "", p, flags=re.IGNORECASE)
        if cleaned_p not in kept:
            kept.append(cleaned_p)

    result = "\n\n".join(kept)
    if len(result) > target_max_chars:
        result = result[: int(target_max_chars * 0.55)] + "\n\n" + result[-int(target_max_chars * 0.45) :]
    return result


# ---------------------------------------------------------------------------
# 黄金轨迹提纯器：状态折叠 + 弯路剪枝 + 空思考过滤 (Gold Trajectory Purifier)
# ---------------------------------------------------------------------------

def purify_segment_into_gold(
    seg: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    对完整项目会话消息流执行黄金提纯：
    1. 状态折叠 (State Folding)：若 Step i 是 `write(path)`，且 Step i+1 是对同一 `path` 的立即修复 `edit`，
       将 Step i+1 的 `edits` 直接合并回 Step i 的 `write` 内容，移除 Step i+1。
    2. 弯路剪枝 (Detour Pruning)：若 Step i 触发了低级命令错误/逻辑失误或下一步开头承认手滑 (`Oops...`)，
       从黄金轨迹中剪除 Step i，并将 (Step i+1 修正版 vs Step i 报错版) 生成一条天然 DPO 偏好对。
    3. 空 `<think>` 过滤：若某步有 `tool_calls` 但 `thinking` 为空（如 GLM 的偶发跳步），若其本身报错则直接剪除，
       若执行成功则自动根据调用的工具为其补齐简明意图推理，杜绝训练出“不思考盲调工具”的坏习惯。
    4. 思维链去噪与静默化：将 >2400 字的 `<think>` 压缩提纯，并清空中间工具调用步的冗余 `text` 播报。
    """
    units: List[Dict[str, Any]] = []
    for m in seg:
        if m["role"] in ("system", "user"):
            units.append({"type": m["role"], "msg": copy.deepcopy(m), "tools": []})
        elif m["role"] == "assistant":
            units.append({"type": "assistant_turn", "msg": copy.deepcopy(m), "tools": []})
        elif m["role"] == "tool":
            if units and units[-1]["type"] == "assistant_turn":
                units[-1]["tools"].append(copy.deepcopy(m))

    pruned_dpo_records: List[Dict[str, Any]] = []
    kept_units: List[Dict[str, Any]] = []

    i = 0
    while i < len(units):
        cur = units[i]
        if cur["type"] != "assistant_turn":
            kept_units.append(cur)
            i += 1
            continue

        cur_msg = cur["msg"]
        cur_tcs = cur_msg.get("tool_calls", [])
        cur_trs = cur["tools"]

        # 检查是否可以与下一个 assistant_turn 进行“状态折叠”或“弯路剪枝”
        if i + 1 < len(units) and units[i + 1]["type"] == "assistant_turn":
            nxt = units[i + 1]
            nxt_msg = nxt["msg"]
            nxt_tcs = nxt_msg.get("tool_calls", [])

            # (A) 状态折叠：Step i 是单次 write，Step i+1 是对同一文件的单次 edit
            if (
                len(cur_tcs) == 1
                and len(nxt_tcs) == 1
                and cur_tcs[0]["name"] == "write"
                and nxt_tcs[0]["name"] == "edit"
                and cur_tcs[0]["arguments"].get("path") == nxt_tcs[0]["arguments"].get("path")
            ):
                content_str = cur_tcs[0]["arguments"].get("content", "")
                edits_list = nxt_tcs[0]["arguments"].get("edits", [])
                if not edits_list and "oldText" in nxt_tcs[0]["arguments"]:
                    edits_list = [nxt_tcs[0]["arguments"]]

                folded_ok = bool(edits_list)
                for ed in edits_list:
                    old_t = ed.get("oldText", "")
                    new_t = ed.get("newText", "")
                    if old_t and old_t in content_str:
                        content_str = content_str.replace(old_t, new_t, 1)
                    else:
                        folded_ok = False
                        break

                if folded_ok:
                    orig_unfolded_msg = copy.deepcopy(cur_msg)
                    cur_tcs[0]["arguments"]["content"] = content_str
                    cur_msg["folded_from_msg_id"] = nxt_msg["id"]
                    pruned_dpo_records.append({
                        "reason": "state_folding_bug_fix",
                        "context_units": copy.deepcopy(kept_units),
                        "chosen_msg": copy.deepcopy(cur_msg),
                        "rejected_msg": orig_unfolded_msg,
                    })
                    kept_units.append(cur)
                    i += 2
                    continue

            # (B) 弯路剪枝：当前步触发了明显的命令错误/逻辑失误，或下一步开头明确说 Oops / 传错目录
            combined_tr_out = "\n".join(t.get("content", "") for t in cur_trs)
            has_low_level_err = any(t.get("isError") for t in cur_trs) or bool(
                LOW_LEVEL_MISTAKE_OUTPUT_REGEX.search(combined_tr_out)
            )
            nxt_admits_mistake = bool(NEXT_STEP_OOPS_REGEX.search((nxt_msg.get("thinking") or "").strip()))

            if (has_low_level_err or nxt_admits_mistake) and len(nxt_tcs) > 0:
                cleaned_nxt = copy.deepcopy(nxt)
                cleaned_think = re.sub(
                    r"^(Oops[^.!?\n]*[.!?\n]+|It didn't recognize because[^.!?\n]*[.!?\n]+|"
                    r"`zip` not available[.!?\n]*)",
                    "",
                    (cleaned_nxt["msg"].get("thinking") or "").strip(),
                    flags=re.IGNORECASE,
                ).strip()
                if not cleaned_think and cur_msg.get("thinking"):
                    cleaned_think = compact_ruminating_thinking(cur_msg["thinking"], 1200)
                cleaned_nxt["msg"]["thinking"] = cleaned_think
                cleaned_nxt["msg"]["pruned_detour_msg_id"] = cur_msg["id"]

                pruned_dpo_records.append({
                    "reason": "detour_pruned_error_or_oops",
                    "context_units": copy.deepcopy(kept_units),
                    "chosen_msg": copy.deepcopy(cleaned_nxt["msg"]),
                    "rejected_msg": copy.deepcopy(cur_msg),
                })
                units[i + 1] = cleaned_nxt
                i += 1
                continue

        # 过滤无思考且报错的废步；对无思考但成功的步，若其有 SAY 文本则将 SAY 内化为 thinking
        if not cur_msg.get("thinking") and cur_tcs:
            if any(t.get("isError") for t in cur_trs):
                i += 1
                continue
            if cur_msg.get("text"):
                cur_msg["thinking"] = cur_msg["text"].strip()
                cur_msg["text"] = ""
            else:
                # 跳过完全无思考且无文本说明的盲调步
                i += 1
                continue

        # 对保留的回合进行思维链提纯与工具命令规范化
        if len(cur_msg.get("thinking", "")) > 2400:
            cur_msg["thinking"] = compact_ruminating_thinking(cur_msg["thinking"], target_max_chars=1600)
        # 中间步骤（有 tool_calls）剥离多余的 SAY 文本，强化静默执行
        if cur_tcs and cur_msg.get("text"):
            cur_msg["text"] = ""
        # 规范化 bash 中的盲等 sleep 轮询
        for tc in cur_tcs:
            if tc["name"] == "bash" and "command" in tc.get("arguments", {}):
                tc["arguments"]["command"] = normalize_polling_command(tc["arguments"]["command"])

        kept_units.append(cur)
        i += 1

    gold_seg: List[Dict[str, Any]] = []
    for u in kept_units:
        gold_seg.append(u["msg"])
        gold_seg.extend(u["tools"])

    return gold_seg, pruned_dpo_records


# ---------------------------------------------------------------------------
# 会话解析与多维数据集生成核心
# ---------------------------------------------------------------------------

def parse_session_file(
    file_path: Path,
    rel_path: str,
    max_tool_output_chars: int,
    teacher_patterns: List[str],
    step_window_turns: int = 8,
) -> Dict[str, Any]:
    """
    解析单个工程分析 `.pi` 会话 `.jsonl` 文件：
    1. 自动剔除前置与中途的模型配置、`hi` 测试与报错排查 Turn；
    2. 将同一会话内的有效工程分析 Turn 合并在同一个根任务下（避免无头 Prompt）；
    3. 区分 Teacher 与 Student 会话，生成黄金轨迹、单步窗口、DPO 对与诊断库。
    """
    session_id = file_path.stem
    session_cwd = ""
    session_start_ts = ""
    current_model = "unknown"
    current_provider = "unknown"
    current_thinking_level = "default"

    raw_events: List[Dict[str, Any]] = []
    deleted_ids: Set[str] = set()
    replacements: Dict[str, Any] = {}

    line_count = 0
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            raw_events.append(obj)
            if obj.get("type") == "context_edit":
                target_id = obj.get("targetId")
                rep = obj.get("replacement")
                if target_id:
                    if rep is None:
                        deleted_ids.add(target_id)
                    else:
                        replacements[target_id] = rep

    raw_messages: List[Dict[str, Any]] = []
    last_assistant_entry: Optional[Dict[str, Any]] = None

    for ev in raw_events:
        ev_type = ev.get("type")
        ev_id = ev.get("id", "")
        if ev_id in deleted_ids:
            continue

        if ev_type == "session":
            session_id = ev.get("id", session_id)
            session_cwd = ev.get("cwd", "")
            session_start_ts = ev.get("timestamp", "")

        elif ev_type == "model_change":
            current_model = ev.get("modelId") or current_model
            current_provider = ev.get("provider") or current_provider

        elif ev_type == "thinking_level_change":
            current_thinking_level = ev.get("thinkingLevel") or current_thinking_level

        elif ev_type == "message":
            msg = replacements.get(ev_id, ev.get("message", {}))
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", [])
            ts = ev.get("timestamp", "")

            if role == "user":
                user_texts: List[str] = []
                if isinstance(content, str):
                    user_texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_texts.append(item.get("text", ""))
                text_joined = "\n".join(t for t in user_texts if t).strip()
                if text_joined:
                    raw_messages.append({
                        "id": ev_id,
                        "role": "user",
                        "content": text_joined,
                        "timestamp": ts,
                    })
                    last_assistant_entry = None

            elif role == "assistant":
                mdl = msg.get("model") or current_model
                prv = msg.get("provider") or current_provider
                stop_reason = msg.get("stopReason", "")
                error_msg = msg.get("errorMessage")
                if error_msg and not content:
                    continue

                think_parts: List[str] = []
                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []

                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        ctype = item.get("type")
                        if ctype == "thinking":
                            t_str = item.get("thinking", "")
                            if t_str:
                                think_parts.append(t_str)
                        elif ctype == "text":
                            txt_str = item.get("text", "")
                            if txt_str:
                                text_parts.append(txt_str)
                        elif ctype == "toolCall":
                            tc_id = item.get("id") or f"call_{ev_id}_{len(tool_calls)}"
                            tc_name = item.get("name", "unknown")
                            tc_args = item.get("arguments", {})
                            tool_calls.append({
                                "id": tc_id,
                                "name": tc_name,
                                "arguments": tc_args,
                            })

                thinking_str = "\n".join(think_parts).strip()
                text_str = "\n".join(text_parts).strip()

                if not thinking_str and not text_str and not tool_calls:
                    continue

                entry = {
                    "id": ev_id or f"asst_{len(raw_messages)}",
                    "role": "assistant",
                    "model": mdl,
                    "provider": prv,
                    "thinking_level": current_thinking_level,
                    "thinking": thinking_str,
                    "text": text_str,
                    "tool_calls": tool_calls,
                    "tool_results": [],
                    "stop_reason": stop_reason,
                    "timestamp": ts,
                }
                raw_messages.append(entry)
                last_assistant_entry = entry

            elif role == "toolResult":
                tc_id = msg.get("toolCallId", "")
                t_name = msg.get("toolName", "tool")
                is_err = bool(msg.get("isError", False))
                tr_texts: List[str] = []
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                tr_texts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                tr_texts.append("[Image Output]")
                elif isinstance(content, str):
                    tr_texts.append(content)

                raw_tr_text = "\n".join(tr_texts)
                trunc_tr_text = truncate_middle(raw_tr_text, max_tool_output_chars)

                tr_obj = {
                    "id": ev_id or f"tr_{len(raw_messages)}",
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": t_name,
                    "isError": is_err,
                    "content": trunc_tr_text,
                    "raw_len": len(raw_tr_text),
                    "timestamp": ts,
                }
                raw_messages.append(tr_obj)
                if last_assistant_entry is not None:
                    last_assistant_entry["tool_results"].append(tr_obj)

    project_group = infer_project_group(session_cwd, rel_path)

    # -----------------------------------------------------------------------
    # [Layer 2 & Layer 3] 按 User Turn 切块，剥离前置/中途的模型配置与测试杂音，
    # 并将纪律纠偏（如“不要参考其他的目录”）内化合并，形成纯净的工程分析消息流
    # -----------------------------------------------------------------------
    user_blocks: List[List[Dict[str, Any]]] = []
    cur_block: List[Dict[str, Any]] = []
    for m in raw_messages:
        if m["role"] == "user":
            if cur_block:
                user_blocks.append(cur_block)
            cur_block = [m]
        else:
            if cur_block:
                cur_block.append(m)
    if cur_block:
        user_blocks.append(cur_block)

    normalized_messages: List[Dict[str, Any]] = [
        {
            "id": f"sys_{session_id}",
            "role": "system",
            "content": DEFAULT_CRACK_SYSTEM_PROMPT,
            "timestamp": session_start_ts,
        }
    ]

    crack_started = False
    dropped_irrelevant_turns = 0

    for blk_idx, blk in enumerate(user_blocks):
        u_msg = blk[0]
        u_text = u_msg["content"].strip()
        blk_assts = [x for x in blk[1:] if x["role"] == "assistant"]

        # 跳过因 API 502/401 报错而完全没有任何有效 Assistant 回复的空 Turn
        if not blk_assts:
            dropped_irrelevant_turns += 1
            continue

        # 特殊过滤：01a1577a Turn 0 误用 glm-5.3 仅跑 3 步就被用户 Abort 紧接着重发相同 Prompt
        if (
            blk_idx + 1 < len(user_blocks)
            and len(blk_assts) <= 3
            and user_blocks[blk_idx + 1][0]["content"].strip() == u_text
        ):
            dropped_irrelevant_turns += 1
            continue

        # 在首个真正的工程分析指令出现之前，丢弃一切配模型/装工具/hi 测试回合
        if not crack_started:
            if CRACK_TASK_START_REGEX.search(u_text):
                crack_started = True
                # 规范化初始根任务指令（例如将“我要破解这个软件，你看看需要安装什么工具”统一为标准全量工程分析目标）
                if "jigsawcard" in u_text.lower():
                    u_msg = copy.deepcopy(u_msg)
                    u_msg["content"] = (
                        "帮我破解 E:\\Download\\com.oakever.jigsawcard@arm64-v8a.xapk 游戏，"
                        "获取 资源，配置，代码 所有解密文件保存在本文件夹"
                    )
                elif "meowdoku" in u_text.lower():
                    u_msg = copy.deepcopy(u_msg)
                    u_msg["content"] = (
                        "帮我破解 E:\\Download\\com.oakever.meowdoku@arm64-v8a.xapk 游戏，"
                        "获取 资源，配置，代码 所有文件保存在本文件夹"
                    )
                blk = [u_msg] + blk[1:]
            else:
                dropped_irrelevant_turns += 1
                continue
        else:
            # 任务开启后，若中途穿插了修改 models.json / 排查隔壁报错 / 硬件闲聊 / 磁盘清理等无关指令，直接剥离！
            if is_irrelevant_user_turn(u_text):
                dropped_irrelevant_turns += 1
                continue

        # 若当前 Turn 是“过程催促/确认（继续、可以、补全结构）”或“不要参考其他的目录”纪律纠偏：
        # 1) 不把该句作为孤立 User Turn 暴露给 SFT；
        # 2) 若上一轮末尾 Assistant 消息仅仅是“停下来向用户汇报阶段进度并等待继续”的纯文本消息（无 tool_calls），
        #    将其移除，使前后工具调用无缝衔接为一条全自主执行流！
        if CONTINUATION_OR_CORRECTION_TURN_REGEX.search(u_text):
            if (
                normalized_messages
                and normalized_messages[-1]["role"] == "assistant"
                and not normalized_messages[-1].get("tool_calls")
            ):
                normalized_messages.pop()
            normalized_messages.extend(blk[1:])
        else:
            normalized_messages.extend(blk)

    asst_count = 0
    clean_asst_count = 0
    models_counter: Counter = Counter()

    for m in normalized_messages:
        if m["role"] == "assistant":
            asst_count += 1
            models_counter[m["model"]] += 1
            q = analyze_step_quality(
                thinking=m["thinking"],
                text=m["text"],
                tool_calls=m["tool_calls"],
                tool_results=m["tool_results"],
                session_cwd=session_cwd,
            )
            m["quality"] = q
            if q["is_clean"]:
                clean_asst_count += 1

    # 消息格式转换器（支持 strip_thinking 用于剥离历史步的 <think>）
    def to_openai_message(
        m: Dict[str, Any],
        override_thinking: Optional[str] = None,
        strip_thinking: bool = False,
    ) -> Dict[str, Any]:
        role = m["role"]
        if role in ("system", "user"):
            return {"role": role, "content": m["content"]}
        elif role == "tool":
            return {
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "name": m["name"],
                "content": m["content"],
            }
        elif role == "assistant":
            thk = "" if strip_thinking else (override_thinking if override_thinking is not None else m["thinking"])
            txt = m["text"]
            combined_content = f"<think>\n{thk}\n</think>\n{txt}".strip() if thk else txt

            out: Dict[str, Any] = {
                "role": "assistant",
                "content": combined_content,
            }
            if thk:
                out["reasoning_content"] = thk
            if m["tool_calls"]:
                out["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            return out
        return {"role": "user", "content": str(m)}

    def to_sharegpt_turn(
        m: Dict[str, Any],
        override_thinking: Optional[str] = None,
        strip_thinking: bool = False,
    ) -> List[Dict[str, str]]:
        role = m["role"]
        if role == "system":
            return [{"from": "system", "value": m["content"]}]
        elif role == "user":
            return [{"from": "human", "value": m["content"]}]
        elif role == "tool":
            return [{"from": "observation", "value": m["content"]}]
        elif role == "assistant":
            thk = "" if strip_thinking else (override_thinking if override_thinking is not None else m["thinking"])
            txt = m["text"]
            prefix = f"<think>\n{thk}\n</think>\n" if thk else ""
            turns: List[Dict[str, str]] = []
            if m["tool_calls"]:
                tc_blocks = []
                for tc in m["tool_calls"]:
                    tc_payload = json.dumps(
                        {"name": tc["name"], "arguments": tc["arguments"]},
                        ensure_ascii=False,
                    )
                    tc_blocks.append(f"<tool_call>\n{tc_payload}\n</tool_call>")
                full_val = f"{prefix}{txt}\n" + "\n".join(tc_blocks)
                turns.append({"from": "gpt", "value": full_val.strip()})
            else:
                turns.append({"from": "gpt", "value": f"{prefix}{txt}".strip()})
            return turns
        return []

    trajectories_openai: List[Dict[str, Any]] = []
    trajectories_sharegpt: List[Dict[str, Any]] = []
    gold_trajectories_sharegpt: List[Dict[str, Any]] = []
    step_windows: List[Dict[str, Any]] = []
    dpo_pairs: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    if asst_count == 0:
        return {
            "session_id": session_id,
            "cwd": session_cwd,
            "project_group": project_group,
            "line_count": line_count,
            "asst_count": 0,
            "clean_asst_count": 0,
            "dropped_irrelevant_turns": dropped_irrelevant_turns,
            "models_counter": {},
            "trajectories_openai": [],
            "trajectories_sharegpt": [],
            "gold_trajectories_sharegpt": [],
            "step_windows": [],
            "dpo_pairs": [],
            "diagnostics": [],
        }

    # -----------------------------------------------------------------------
    # 将整个项目会话作为 1 条完整的连续多轮轨迹（不再把 `继续`/`可以` 拆成无头碎片！）
    # -----------------------------------------------------------------------
    seg = normalized_messages
    seg_assts = [x for x in seg if x["role"] == "assistant"]
    dominant_model, _ = models_counter.most_common(1)[0]
    dominant_provider = seg_assts[0]["provider"]
    is_teacher = is_teacher_model(dominant_model, teacher_patterns)

    avg_score = sum(x["quality"]["score"] for x in seg_assts) / len(seg_assts)
    clean_ratio = sum(1 for x in seg_assts if x["quality"]["is_clean"]) / len(seg_assts)

    # 黄金提纯与弯路 DPO 挖掘对所有项目会话开放（从而也能从 Student 会话中挖掘 state_folding/detour DPO）
    gold_seg, pruned_dpos = purify_segment_into_gold(seg)
    gold_assts = [x for x in gold_seg if x["role"] == "assistant"]

    # 仅将 Teacher 模型的完整项目会话（DeepSeek / GLM）收入 SFT 轨迹库
    if is_teacher and len(gold_assts) >= 10:
        sample_id = f"traj::{session_id}"
        oai_msgs = [to_openai_message(m) for m in seg]
        sgpt_convs: List[Dict[str, str]] = []
        for m in seg:
            sgpt_convs.extend(to_sharegpt_turn(m))

        meta = {
            "sample_id": sample_id,
            "session_id": session_id,
            "session_file": rel_path,
            "project_group": project_group,
            "cwd": session_cwd,
            "dominant_model": dominant_model,
            "provider": dominant_provider,
            "is_teacher_model": is_teacher,
            "assistant_steps": len(seg_assts),
            "dropped_irrelevant_turns": dropped_irrelevant_turns,
            "clean_step_ratio": round(clean_ratio, 3),
            "avg_quality_score": round(avg_score, 2),
            "timestamp": session_start_ts,
        }

        trajectories_openai.append({
            "sample_id": sample_id,
            "session_file": rel_path,
            "messages": oai_msgs,
            "tools": DEFAULT_PI_TOOLS,
            "metadata": meta,
        })

        trajectories_sharegpt.append({
            "sample_id": sample_id,
            "session_file": rel_path,
            "conversations": sgpt_convs,
            "tools": json.dumps(DEFAULT_PI_TOOLS, ensure_ascii=False),
            "metadata": meta,
        })

        gold_sgpt_convs: List[Dict[str, str]] = []
        for gm in gold_seg:
            gold_sgpt_convs.extend(to_sharegpt_turn(gm))

        gold_id = f"gold_traj::{session_id}"
        gold_meta = dict(meta)
        gold_meta.update({
            "sample_id": gold_id,
            "raw_assistant_steps": len(seg_assts),
            "gold_assistant_steps": len(gold_assts),
            "pruned_or_folded_steps": len(seg_assts) - len(gold_assts),
        })
        gold_trajectories_sharegpt.append({
            "sample_id": gold_id,
            "session_file": rel_path,
            "conversations": gold_sgpt_convs,
            "tools": json.dumps(DEFAULT_PI_TOOLS, ensure_ascii=False),
            "metadata": gold_meta,
        })

    # 将剪枝/状态折叠捕获到的天然弯路对加入 DPO 数据集（强制附带 System + 根 User 任务指令）
    sys_msg = normalized_messages[0] if normalized_messages and normalized_messages[0]["role"] == "system" else None
    root_user_msg = next((m for m in normalized_messages if m["role"] == "user"), None)

    for pd in pruned_dpos:
        ctx_convs: List[Dict[str, str]] = []
        if sys_msg:
            ctx_convs.extend(to_sharegpt_turn(sys_msg))
        if root_user_msg:
            ctx_convs.extend(to_sharegpt_turn(root_user_msg))
        for u in pd["context_units"][-step_window_turns:]:
            if u["msg"]["role"] == "system" or (root_user_msg and u["msg"]["id"] == root_user_msg["id"]):
                continue
            ctx_convs.extend(to_sharegpt_turn(u["msg"], strip_thinking=True))
            for tr_m in u["tools"]:
                ctx_convs.extend(to_sharegpt_turn(tr_m))
        if ctx_convs and pd["chosen_msg"].get("thinking"):
            dpo_id = f"dpo_pruned::{session_id}::{pd['rejected_msg']['id']}"
            dpo_pairs.append({
                "sample_id": dpo_id,
                "session_file": rel_path,
                "conversations": ctx_convs,
                "chosen": to_sharegpt_turn(pd["chosen_msg"])[0],
                "rejected": to_sharegpt_turn(pd["rejected_msg"])[0],
                "tools": json.dumps(DEFAULT_PI_TOOLS, ensure_ascii=False),
                "metadata": {
                    "sample_id": dpo_id,
                    "session_id": session_id,
                    "session_file": rel_path,
                    "project_group": project_group,
                    "msg_id": pd["rejected_msg"]["id"],
                    "cwd": session_cwd,
                    "model": dominant_model,
                    "pathologies": [pd["reason"]],
                    "timestamp": pd["rejected_msg"].get("timestamp", ""),
                },
            })

    # -----------------------------------------------------------------------
    # 构建单步决策窗口样本 (Step-level Windows，剥离历史 <think> + 永久锚定根任务指令)
    # -----------------------------------------------------------------------
    # 注意：SFT 单步窗口基于黄金提纯后的 `gold_seg` 构建（若为 Teacher 会话），确保不把剪掉的弯路放进单步 SFT！
    source_for_steps = gold_seg if is_teacher else normalized_messages
    all_asst_indices = [idx for idx, m in enumerate(source_for_steps) if m["role"] == "assistant"]
    total_asst_in_session = len(all_asst_indices)
    asst_rank_map = {msg_idx: rank for rank, msg_idx in enumerate(all_asst_indices)}

    for idx, m in enumerate(source_for_steps):
        if m["role"] != "assistant":
            continue

        q = m.get("quality") or analyze_step_quality(
            thinking=m["thinking"],
            text=m["text"],
            tool_calls=m["tool_calls"],
            tool_results=m.get("tool_results", []),
            session_cwd=session_cwd,
        )
        msg_id = m["id"]
        mdl = m["model"]
        step_is_teacher = is_teacher_model(mdl, teacher_patterns)
        step_rank = asst_rank_map[idx]

        # 收集上下文窗口：强制锚定 [System Prompt, Root Crack User Prompt] + 最近 N 条交互历史（历史步剥离 <think>）
        history_slice = source_for_steps[max(0, idx - step_window_turns) : idx]
        while history_slice and history_slice[0]["role"] == "tool":
            history_slice = history_slice[1:]

        anchored_prefix: List[Dict[str, Any]] = []
        if sys_msg and (not history_slice or history_slice[0]["role"] != "system"):
            anchored_prefix.append(sys_msg)
        if root_user_msg and not any(hm.get("id") == root_user_msg["id"] for hm in history_slice):
            anchored_prefix.append(root_user_msg)

        full_history = anchored_prefix + [
            hm for hm in history_slice if hm["role"] != "system"
        ]
        # 确保紧接在 root_user_msg 之后的不是悬空的 toolResult
        cleaned_history: List[Dict[str, Any]] = []
        for hm in full_history:
            if hm["role"] == "tool" and (not cleaned_history or cleaned_history[-1]["role"] not in ("assistant", "tool")):
                continue
            cleaned_history.append(hm)

        if not cleaned_history:
            continue

        prompt_oai_msgs = [to_openai_message(hm, strip_thinking=True) for hm in cleaned_history]
        prompt_sgpt_convs: List[Dict[str, str]] = []
        for hm in cleaned_history:
            prompt_sgpt_convs.extend(to_sharegpt_turn(hm, strip_thinking=True))

        # 判定当前步所属阶段 (Stage) 与推荐过采样权重 (Sample Weight)
        if step_rank < 3:
            stage = "opening"
            sample_weight = 3
        elif step_rank >= max(3, total_asst_in_session - 3):
            stage = "closing"
            sample_weight = 3
        elif len(m["tool_calls"]) >= 2 or "script_verification" in q["strengths"] or any(
            tc["name"] in ("write", "edit") for tc in m["tool_calls"]
        ):
            stage = "key_decision"
            sample_weight = 2
        else:
            stage = "routine"
            sample_weight = 1

        # (A) SFT 单步正样本入库：严格要求来自 Teacher 模型（或 01a15c80 中 OpenGDS/Variant 解密成功且无报错的高分步骤）
        is_whitelisted_student_step = (
            "01a15c80" in session_id
            and 50 <= (step_rank + 1) <= 106
            and q["is_clean"]
            and len(m["thinking"]) >= 100
        )
        if q["is_clean"] and len(m["thinking"]) > 0 and (step_is_teacher or is_whitelisted_student_step):
            step_sample_id = f"step::{session_id}::{msg_id}"
            clean_thk = compact_ruminating_thinking(m["thinking"], 1800) if len(m["thinking"]) > 2400 else m["thinking"]
            target_oai = to_openai_message(m, override_thinking=clean_thk, strip_thinking=False)
            step_windows.append({
                "sample_id": step_sample_id,
                "session_file": rel_path,
                "messages": prompt_oai_msgs + [target_oai],
                "tools": DEFAULT_PI_TOOLS,
                "metadata": {
                    "sample_id": step_sample_id,
                    "session_id": session_id,
                    "session_file": rel_path,
                    "project_group": project_group,
                    "root_task": root_user_msg["content"][:120] if root_user_msg else "",
                    "msg_id": msg_id,
                    "step_rank": step_rank + 1,
                    "total_steps": total_asst_in_session,
                    "stage": stage,
                    "sample_weight": sample_weight,
                    "cwd": session_cwd,
                    "model": mdl,
                    "provider": m["provider"],
                    "is_teacher_model": step_is_teacher,
                    "quality_score": q["score"],
                    "strengths": q["strengths"],
                    "metrics": q["metrics"],
                    "timestamp": m["timestamp"],
                },
            })

    # (B) 遍历原始 normalized_messages 收集全部病态诊断与反刍/心算/跨目录 DPO 偏好对
    raw_asst_indices = [idx for idx, m in enumerate(normalized_messages) if m["role"] == "assistant"]
    for idx, m in enumerate(normalized_messages):
        if m["role"] != "assistant":
            continue
        q = m["quality"]
        if not q["pathologies"]:
            continue

        msg_id = m["id"]
        mdl = m["model"]
        diag_id = f"diag::{session_id}::{msg_id}"
        diagnostics.append({
            "sample_id": diag_id,
            "session_id": session_id,
            "session_file": rel_path,
            "project_group": project_group,
            "msg_id": msg_id,
            "cwd": session_cwd,
            "model": mdl,
            "provider": m["provider"],
            "quality_score": q["score"],
            "pathologies": q["pathologies"],
            "metrics": q["metrics"],
            "thinking_preview": truncate_middle(m["thinking"], 600),
            "tool_calls": [tc["name"] for tc in m["tool_calls"]],
            "timestamp": m["timestamp"],
        })

        if any(
            p in ("over_rumination_length", "over_rumination_doubt_loop", "mental_hex_checkmark")
            for p in q["pathologies"]
        ):
            compacted_think = compact_ruminating_thinking(m["thinking"], target_max_chars=1500)
            if len(compacted_think) < len(m["thinking"]) * 0.75:
                history_slice = normalized_messages[max(0, idx - step_window_turns) : idx]
                while history_slice and history_slice[0]["role"] == "tool":
                    history_slice = history_slice[1:]
                prompt_sgpt_convs = []
                if sys_msg:
                    prompt_sgpt_convs.extend(to_sharegpt_turn(sys_msg))
                if root_user_msg and not any(hm.get("id") == root_user_msg["id"] for hm in history_slice):
                    prompt_sgpt_convs.extend(to_sharegpt_turn(root_user_msg))
                for hm in history_slice:
                    if hm["role"] == "system":
                        continue
                    prompt_sgpt_convs.extend(to_sharegpt_turn(hm, strip_thinking=True))

                dpo_id = f"dpo::{session_id}::{msg_id}"
                chosen_turn = to_sharegpt_turn(m, override_thinking=compacted_think)[0]
                rejected_turn = to_sharegpt_turn(m)[0]
                dpo_pairs.append({
                    "sample_id": dpo_id,
                    "session_file": rel_path,
                    "conversations": prompt_sgpt_convs,
                    "chosen": chosen_turn,
                    "rejected": rejected_turn,
                    "tools": json.dumps(DEFAULT_PI_TOOLS, ensure_ascii=False),
                    "metadata": {
                        "sample_id": dpo_id,
                        "session_id": session_id,
                        "session_file": rel_path,
                        "project_group": project_group,
                        "msg_id": msg_id,
                        "cwd": session_cwd,
                        "model": mdl,
                        "pathologies": q["pathologies"],
                        "rejected_think_len": len(m["thinking"]),
                        "chosen_think_len": len(compacted_think),
                        "timestamp": m["timestamp"],
                    },
                })

    return {
        "session_id": session_id,
        "cwd": session_cwd,
        "project_group": project_group,
        "line_count": line_count,
        "asst_count": asst_count,
        "clean_asst_count": clean_asst_count,
        "dropped_irrelevant_turns": dropped_irrelevant_turns,
        "models_counter": dict(models_counter),
        "trajectories_openai": trajectories_openai,
        "trajectories_sharegpt": trajectories_sharegpt,
        "gold_trajectories_sharegpt": gold_trajectories_sharegpt,
        "step_windows": step_windows,
        "dpo_pairs": dpo_pairs,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# 增量数据集存储与原子更新引擎
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "gold_trajectories_sharegpt": "sft_gold_trajectories_sharegpt.jsonl",
    "trajectories_openai": "sft_trajectories_openai.jsonl",
    "trajectories_sharegpt": "sft_trajectories_sharegpt.jsonl",
    "step_windows": "sft_step_windows.jsonl",
    "dpo_pairs": "dpo_preference_pairs.jsonl",
    "diagnostics": "negative_diagnostics.jsonl",
}


def load_existing_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl_atomic(file_path: Path, records: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(file_path)


def write_llama_factory_dataset_info(output_dir: Path) -> Path:
    info = {
        "pi_sft_gold_trajectories": {
            "file_name": DATASET_FILES["gold_trajectories_sharegpt"],
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "tools": "tools",
            },
        },
        "pi_sft_trajectories_sharegpt": {
            "file_name": DATASET_FILES["trajectories_sharegpt"],
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "tools": "tools",
            },
        },
        "pi_sft_trajectories_openai": {
            "file_name": DATASET_FILES["trajectories_openai"],
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "observation_tag": "tool",
            },
        },
        "pi_sft_step_windows": {
            "file_name": DATASET_FILES["step_windows"],
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "observation_tag": "tool",
            },
        },
        "pi_dpo_preference_pairs": {
            "file_name": DATASET_FILES["dpo_pairs"],
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "tools": "tools",
            },
        },
    }
    info_path = output_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info_path


def run_sync_once(
    sessions_dir: Path,
    output_dir: Path,
    max_tool_output_chars: int = 10000,
    step_window_turns: int = 8,
    teacher_patterns: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "sync_state.json"
    patterns = teacher_patterns or DEFAULT_TEACHER_PATTERNS

    if state_path.exists() and not force_rebuild:
        try:
            with state_path.open("r", encoding="utf-8") as f:
                sync_state = json.load(f)
        except Exception:
            sync_state = {"version": 3, "files": {}}
    else:
        sync_state = {"version": 3, "files": {}}

    known_files: Dict[str, Dict[str, Any]] = sync_state.get("files", {})

    # [Layer 1 会话级准入过滤] 仅扫描并跟踪工程分析会话 (--D--crack-*--)
    disk_files: Dict[str, Path] = {}
    ignored_non_crack_files = 0
    if sessions_dir.exists():
        for p in sorted(sessions_dir.rglob("*.jsonl")):
            rel = str(p.relative_to(sessions_dir)).replace("\\", "/")
            if is_crack_session_path(rel):
                disk_files[rel] = p
            else:
                ignored_non_crack_files += 1

    new_rels: List[str] = []
    updated_rels: List[str] = []
    unchanged_rels: List[str] = []
    deleted_rels: List[str] = []
    current_fps: Dict[str, Dict[str, Any]] = {}

    for rel, p in disk_files.items():
        fp = compute_file_fingerprint(p)
        current_fps[rel] = fp
        old_entry = known_files.get(rel)
        if old_entry is None or force_rebuild:
            new_rels.append(rel)
        elif not is_fingerprint_equal(old_entry.get("fingerprint"), fp):
            updated_rels.append(rel)
        else:
            unchanged_rels.append(rel)

    for rel in list(known_files.keys()):
        if rel not in disk_files:
            deleted_rels.append(rel)

    dirty_rels: Set[str] = set(new_rels) | set(updated_rels) | set(deleted_rels)

    if not dirty_rels and not force_rebuild:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "NO_CHANGES",
            "elapsed_ms": elapsed_ms,
            "crack_session_files": len(disk_files),
            "ignored_non_crack_files": ignored_non_crack_files,
            "new_files": 0,
            "updated_files": 0,
            "unchanged_files": len(unchanged_rels),
            "deleted_files": 0,
            "dataset_counts": sync_state.get("dataset_counts", {}),
        }

    datasets_in_mem: Dict[str, List[Dict[str, Any]]] = {}
    for key, fname in DATASET_FILES.items():
        fpath = output_dir / fname
        if force_rebuild:
            datasets_in_mem[key] = []
        else:
            existing = load_existing_jsonl(fpath)
            datasets_in_mem[key] = [
                rec for rec in existing
                if rec.get("session_file") not in dirty_rels and is_crack_session_path(rec.get("session_file", ""))
            ]

    for rel in deleted_rels:
        known_files.pop(rel, None)

    to_process = new_rels + updated_rels
    for rel in to_process:
        p = disk_files[rel]
        parsed = parse_session_file(
            file_path=p,
            rel_path=rel,
            max_tool_output_chars=max_tool_output_chars,
            teacher_patterns=patterns,
            step_window_turns=step_window_turns,
        )
        for key in DATASET_FILES.keys():
            datasets_in_mem[key].extend(parsed[key])

        known_files[rel] = {
            "session_id": parsed["session_id"],
            "cwd": parsed["cwd"],
            "project_group": parsed["project_group"],
            "fingerprint": current_fps[rel],
            "line_count": parsed["line_count"],
            "asst_count": parsed["asst_count"],
            "clean_asst_count": parsed.get("clean_asst_count", 0),
            "dropped_irrelevant_turns": parsed.get("dropped_irrelevant_turns", 0),
            "models": parsed.get("models_counter", {}),
            "counts": {k: len(parsed[k]) for k in DATASET_FILES.keys()},
            "last_synced_at": datetime.now(timezone.utc).isoformat(),
        }

    dataset_counts: Dict[str, int] = {}
    for key, fname in DATASET_FILES.items():
        fpath = output_dir / fname
        recs = datasets_in_mem[key]
        save_jsonl_atomic(fpath, recs)
        dataset_counts[key] = len(recs)

    write_llama_factory_dataset_info(output_dir)

    total_asst_steps = sum(v.get("asst_count", 0) for v in known_files.values())
    total_clean_steps = sum(v.get("clean_asst_count", 0) for v in known_files.values())
    global_models: Counter = Counter()
    for v in known_files.values():
        for m_name, cnt in v.get("models", {}).items():
            global_models[m_name] += cnt

    sync_state = {
        "version": 3,
        "last_sync_time": datetime.now(timezone.utc).isoformat(),
        "sessions_dir": str(sessions_dir),
        "output_dir": str(output_dir),
        "crack_session_files": len(disk_files),
        "ignored_non_crack_files": ignored_non_crack_files,
        "total_assistant_steps": total_asst_steps,
        "total_clean_steps": total_clean_steps,
        "global_models": dict(global_models.most_common()),
        "dataset_counts": dataset_counts,
        "files": known_files,
    }
    tmp_state = state_path.with_suffix(".json.tmp")
    with tmp_state.open("w", encoding="utf-8") as f:
        json.dump(sync_state, f, ensure_ascii=False, indent=2)
    tmp_state.replace(state_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "SYNCED",
        "elapsed_ms": elapsed_ms,
        "crack_session_files": len(disk_files),
        "ignored_non_crack_files": ignored_non_crack_files,
        "new_files": len(new_rels),
        "updated_files": len(updated_rels),
        "unchanged_files": len(unchanged_rels),
        "deleted_files": len(deleted_rels),
        "dataset_counts": dataset_counts,
        "global_models": dict(global_models.most_common(15)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GoldThread · 增量提取 .pi 会话日志并同步生成黄金 SFT / DPO 训练数据集"
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSIONS_DIR,
        help=f"会话日志根目录 (默认: {DEFAULT_SESSIONS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"训练数据集输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-tool-chars",
        type=int,
        default=10000,
        help="单条工具输出保留的最大字符数，超出部分首尾截断 (默认: 10000)",
    )
    parser.add_argument(
        "--window-turns",
        type=int,
        default=8,
        help="单步决策样本保留的上下文消息窗口大小 (默认: 8)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="忽略增量状态，强制全量重新解析所有项目会话并重建数据集",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="开启持续监听模式，每隔 SECONDS 秒自动扫描并增量同步新会话 (例如 --watch 10)",
    )
    args = parser.parse_args()

    if args.watch <= 0:
        res = run_sync_once(
            sessions_dir=args.sessions_dir,
            output_dir=args.output_dir,
            max_tool_output_chars=args.max_tool_chars,
            step_window_turns=args.window_turns,
            force_rebuild=args.force_rebuild,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(
            f"[Watch 模式已启动] 正在监听 {args.sessions_dir} 下的工程分析会话 (每 {args.watch} 秒增量同步至 {args.output_dir})..."
        )
        first = True
        try:
            while True:
                res = run_sync_once(
                    sessions_dir=args.sessions_dir,
                    output_dir=args.output_dir,
                    max_tool_output_chars=args.max_tool_chars,
                    step_window_turns=args.window_turns,
                    force_rebuild=(args.force_rebuild and first),
                )
                first = False
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if res["status"] == "SYNCED":
                    print(
                        f"[{now_str}] 检测到变更并完成同步 (耗时 {res['elapsed_ms']}ms): "
                        f"新增={res['new_files']}, 更新={res['updated_files']}, "
                        f"跳过={res['unchanged_files']} | 样本统计={res['dataset_counts']}"
                    )
                else:
                    print(
                        f"[{now_str}] 无新增/变更项目会话 (扫描 {res['crack_session_files']} 个项目会话耗时 {res['elapsed_ms']}ms)"
                    )
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[Watch 模式已停止]")


if __name__ == "__main__":
    main()
