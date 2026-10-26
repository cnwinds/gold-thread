#!/usr/bin/env bash
# 用 vLLM 起规范推理服务：reasoning parser 自动拆分 think 标签 + 结构化 tool_calls
#
# 用法：
#   bash serve_vllm.sh [MODEL_PATH] [PORT] [API_KEY] [SERVED_NAME]
# 例：
#   bash serve_vllm.sh /data/models/qwen3-8b-merged 8000 gt-2026 qwen3-8b-goldthread-sft
#
# 环境变量可覆盖：
#   MAX_LEN=32768  GPU_UTIL=0.90  REASONING_PARSER=qwen3  TOOL_PARSER=hermes
set -euo pipefail

MODEL="${1:-/data/models/qwen3-8b-merged}"
PORT="${2:-8000}"
KEY="${3:-gt-2026}"
SERVED_NAME="${4:-${API_MODEL_NAME:-qwen3-8b-goldthread-sft}}"

MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"

# reasoning parser 名称随 vLLM 版本变化：
#   vLLM >= 0.10 提供 qwen3 专用 parser；老版本用通用的 deepseek_r1（同为 think/think 标签）
if [ -z "${REASONING_PARSER:-}" ]; then
  if vllm serve --help 2>/dev/null | grep -q "qwen3"; then
    REASONING_PARSER="qwen3"
  else
    REASONING_PARSER="deepseek_r1"
  fi
fi

echo "[serve_vllm] model=$MODEL"
echo "[serve_vllm] served-name=$SERVED_NAME  port=$PORT  max-len=$MAX_LEN"
echo "[serve_vllm] reasoning-parser=$REASONING_PARSER  tool-parser=$TOOL_PARSER"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --api-key "$KEY" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
  --kv-cache-dtype auto