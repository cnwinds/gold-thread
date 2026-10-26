#!/usr/bin/env bash
# 用 vLLM 提供规范推理服务（支持 reasoning parser，自动拆分 thinking → reasoning_content）
# 用法：bash serve_vllm.sh [MODEL_PATH] [PORT] [API_KEY]
set -euo pipefail

MODEL="${1:-/data/models/qwen3-8b-merged}"
PORT="${2:-8000}"
KEY="${3:-gt-2026}"

# vLLM 对 Qwen3 模型会自动识别 qwen3 chat template，
# 加 --reasoning-parser deepseek_r1 把防腐剂标签拆分成 reasoning_content
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --api-key "$KEY" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --kv-cache-dtype auto
  # 注：若 48G 卡 256K 显存紧张，在以上参数基础上加：
  #   --max-model-len 65536 --kv-cache-dtype fp8
