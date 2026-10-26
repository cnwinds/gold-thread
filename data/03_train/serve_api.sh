#!/usr/bin/env bash
# 把 SFT 合并模型起成 OpenAI 兼容 API（LLaMA-Factory 自带服务，hf 后端）
# 用法：bash serve_api.sh [端口] [API KEY]
set -euo pipefail

PORT="${1:-8000}"
KEY="${2:-gt-2026}"
MODEL="${MODEL:-/data/models/qwen3-8b-merged}"

export API_HOST=0.0.0.0
export API_PORT="$PORT"
export API_MODEL_NAME="${API_MODEL_NAME:-qwen3-8b-goldthread-sft}"
export API_KEY="$KEY"
export DISABLE_VERSION_CHECK=1

exec llamafactory-cli api \
  --model_name_or_path "$MODEL" \
  --template qwen3 \
  --infer_backend huggingface \
  --infer_dtype bfloat16 \
  --trust_remote_code
