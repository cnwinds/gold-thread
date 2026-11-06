#!/usr/bin/env bash
# 安装/修复训练环境。可重复执行：版本已对的包直接跳过。
#   bash setup_env.sh                 # 用 PATH 里的 python
#   PYTHON=/path/to/python bash setup_env.sh
# 版本组合在 2×A800 / CUDA 12.6 / Ubuntu 22.04 上实测通过。
set -uo pipefail

PY=${PYTHON:-python}
LF_DIR=${LF_DIR:-/opt/LLaMA-Factory}
ROOT=$(cd "$(dirname "$0")" && pwd)

# 发行包名=版本（与 autotrain.py 的 PINNED 保持一致）
PINS=(
  "transformers=5.8.0"
  "accelerate=1.11.0"
  "peft=0.18.1"
  "trl=0.24.0"
  "datasets=4.0.0"
  "deepspeed=0.19.9"
)

log()  { echo -e "\033[36m[setup]\033[0m $*"; }
warn() { echo -e "\033[33m[setup][WARN]\033[0m $*"; }
die()  { echo -e "\033[31m[setup][FAIL]\033[0m $*"; exit 1; }
ver()  { "$PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null; }

"$PY" -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>/dev/null \
  || warn "期望 Python 3.12，当前 $("$PY" -V 2>&1)（通常也能跑，注意预编译轮子的匹配）"

# ---------- 网络：PyPI 不通就用清华镜像，GitHub 不通就找代理 ----------
if ! curl -s -m 6 -o /dev/null https://pypi.org/simple/pip/; then
  export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
  log "PyPI 不可达，使用清华镜像"
fi
export PIP_DEFAULT_TIMEOUT=120
GH_PREFIX=""
if ! curl -s -m 6 -o /dev/null https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md; then
  for p in https://ghfast.top https://gh-proxy.com https://ghproxy.net; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$p/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md")
    if [ "$code" = "200" ]; then GH_PREFIX="$p/"; log "GitHub 走代理 $p"; break; fi
  done
fi
grep -q HF_ENDPOINT ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# ---------- torch ----------
TV=$(ver torch)
if [ -z "$TV" ]; then
  log "安装 torch 2.7.1+cu126"
  "$PY" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
    || die "torch 安装失败。手动装与驱动匹配的 torch 2.7.1 后重跑"
elif [[ "$TV" != 2.7.1* ]]; then
  warn "torch 是 $TV（验证版本 2.7.1）。不自动替换 torch，避免弄坏 CUDA 栈"
fi

# ---------- LLaMA-Factory（必须 main：qwen3_8 模板与 qwen3_5 架构只在 main 上）----------
if ! "$PY" -c "import llamafactory" 2>/dev/null; then
  if [ ! -d "$LF_DIR/.git" ]; then
    log "克隆 LLaMA-Factory 到 $LF_DIR"
    git clone --depth 1 "${GH_PREFIX}https://github.com/hiyouga/LLaMA-Factory.git" "$LF_DIR" \
      || git clone --depth 1 https://gitee.com/mirrors/LLaMA-Factory.git "$LF_DIR" \
      || die "LLaMA-Factory 克隆失败"
  fi
  # --no-deps：否则 transformers 会被换成别的版本
  (cd "$LF_DIR" && "$PY" -m pip install -e ".[metrics]" --no-deps) || die "LLaMA-Factory 安装失败"
fi

# ---------- 钉版本的包（LLaMA-Factory 之后装，防止被覆盖）----------
need=()
for kv in "${PINS[@]}"; do
  pkg=${kv%%=*}; want=${kv#*=}
  [ "$(ver "$pkg")" = "$want" ] || need+=("$pkg==$want")
done
if [ ${#need[@]} -gt 0 ]; then
  log "安装 ${need[*]}"
  "$PY" -m pip install "${need[@]}" || die "安装失败：${need[*]}"
fi
if [ "$(ver transformers)" != "5.8.0" ] && ! "$PY" -m pip install transformers==5.8.0; then
  log "PyPI 没有 transformers 5.8.0，改从 GitHub 安装"
  "$PY" -m pip install "git+${GH_PREFIX}https://github.com/huggingface/transformers.git@v5.8.0" \
    || die "transformers 5.8.0 安装失败"
fi

# ---------- 线性注意力 kernel（27B 必须）----------
if ! "$PY" -c "import fla, causal_conv1d" 2>/dev/null; then
  log "安装 flash-linear-attention / causal-conv1d"
  "$PY" -m pip install "flash-linear-attention==0.4.2" causal-conv1d --no-build-isolation \
    || die "flash-linear-attention / causal-conv1d 安装失败"
fi

# ---------- flash-attn（可选；失败时 autotrain 自动改用 sdpa）----------
if ! "$PY" -c "import flash_attn" 2>/dev/null; then
  log "尝试安装 flash-attn（最多 15 分钟，开始源码编译说明没有匹配的预编译轮子）"
  MAX_JOBS=8 timeout 900 "$PY" -m pip install flash-attn --no-build-isolation \
    || warn "flash-attn 没装上：训练会自动改用 sdpa（16K 附近多占约 3-5GB）"
fi

# ---------- 收尾 ----------
"$PY" -m pip install -q pyyaml psutil >/dev/null 2>&1 || true
log "版本核对："
"$PY" "$ROOT/autotrain.py" env || die "环境检查未通过，见上面的 [FAIL]"
log "完成"
