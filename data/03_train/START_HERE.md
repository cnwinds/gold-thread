# START HERE · GoldThread 训练集

> 这个包只含**训练需要的东西**（19MB）。完整说明见 `handoff.md`，本文件是最短启动路径。

## 0. 放置位置

```bash
mkdir -p /data/goldthread/03_train
# 把本包内容解压到 /data/goldthread/03_train/
```

目录里应该有：

```
03_train/
├── sft_step_windows_sharegpt.train.jsonl   ← SFT 训练集
├── sft_step_windows_sharegpt.eval.jsonl    ← SFT 验证集（整项目留出）
├── dpo_preference_pairs.train.jsonl        ← DPO 训练集
├── dpo_preference_pairs.eval.jsonl
├── sft_gold_trajectories_sharegpt.*.jsonl  ← 完整长轨迹（可选混训）
├── dataset_info.json                       ← 必须在，LLaMA-Factory 靠它识别数据集
├── train_sft_qwen3_8b.yaml                 ← 【当前阶段主配置】8B 全量
├── train_dpo_qwen3_8b.yaml                 ← 8B 的 DPO
├── train_sft_smoke_qwen3_8b.yaml           ← 8B 冒烟（200 条）
├── train_dpo_smoke_qwen3_8b.yaml           ← DPO 冒烟（50 对）
├── train_sft.yaml / train_dpo.yaml         ← 27B 正式配置（后阶段用）
├── train_sft_qlora.yaml                    ← 27B 的 4bit 备选
├── check_env.py / check_data.py / train_monitor.py   ← 训练前自检 + 进程监控（推荐）
├── verify_longctx.py / verify_longctx_api.py / probe_24g.py
└── handoff.md                              ← 完整文档（架构、显存、排障、推理机交接）
```

## 1. 网络（国内机器必做）

```bash
export GH_PROXY=https://ghfast.top
for p in https://ghfast.top https://gh-proxy.com https://ghproxy.net; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 \
    "$p/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md")
  [ "$code" = "200" ] && export GH_PROXY=$p && break
done
git config --global url."$GH_PROXY/https://github.com/".insteadOf "https://github.com/"

mkdir -p ~/.pip && cat > ~/.pip/pip.conf <<'INI'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
INI
```

## 2. 装训练栈（版本是硬约束）

```bash
pip install "transformers==5.8.0" "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0" --no-cache-dir
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory
cd /opt/LLaMA-Factory && pip install -e ".[metrics]" --no-deps
pip install flash-attn --no-build-isolation || echo "装不上就改 YAML 里的 sdpa"
llamafactory-cli env
```

## 3. 下模型（8B，16GB）

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B
du -sh /data/models/Qwen3-8B          # 期望 ≈16G
```

## 4. 训练前自检（CPU 即可，停在这步也行）

```bash
cd /data/goldthread/03_train
python check_env.py --model /data/models/Qwen3-8B --data .
python check_data.py --model /data/models/Qwen3-8B --template qwen3
```

**验收**：`check_env` 无 FAIL；`check_data` 每条样本 ` thinking×1  ×1`。
不是 1 对就停下来解决模板问题，别开训。

## 5. 训练（推荐用 `train_monitor.py`，OOM 秒级诊断，进程退出立刻知道）

```bash
cd /data/goldthread/03_train

# 冒烟：200 条 × 1 epoch，约 20 分钟
python train_monitor.py --timeout-hours 1 --log logs/smoke_monitored.log -- \
  llamafactory-cli train /data/goldthread/03_train/train_sft_smoke_qwen3_8b.yaml

# 全量：548 条 × 3 epoch
python train_monitor.py --timeout-hours 3 --log logs/full_monitored.log -- \
  llamafactory-cli train /data/goldthread/03_train/train_sft_qwen3_8b.yaml
```

如果你习惯裸跑，也能直接用 `llamafactory-cli train ...`，但遇到 OOM 或被系统 kill 时不会自动通知。

## 6. 合并 + 推理验证

```bash
llamafactory-cli export --model_name_or_path /data/models/Qwen3-8B \
  --adapter_name_or_path /data/saves/qwen3-8b-goldthread-sft \
  --template qwen3 --finetuning_type lora \
  --export_dir /data/models/qwen3-8b-merged

vllm serve /data/models/qwen3-8b-merged --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --enable-auto-tool-choice \
  --tool-call-parser hermes --port 8000
```

然后验工具调用（handoff §10.4 第三条），必须能解析出结构化 `tool_calls`。

## 7. DPO（可选但建议，推荐用 `train_monitor.py`）

```bash
python train_monitor.py --timeout-hours 1 --log logs/dpo_monitored.log -- \
  llamafactory-cli train /data/goldthread/03_train/train_dpo_qwen3_8b.yaml
```

## 换 27B 时只改三处

```diff
- model_name_or_path: /data/models/Qwen3-8B
+ model_name_or_path: /data/models/Qwen3.8-27B
- template: qwen3
+ template: qwen3_8
- output_dir: /data/saves/qwen3-8b-goldthread-sft
+ output_dir: /data/saves/qwen38-27b-goldthread-sft
```
另需：`enable_liger_kernel: false`、`lora_target: mlp`（27B 保长上下文）、不设 deepspeed。

---

**遇到问题先查 `handoff.md`**：§3 架构核实、§4.4 目录规范、§6.0.1 执行顺序、
§11 排障表、§10 推理机交接。