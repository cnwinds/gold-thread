# START HERE · GoldThread 训练包

> 最短可执行路径。完整原理/排障见 `handoff.md`，实跑记录与踩坑见 `RESULT.md`。
> 本文件同时适用于 **8B 冒烟验证** 和 **27B 正式训练**，两条路径在第 5/6 节分开。

---

## 0. 放置位置

```bash
mkdir -p /data/goldthread
tar xzf 03_train_*.tar.gz -C /data/goldthread
cd /data/goldthread/03_train
```

目录里应该是（**实际训练用的是 `.v2` 文件，不是无后缀的旧版**）：

```
03_train/
│
├── 【数据】真正被 YAML 引用的文件
│   ├── sft_step_windows_sharegpt.train.v2.jsonl   ← SFT 训练集 548 条
│   ├── sft_step_windows_sharegpt.eval.v2.jsonl    ← SFT 验证集 221 条（整项目留出）
│   ├── sft_gold_trajectories_sharegpt.*.v2.jsonl  ← 完整长轨迹（可选混训）
│   ├── dpo_preference_pairs.train.v2w.jsonl       ← DPO 训练（窗口裁剪，48G 首选）
│   ├── dpo_preference_pairs.eval.v2w.jsonl        ← DPO 验证
│   ├── dpo_preference_pairs.train.v2.jsonl        ← DPO 训练（全窗口，显存充足时用）
│   ├── dpo_preference_pairs.eval.v2.jsonl
│   └── dataset_info.json                          ← 必须；LLaMA-Factory 靠它认数据集
│
├── 【配置】8B 冒烟
│   ├── train_sft_qwen3_8b.yaml
│   ├── train_dpo_qwen3_8b.yaml
│   ├── train_sft_smoke_qwen3_8b.yaml              ← 200 条
│   ├── train_dpo_smoke_qwen3_8b.yaml              ← 50 对
│   └── eval_sft_qwen3_8b.yaml                     ← 训练后单独跑 eval
│
├── 【配置】27B 正式
│   ├── train_sft.yaml
│   ├── train_sft_qlora.yaml                       ← 4bit 备选
│   └── train_dpo.yaml
│
├── 【脚本】
│   ├── check_env.py / check_data.py               ← 上机自检
│   ├── train_lowmem.py                            ← 27B SFT/eval 启动器（替代 llamafactory-cli train）
│   ├── train_monitor.py                           ← 训练监控（OOM 秒级诊断）
│   ├── smoke_infer_merged.py                      ← 合并后推理冒烟
│   ├── verify_longctx.py / verify_longctx_api.py  ← 长上下文验收
│   ├── probe_24g.py                               ← 小显存可行性探测
│   ├── prep_dpo_window.py                         ← DPO 窗口裁剪
│   ├── serve_api.sh / serve_vllm.sh               ← 起推理服务
│   └── reasoning_proxy.py                         ← reasoning 拆分过渡代理
│
└── 【文档】README.md / handoff.md / RESULT.md / stats.json / index.json
```

---

## 1. 网络（国内机器必做）

```bash
export GH_PROXY=https://ghfast.top
for p in https://ghfast.top https://gh-proxy.com https://ghproxy.net; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 \
    "$p/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md")
  echo "$p -> $code"
  [ "$code" = "200" ] && export GH_PROXY=$p && break
done
git config --global url."$GH_PROXY/https://github.com/".insteadOf "https://github.com/"

mkdir -p ~/.pip && cat > ~/.pip/pip.conf <<'INI'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
INI

echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
```

---

## 2. 装训练栈

```bash
# 2.1 钉死三个关键版本（顺序很重要）
pip install "transformers==5.8.0" "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0" --no-cache-dir

# 2.2 LLaMA-Factory（必须 main 分支：qwen3_8 模板 / qwen3_5 架构支持都在 main）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory
cd /opt/LLaMA-Factory && pip install -e ".[metrics]" --no-deps

# 2.3 Liger Kernel —— 【必须】8B 配置里 enable_liger_kernel: true 是硬依赖
#     不装，第一个 step 就在 cross_entropy OOM（8.7K token × bs2 的 fp32 logits 要 10.1GB）
pip install "liger-kernel>=0.8,<0.9"

# 2.4 flash-attn（装不上不影响开训，YAML 里 fa2 改 sdpa 即可）
pip install flash-attn --no-build-isolation || echo "[warn] flash-attn 失败 → 改 YAML: flash_attn: sdpa"

# 2.5 bitsandbytes（仅 QLoRA 4bit / DPO ref_model 8bit 量化需要，8B 纯 SFT 可不装）
pip install bitsandbytes

# 2.6 线性注意力 kernel —— 27B【必装】：48 层 linear_attention 没有它会退回 PyTorch 实现
#     （日志出现 "fast path is not available ... Falling back to torch implementation"）
pip install flash-linear-attention causal-conv1d --no-build-isolation

# 2.7 验收
llamafactory-cli env
python -c "import transformers;print('transformers',transformers.__version__)"      # 期望 5.8.x
python -c "import liger_kernel;print('liger',liger_kernel.__version__)"   || echo "[8B 必装 / 27B 可忽略] liger_kernel 未安装"
```

> `transformers` 不是 5.8.x = 被 LLaMA-Factory 依赖覆盖了 → 重装 2.1 并保留 `--no-deps`。
> `liger_kernel` 无法 import → 走 8B 路径必须重装 2.3；只跑 27B 可以忽略（27B 用 `enable_liger_kernel: false`）。

---

## 3. 下模型

**8B（冒烟用，≈16GB）**

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B
du -sh /data/models/Qwen3-8B          # 期望 ≈16G
```

**27B（正式用，≈55GB，磁盘需 ≥70GB 可用）**

```bash
df -h /data                           # 确认可用 ≥70GB
modelscope download --model Qwen/Qwen3.8-27B --local_dir /data/models/Qwen3.8-27B
du -sh /data/models/Qwen3.8-27B       # 期望 ≈55G
```

下完**必须**验证三件事：

```bash
python - <<'PY'
import json, pathlib
d = pathlib.Path("/data/models/Qwen3.8-27B")   # 8B 换成 /data/models/Qwen3-8B
cfg = json.load(open(d / "config.json"))
print("architectures:", cfg.get("architectures"))
print("model_type   :", cfg.get("model_type"))
print("max_position :", cfg.get("max_position_embeddings"), "| rope:", cfg.get("rope_scaling"))
idx = d / "model.safetensors.index.json"
if idx.exists():
    want = set(json.load(open(idx))["weight_map"].values())
    have = {p.name for p in d.glob("*.safetensors")}
    print("缺失分片:", sorted(want - have) or "无（完整）")
PY
```

---

## 4. 训练前自检（CPU 即可，停在这步也行）

```bash
cd /data/goldthread/03_train

# 8B
python check_env.py --model /data/models/Qwen3-8B --data . --cutoff 32768
python check_data.py --model /data/models/Qwen3-8B --template qwen3

# 27B
python check_env.py --model /data/models/Qwen3.8-27B --data . --cutoff 16384
python check_data.py --model /data/models/Qwen3.8-27B --template qwen3_8
```

**验收（必须全过才开训）**

- `check_env` 无 `[FAIL]`；`liger_kernel` 那行不能有 `[WARN]`
- `check_data` 里每条样本 `&lt;think&gt;×1`、`&lt;/think&gt;×1`
- `check_env` 的长度分布确认 `cutoff_len` 覆盖 max（8B:32768 / 27B:16384）

> `&lt;think&gt;` 与 `&lt;/think&gt;` 不是 1 对 → 停，先解决模板问题（8B 用 `qwen3`，27B 用 `qwen3_8`）。

---

## 5. 路径 A：8B 冒烟验证全链路（48G 卡，约 2 小时）

目的：先用同族小模型把「数据 → SFT → 合并 → 工具调用 → DPO」全部验通，再上 27B。

```bash
cd /data/goldthread/03_train

# 5.1 冒烟训练（200 条 × 1 epoch，≈20 min）
python train_monitor.py --timeout-hours 1 --log logs/smoke_sft.log -- \
  llamafactory-cli train train_sft_smoke_qwen3_8b.yaml

# 5.2 全量训练（548 条 × 3 epoch ≈207 步，≈1h15m）
python train_monitor.py --timeout-hours 3 --log logs/full_sft.log -- \
  llamafactory-cli train train_sft_qwen3_8b.yaml

# 5.3 单独跑 eval（训练时 eval_strategy 已关，因为 eval 不走 Liger 会 OOM）
python train_monitor.py --timeout-hours 1 --log logs/eval_sft.log -- \
  llamafactory-cli train eval_sft_qwen3_8b.yaml
```

**5.4 合并 LoRA**

```bash
llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3-8B \
  --adapter_name_or_path /data/saves/qwen3-8b-goldthread-sft \
  --template qwen3 --finetuning_type lora \
  --export_dir /data/models/qwen3-8b-merged
```

**5.5 推理冒烟（必须看到 `&lt;think&gt;…&lt;/think&gt;` + `&lt;tool_call&gt;`）**

```bash
python smoke_infer_merged.py --model /data/models/qwen3-8b-merged \
  --n 5 --max-new-tokens 1024 --sample
# 期望：5/5 有 think + tool_call，多数 JSON 可解析
# 注意：必须开采样（temp .6 / top_p .95 / top_k 20），greedy 会陷长思考不收敛
```

**5.6 DPO**（`train_dpo_qwen3_8b.yaml` 以 `.../qwen3-8b-merged` 为 base 训新 LoRA，ref = 关掉 adapter 的 SFT 模型，依赖 5.4 产物）

```bash
python train_monitor.py --timeout-hours 1 --log logs/dpo.log -- \
  llamafactory-cli train train_dpo_qwen3_8b.yaml
```

**5.7 起推理服务（最后一步，会常驻占显存）**

```bash
# 推荐：vLLM（规范的 reasoning parser + tool parser）
bash serve_vllm.sh /data/models/qwen3-8b-merged 8000 gt-2026

# 或用 LLaMA-Factory 自带 API（无 reasoning 拆分，需配 reasoning_proxy.py）
# bash serve_api.sh 8000 gt-2026
# python reasoning_proxy.py --upstream http://localhost:8000/v1 --port 9000 &
```

验证三条：

```bash
curl -s localhost:8000/v1/models | head -c 200

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-goldthread-sft","messages":[{"role":"user","content":"列出当前目录文件"}],
       "tools":[{"type":"function","function":{"name":"bash","description":"run shell",
       "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
       "max_tokens":256}' \
  | python -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];\
print('reasoning:',(m.get('reasoning_content') or '')[:80]);\
print('tool_calls:',m.get('tool_calls'))"
```

**期望**：`tool_calls` 非空且 `function.name=="bash"`；用 vLLM 时 `reasoning_content` 有内容。

---

## 6. 路径 B：27B 正式训练（单卡 80G）

> 27B 有现成配置 **`train_sft.yaml` / `train_dpo.yaml`**，不用改 8B 的 YAML。
> 关键差异（已预置好）：`template: qwen3_8`、`enable_liger_kernel: false`、`lora_target: mlp`、`cutoff_len: 16384`。
>
> **27B 的 SFT / eval 必须用 `python train_lowmem.py <yaml>` 启动，不要用 `llamafactory-cli train`。**
> 原因：词表 248320 且拿不到 Liger 融合 CE，原生路径对 16K 全部位置算 fp32 logits ≈ 40GB，
> 加 54G 权重在单卡 80G 上必 OOM（实测 16K 样本即炸）。`train_lowmem.py` 只对 `mask_history`
> 留下的监督位置算 logits，loss/梯度与原生数学等价，YAML 与命令行参数完全不变。
> 启动后日志前 3 步会打印 `[lowmem] seq_len=… 监督位置=… 峰值显存=…`，用来确认补丁生效。

```bash
cd /data/goldthread/03_train

# 6.1 SFT（548 条 × 3 epoch）
#     训练中 eval_strategy 已设 "no"（27B 显存紧，eval 会额外占激活）
#     机器上有多张卡时务必限定单卡，否则 HF 会退化成 DataParallel（脚本会直接拒绝启动）
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 5 --log logs/27b_sft.log -- \
  python train_lowmem.py train_sft.yaml
#     多卡 DDP：python train_monitor.py ... -- torchrun --nproc_per_node 2 train_lowmem.py train_sft.yaml

# 6.1b 训完单独跑 eval（bs=1，拿到 eval_loss）
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 1 --log logs/27b_eval.log -- \
  python train_lowmem.py eval_sft.yaml

# 6.2 合并（CPU 内存峰值约 60GB）
llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3.8-27B \
  --adapter_name_or_path /data/saves/qwen38-27b-goldthread-sft \
  --template qwen3_8 --finetuning_type lora \
  --export_dir /data/models/Qwen3.8-27B-merged

# 6.3 推理冒烟
python smoke_infer_merged.py --model /data/models/Qwen3.8-27B-merged \
  --n 3 --max-new-tokens 1024 --sample

# 6.4 长上下文验收（不达标就不要对外宣称"保住 256K"）
python verify_longctx.py --model /data/models/Qwen3.8-27B-merged \
  --lengths 8192,32768,65536

# 6.5 DPO（可选；依赖 6.2 的 merged 产物；同样必须走 train_lowmem.py）
#     先冒烟：命令行追加 max_samples=20（覆盖 YAML），看到 [lowmem-dpo] 日志与峰值显存再跑全量
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 1 --log logs/27b_dpo_smoke.log -- \
  python train_lowmem.py train_dpo.yaml max_samples=20 output_dir=/data/saves/qwen38-27b-dpo-smoke
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 3 --log logs/27b_dpo.log -- \
  python train_lowmem.py train_dpo.yaml

# 6.6 合并 DPO 适配器：base 是 SFT 合并模型（不是原始 Qwen3.8-27B）
llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3.8-27B-merged \
  --adapter_name_or_path /data/saves/qwen38-27b-goldthread-dpo \
  --template qwen3_8 --finetuning_type lora \
  --export_dir /data/models/Qwen3.8-27B-dpo-merged
```

**27B DPO 显存方案**：`train_dpo.yaml` 以 `Qwen3.8-27B-merged`（SFT 合并模型）为 base 训新的 DPO LoRA，
不设 `ref_model`——LLaMA-Factory 对「LoRA + 无 ref_model」用同一模型 `disable_adapter()` 做 ref 前向，
ref 就是 SFT 模型，不再额外加载 8bit ref。显存 ≈ bf16 权重 54G + 激活；`train_lowmem.py` 自动加载
`lowmem_dpo.py`，policy / ref 都只对 response 位置算 logits 并分块升 fp32（原生全量 logits ≈ 40G 是 OOM 主因）。
日志前 3 步打印 `[lowmem-dpo] … 监督位置=… 峰值显存=…` 确认补丁生效。
OOM 时按顺序：`LOWMEM_DPO_CHUNK=512` → `cutoff_len 8192 → 6144 → 4096` → `eval_strategy: "no"`。

**27B 训练首步必看**：`[lowmem]` 日志的峰值显存 / `nvidia-smi` 峰值 >78GB，先确认 2.6 的线性注意力 kernel 已装，
再把 `cutoff_len` 降到 12288（仍覆盖 p99）。

---

## 7. 换卡 / 排障速查

| 现象 | 处置 |
| :--- | :--- |
| 首步就 OOM（8B） | 确认装了 `liger-kernel`；`enable_liger_kernel: true`；`cutoff_len` 降档 |
| 首步就 OOM（27B） | 确认用的是 `python train_lowmem.py` 而不是 `llamafactory-cli train`；装 `flash-linear-attention` / `causal-conv1d`；`cutoff_len` 降档 |
| 训练中 eval 阶段 OOM | 正常 —— eval 不吃 Liger 融合 CE。设 `eval_strategy: "no"`，训完单独跑 `eval_*.yaml` |
| 进程突然消失、日志无 traceback | 大概率系统 OOM Killer（SIGKILL-9）。`train_monitor.py` 会直接告诉你；处置：加内存/swap |
| loss 前 100 步不降 | 模板错（think 重复注入）→ 重跑 `check_data.py`，8B 用 `qwen3`、27B 用 `qwen3_8` |
| 输出只有 tool_call 没有 think | 历史步参与了 loss → 确认 `mask_history: true` |
| DPO 后模型崩（乱码/think 不闭合） | ref 不是 SFT 模型：DPO 的 `model_name_or_path` 必须是 SFT 合并目录（ref = 关 adapter 后的它）→ 见 RESULT.md 的 DPO 根因 |
| 客户端看到 `&lt;think&gt;` 原文 | 服务层缺 reasoning parser → 用 `serve_vllm.sh`（含 `--reasoning-parser`） |
| flash-attn 编译失败 | 改 `flash_attn: sdpa`，别在编译上耗时间 |

---

**深入阅读**：`handoff.md` §3 架构核实 · §4.4 目录规范 · §6.3 OOM 阶梯 · §9 显存精算 · §10 推理机交接 · §11 排障表