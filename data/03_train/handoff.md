# 数据交接说明（handoff）

> 由 GoldThread `stage3` 自动生成，数据统计取自本次运行的真实产物。
> 数据集目录：`data/03_train`

**目标训练环境**：1 × NVIDIA A800 80GB（Ampere / SM80 / bf16） · Ubuntu 22.04 · CUDA 12.6 · PyTorch 2.7.1 · Python 3.12

---

## 0. 这批数据是什么

从 pi 会话日志提纯出的 **Agent 工程分析轨迹**（解包 / 反编译 / 资源与配置表还原 / 二进制格式破解）。
三方分工：

- **程序**：机械去噪、静默执行、命令规范化、窗口拼装、血缘记录、切分
- **模型**：判断删哪些步、接缝是否平滑、整条会话值不值得训练、偏好对是否成立
- **人工**：逐条复核改写与问题步，最后定稿

来源模型分布：`deepseek-v4.1-flash-1017` × 2, `glm-5.3` × 1

## 1. 数据集清单

| 用途 | 文件 | dataset_info 名字 | 条数 |
| :--- | :--- | :--- | :--- |
| **SFT 训练（主）** | `sft_step_windows_sharegpt.train.jsonl` | `pi_sft_step_windows_train` | 548 |
| SFT 验证 | `sft_step_windows_sharegpt.eval.jsonl` | `pi_sft_step_windows_eval` | 221 |
| 完整轨迹训练（可选） | `sft_gold_trajectories_sharegpt.train.jsonl` | `pi_sft_gold_trajectories_train` | 2 |
| 完整轨迹验证 | `sft_gold_trajectories_sharegpt.eval.jsonl` | `pi_sft_gold_trajectories_eval` | 1 |
| **DPO 训练** | `dpo_preference_pairs.train.jsonl` | `pi_dpo_preference_pairs_train` | 100 |
| DPO 验证 | `dpo_preference_pairs.eval.jsonl` | `pi_dpo_preference_pairs_eval` | 143 |
| OpenAI 格式备选 | `sft_step_windows_openai.train.jsonl` | 未注册（同数据另一种格式） | 548 |
| 病态诊断（不进训练） | `negative_diagnostics.jsonl` | — | 814 |

统计：

- 阶段权重分布（train）：opening 9, key_decision 418, routine 279, highlight 54, closing 9
  （`highlight` = 模型判定的关键决策步，权重最高；若你的 collator 支持 `metadata.sample_weight` 可直接用）
- 项目分布：`game::jigsawcard` 548, `game::meowdoku` 221
- 训练/验证切分（**按项目整包**，防相邻步泄漏）：
  - train：`game::altosodyssey`, `game::cookingmadness`, `game::jigsawcard`, `game::mahjong`
  - eval：`game::meowdoku`, `game::riddle`
- 入库审计：clean_gold 499, rescued_by_rewrite 218, human_kept 1, whitelisted_subsegment 51
- 剔除原因：excluded_by_review 9, pathology 28, empty_thinking 5, model_discard 708, label_excluded 791, outside_subsegment_whitelist 341, session_blocked 298
- 会话标签：positive 6, pending 5, negative 6
- 复核状态：approved 13, held 3, unreviewed 1
- DPO 来源：stage2_pruned 190, negative_step 53
- DPO 复核：检查 305 对，
  调换 10、丢弃 62

## 2. 显存与长度核算（决定 cutoff_len）

**训练样本长度实测**（train，548 条）：

| 分位 | p50 | p90 | p99 | max |
| :--- | :--- | :--- | :--- | :--- |
| 字符数 | 9,471 | 16,250 | 24,809 | 25,770 |
| ≈tokens（中英混合 2.2 字符/token） | 4,305 | 7,386 | 11,277 | 11,714 |

- 最长样本约 **11,714 tokens**（按纯英文 3.0 字符/token 估则约 8,590）
- 因此 **`cutoff_len: 16384` 覆盖全部样本，0 截断**；若降到 8192，约 36 条会被截断（含目标步，直接丢弃该能力）

**27B bf16 显存预算（单卡 80GB，cutoff_len 16384）**

| 项 | 占用 | 说明 |
| :--- | :--- | :--- |
| 文本权重（bf16） | ≈ 54 GB | 全参数常驻（27B dense 为主体） |
| 视觉塔（vision_config 27 层 / hidden 1152） | ≈ 0.5 ~ 1.5 GB | 多模态结构，纯文本训练时也常驻 |
| LoRA 适配器 + 优化器 | < 1 GB | rank 32，只训 mlp |
| 激活值（bs1 + 梯度检查点 + FA2） | 8 ~ 16 GB | 64 层里有 48 层是线性注意力，激活随长度线性增长（不是平方），比标准 Transformer 省 |
| **峰值合计** | **约 64 ~ 72 GB** | 80GB 有余量；仍需首个 epoch 盯 nvidia-smi |

**磁盘（你已确认 200 GB）**：底座 54G + 合并版 54G + 环境 10G + ckpt 3G ≈ **121G**，
可在同一块盘上同时保留「原始底座」和「合并后的模型」，无需来回删。
**主机内存（你已确认 >100 GB）**：合并 27B bf16 时 CPU 峰值约 60G，100G+ 完全够。

## 3. 底座模型核实结果（必须在开机时复核）

我从 ModelScope 的 `config.json` 实测到的事实（**与"标准 27B dense"的假设差别很大**）：

| 项 | 实测值 | 含义 |
| :--- | :--- | :--- |
| architectures | `Qwen3_5ForConditionalGeneration` | 专用架构，不是 LlamaForCausalLM |
| model_type | `qwen3_5` | 决定 transformers / 框架能否加载 |
| 多模态 | vision_config 存在（depth 27 / hidden 1152 / patch 16），有 image_token_id 与 video_token_id | 纯文本训练可行，但走的是多模态代码路径 |
| 网络结构 | 64 层 = 48 × linear_attention + 16 × full_attention（full_attention_interval=4） | 混合线性注意力 |
| 尺寸 | 5120 / intermediate 17408 / 24 heads / 4 KV heads / head_dim 256 | GQA 6:1，head_dim 256 |
| 词表 | 248320（tie_word_embeddings=False） | embedding 约 1.27B 参数 |
| 上下文 | max_position_embeddings 262144，rope_scaling=None（原生 256K，勿加 yarn） | 原生 256K，已正确 |
| transformers | config 里声明 transformers_version=5.8.0.dev0 | 关键约束，见 §4.2 |

### 3.1 混合线性注意力带来的四个直接影响

1. **`flash_attn: fa2` 只对 16/64 层 full_attention 生效**，48 层 linear_attention 走模型自带 kernel。
   仍建议装（省那 16 层的显存与时间），但别指望它像纯 Transformer 那样大幅降显存。
2. **训练显存比预期省**：线性注意力层的状态是固定大小（不随序列增长），
   16K 序列的激活主要是 MLP 部分，实测预算见 §2。
3. **序列并行（Ulysses）基本不可用**：linear_attention 层不支持按序列切分。
   → 想训 100K+ 全长轨迹必须走"多卡 TP/PP + 显存优化"，不要指望 `sequence_parallel_size`。
4. **`enable_liger_kernel` 先关掉**：Liger 只覆盖特定模型族的融合算子，
   `qwen3_5` 的线性注意力层不在其内，开了大概率直接报错。跑通 SFT 后再单独试。

### 3.2 关于"原生 256K"的正确理解

1. **训练长度 ≠ 能力长度**。本数据集滑窗最长约 11,714 tokens，
   用 `cutoff_len 16384` 训练不会改变 `max_position_embeddings`，
   合并后 vLLM 仍接受 256K 输入（接口层面成立）。
2. **但长上下文质量不会被强化**，所以默认 `lora_target: mlp`（不动注意力投影层）+ §7 有验收脚本。
3. **单卡 80GB 跑 256K 推理也不现实**，算一下 KV cache：
   ```
   每 token 每 full 层 KV = 2(K,V) × 4(kv_heads) × 256(head_dim) × 2(bf16) = 4 KB
   × 16 个 full_attention 层 = 64 KB / token
   256K 上下文 → 64KB × 262144 ≈ 16 GB
   ```
   权重 55G + KV 16G = 71G，再加激活/碎片 → **单卡 80G 顶不住 256K**。
   务实区间：`--max-model-len 32768 ~ 65536`；要更长就用 2 卡 TP 或 fp8 KV cache（KV 降到 8G）。

## 4. 环境搭建（Ubuntu 22.04 / CUDA 12.6 / torch 2.7.1 / py3.12）

镜像选型：平台「基础镜像 → PyTorch → Ubuntu 22.04 / CUDA 12.6 / 2.7.1 / 3.12」。
**不要选 CUDA 13.x**：flash-attn 等轮子跟进慢，容易退化成源码编译。

### 4.0 无卡阶段先做这些（不用等 GPU）

平台支持"无卡启动"时，**先把软件栈、模型、数据、模板全部验完**，挂上卡就能直接开训：

```bash
# 一条命令跑完 A~E 五组检查（无卡会自动把 GPU 项标记为 SKIP）
python check_env.py --model /data/models/Qwen3.8-27B --data /data/goldthread/03_train

# 单独验数据 + 模板（CPU 即可，最值得先跑）
python check_data.py --model /data/models/Qwen3.8-27B --template qwen3_8
```

`check_env.py` 会输出 `[PASS] / [FAIL] / [SKIP]` 三类：
- `FAIL` → 必须修（缺包、版本不对、数据不全、磁盘不足）
- `SKIP` → 无卡时正常，挂卡后必须再验（nvidia-smi、bf16、flash-attn、显存峰值）

**无卡阶段能验的**：python/torch 版本、全部依赖与版本区间、模型 config 架构一致性、
权重体积完整性、tokenizer 与 chat_template、**用真实 tokenizer 统计样本长度分布**、
数据文件齐全性、磁盘与主存。

**无卡阶段验不了的**（挂卡后按 4.1 做）：CUDA 可用性、bf16 支持、flash-attn kernel 加载、
真实显存峰值、20 步冒烟训练、DPO 训练。

### 4.1 挂上卡之后的 5 条验收（不通过先别装训练栈）

```bash
nvidia-smi | head -12
# 期望：A800 80GB，Driver ≥ 535

python -V
# 期望：Python 3.12

python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
# 期望：2.7.1 12.6 True

python -c "import flash_attn;print('fa2', flash_attn.__version__)" 2>/dev/null || echo "需要安装 flash-attn（见 4.2）"

df -h /    # 期望可用 ≥100GB
```

### 4.2 网络加速（国内机器必做）

这台机器访问 GitHub / PyPI 会超时，**先把加速配好再装依赖**，否则后面每步都会卡。

> 下面用 `GH_PROXY` 变量统一表示你的代理地址。把它换成你实际在用的（常见形式：
> `https://ghfast.top`、`https://gh-proxy.com`、`https://ghproxy.net`，
> 规则都是 `<代理>/https://github.com/...`）。**先跑一次连通性检查再批量配置。**

```bash
# 0) 选出能用的代理（把三个候选依次试一遍，取第一个返回 200 的）
export GH_PROXY=https://ghfast.top
for p in https://ghfast.top https://gh-proxy.com https://ghproxy.net; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$p/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md")
  echo "$p -> $code"
  [ "$code" = "200" ] && export GH_PROXY=$p && break
done
echo "使用代理: $GH_PROXY"

# 1) git 全局改写：之后所有 https://github.com/... 自动走代理
git config --global url."$GH_PROXY/https://github.com/".insteadOf "https://github.com/"
git config --global --get-regexp 'url\..*insteadof'   # 确认写入

# 2) PyPI 镜像（清华，国内稳定）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'INI'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
INI

# 3) HuggingFace 镜像（本方案主用 ModelScope，这里只是备选）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
```

**哪些需要代理、哪些不需要**：

| 场景 | 是否走代理 | 做法 |
| :--- | :--- | :--- |
| `git clone github.com/...` | 需要 | 已由 1) 的 insteadOf 自动处理 |
| `raw.githubusercontent.com/...` | 需要 | 手动加前缀：`$GH_PROXY/https://raw.githubusercontent.com/...` |
| GitHub **Releases** 下载（如 flash-attn wheel） | 需要 | 同上，加前缀 |
| PyPI（pip install） | 不需要 | 走 2) 的清华镜像 |
| ModelScope 模型下载 | 不需要 | 国内直连，比 HF 快 |
| HuggingFace 模型 | 需要 | 走 3) 的 hf-mirror |

**flash-attn 装不上时的代理装法**（用预编译 wheel 比源码编译省 20 分钟）：

```bash
# 先看你的组合该用哪个 wheel：cp312 + torch2.7 + cu12
python - <<'PY'
import torch, sys
print(f"py3{sys.version_info.minor} torch{torch.__version__.split('+')[0]} cu{torch.version.cuda}")
PY

# 从 Releases 走代理下载（版本号按上面的输出选，示例为 cp312-torch2.7-cu12）
pip install "$GH_PROXY/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"   --no-build-isolation --no-cache-dir
```

**代理失效时的回退**（别卡在网络上）：

```bash
# 临时禁用 insteadOf
git config --global --unset url."$GH_PROXY/https://github.com/".insteadOf
# 或换候选代理后重新配置
# 或用 gitee 镜像的 LLaMA-Factory（更新略滞后）
git clone --depth 1 https://gitee.com/mirrors/LLaMA-Factory.git /opt/LLaMA-Factory
```

### 4.3 安装训练栈（版本是硬约束，别跳步）

模型的 `config.json` 声明 `transformers_version=5.8.0.dev0`，
而 LLaMA-Factory（main 分支）约束是 `transformers>=4.55.0,<=5.8.0`。
两者的交集就是 **`transformers==5.8.0`**（release 版已包含 dev0 的能力）。

```bash
# 1) 关键版本先钉死（顺序很重要：先装 transformers，再装框架）
pip install "transformers==5.8.0" "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0"

# 2) LLaMA-Factory 必须用 main 分支（qwen3_8 模板、qwen3_5 架构支持都在 main 上）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]" --no-deps   # 避免 pip 把 transformers 改回 4.x

# 3) flash-attn（有轮子时十几秒；若开始源码编译 → Ctrl+C，改用 SDPA 兜底）
pip install flash-attn --no-build-isolation

# 3b) Liger Kernel —— 8B 冒烟阶段【必装】，27B（qwen3_5）【必关】不用装
#     8B 的 train_sft_qwen3_8b.yaml 里 enable_liger_kernel: true 是硬依赖：
#     不装，第一个 step 就在 cross_entropy OOM（8.7K token × bs2 的 fp32 logits 要 10.1GB）
pip install "liger-kernel>=0.8,<0.9"

# 3c) bitsandbytes —— 仅 QLoRA 4bit / DPO ref_model 8bit 量化需要
pip install bitsandbytes

# 4) 验证三件事
python -c "import transformers;print('transformers', transformers.__version__)"   # 期望 5.8.0
python -c "from transformers import AutoConfig;c=AutoConfig.from_pretrained('/data/models/Qwen3.8-27B');print(c.model_type, c.architectures)"
llamafactory-cli env
```

**若 `transformers==5.8.0` 在 PyPI 找不到**（尚未发版），则从源码装对应 tag：

```bash
pip install "git+https://github.com/huggingface/transformers.git@v5.8.0"
```

> 不要随手 `pip install -U transformers`：5.9+ 会超出 LLaMA-Factory 的 `<=5.8.0` 约束，
> 表现为 `ImportError: cannot import name ...` 这类莫名其妙的错。

**flash-attn 装不上怎么办**（不影响开训）：把 YAML 里
`flash_attn: fa2` 改成 `flash_attn: sdpa`，torch 原生实现，16K 峰值大约多 3~5 GB。

**Liger 装不上的兜底**：把 8B YAML 里 `enable_liger_kernel: true` 改成 `false`
（同时 `cutoff_len` 降到 16384），代价是峰值显存多 10GB 左右；27B 本来就是 false。

### 4.4 目录规范（统一按数据盘 /data 布局）

**所有大文件都放数据盘**，这样换机器时只挂盘、不改路径。下文所有命令都按这套路径写：

```
/data                                ← 数据盘挂载点（你已确认 200GB）
├── models/
│   ├── Qwen3.8-27B/                 ← 底座，bf16，约 55 GB     【必留】
│   ├── Qwen3-8B/                    ← 冒烟用小模型，约 16 GB   【可选，验完可删】
│   ├── Qwen3.8-27B-merged/          ← LoRA 合并产物，约 55 GB  【48G 推理要用】
│   ├── Qwen3.8-27B-w4a16/           ← 4bit 量化产物，约 15 GB  【24G 推理才需要】
│   └── smoke-qwen3-8b-merged/       ← 小模型合并产物，约 16 GB  【验完可删】
├── goldthread/
│   ├── 03_train/                    ← 本数据集（从开发机拷来，约 60 MB）
│   ├── handoff.md / train_*.yaml / *.py
│   └── inference_report.md          ← 推理机验收报告（见 §10.8）
├── saves/                           ← 训练输出（adapter 每个约 0.5~1.5 GB）
│   ├── qwen38-27b-goldthread-sft/
│   ├── qwen38-27b-goldthread-dpo/
│   └── smoke-qwen3-8b-sft/
└── logs/
```

**为什么必须用绝对路径**：`output_dir: saves/...` 是相对**当前工作目录**的，
你若在 `LLaMA-Factory/` 里执行，产物会落到 `LLaMA-Factory/saves/`，
换机器或换目录就找不到了。本 handoff 生成的 YAML 已全部用 `/data/saves/...` 绝对路径。

**首次准备**：

```bash
mkdir -p /data/{models,goldthread,saves,logs}
cp -r <本目录>/* /data/goldthread/03_train/     # 把 03_train 拷到数据盘
```

体积核算（200GB 盘）：

| 项 | 体积 | 说明 |
| :--- | :--- | :--- |
| Qwen3.8-27B 底座 | 55 GB | 训练与合并都要它 |
| Qwen3.8-27B-merged | 55 GB | 推理产物 |
| Qwen3-8B（冒烟） | 16 GB | 验完流程可删 |
| 4bit 量化 | 15 GB | 只有 24G 机器才需要 |
| 训练输出 + checkpoint | 3 ~ 6 GB | save_total_limit=2 |
| 数据集 + 环境 | 10 GB | 含 pip 包 |
| **合计** | **约 154 GB** | 200GB 够用；若只做 48G 推理可只留 底座+merged |

> 磁盘紧张时的取舍：**48G 推理只需要 `models/Qwen3.8-27B-merged`（55G）+ 免底座**，
> 因为合并版已含权重；但若还是要重训，就必须保留原始底座。

### 4.5 下载底座模型

> 仓库已核实：**`Qwen/Qwen3.8-27B`**（ModelScope），架构见 §3。
> 下载前先确认磁盘：`df -h /data` 需要 **≥ 70 GB 可用**（权重 55G + 余量）。

```bash
# 推荐：ModelScope（国内快，支持断点续传）
pip install -U modelscope
modelscope download --model Qwen/Qwen3.8-27B --local_dir /data/models/Qwen3.8-27B

# 备选：HuggingFace
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.8-27B --local-dir /data/models/Qwen3.8-27B
```

**下载完必须校验体积与文件数**（中断会留下残缺权重，训练时报错很难定位）：

```bash
du -sh /data/models/Qwen3.8-27B                       # 期望 ≈ 55G
ls /data/models/Qwen3.8-27B/*.safetensors | wc -l     # 应与 index.json 里的 shard 数一致
python - <<'PY'
import json, pathlib
d = pathlib.Path("/data/models/Qwen3.8-27B")
idx = d / "model.safetensors.index.json"
if idx.exists():
    want = set(json.load(open(idx))["weight_map"].values())
    have = {p.name for p in d.glob("*.safetensors")}
    print("缺失分片:", sorted(want - have) or "无（完整）")
else:
    print("单文件权重，跳过分片校验")
PY
```

小模型（冒烟用，见 §6.0）：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B   # ≈16G
```

下载后**必须**确认三件事：

```bash
python - <<'PY'
import json, pathlib
cfg = json.load(open("/data/models/Qwen3.8-27B/config.json"))
print("architectures      :", cfg.get("architectures"))
print("max_position_emb   :", cfg.get("max_position_embeddings"))
print("model_type         :", cfg.get("model_type"))
print("hidden/layers      :", cfg.get("hidden_size"), "/", cfg.get("num_hidden_layers"))
print("rope_scaling       :", cfg.get("rope_scaling"))
# 已核实的期望值（与 §3 一致，若不同则需重新调整训练配置）：
#   architectures = Qwen3_5ForConditionalGeneration
#   model_type    = qwen3_5
#   max_position_embeddings = 262144   rope_scaling = None
PY
```

`rope_scaling` 若已是 null 且 `max_position_embeddings=262144`，**不要**再加 `yarn`。

## 5. 数据格式与对话模板（最容易踩的坑）

- ShareGPT：`conversations: [{from, value}]`
  `system` 工程纪律提示词 / `human` 根任务 / `gpt` 助手步 / `observation` 工具输出
- 助手步内容：` thinking…推理…` + `<tool_call>{"name":"bash","arguments":{...}}</tool_call>`
- `tools` 列是 JSON 字符串（5 个函数声明）。框架不支持时删掉即可（见 `check_data.py` 的 `--drop-tools`）
- 单步窗口结构：`[system, 根任务] + 最近 N 步历史 + 目标步`，
  **历史步 thinking 已剥离，只有目标步带 thinking**

### 5.1 模板必须用 `qwen3_8`（不是 qwen3）

LLaMA-Factory 为这个模型族提供了**专用模板**：

| 要点 | 取值 | 说明 |
| :--- | :--- | :--- |
| 模板名 | `qwen3_8` | `template_class=Qwen38ReasoningTemplate`，`preserve_thinking=True`，内置 `tool_format=qwen3_8` |
| 坑 1 | 不要设 `tool_format` | 该模板内置工具格式，另设会直接抛 ValueError |
| 坑 2 | 不要用 `qwen3` | 通用模板不理解 qwen3_8 的推理段切分，可能重复注入 ` thinking` |
| 坑 3 | 多模态路径 | 模板带 mm_plugin，纯文本数据集应可跑；若报缺 images 字段再按 §5.2 处理 |

开机后先确认模板存在：

```bash
python -c "from llamafactory.data.template import TEMPLATES;print([k for k in TEMPLATES if 'qwen3' in k])"
# 期望输出里包含 qwen3_8 / qwen3_5 等
```

再跑数据自检：

```bash
python check_data.py --model /data/models/Qwen3.8-27B --template qwen3_8
# 期望：每条样本 ` thinking` 与 `` 各出现 1 次；tool_call 数与目标步一致；
#       渲染后 token 数与 §2 的估算吻合
```

### 5.2 loss 掩码：用 `mask_history: true`（已解决）

我们数据的设计是「历史步没有 thinking，只有目标步带 thinking」，
所以**只应该对最后一步算 loss**。LLaMA-Factory 恰好有这个开关：

```yaml
mask_history: true     # 屏蔽历史，只训练最后一轮（正是我们要的）
train_on_prompt: false # 与 mask_history 互斥，保持 false
```

不设 `mask_history` 的后果：历史段也参与 loss → 模型会学到"不说话直接调工具"，
与你想要的行为相反。**这是本数据集唯一必须手动开的开关。**

## 6. 训练配置

同目录已生成：

- `train_sft.yaml` —— **bf16 LoRA 单卡主配置**（本环境直接用）
- `train_sft_qlora.yaml` —— 4-bit QLoRA 备选（显存紧张/换 48G 卡时）
- `train_dpo.yaml` —— SFT 之后的偏好对齐（单卡适配版）

关键超参说明：

| 参数 | 值 | 原因 |
| :--- | :--- | :--- |
| `template` | qwen3_8 | **专用模板**；不要设 tool_format |
| `mask_history` | true | **只训练最后一轮**（本数据集的关键开关） |
| `cutoff_len` | 16384 | 实测最长 11,714 tokens，0 截断 |
| `lora_target` | mlp | 不动注意力投影层，尽量保住原生长上下文能力 |
| `lora_rank` | 32 | 27B 用 32~64 均可 |
| `learning_rate` | 1.0e-4 | LoRA 常规区间；总步数 ~411 |
| `gradient_accumulation_steps` | 4 | 有效 batch 4，每 epoch ≈137 步 |
| `num_train_epochs` | 3.0 | 823 条量级下 3 epoch 足够，再多易过拟合 |
| `packing` | false | **必须关闭**：拼包会破坏「目标步在末尾」的结构 |
| `bf16` | true | A800 原生支持；不要用 fp16 |
| `gradient_checkpointing` | true | 16K 序列必须开 |
| `flash_attn` | fa2 | 仅对 16/64 层 full_attention 生效；装不上换 sdpa |
| `enable_liger_kernel` | false | qwen3_5 不在 Liger 支持列表，先关 |
| `deepspeed` | （不设） | 单卡无收益，且会强制走 torchrun |

### 6.0 先用小模型跑通流程（48G 卡，强烈建议）

27B 这个底座有两个"非常规"之处（`qwen3_5` 混合线性注意力 + 多模态），
一旦训练脚本出问题，很难分清是**数据/模板**还是**架构**的问题。
所以建议先用一个"同族、标准架构"的小模型把全链路跑一遍：

| 模型 | bf16 权重 | 48G 上可用上下文 | 为什么选它 |
| :--- | :--- | :--- | :--- |
| **Qwen3-8B**（推荐） | 16.4 GB（bf16） | 16K 约 28G / 32K 约 35G | 与 27B 同族：`template: qwen3`、`<tool_call>` JSON 格式一致 |
| Qwen3-4B | 8 GB | 16K 约 18G | 最快迭代，适合反复调数据 |
| Qwen3-14B | 28 GB | 16K 约 40G | 更接近大模型行为，但 48G 上余量小 |

> ⚠️ **8B 不能用来验 256K**（这点必须先说清）
>
> | 模型 | 原生上下文 | KV/token | 原因 | 256K KV（bf16/fp8） | 256K 可用性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B | **40,960（40K）** | 144 KB | 36 层全是 full attention | 36 G / 18 G | 需 YaRN 外推，质量不可用 |
| Qwen3.8-27B（目标） | **262,144（256K）** | 64 KB | 仅 16/64 层 full attention | 16 G / 8 G | **原生支持** |
>
> 两个结论：
> 1. **8B 的 256K 要 YaRN 外推**（官方推荐只在 131K 内用 YaRN），256K 远超训练分布，质量会崩，
>    即使跑通也不代表 27B 的行为；
> 2. **8B 的 KV cache 比 27B 贵 2.25 倍**（144 KB vs 64 KB/token），
>    因为 8B 36 层全是全注意力，而 27B 只有 16/64 层是全注意力。
>    → 在 48G 上，27B 的 256K 反而比 8B 的 256K 更容易跑。**别用 8B 排练长上下文**。

**它能验证什么**（覆盖 90% 的踩坑点）：
模板渲染是否正确、`mask_history` 是否生效、`tools` 列能否被读取、
单步窗口结构是否被正确解析、DPO 数据格式、LoRA 合并、vLLM 起服务与工具调用解析。

**它不能证明什么**：不能证明 `qwen3_5` 架构在 LLaMA-Factory 下也没问题
（那是模型适配问题，与数据无关）；显存与吞吐数字也不能外推到 27B。

开机后先跑这套（约 20 分钟）：

```bash
# 1) 下载小模型
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B

# 2) 数据 + 模板自检（把 --model 换成小模型，template 换成 qwen3）
python check_data.py --model /data/models/Qwen3-8B --template qwen3

# 3) 冒烟训练：只跑 200 条 × 1 epoch
llamafactory-cli train data/03_train/train_sft_smoke_qwen3_8b.yaml
#    看三件事：loss 从 2.x 降到 1.x；nvidia-smi 峰值（应 < 35G）；无 OOM

# 4) 合并 + 起服务 + 验工具调用（走 §10.4 的三条冒烟）
llamafactory-cli export --model_name_or_path /data/models/Qwen3-8B   --adapter_name_or_path /data/saves/smoke-qwen3-8b-sft --template qwen3   --finetuning_type lora --export_dir /data/models/smoke-qwen3-8b-merged

# 5) DPO 冒烟（只用 50 对）
llamafactory-cli train data/03_train/train_dpo_smoke_qwen3_8b.yaml
```

**验收清单**（全过再上 27B）：

- [ ] 训练日志里没有 `mask_history` / `tools` 相关的告警
- [ ] `loss` 明显下降（2.0+ → 1.2 以下）
- [ ] 生成结果里同时出现 `thinking` 与 `<tool_call>`
- [ ] vLLM `/v1/chat/completions` 能解析出结构化 `tool_calls`
- [ ] DPO 冒烟能跑完 1 epoch 不报错
- [ ] 合并后的模型能被 vLLM 正常加载

### 6.0.1 当前阶段：Qwen3-8B 完整执行顺序（逐条照做）

目标：用 8B 把「数据 → SFT → 合并 → vLLM 工具调用 → DPO」整条链路跑出**可用模型**，
再换 27B。每步都有验收点，不过就停在那一步排查。

> 💡 **推荐用 `train_monitor.py` 代替裸跑 `llamafactory-cli train`**：
> 实时 tail 日志、OOM 或被系统 kill 时**立刻**给出诊断与建议、超时可自动终止、Ctrl-C 优雅清理进程组。
> 用法：`python train_monitor.py --timeout-hours 3 --log logs/train.log -- llamafactory-cli train train_sft_qwen3_8b.yaml`
> 以下命令仍保留原始形式，只需在前面加上 `python train_monitor.py --log logs/xxx.log -- ` 即可。

**Step 0 · 装依赖（无卡阶段就能做，约 6 GB）**

```bash
python -V                                     # 期望 3.12
pip install "transformers==5.8.0" "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0" --no-cache-dir
# 先按 §4.2 配好 GH_PROXY 与 git insteadOf，再执行下面这条
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory
cd /opt/LLaMA-Factory && pip install -e ".[metrics]" --no-deps
pip install flash-attn --no-build-isolation || echo "装不上就改用 sdpa（见下）"
llamafactory-cli env
```
> `--no-cache-dir` 省磁盘；`--no-deps` 防止 pip 把 transformers 降回 4.x。
> flash-attn 失败不影响开训：把 YAML 里 `flash_attn: fa2` 改成 `sdpa`。

**Step 1 · 下载 8B（16 GB）**

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B
du -sh /data/models/Qwen3-8B                  # 期望 ≈16G
```

**Step 2 · 数据与模板自检（CPU 即可）**

```bash
cd /data/goldthread/03_train
python check_env.py --model /data/models/Qwen3-8B --data .
python check_data.py --model /data/models/Qwen3-8B --template qwen3
```
验收：`check_env` 无 FAIL；`check_data` 里模板渲染每条 ` thinking×1  ×1`。
**若 think 不是 1 对 → 停，先解决模板问题**（换模板或去掉数据里的 thinking）。

**Step 3 · 冒烟训练（200 条 × 1 epoch，约 20 分钟）**

```bash
llamafactory-cli train /data/goldthread/03_train/train_sft_smoke_qwen3_8b.yaml
```
验收：`loss` 从 2.x 降到 1.x；无 OOM；日志里没有 `mask_history` / `tools` 告警。

**Step 4 · 全量训练（548 条 × 3 epoch）**

```bash
llamafactory-cli train /data/goldthread/03_train/train_sft_qwen3_8b.yaml
```
48G 卡上 8B 很宽松（16G 权重 + 32K 激活 ≈ 35G），也可在 80G 上跑。
预期：3 epoch、bs2×ga4 → 每 epoch 约 69 步，总约 207 步。

**Step 5 · 合并并做工具调用验收**

```bash
llamafactory-cli export --model_name_or_path /data/models/Qwen3-8B   --adapter_name_or_path /data/saves/qwen3-8b-goldthread-sft   --template qwen3 --finetuning_type lora   --export_dir /data/models/qwen3-8b-merged

vllm serve /data/models/qwen3-8b-merged --max-model-len 32768   --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes   --port 8000
```
然后按 **§10.4** 跑三条冒烟（存活 / 生成 / **工具调用**）。

**Step 6 · DPO（可选但建议，验数据格式）**

```bash
llamafactory-cli train /data/goldthread/03_train/train_dpo_qwen3_8b.yaml
```

**Step 7 · 记录基线（换 27B 前必做）**

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
# 记录：峰值显存 / 每步耗时 / 2 个 epoch 时的 loss
```
这些数字是换 27B 后判断"是否正常"的唯一参照：27B 的显存约为 8B 的 3.4 倍、
步时约为 3~4 倍，超出太多说明配置有问题。

> **换 27B 只需改三处**：`model_name_or_path` → `/data/models/Qwen3.8-27B`；
> `template` → `qwen3_8`；`output_dir` → `/data/saves/qwen38-27b-goldthread-sft`。
> 其余（`mask_history` / `packing: false` / `lora_target: mlp` / 无 deepspeed）保持不变，
> 另需按 §3.1 关掉 `enable_liger_kernel`。

### 6.1 启动

```bash
cd /data/goldthread/03_train
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml
# 多卡 DDP：torchrun --nproc_per_node 2 train_lowmem.py train_sft.yaml
```

**27B 不要用 `llamafactory-cli train`**：词表 248320、qwen3_5 拿不到 Liger 融合 CE，
原生路径对 16K 全部位置算 logits 并升 fp32（bf16 8G + fp32 16G + 反向梯度 16G ≈ 40GB），
加 54G 权重与 11G 检查点激活 ≈ 105GB，单卡 80G 在 16K 样本上必 OOM。
`train_lowmem.py` 接管 `Trainer.compute_loss`，只对 `labels != -100` 的位置过 `lm_head`
（`mask_history` 下通常 < 2K token），CE 分块 + checkpoint；loss/梯度与原生数学等价，
归一化与梯度累积缩放沿用 Trainer 原约定。日志前 3 步打印 `[lowmem] seq_len / 监督位置 / 峰值显存`。

### 6.2 训练中要盯的三件事

1. **首 50 步看 `nvidia-smi`**：峰值 > 78 GB 就把 `cutoff_len` 降到 12288（仍覆盖 p99）
2. **loss 曲线**：应从 ~1.5-2.0 单调下降到 0.6-1.0 区间；若卡在 2.0 附近不动 → 检查模板（§5.1）
3. **每 100 步抽 5 条 eval 样本续写**：期望输出 `thinking` + `<tool_call>`，而不是纯文本或空思考

### 6.3 OOM 处置阶梯（按顺序试）

确认用 `train_lowmem.py` 启动 → 装 `flash-linear-attention` / `causal-conv1d`（日志无 "Falling back to torch implementation"）
→ `cutoff_len 16384 → 12288` → `flash_attn: sdpa` 换回 `fa2`
→ `gradient_accumulation_steps 4 → 8`（不等于降显存，但可把 micro-batch 保持 1）
→ 仍未解决才考虑 `train_sft_qlora.yaml`（4-bit，质量略降）

## 7. 评估与长上下文验收

**损失**：本数据集的 eval 是**整项目留出**（`game::meowdoku`, `game::riddle`），
所以 `eval_loss` 明显高于 `train_loss` 是正常的，它衡量"换一个程序还能不能干活"。

**能力抽检**（人工打分，各 10 条）：

1. 是否先观察再动手（不盲调工具）
2. 报错后是否查证再重试（不是重复同一条错误命令）
3. 是否避免心算偏移/密钥并声称已验证
4. 是否遵守"只在当前工作目录内作业"
5. 工具调用 JSON 是否格式合法、参数完整

**长上下文验收**（验证 §3 的结论，不达标就别宣称"保住 256K"）：

```bash
python verify_longctx.py --model /data/models/Qwen3.8-27B-merged \
    --lengths 8192,32768,65536,131072,262144
# 逐档构造"前文 + 尾部提问"，记录是否 OOM/报错、首 token 延迟、生成是否连贯
```

## 8. DPO 偏好对齐（SFT 达标后再做）

```bash
cd data/03_train
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_dpo.yaml   # 不要用 llamafactory-cli train（全量 logits 会 OOM）
```

单卡注意：

- **reference 必须是 SFT 模型**。LoRA 做 DPO 且不设 `ref_model` 时，LLaMA-Factory 的 reference 是
  「关掉 adapter 的 `model_name_or_path`」；若那是原始 base，分布与 SFT 差太远 → 会把模型推崩
  （8B 实测：输出乱码、think 不闭合、无 tool_call）。
  现方案：`model_name_or_path: /data/models/Qwen3.8-27B-merged`（SFT 合并产物）+ 新 DPO LoRA、不设 `ref_model`，
  reference = 关 adapter 的 merged = SFT 模型，且不额外加载权重。8B 的 `train_dpo_qwen3_8b.yaml` 同方案。
- `cutoff_len` 已降到 **8192**（DPO 同时跑 policy + reference 两路前向，激活约翻倍）；
  数据用 **v2w 窗口裁剪版**（`pi_dpo_preference_pairs_train_v2w`），避免 LLaMA-Factory
  从 prompt **尾部**截断（那会丢掉最近的 observation）
- **显存**：bf16 权重 ≈54G + 激活；`train_lowmem.py` 加载 `lowmem_dpo.py`，policy / ref 只对 response 位置算 logits、
  分块升 fp32（原生 2×8192×248320 全量 fp32 logits ≈ 40G 是 OOM 主因）。
  OOM 依次：`LOWMEM_DPO_CHUNK=512` → `cutoff_len 8192 → 6144 → 4096` → `eval_strategy: "no"`
- 导出：DPO 适配器叠在 merged 上，`llamafactory-cli export --model_name_or_path /data/models/Qwen3.8-27B-merged
  --adapter_name_or_path /data/saves/qwen38-27b-goldthread-dpo ...`（不是原始 base）
- `pref_beta: 0.1`；lr 用 **1e-5**（5e-5 会把模型推崩）；数据约 94 对，1 epoch 即可

## 9. 训练产物与迁移到 48G 推理机

### 9.1 数据盘上应该留什么

| 类型 | 路径 | 体积 | 说明 |
| :--- | :--- | :--- | :--- |
| LoRA 适配器 | `/data/saves/qwen38-27b-goldthread-sft/` | ≈ 0.5 GB | **必留**；但单独无法推理，必须配底座 |
| 原始底座 | `/data/models/Qwen3.8-27B` | ≈ 55 GB | **必留**；适配器要挂在它上面 |
| 合并后 bf16 | `/data/models/Qwen3.8-27B-merged` | ≈ 55 GB | 交给 24G 机器前建议先在训练机量化，见 9.2 |
| 量化产物 | `…-merged-4bit/` 或 `*.gguf` | 15 ~ 17 GB | 24G 机器的实际推理载体 |

⚠️ **只拷适配器（0.5G）到新机器是跑不起来的**——必须先有底座，或者先在训练机合并/量化。

### 9.2 24GB 显卡能不能推理？—— 分三种情况

**结论：bf16 直接加载不可能（54G > 24G），必须量化。**
而这个模型是 `Qwen3_5ForConditionalGeneration`（混合线性注意力 + 多模态），
量化工具对它的支持需要实测，不能假设。

按顺序尝试，前一个不通再走下一个：

| 方案 | 做法 | 显存 | 说明 |
| :--- | :--- | :--- | :--- |
| ① bnb 4-bit 动态量化 | transformers + bitsandbytes，加载时量化，无需转换 | 15G 权重 + KV | **最快验证**；但对 linear_attention 层的支持未知 |
| ② GPTQ / AWQ 离线量化 | llm-compressor / autoawq 先转 4bit 权重再加载 | 14~15G | 质量/速度更好；需工具支持该 arch |
| ③ GGUF（llama.cpp） | convert_hf_to_gguf.py → Q4_K_M，llama.cpp 推理 | 16~17G | 最省显存、可 CPU/GPU 混合；需 llama.cpp 支持 qwen3next 系架构 |

### 9.3 先探测，再决定买/换卡（在训练机上跑，不用等换机）

数据盘挂到新机器前，**先在 80G 训练机上把 24G 场景模拟一遍**：

```bash
# 用 22GB 上限模拟 24G 卡（多卡卡也能跑，只限制显存预算）
CUDA_VISIBLE_DEVICES=0 python probe_24g.py     --model /data/models/Qwen3.8-27B-merged     --limit-gb 22
```

脚本会依次尝试：bf16 / 4-bit(bnb) / 8-bit，报告每种的**是否成功、峰值显存、首token延迟**。
根据结果：

- **4-bit 成功** → 24G 机器可直接 `load_in_4bit` 推理（注意 KV 预算，见 9.4）
- **只有 8-bit 成功** → 8bit 约 28G，24G 仍不够 → 需要 GGUF 路线
- **全部失败** → 走 GGUF（`llama.cpp`）；若 llama.cpp 也不支持该架构 →
  24G 只能用 CPU offload（约 1~3 tok/s）或换 48G/80G

### 9.4 如果要 256K 上下文推理：显存精算

基于实测架构（16 个 full_attention 层 × 4 KV heads × head_dim 256）：

```
KV / token  = 2(K,V) × 4 heads × 256 dim × bytes × 16 层
            = 64 KB (bf16)   /   32 KB (fp8)
256K KV     = 16 GB (bf16)    /    8 GB (fp8)
```
线性注意力那 48 层状态是固定大小，不随序列增长 —— 这是这个架构唯一的便宜处。

| 权重档 | KV(bf16) | 合计+3G | KV(fp8) | 合计+3G | 最小显卡 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bf16 权重 55G | 16 G | **74 G** | 8 G | 66 G | 80G（0.9 利用率，紧） |
| 8bit 权重 28G | 16 G | 47 G | 8 G | **39 G** | 48G |
| 4bit 权重 15G | 16 G | 34 G | 8 G | **26 G** | 48G |

**结论**：

- **24G 单卡做不了 256K**：最省的组合（4bit + fp8 KV）也要 15 + 8 + 3 = **26 G > 24 G**
- **2 × 24G + TP=2 理论可行**（每卡约 15G），但依赖 vLLM 对混合线性注意力的张量并行支持，
  且 prefill 256K 的 TTFT 会到分钟级 → 只适合离线批处理，不适合交互
- **已定方案：1 × 48G** 跑 `4bit + bf16 KV`（31G），256K 可行且留 2 并发余量；
  具体执行步骤见 **§10**（AI 可直接照做）

### 9.4.1 24G 实际能跑多长

```
4-bit 权重 15 GB → 剩余约 7 GB
   bf16 KV：7G / 64KB ≈ 110K tokens（理论），实际建议 16K~32K（要给激活留余量）
   fp8 KV ：7G / 32KB ≈ 220K tokens（理论），实际建议 32K~64K
```
即 **24G 的舒适区间是 4K~16K，开 fp8 KV 可到 32K**，够跑单步窗口式的 agent 调用。

### 9.4.2 vLLM 启动参数（按卡型选）

```bash
# 48G：4bit + bf16 KV，256K 可行
vllm serve /data/models/Qwen3.8-27B-merged-4bit   --max-model-len 262144   --gpu-memory-utilization 0.92   --kv-cache-dtype auto   --max-num-seqs 1   --enable-chunked-prefill   --max-num-batched-tokens 8192   --enable-auto-tool-choice --tool-call-parser hermes

# 80G：bf16 权重 + fp8 KV，质量最好
vllm serve /data/models/Qwen3.8-27B-merged   --max-model-len 262144   --kv-cache-dtype fp8   --gpu-memory-utilization 0.92 --max-num-seqs 1   --enable-chunked-prefill --max-num-batched-tokens 8192   --enable-auto-tool-choice --tool-call-parser hermes

# 2×24G：4bit + fp8 KV + TP=2（先确认 vLLM 支持该架构的 TP）
vllm serve /data/models/Qwen3.8-27B-merged-4bit   --tensor-parallel-size 2 --max-model-len 262144   --kv-cache-dtype fp8 --gpu-memory-utilization 0.95   --max-num-seqs 1 --enable-chunked-prefill --max-num-batched-tokens 4096
```

**必须加 `--max-num-seqs 1`**：256K 下一个请求就能吃满 KV，并发会直接 OOM。
**必须开 `--enable-chunked-prefill`**：否则 256K prefill 的瞬时激活会炸。

### 9.4.3 关于"256K 质量"的现实提醒

我们的 LoRA 只在 ≤16K 上训练过。因此：

- 接口上能喂 256K（`max_position_embeddings=262144`，`rope_scaling=None`）
- 但 100K+ 的实际表现**未被训练强化**，可能不如原始底座
- 换机前务必用 `verify_longctx.py` 在 8K/32K/64K/131K/256K 逐档跑一遍，
  确认不是"能加载但答非所问"

### 9.5 合并与量化命令

```bash
# 1) 合并 LoRA（训练机上做，输出到数据盘）
llamafactory-cli export   --model_name_or_path /data/models/Qwen3.8-27B   --adapter_name_or_path /data/saves/qwen38-27b-goldthread-sft   --template qwen3_8 --finetuning_type lora   --export_dir /data/models/Qwen3.8-27B-merged

# 2a) 方案①：不转换，直接动态 4-bit（最省事，先试这个）
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "/data/models/Qwen3.8-27B-merged"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    m, load_in_4bit=True, device_map={"": 0},
    max_memory={0: "20GiB"}, trust_remote_code=True)
print("loaded, 4bit OK")
PY

# 2b) 方案②：GPTQ/AWQ 离线量化（需工具支持该架构）
pip install llm-compressor
python -m llmcompressor.transformers.entrypoints.oneshot   --model /data/models/Qwen3.8-27B-merged --scheme W4A16   --output_dir /data/models/Qwen3.8-27B-w4a16

# 2c) 方案③：GGUF（llama.cpp）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py /data/models/Qwen3.8-27B-merged   --outfile /data/models/qwen38-27b-Q4_K_M.gguf --outtype q4_k_m
# 若报 "unsupported model architecture: qwen3_5"，说明 llama.cpp 尚未支持，此路不通
```

### 9.6 新机器（24G）上的最小推理验证

```bash
# 4-bit 加载 + 一次生成
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = "/data/models/Qwen3.8-27B-merged"   # 数据盘挂载点
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, load_in_4bit=True,
        device_map={"": 0}, max_memory={0: "21GiB"}, trust_remote_code=True)
msgs = [{"role": "user", "content": "帮我破解 xxx.xapk，先告诉我第一步做什么"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=128)[0], skip_special_tokens=True))
PY
```

**验收三条**：能加载、能生成 `thinking + <tool_call>`、`nvidia-smi` 峰值 < 23GB。

## 10. 48G 推理机交接清单（AI Agent 可直接照做）

> **目标**：在 1 × 48GB（L40S / RTX A6000 Ada / A100-40） 上，以 **4-bit 权重 + 256K 上下文**提供推理服务。
> 环境与训练机一致：Ubuntu 22.04 / CUDA 12.6 / torch 2.7.1 / py3.12，数据盘整盘挂载。

### 10.0 决策表：先看数据盘上有什么

| 数据盘产物 | 下一步 |
| :--- | :--- |
| `*-merged/`（bf16, 55G）+ adapter | → 10.2 装依赖 → 10.3-A 在线 4bit 启动 |
| 只有 adapter（0.5G） | → **无法推理**，回训练机执行 §9.5 合并 |
| `*-4bit/` 或 `*gptq*` | → 10.3-B 直接加载量化产物（更快） |
| 只有 `*.gguf` | → 10.5 走 llama.cpp |

### 10.1 硬件与环境确认（30 秒，不通过先别往下走）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -V
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
df -h /data ; free -g | head -2
```

通过标准：显存 **≥ 44 GB**；`torch 2.7.1` / cuda `12.6` / bf16 `True`；`/data` 可用 ≥ 40 GB。

### 10.2 安装推理依赖

```bash
pip install "vllm>=0.11"                    # 需含 Qwen3_5 支持
pip install "transformers==5.8.0" "accelerate<=1.11.0"

# 关键：确认 vLLM 认识这个架构（False 就别往下走，见 10.6）
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; print('Qwen3_5ForConditionalGeneration' in R.get_supported_archs())"
```

### 10.3 启动服务（48G 推荐参数）

**A. 在线 4-bit（最省事，先用这个）**

```bash
vllm serve /data/models/Qwen3.8-27B-merged   --quantization bitsandbytes --load-format bitsandbytes   --max-model-len 262144   --kv-cache-dtype bf16   --gpu-memory-utilization 0.90   --max-num-seqs 2   --enable-chunked-prefill --max-num-batched-tokens 8192   --enable-auto-tool-choice --tool-call-parser hermes   --port 8000
```

显存账：4-bit 权重 ≈15G + KV(bf16) ≈16G = **31G**，48G 有余量 → 可留 2 并发。
**若 OOM，按这个顺序降**：`--max-model-len 131072` → `--kv-cache-dtype fp8` → `--max-num-seqs 1`。

**B. 离线量化（A 能跑但嫌加载慢/吞吐低时）**

```bash
pip install llm-compressor
python -m llmcompressor.transformers.entrypoints.oneshot   --model /data/models/Qwen3.8-27B-merged --scheme W4A16   --output_dir /data/models/Qwen3.8-27B-w4a16

vllm serve /data/models/Qwen3.8-27B-w4a16   --quantization compressed-tensors   --max-model-len 262144 --kv-cache-dtype bf16   --gpu-memory-utilization 0.90 --max-num-seqs 2   --enable-chunked-prefill --max-num-batched-tokens 8192   --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

### 10.4 冒烟测试（3 条，必须全过才算部署成功）

```bash
# ① 服务存活
curl -s localhost:8000/v1/models | head -c 300

# ② 普通生成（检查是否仍带 thinking）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json'  -d '{"model":"Qwen3.8-27B-merged","messages":[{"role":"user","content":"用一句话说明你会怎么开始分析一个 xapk 包"}],"max_tokens":128}'  | python -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:400])"
```

**③ 工具调用**（本数据集的核心能力，必须单独验证）：

```bash
cat > /tmp/tools.json <<'JSON'
{"model":"Qwen3.8-27B-merged",
 "messages":[{"role":"user","content":"列出当前目录下的文件"}],
 "tools":[{"type":"function","function":{"name":"bash","description":"run shell",
   "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
 "tool_choice":"auto","max_tokens":256}
JSON
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/tools.json  | python -c "import json,sys;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('tool_calls:',m.get('tool_calls'));print('content:',(m.get('content') or '')[:200])"
```

期望：`tool_calls` 非空且 `function.name == "bash"`。若只有纯文本 → `--tool-call-parser` 不匹配，
查模型文档换 parser（如 `qwen3_coder` / `hermes`）。

### 10.5 256K 长上下文验收

```bash
python /data/tools/verify_longctx_api.py   --base-url http://localhost:8000/v1 --model Qwen3.8-27B-merged   --lengths 8192,32768,65536,131072,262144
```

通过标准：五档都不报错；每档 TTFT 可接受（chunked prefill 下 256K 首次可能 60~180s）；
回答与前文相关（脚本会打印摘要，人工看一眼）。

**若 256K 档 OOM**：说明 KV 预算不足 → 加 `--kv-cache-dtype fp8`，或把 `--max-model-len` 降到 131072
并把实测上限写进交付报告（不要让下游以为能跑 256K）。

> **若这次验的是 Qwen3-8B（冒烟模型）**：它原生只有 40,960，跑 256K 必须加 YaRN，
> 且 KV 需 36G（bf16）/18G（fp8），48G 上只能开 fp8 KV 勉强跑：
> ```bash
> vllm serve /data/models/qwen3-8b-merged >   --max-model-len 262144 --kv-cache-dtype fp8 >   --rope-scaling '{"type":"yarn","factor":6.4,"original_max_position_embeddings":40960}' >   --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 1
> ```
> 这样跑通只证明"机制能跑"，**不能作为 27B 的 256K 结论**。

### 10.6 失败回退路径（按顺序试，别在死路上耗时间）

| 失败点 | 退路 |
| :--- | :--- |
| `bitsandbytes` 4bit 加载报错（不支持该 arch） | 走 10.3-B 离线 W4A16；再不行走 10.7 GGUF |
| vLLM 不认识 `Qwen3_5ForConditionalGeneration` | 升级 vLLM；或用 HF `transformers` + `load_in_4bit`（慢但能跑）；或 10.7 |
| 256K OOM | `--kv-cache-dtype fp8` → `--max-model-len 131072` → 记录实际上限 |
| 工具调用解析失败 | 换 `--tool-call-parser`；或退化为让模型在 content 里输出 `<tool_call>` JSON 自行解析 |
| 生成质量明显下降 | 4bit 量化损伤；改 8bit（28G+16G=44G，48G 勉强）或换 80G 跑 bf16+fp8 KV |

### 10.7 最后一条退路：llama.cpp + GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py /data/models/Qwen3.8-27B-merged   --outfile /data/models/qwen38-27b-Q4_K_M.gguf --outtype q4_k_m
# 报 "unsupported model architecture: qwen3_5" 则此路不通

llama-server -m /data/models/qwen38-27b-Q4_K_M.gguf   -c 262144 -ngl 99 --host 0.0.0.0 --port 8000
```
GGUF Q4 约 17G，KV 可用 `-ctk q8_0 -ctv q8_0` 压到 8G → 48G 上 256K 也能塞下，缺点是不支持 tools 自动解析。

### 10.8 交付验收清单（逐项打勾）

- [ ] `nvidia-smi` 峰值显存 **< 46 GB**
- [ ] `/v1/models` 能列出模型
- [ ] 普通生成正常，且输出仍包含 `thinking` 与 `<tool_call>`
- [ ] 工具调用能被解析成结构化 `tool_calls`
- [ ] 256K 档不崩（或已记录实际可用上限）
- [ ] 把**实际命令、包版本、峰值显存、256K 实测 TTFT** 写进 `inference_report.md`

```bash
# inference_report.md 建议模板
cat > /data/inference_report.md <<'MD'
# 推理机验收报告
- GPU: <nvidia-smi 输出>
- torch / cuda / transformers / vllm: <版本>
- 权重: <路径>  量化: <bnb-4bit | w4a16 | gguf>
- 启动参数: <完整命令>
- 峰值显存: <GB>
- 256K TTFT: <秒>   256K 是否通过: <是/否>
- 工具调用: <通过/失败 + parser>
MD
```

---

## 11. 排障表

| 现象 | 原因 / 处置 |
| :--- | :--- |
| 启动就 OOM | `cutoff_len` 降 12288；确认 `gradient_checkpointing: true`、`flash_attn: fa2` |
| 报不认识 `tools` 列 | 升级 LLaMA-Factory；或 `python check_data.py --drop-tools` 后重训 |
| loss 前 100 步不降 | 模板问题（think 重复注入）→ 跑 `check_data.py --model ... --template qwen3` |
| 输出只有 `<tool_call>` 没有 thinking | 历史段参与 loss 导致（§5.2）；临时把 `synth.window_turns` 降到 4 重跑 |
| `flash-attn` 编译失败 | 改 `flash_attn: sdpa` 继续，别在编译上耗时间 |
| 合并导出时 CPU OOM | 主机 RAM 需 ≥64GB；或分片导出 `--export_size 4` |
| eval_loss 居高不下 | 正常（整项目留出）；看人工抽检而非只看 loss |
| DPO 训练发散 | `pref_beta 0.1 → 0.05`；`learning_rate 5e-5 → 2e-5` |

## 12. 可复现性

- 生成配置：`config/pipeline.yaml`（`config_hash` 见 `stats.json`）
- 每条样本带 `metadata.sample_id / session_id / project_group / stage / split / sample_weight`
- 血缘可回溯：`data/02_gold/sessions/<项目>/<会话>.gold.jsonl`（含 `raw_thinking`、`pruned_details`、`model_report`）
- 重跑：`python -m pipeline.run all`（增量，模型调用命中缓存）

## 13. 使用边界

- eval 项目（`game::meowdoku`, `game::riddle`）**不要再拿去训练**，否则评估失真
- `negative_diagnostics.jsonl` 仅用于分析（无标准答案），不要喂进 SFT/DPO
- 数据来自自有/已授权应用的逆向分析，使用范围请遵守你的授权约定
