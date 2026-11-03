# 训练交接

给在训练机上把这批数据训成模型的人。按章节顺序做。每节末尾的「停」是不过就不能往下走的条件。

机器按这个配过：1 × NVIDIA A800 80GB，Ubuntu 22.04，CUDA 12.6，PyTorch 2.7.1，Python 3.12，数据盘约 200GB，内存 100GB 以上。8B 冒烟在 48GB 卡上也能跑。

三份材料各管一件事：

- 本文件：怎么装、怎么训、失败了改哪一项
- 同目录 YAML：真正生效的参数。`dataset`、`dataset_dir`、`model_name_or_path`、`cutoff_len` 以 YAML 为准，不要按记忆改
- `manifest.json`：本包条数与 sha256。`RESULT.md`、`RESULT_27b_run.md` 是已发生过的实跑记录

交付 zip 解压后，`dataset_dir` 已是 `./data`，数据集键是 `gt_*_lf`。在解压根目录执行下面的命令。

---

## 1. 目录

大文件放数据盘，YAML 里用绝对路径。相对路径会跟当前工作目录走，换目录就找不到产物。

```
/data
├── models/
│   ├── Qwen3-8B/                  冒烟底座，约 16GB
│   ├── qwen3-8b-merged/           8B SFT 合并结果
│   ├── Qwen3.8-27B/               27B 底座，约 52–55GB，重训必须留
│   ├── Qwen3.8-27B-merged/        27B SFT 合并结果，DPO 的底座
│   └── Qwen3.8-27B-dpo-merged/    DPO 合并结果
├── goldthread/03_train/           本目录（交付包则是解压根目录）
└── saves/                         adapter 与 checkpoint
```

```bash
mkdir -p /data/{models,goldthread,saves,logs}
# 仓库训练目录：把本目录放到 /data/goldthread/03_train
# 交付包：解压后直接 cd 进该目录，不必再拷一层
```

磁盘：底座 55GB + 合并模型 55GB + 环境与 checkpoint 约 20GB。只做推理可以只留合并模型。还要重训就必须留下原始底座。

---

## 2. 开工前验收

无卡阶段先做软件和数据。有卡之后再看 GPU。任一项 FAIL 先修，再装训练栈。

```bash
python -V
# Python 3.12

df -h /data
free -g | head -2
# 下 27B 前 /data 可用 ≥ 70GB；合并时内存 ≥ 64GB
```

挂上卡之后：

```bash
nvidia-smi | head -12
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
```

通过标准：A800 80GB（或你明确改用的卡）、驱动能跑 CUDA 12.6、`torch 2.7.1`、`12.6`、`bf16 True`。

**停：** `is_bf16_supported()` 不是 `True`。这台机器不能按本文的 bf16 配置训练。

---

## 3. 安装

国内机器先打通 GitHub / PyPI，再装包。

```bash
export GH_PROXY=https://ghfast.top
for p in https://ghfast.top https://gh-proxy.com https://ghproxy.net; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 \
    "$p/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/README.md")
  echo "$p -> $code"
  [ "$code" = "200" ] && export GH_PROXY=$p && break
done

git config --global url."$GH_PROXY/https://github.com/".insteadOf "https://github.com/"

mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'INI'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
INI

grep -q HF_ENDPOINT ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

然后按这个顺序装。顺序错了，`transformers` 会被装回 4.x。

```bash
pip install "transformers==5.8.0" "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0" --no-cache-dir

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory
cd /opt/LLaMA-Factory && pip install -e ".[metrics]" --no-deps

pip install "liger-kernel>=0.8,<0.9"          # 8B 必装；27B 配置里是关闭的
pip install "flash-linear-attention==0.4.2" "causal-conv1d" --no-build-isolation
                                              # 27B 必装。qwen3_5 的线性注意力在 LLaMA-Factory 里依赖它
pip install bitsandbytes                      # 仅 4bit QLoRA 需要
pip install flash-attn --no-build-isolation || echo "[warn] flash-attn 失败，YAML 改 flash_attn: sdpa"

llamafactory-cli env
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import liger_kernel; print('liger', liger_kernel.__version__)"
python -c "import fla, causal_conv1d; print('fla ok')"
```

`transformers` 必须是 5.8.x。模型 `config.json` 写的是 `5.8.0.dev0`，LLaMA-Factory main 的上限是 `<=5.8.0`，交集就是 5.8.0。LLaMA-Factory 必须用 main：`qwen3_8` 模板和 `qwen3_5` 架构都在 main 上。

| 现象 | 处理 |
| --- | --- |
| 三个代理都不是 200 | 换可用的 `GH_PROXY` 后再配 `insteadOf`。临时取消：`git config --global --unset url."<代理>/https://github.com/".insteadOf`。GitHub 完全不可达时，`git clone --depth 1 https://gitee.com/mirrors/LLaMA-Factory.git /opt/LLaMA-Factory`，装完后仍要核对 `transformers==5.8.0` |
| PyPI 没有 `transformers==5.8.0` | `pip install "git+$GH_PROXY/https://github.com/huggingface/transformers.git@v5.8.0"` |
| `transformers` 不是 5.8.x | LLaMA-Factory 把版本盖掉了。重装上面第一行，并保留 `--no-deps` |
| `ImportError: cannot import name ...` | 多半升到了 5.9+。钉回 `transformers==5.8.0`，不要 `pip install -U transformers` |
| `flash-attn` 开始源码编译 | Ctrl+C。把 YAML 的 `flash_attn: fa2` 改成 `sdpa`，然后继续。16K 附近大约多占 3–5GB |
| 想用预编译 wheel | `python -c "import torch,sys; print(f'cp3{sys.version_info.minor} torch{torch.__version__.split(\"+\")[0]} cu{torch.version.cuda}')"`，再从 flash-attention Releases 选对应 `cp312 + torch2.7 + cu12` 的 wheel，URL 前加 `$GH_PROXY/` |
| `import fla` 失败 | 27B 不能开训。重装 `flash-linear-attention==0.4.2`。日志里的 `Falling back to torch implementation` 或 `requires flash-linear-attention` 是同一件事 |
| 只跑 27B、liger 装不上 | 可以忽略。8B 路径不能忽略 |

---

## 4. 下载底座

```bash
pip install -U modelscope
df -h /data

modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B
modelscope download --model Qwen/Qwen3.8-27B --local_dir /data/models/Qwen3.8-27B

du -sh /data/models/Qwen3-8B /data/models/Qwen3.8-27B
# 期望约 16G 和 52–55G
```

27B 下载后核对架构和分片。中断会留下能列出、但不能训练的残缺权重。

```bash
python - <<'PY'
import json, pathlib
d = pathlib.Path("/data/models/Qwen3.8-27B")
cfg = json.load(open(d / "config.json"))
print("architectures:", cfg.get("architectures"))
print("model_type   :", cfg.get("model_type"))
print("layers/hidden:", cfg.get("num_hidden_layers"), cfg.get("hidden_size"))
print("max_position :", cfg.get("max_position_embeddings"), "rope:", cfg.get("rope_scaling"))
idx = d / "model.safetensors.index.json"
want = set(json.load(open(idx))["weight_map"].values())
have = {p.name for p in d.glob("*.safetensors")}
print("缺失分片:", sorted(want - have) or "无")
PY
```

通过标准：

- `architectures = Qwen3_5ForConditionalGeneration`
- `model_type = qwen3_5`
- 64 层，hidden 5120，词表 248320
- `max_position_embeddings = 262144`，`rope_scaling` 为空
- 缺失分片为「无」

这是多模态混合模型：48 层线性注意力 + 16 层全注意力，视觉塔在纯文本训练时也常驻。`flash_attn: fa2` 只作用在那 16 层上。`enable_liger_kernel` 对 `qwen3_5` 保持 `false`。`rope_scaling` 已是原生 256K，不要再加 YaRN。

**停：** 架构、`model_type` 或分片对不上。删掉目录重新下载，不要在残缺权重上改 YAML。

HuggingFace 备选：`pip install -U "huggingface_hub[cli]" && hf download Qwen/Qwen3.8-27B --local-dir /data/models/Qwen3.8-27B`。

---

## 5. 开训前检查

在训练目录执行。无 GPU 也可以跑；GPU 项显示 `[SKIP]` 时，挂卡后必须再跑一次。

```bash
cd /data/goldthread/03_train          # 交付包则 cd 到解压根目录

# 8B
python check_env.py --model /data/models/Qwen3-8B --data . --cutoff 32768
python check_data.py --model /data/models/Qwen3-8B --template qwen3

# 27B
python check_env.py --model /data/models/Qwen3.8-27B --data . --cutoff 12288
python check_data.py --model /data/models/Qwen3.8-27B --template qwen3_8
```

通过标准：

- `check_env` 没有 `[FAIL]`
- 走 8B 时 `liger_kernel` 没有 `[WARN]`
- `check_data` 里每条样本的 `` `<think>` `` 与 `` `</think>` `` 各 1 次，`<tool_call>` 次数与目标步一致
- 长度分布的 max 不超过该 YAML 的 `cutoff_len`。超长样本由 `train_lowmem.py` 在开训前按 cutoff 过滤；被滤掉的是整条样本，不是悄悄裁掉目标步

数据是 ShareGPT 单步窗口：`system` / `human` / `gpt` / `observation`。历史步没有思考，只有目标步有 `` `<think>` ``。因此 `mask_history: true`，`packing: false`。`packing: true` 会把目标步从末尾拆走。

模板：8B 用 `qwen3`，27B 用 `qwen3_8`。`qwen3_8` 已内置工具格式，YAML 里不要再写 `tool_format`。

```bash
python -c "from llamafactory.data.template import TEMPLATES; print([k for k in TEMPLATES if 'qwen3' in k])"
# 输出里要有 qwen3_8
```

**停：**

| 现象 | 处理 |
| --- | --- |
| think 不是 1 对 | 8B 确认 `--template qwen3`，27B 确认 `qwen3_8`。模板名对了仍不是 1 对，停止训练，先修数据或模板 |
| `ValueError` 提到 `tool_format` | 删掉 YAML 里的 `tool_format` |
| `dataset ... not found` | YAML 的 `dataset` 在 `dataset_info.json` 里没有同名键。交付包应使用 `gt_sft_train_lf` 等键；不要手改文件名 |
| 报不认识 `tools` 列 | `python check_data.py --drop-tools` 生成去掉该列的副本，再把 YAML 指到新文件 |
| 只想确认条数 | 读 `manifest.json` 的 `files.*.count`，不要用旧文档里的 548 |

`observation` 里偶发 U+FFFD（界面上常是连续问号或方框）是采集工具 stdout 时的编码损坏。`system` / `human` / `gpt` 里没有。不影响开训。

---

## 6. 8B：先跑通整条链路

用 Qwen3-8B 验证数据、模板、`mask_history`、合并、工具调用和 DPO 格式。它证明不了 `qwen3_5` 的显存，也不能拿来验收 256K（8B 原生上下文 40K，KV 比 27B 更贵）。

全程在训练目录。建议用 `train_monitor.py`：OOM 或进程被杀掉时它会直接说明原因。

```bash
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 200 条 × 1 epoch
python train_monitor.py --timeout-hours 1 --log logs/smoke_sft.log -- \
  llamafactory-cli train train_sft_smoke_qwen3_8b.yaml

# 全量 × 3 epoch。训练中 eval 已关闭
python train_monitor.py --timeout-hours 3 --log logs/full_sft.log -- \
  llamafactory-cli train train_sft_qwen3_8b.yaml

# 训完再评测
python train_monitor.py --timeout-hours 1 --log logs/eval_sft.log -- \
  llamafactory-cli train eval_sft_qwen3_8b.yaml

llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3-8B \
  --adapter_name_or_path /data/saves/qwen3-8b-goldthread-sft \
  --template qwen3 --finetuning_type lora \
  --export_dir /data/models/qwen3-8b-merged

python smoke_infer_merged.py --model /data/models/qwen3-8b-merged \
  --n 5 --max-new-tokens 1024 --sample
```

通过标准：loss 从 2.x 降到 1.x 附近；5 条生成里都有闭合的思考和 `<tool_call>`，多数 JSON 可解析。必须带 `--sample`（温度 0.6 / top_p 0.95 / top_k 20）。贪心解码会停在长思考里。

8B DPO 依赖上面的合并目录。`train_dpo_qwen3_8b.yaml` 的 `model_name_or_path` 应是 `/data/models/qwen3-8b-merged`，不另挂一份 reference。

```bash
python train_monitor.py --timeout-hours 1 --log logs/dpo_8b.log -- \
  llamafactory-cli train train_dpo_qwen3_8b.yaml
```

**停：**

| 现象 | 处理 |
| --- | --- |
| 第一步就 OOM | 确认装了 liger，且 YAML 是 `enable_liger_kernel: true`。仍 OOM 就把该 YAML 的 `cutoff_len` 降到 16384 |
| 训练步正常，一进 eval 就 OOM | 保持 `eval_strategy: "no"`，用 `eval_sft_qwen3_8b.yaml` 单独跑 |
| 日志突然结束，没有 traceback | 系统 OOM killer。`train_monitor.py` 会写明；加大内存或 swap，并带上 `expandable_segments:True` |
| loss 前 100 步几乎不动 | 模板错了，通常是 think 被注入了两次。重跑第 5 节的 `check_data.py` |
| 生成只有 tool_call，没有思考 | `mask_history` 没生效。确认 YAML 为 `true` 后重训 |
| 冒烟 1024 token 里 JSON 被截断 | 格式问题要看有没有闭合标签。只是撞上长度上限时，加大 `--max-new-tokens` 再看一条 |
| DPO 后乱码、think 不闭合、不再调用工具 | reference 变成了原始 8B。`model_name_or_path` 改成 SFT 合并目录，学习率用 `1e-5`，丢掉这次 adapter 重训 |

---

## 7. 27B SFT

配置文件是 `train_sft.yaml`，不要改 8B 的 YAML 来顶替。和 8B 的差别已经写在文件里：`template: qwen3_8`，`enable_liger_kernel: false`，`lora_target: gate_proj,up_proj,down_proj`，`cutoff_len: 12288`。

启动器是 `train_lowmem.py`。词表 248320，又不能用 Liger 融合交叉熵；原生实现会对整段序列物化 fp32 logits，单卡 80GB 在长样本上会 OOM。`train_lowmem.py` 只在 `mask_history` 留下的监督位置上算 logits，loss 与原实现一致。

```bash
cd /data/goldthread/03_train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 8 --log logs/27b_sft.log -- \
  python train_lowmem.py train_sft.yaml
```

多卡要用 DDP，不能让 HuggingFace 自己走 DataParallel：

```bash
python train_monitor.py --timeout-hours 8 --log logs/27b_sft.log -- \
  torchrun --nproc_per_node 2 train_lowmem.py train_sft.yaml
```

前 3 步日志里要有 `[lowmem] seq_len=… 监督位置=… 峰值显存=…`。同时看 `nvidia-smi`。已跑通过的量级是：约 20 秒/步，3 epoch 两小时出头，峰值 76GB 上下。loss 应从 1.x 降到 0.5 附近；eval 高于 train 是正常的，验证集是整包留出的项目。

`mask_history: true` 时超长序列从最早的历史步裁，目标步保留。所以降 `cutoff_len` 丢掉的是旧上下文，不是要学的那一步。

**停：**

| 现象 | 处理 |
| --- | --- |
| 启动打印检测到多张卡并退出码 2 | 单卡加上 `CUDA_VISIBLE_DEVICES=0`，或多卡改用上面的 `torchrun` |
| `ImportError` / `requires flash-linear-attention` | 回到第 3 节装 `flash-linear-attention==0.4.2` 和 `causal-conv1d` |
| `Target module Qwen3_5MLP ... is not supported` | `lora_target` 写成了 `mlp`。改成 `gate_proj,up_proj,down_proj` |
| 日志里没有 `[lowmem]`，随后 OOM | 用了 `llamafactory-cli train`。停掉，改回上面的启动命令 |
| 峰值 > 78GB，或第一步 OOM | 确认线性注意力包已装。然后把 `cutoff_len` 按 12288 → 10240 → 8192 降。`gradient_accumulation_steps` 不降低峰值 |
| 训练中 eval OOM | `train_sft.yaml` 应已是 `eval_strategy: "no"`。评测用第 8 节 |
| 进程消失且无 traceback | OOM killer。超时给到 8 小时，避免 monitor 先杀掉一次正常的长跑 |
| loss 不降，或只有工具调用没有思考 | 与第 6 节相同：先查模板，再查 `mask_history` |
| 4bit 仍想试 | `train_sft_qlora.yaml` 不是主路径。使用前把 `lora_target`、`cutoff_len`、`dataset` 改成与 `train_sft.yaml` 相同，并用 `train_lowmem.py` 启动 |

单卡不要设置 `deepspeed`。

---

## 8. 合并、评测、训练机冒烟

评测和合并都在 SFT 结束之后做。合并时 CPU 内存峰值约 60GB。

```bash
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 3 --log logs/27b_eval.log -- \
  python train_lowmem.py eval_sft.yaml

llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3.8-27B \
  --adapter_name_or_path /data/saves/qwen38-27b-goldthread-sft \
  --template qwen3_8 --finetuning_type lora \
  --export_dir /data/models/Qwen3.8-27B-merged \
  --export_size 5

python smoke_infer_merged.py --model /data/models/Qwen3.8-27B-merged \
  --n 5 --max-new-tokens 1024 --sample

python verify_longctx.py --model /data/models/Qwen3.8-27B-merged \
  --lengths 8192,32768,65536
```

`eval_sft.yaml` 的 `adapter_name_or_path` 要指向这次 SFT 的 `output_dir`。`train_sft.yaml` 里的目录名如果和默认值不同，两处一起改。

通过标准：eval 能跑完；5 条冒烟都闭合思考并出现 `<tool_call>`；8K / 32K / 64K 三档不报错。已测过的 64K 可以当作单卡生成的实际上限。接口能接收 256K，不代表这次 LoRA 强化过 256K。对外说上下文能力时，说实测到的档位。

**停：**

| 现象 | 处理 |
| --- | --- |
| 合并时 CPU OOM | 确认可用内存 ≥ 64GB，保留 `--export_size 5`。不要在内存不足时重试多次把盘写满 |
| `smoke_infer_merged.py` 报没有思考 | 官方模板会把开标签放进 prompt，生成里主要看闭标签。脚本应同时认 prompt 里的开标签和生成里的闭标签 |
| 长上下文某一档 OOM | 该档就是当前机器的上限，写进验收记录。不要把 `--max-model-len` 报到没跑过的长度 |
| 只拷了 `saves/` 里的 adapter | 新机器无法推理。adapter 必须配原始底座，或者拷贝合并后的整模 |

人工抽 10 条看行为：先观察再调用工具、报错后会改命令、不声称心算过偏移或密钥、只在当前目录作业、工具 JSON 可解析。`eval_loss` 高于 `train_loss` 不是失败。

---

## 9. 27B DPO

SFT 冒烟通过后再做。`train_dpo.yaml` 的底座是 `/data/models/Qwen3.8-27B-merged`，在上面新训一个 LoRA，不设置 `ref_model`。reference 前向就是关掉 adapter 的这份合并权重，也就是 SFT 模型，不会再加载第二份 27B。

学习率保持 `1e-5`，`pref_beta: 0.1`，`cutoff_len: 5120`。数据用 YAML 里登记的那份（交付包是 `gt_dpo_train_lf`）。截断发生在 prompt 尾部，所以 cutoff 不要随意加大去“多看点历史”。

```bash
# 先 20 条，确认 [lowmem-dpo] 和峰值
CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 1 --log logs/27b_dpo_smoke.log -- \
  python train_lowmem.py train_dpo.yaml max_samples=20 output_dir=/data/saves/qwen38-27b-dpo-smoke

CUDA_VISIBLE_DEVICES=0 python train_monitor.py --timeout-hours 3 --log logs/27b_dpo.log -- \
  python train_lowmem.py train_dpo.yaml

llamafactory-cli export \
  --model_name_or_path /data/models/Qwen3.8-27B-merged \
  --adapter_name_or_path /data/saves/qwen38-27b-goldthread-dpo \
  --template qwen3_8 --finetuning_type lora \
  --export_dir /data/models/Qwen3.8-27B-dpo-merged \
  --export_size 5
```

导出时的底座是 SFT 合并目录，不是 `/data/models/Qwen3.8-27B`。

**停：**

| 现象 | 处理 |
| --- | --- |
| 训完后乱码、think 不闭合、工具调用消失 | reference 是原始底座。确认 `model_name_or_path` 为 SFT 合并目录，且 YAML 里没有 `ref_model`。删除这次 DPO adapter 后重训 |
| OOM | 依次做：`LOWMEM_DPO_CHUNK=512`，`cutoff_len` 5120 → 4096，`eval_strategy: "no"`。日志里要有 `[lowmem-dpo]` |
| 另加载 8bit/4bit reference 后仍然 OOM | 单卡 80GB 上，policy 与独立 reference 的两套 logits 放不下。回到“合并模型 + 关闭 adapter”这一种，不要叠加第二份权重 |
| 用了 `run_train.py` / 分块 CE 补丁 | DPO 需要完整 logits 算 logps，那个补丁会把 logits 变成空。DPO 只用 `train_lowmem.py` |

---

## 10. 起推理服务

vLLM 放在单独的 conda 环境里，不装进训练环境。首次启动需要系统 `g++`（flashinfer 会 JIT）。

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install "vllm>=0.11"

# 8B
bash serve_vllm.sh /data/models/qwen3-8b-merged 8000 gt-2026 qwen3-8b-goldthread-sft

# 27B。上下文用实测过的 32K；64K 仅在训练机冒烟通过后再开
MAX_LEN=32768 GPU_UTIL=0.90 \
  bash serve_vllm.sh /data/models/Qwen3.8-27B-dpo-merged 8000 gt-2026 Qwen3.8-27B-sft-dpo
```

两个 parser 都要有：`--reasoning-parser qwen3`（旧 vLLM 没有这项时脚本会退到 `deepseek_r1`），`--tool-call-parser hermes`。脚本已经带上。XML 工具格式的模型把 `TOOL_PARSER` 换成 `qwen3_coder`。

加载视觉塔失败或因此 OOM 时，在 `vllm serve` 上加 `--language-model-only`。

```bash
curl -s localhost:8000/v1/models | head -c 200

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer gt-2026' \
  -d '{"model":"Qwen3.8-27B-sft-dpo","messages":[{"role":"user","content":"列出当前目录文件"}],
       "tools":[{"type":"function","function":{"name":"bash","description":"run shell",
         "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
       "max_tokens":256}' \
  | python -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('reasoning:', (m.get('reasoning_content') or m.get('reasoning') or '')[:80]); print('tool_calls:', m.get('tool_calls'))"
```

通过标准：`tool_calls` 非空且 `function.name` 为 `bash`；reasoning 字段有内容，正文里没有裸的 `` `</think>` ``。A800 80GB、32K、bf16 的实测吞吐大约 28 token/s。reasoning 字段名随 vLLM 版本是 `reasoning_content` 或 `reasoning`。

**停：**

| 现象 | 处理 |
| --- | --- |
| 训练环境里 `pip install vllm` 失败或弄坏 torch | 删掉这次安装，改到上面的 `vllm` 环境 |
| 正文里能看到 `` `</think>` `` | 没加 reasoning parser，或客户端读错字段 |
| `<tool_call>` 整段在 `content` 里 | `--tool-call-parser` 与训练格式不一致。JSON 工具用 `hermes`，XML 用 `qwen3_coder` |
| 24GB 卡加载 bf16 合并模型 | 权重约 52GB，装不下。换 48GB/80GB，或先在大卡上做成该架构支持的 4bit/GGUF 再搬。不支持 `qwen3_5` 的量化工具会直接报架构错误，换工具，不要反复重试同一条命令 |
| 服务起来但 256K 请求 OOM | 把 `--max-model-len` 降到已用 `verify_longctx.py` 通过的长度 |

没有 vLLM 时，可以用 `serve_api.sh` 加 `reasoning_proxy.py` 做过渡。那个组合不会自己拆 reasoning，代理是必须的。

---

## 11. 排障索引

训练过程中的失败，优先改启动方式和 YAML，不要先改数据。

| 现象 | 处理 |
| --- | --- |
| 8B 第一步 OOM | liger 打开；`cutoff_len` 降到 16384 |
| 27B 第一步 OOM | `train_lowmem.py` + `CUDA_VISIBLE_DEVICES=0` + fla/causal-conv1d；`cutoff_len` 12288 → 10240 → 8192 |
| 27B 多卡又慢又占显存 | 没走 torchrun，被 DataParallel 接管。按第 7 节重开 |
| eval 阶段 OOM | 训练 YAML 设 `eval_strategy: "no"`，训完跑 `eval_*.yaml` |
| 进程无 traceback 消失 | OOM killer。看 `train_monitor.py` 的日志，加内存，保留 `expandable_segments:True` |
| loss 长时间不降 | 模板：8B `qwen3`，27B `qwen3_8`。重跑 `check_data.py` |
| 只有 tool_call | `mask_history: true` 且 `packing: false`，然后重训 |
| DPO 后模型崩 | 底座改为 SFT 合并目录，去掉独立 `ref_model`，lr `1e-5` |
| `flash-attn` 编译失败 | `flash_attn: sdpa` |
| 合并 CPU OOM | 内存 ≥ 64GB，`--export_size 5` |
| 客户端看不到结构化工具调用 | 查第 10 节的两个 parser，以及 curl 里的 `model` 是否等于 `--served-model-name` |
| 数据集键找不到 | 以当前目录的 YAML 和 `dataset_info.json` 为准 |

改完任何一项后，从失败的那一条命令重跑验收，不要只看进程退出码为 0。
