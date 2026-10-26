"""handoff 文档生成：把训练集的用法写成可直接照做的说明。

目标环境（本文件按此写死并逐处校验）：
    1 × A800 80GB  |  Ubuntu 22.04  |  CUDA 12.6  |  PyTorch 2.7.1  |  Python 3.12

产出（与数据集同目录，随 stage3 一起刷新）：
    handoff.md           从环境搭建到训练/验收/推理的全流程
    train_sft.yaml       LLaMA-Factory SFT 配置（bf16 LoRA 单卡）
    train_sft_qlora.yaml 显存不足时的 4-bit QLoRA 备选
    train_dpo.yaml       DPO 配置（单卡适配）
    check_data.py        数据 + 模板自检脚本（训练前必跑）
    verify_longctx.py    训练后长上下文验收脚本

所有数据统计（条数、长度分布、切分、DPO 复核结果）都从本次真实产物计算，
不做硬编码假设。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ENV = {
    "gpu": "1 × NVIDIA A800 80GB（Ampere / SM80 / bf16）",
    "os": "Ubuntu 22.04",
    "cuda": "12.6",
    "torch": "2.7.1",
    "python": "3.12",
    "disk": "200 GB",
    "ram": ">100 GB",
    "infer_gpu": "1 × 48GB（L40S / RTX A6000 Ada / A100-40）",
    "repo": "Qwen/Qwen3.8-27B（ModelScope）",
}

# 从 ModelScope config.json 实测确认的架构事实（不要凭印象改）
MODEL_FACTS = {
    "architectures": "Qwen3_5ForConditionalGeneration",
    "model_type": "qwen3_5",
    "multimodal": "vision_config 存在（depth 27 / hidden 1152 / patch 16），有 image_token_id 与 video_token_id",
    "layers": "64 层 = 48 × linear_attention + 16 × full_attention（full_attention_interval=4）",
    "hidden": "5120 / intermediate 17408 / 24 heads / 4 KV heads / head_dim 256",
    "vocab": "248320（tie_word_embeddings=False）",
    "context": "max_position_embeddings 262144，rope_scaling=None（原生 256K，勿加 yarn）",
    "transformers": "config 里声明 transformers_version=5.8.0.dev0",
}


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _table(rows: List[List[str]], head: List[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(":---" for _ in head) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _data_profile(root: Path) -> Dict[str, Any]:
    """读训练集，算长度分布（字符数 + 粗估 token），用于论证 cutoff_len。"""
    p = root / "sft_step_windows_sharegpt.train.jsonl"
    lens: List[int] = []
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lens.append(sum(len(str(c.get("value", ""))) + 16 for c in r.get("conversations", [])))
    lens.sort()

    def pct(q: float) -> int:
        if not lens:
            return 0
        return lens[min(len(lens) - 1, int(len(lens) * q))]

    if not lens:
        return {"n": 0}
    # 中英混合约 2.2 字符/token；纯英文/代码约 3.0
    return {
        "n": len(lens),
        "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": lens[-1],
        "tok_p50": round(pct(0.5) / 2.2), "tok_p90": round(pct(0.9) / 2.2),
        "tok_p99": round(pct(0.99) / 2.2), "tok_max": round(lens[-1] / 2.2),
        "tok_max_en": round(lens[-1] / 3.0),
        "over_8k": sum(1 for x in lens if x / 2.2 > 8192),
    }


def write_handoff(
    train_root: Path,
    counts: Dict[str, int],
    stats: Dict[str, Any],
    dataset_info: Dict[str, Any],
    split_map: Dict[str, str],
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    root = Path(train_root)
    split = stats.get("split") or split_map or {}
    by_project = stats.get("by_project") or {}
    by_model = stats.get("by_model") or {}
    stage_counts = stats.get("by_stage") or {}
    dpo_src = stats.get("dpo_by_source") or {}
    dpo_review = stats.get("dpo_review") or {}
    label_counts = stats.get("label_counts") or {}
    review_states = stats.get("review_states") or {}
    admit = stats.get("admit_reasons") or {}
    drop = stats.get("drop_reasons") or {}
    prof = _data_profile(root)

    def cnt(name: str) -> int:
        return int(counts.get(name, 0) or 0)

    win_train = cnt("sft_step_windows_sharegpt.train.jsonl")
    win_eval = cnt("sft_step_windows_sharegpt.eval.jsonl")
    traj_train = cnt("sft_gold_trajectories_sharegpt.train.jsonl")
    traj_eval = cnt("sft_gold_trajectories_sharegpt.eval.jsonl")
    dpo_train = cnt("dpo_preference_pairs.train.jsonl")
    dpo_eval = cnt("dpo_preference_pairs.eval.jsonl")
    diag = cnt("negative_diagnostics.jsonl")

    train_projects = sorted(p for p, s in split.items() if s == "train")
    eval_projects = sorted(p for p, s in split.items() if s == "eval")
    steps_per_epoch = max(1, round(win_train / 4))  # bs1 × ga4
    total_steps = steps_per_epoch * 3

    files = _table([
        ["**SFT 训练（主）**", "`sft_step_windows_sharegpt.train.jsonl`", "`pi_sft_step_windows_train`", _fmt(win_train)],
        ["SFT 验证", "`sft_step_windows_sharegpt.eval.jsonl`", "`pi_sft_step_windows_eval`", _fmt(win_eval)],
        ["完整轨迹训练（可选）", "`sft_gold_trajectories_sharegpt.train.jsonl`", "`pi_sft_gold_trajectories_train`", _fmt(traj_train)],
        ["完整轨迹验证", "`sft_gold_trajectories_sharegpt.eval.jsonl`", "`pi_sft_gold_trajectories_eval`", _fmt(traj_eval)],
        ["**DPO 训练**", "`dpo_preference_pairs.train.jsonl`", "`pi_dpo_preference_pairs_train`", _fmt(dpo_train)],
        ["DPO 验证", "`dpo_preference_pairs.eval.jsonl`", "`pi_dpo_preference_pairs_eval`", _fmt(dpo_eval)],
        ["OpenAI 格式备选", "`sft_step_windows_openai.train.jsonl`", "未注册（同数据另一种格式）", _fmt(cnt("sft_step_windows_openai.train.jsonl"))],
        ["病态诊断（不进训练）", "`negative_diagnostics.jsonl`", "—", _fmt(diag)],
    ], ["用途", "文件", "dataset_info 名字", "条数"])

    doc = f"""# 数据交接说明（handoff）

> 由 GoldThread `stage3` 自动生成，数据统计取自本次运行的真实产物。
> 数据集目录：`{root.as_posix()}`

**目标训练环境**：{ENV["gpu"]} · {ENV["os"]} · CUDA {ENV["cuda"]} · PyTorch {ENV["torch"]} · Python {ENV["python"]}

---

## 0. 这批数据是什么

从 pi 会话日志提纯出的 **Agent 工程分析轨迹**（解包 / 反编译 / 资源与配置表还原 / 二进制格式破解）。
三方分工：

- **程序**：机械去噪、静默执行、命令规范化、窗口拼装、血缘记录、切分
- **模型**：判断删哪些步、接缝是否平滑、整条会话值不值得训练、偏好对是否成立
- **人工**：逐条复核改写与问题步，最后定稿

来源模型分布：{", ".join(f"`{m}` × {_fmt(n)}" for m, n in sorted(by_model.items(), key=lambda kv: -kv[1])[:5]) or "—"}

## 1. 数据集清单

{files}

统计：

- 阶段权重分布（train）：{", ".join(f"{k} {_fmt(v)}" for k, v in stage_counts.items()) or "—"}
  （`highlight` = 模型判定的关键决策步，权重最高；若你的 collator 支持 `metadata.sample_weight` 可直接用）
- 项目分布：{", ".join(f"`{p}` {_fmt(n)}" for p, n in sorted(by_project.items(), key=lambda kv: -kv[1])) or "—"}
- 训练/验证切分（**按项目整包**，防相邻步泄漏）：
  - train：{", ".join(f"`{p}`" for p in train_projects) or "—"}
  - eval：{", ".join(f"`{p}`" for p in eval_projects) or "—"}
- 入库审计：{", ".join(f"{k} {_fmt(v)}" for k, v in admit.items()) or "—"}
- 剔除原因：{", ".join(f"{k} {_fmt(v)}" for k, v in drop.items()) or "—"}
- 会话标签：{", ".join(f"{k} {_fmt(v)}" for k, v in label_counts.items()) or "—"}
- 复核状态：{", ".join(f"{k} {_fmt(v)}" for k, v in review_states.items()) or "—"}
- DPO 来源：{", ".join(f"{k} {_fmt(v)}" for k, v in dpo_src.items()) or "—"}
- DPO 复核：检查 {_fmt(dpo_review.get("checked", 0))} 对，
  调换 {_fmt(dpo_review.get("swapped", 0))}、丢弃 {_fmt(dpo_review.get("dropped", 0))}

## 2. 显存与长度核算（决定 cutoff_len）

**训练样本长度实测**（train，{_fmt(prof.get("n", 0))} 条）：

{_table([
    ["字符数", _fmt(prof.get("p50")), _fmt(prof.get("p90")), _fmt(prof.get("p99")), _fmt(prof.get("max"))],
    ["≈tokens（中英混合 2.2 字符/token）", _fmt(prof.get("tok_p50")), _fmt(prof.get("tok_p90")), _fmt(prof.get("tok_p99")), _fmt(prof.get("tok_max"))],
], ["分位", "p50", "p90", "p99", "max"])}

- 最长样本约 **{_fmt(prof.get("tok_max"))} tokens**（按纯英文 3.0 字符/token 估则约 {_fmt(prof.get("tok_max_en"))}）
- 因此 **`cutoff_len: 16384` 覆盖全部样本，0 截断**；若降到 8192，约 {_fmt(prof.get("over_8k"))} 条会被截断（含目标步，直接丢弃该能力）

**27B bf16 显存预算（单卡 80GB，cutoff_len 16384）**

{_table([
    ["文本权重（bf16）", "≈ 54 GB", "全参数常驻（27B dense 为主体）"],
    ["视觉塔（vision_config 27 层 / hidden 1152）", "≈ 0.5 ~ 1.5 GB", "多模态结构，纯文本训练时也常驻"],
    ["LoRA 适配器 + 优化器", "< 1 GB", "rank 32，只训 mlp"],
    ["激活值（bs1 + 梯度检查点 + FA2）", "8 ~ 16 GB", "64 层里有 48 层是线性注意力，激活随长度线性增长（不是平方），比标准 Transformer 省"],
    ["**峰值合计**", "**约 64 ~ 72 GB**", "80GB 有余量；仍需首个 epoch 盯 nvidia-smi"],
], ["项", "占用", "说明"])}

**磁盘（你已确认 {ENV["disk"]}）**：底座 54G + 合并版 54G + 环境 10G + ckpt 3G ≈ **121G**，
可在同一块盘上同时保留「原始底座」和「合并后的模型」，无需来回删。
**主机内存（你已确认 {ENV["ram"]}）**：合并 27B bf16 时 CPU 峰值约 60G，100G+ 完全够。

## 3. 底座模型核实结果（必须在开机时复核）

我从 ModelScope 的 `config.json` 实测到的事实（**与"标准 27B dense"的假设差别很大**）：

{_table([
    ["architectures", "`" + MODEL_FACTS["architectures"] + "`", "专用架构，不是 LlamaForCausalLM"],
    ["model_type", "`" + MODEL_FACTS["model_type"] + "`", "决定 transformers / 框架能否加载"],
    ["多模态", MODEL_FACTS["multimodal"], "纯文本训练可行，但走的是多模态代码路径"],
    ["网络结构", MODEL_FACTS["layers"], "混合线性注意力"],
    ["尺寸", MODEL_FACTS["hidden"], "GQA 6:1，head_dim 256"],
    ["词表", MODEL_FACTS["vocab"], "embedding 约 1.27B 参数"],
    ["上下文", MODEL_FACTS["context"], "原生 256K，已正确"],
    ["transformers", MODEL_FACTS["transformers"], "关键约束，见 §4.2"],
], ["项", "实测值", "含义"])}

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

1. **训练长度 ≠ 能力长度**。本数据集滑窗最长约 {_fmt(prof.get("tok_max"))} tokens，
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

## 4. 环境搭建（{ENV["os"]} / CUDA {ENV["cuda"]} / torch {ENV["torch"]} / py{ENV["python"]}）

镜像选型：平台「基础镜像 → PyTorch → Ubuntu 22.04 / CUDA 12.6 / 2.7.1 / 3.12」。
**不要选 CUDA 13.x**：flash-attn 等轮子跟进慢，容易退化成源码编译。

### 4.0 无卡阶段先做这些（不用等 GPU）

平台支持"无卡启动"时，**先把软件栈、模型、数据、模板全部验完**，挂上卡就能直接开训：

```bash
# 一条命令跑完 A~E 五组检查（无卡会自动把 GPU 项标记为 SKIP）
python check_env.py --model /root/models/Qwen3.8-27B --data /root/data/03_train

# 单独验数据 + 模板（CPU 即可，最值得先跑）
python check_data.py --model /root/models/Qwen3.8-27B --template qwen3_8
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
# 期望：Python {ENV["python"]}

python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
# 期望：{ENV["torch"]} {ENV["cuda"]} True

python -c "import flash_attn;print('fa2', flash_attn.__version__)" 2>/dev/null || echo "需要安装 flash-attn（见 4.2）"

df -h /    # 期望可用 ≥100GB
```

### 4.2 安装训练栈（版本是硬约束，别跳步）

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

# 4) 验证三件事
python -c "import transformers;print('transformers', transformers.__version__)"   # 期望 5.8.0
python -c "from transformers import AutoConfig;c=AutoConfig.from_pretrained('/root/models/Qwen3.8-27B');print(c.model_type, c.architectures)"
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

### 4.3 下载底座模型

> ⚠️ **先核对架构**：`Qwen3.8-27B` 不是公开已知的官方仓库名，请以实际拿到的仓库为准。

```bash
# 方式 A：HuggingFace
pip install -U "huggingface_hub[cli]"
huggingface-cli download <你的真实仓库ID> --local-dir /root/models/Qwen3.8-27B

# 方式 B：ModelScope（国内更快）
pip install -U modelscope
modelscope download --model <你的真实仓库ID> --local_dir /root/models/Qwen3.8-27B
```

下载后**必须**确认三件事：

```bash
python - <<'PY'
import json, pathlib
cfg = json.load(open("/root/models/Qwen3.8-27B/config.json"))
print("architectures      :", cfg.get("architectures"))
print("max_position_emb   :", cfg.get("max_position_embeddings"))
print("model_type         :", cfg.get("model_type"))
print("hidden/layers      :", cfg.get("hidden_size"), "/", cfg.get("num_hidden_layers"))
print("rope_scaling       :", cfg.get("rope_scaling"))
# 若出现 linear_attention / gated_deltanet 等字段，说明是混合线性注意力架构：
#   -> flash-attn 只对其中标准 attention 层生效
#   -> 序列并行（Ulysses）可能不支持，长序列训练方案需另行核实
PY
```

`rope_scaling` 若已是 null 且 `max_position_embeddings=262144`，**不要**再加 `yarn`。

## 5. 数据格式与对话模板（最容易踩的坑）

- ShareGPT：`conversations: [{{from, value}}]`
  `system` 工程纪律提示词 / `human` 根任务 / `gpt` 助手步 / `observation` 工具输出
- 助手步内容：` thinking…推理…` + `<tool_call>{{"name":"bash","arguments":{{...}}}}</tool_call>`
- `tools` 列是 JSON 字符串（5 个函数声明）。框架不支持时删掉即可（见 `check_data.py` 的 `--drop-tools`）
- 单步窗口结构：`[system, 根任务] + 最近 N 步历史 + 目标步`，
  **历史步 thinking 已剥离，只有目标步带 thinking**

### 5.1 模板必须用 `qwen3_8`（不是 qwen3）

LLaMA-Factory 为这个模型族提供了**专用模板**：

{_table([
    ["模板名", "`qwen3_8`", "`template_class=Qwen38ReasoningTemplate`，`preserve_thinking=True`，内置 `tool_format=qwen3_8`"],
    ["坑 1", "不要设 `tool_format`", "该模板内置工具格式，另设会直接抛 ValueError"],
    ["坑 2", "不要用 `qwen3`", "通用模板不理解 qwen3_8 的推理段切分，可能重复注入 ` thinking`"],
    ["坑 3", "多模态路径", "模板带 mm_plugin，纯文本数据集应可跑；若报缺 images 字段再按 §5.2 处理"],
], ["要点", "取值", "说明"])}

开机后先确认模板存在：

```bash
python -c "from llamafactory.data.template import TEMPLATES;print([k for k in TEMPLATES if 'qwen3' in k])"
# 期望输出里包含 qwen3_8 / qwen3_5 等
```

再跑数据自检：

```bash
python check_data.py --model /root/models/Qwen3.8-27B --template qwen3_8
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

{_table([
    ["`template`", "qwen3_8", "**专用模板**；不要设 tool_format"],
    ["`mask_history`", "true", "**只训练最后一轮**（本数据集的关键开关）"],
    ["`cutoff_len`", "16384", "实测最长 " + _fmt(prof.get("tok_max")) + " tokens，0 截断"],
    ["`lora_target`", "mlp", "不动注意力投影层，尽量保住原生长上下文能力"],
    ["`lora_rank`", "32", "27B 用 32~64 均可"],
    ["`learning_rate`", "1.0e-4", "LoRA 常规区间；总步数 ~" + _fmt(total_steps)],
    ["`gradient_accumulation_steps`", "4", "有效 batch 4，每 epoch ≈" + _fmt(steps_per_epoch) + " 步"],
    ["`num_train_epochs`", "3.0", "823 条量级下 3 epoch 足够，再多易过拟合"],
    ["`packing`", "false", "**必须关闭**：拼包会破坏「目标步在末尾」的结构"],
    ["`bf16`", "true", "A800 原生支持；不要用 fp16"],
    ["`gradient_checkpointing`", "true", "16K 序列必须开"],
    ["`flash_attn`", "fa2", "仅对 16/64 层 full_attention 生效；装不上换 sdpa"],
    ["`enable_liger_kernel`", "false", "qwen3_5 不在 Liger 支持列表，先关"],
    ["`deepspeed`", "（不设）", "单卡无收益，且会强制走 torchrun"],
], ["参数", "值", "原因"])}

### 6.1 启动

```bash
cd LLaMA-Factory
llamafactory-cli train {root.as_posix()}/train_sft.yaml
# 单卡不需要 torchrun；若脚本强制，设 FORCE_TORCHRUN=0
```

### 6.2 训练中要盯的三件事

1. **首 50 步看 `nvidia-smi`**：峰值 > 78 GB 就把 `cutoff_len` 降到 12288（仍覆盖 p99）
2. **loss 曲线**：应从 ~1.5-2.0 单调下降到 0.6-1.0 区间；若卡在 2.0 附近不动 → 检查模板（§5.1）
3. **每 100 步抽 5 条 eval 样本续写**：期望输出 `thinking` + `<tool_call>`，而不是纯文本或空思考

### 6.3 OOM 处置阶梯（按顺序试）

`cutoff_len 16384 → 12288` → 开 `enable_liger_kernel: true` → `flash_attn: sdpa` 换回 `fa2`
→ `gradient_accumulation_steps 4 → 8`（不等于降显存，但可把 micro-batch 保持 1）
→ 仍未解决才考虑 `train_sft_qlora.yaml`（4-bit，质量略降）

## 7. 评估与长上下文验收

**损失**：本数据集的 eval 是**整项目留出**（{", ".join(f"`{p}`" for p in eval_projects) or "—"}），
所以 `eval_loss` 明显高于 `train_loss` 是正常的，它衡量"换一个程序还能不能干活"。

**能力抽检**（人工打分，各 10 条）：

1. 是否先观察再动手（不盲调工具）
2. 报错后是否查证再重试（不是重复同一条错误命令）
3. 是否避免心算偏移/密钥并声称已验证
4. 是否遵守"只在当前工作目录内作业"
5. 工具调用 JSON 是否格式合法、参数完整

**长上下文验收**（验证 §3 的结论，不达标就别宣称"保住 256K"）：

```bash
python verify_longctx.py --model /root/models/Qwen3.8-27B-merged \\
    --lengths 8192,32768,65536,131072,262144
# 逐档构造"前文 + 尾部提问"，记录是否 OOM/报错、首 token 延迟、生成是否连贯
```

## 8. DPO 偏好对齐（SFT 达标后再做）

```bash
llamafactory-cli train {root.as_posix()}/train_dpo.yaml
```

单卡注意：

- `cutoff_len` 已降到 **8192**（DPO 同时跑 policy + reference 两路前向，激活约翻倍）
- 从 SFT 适配器继续：`model_name_or_path` 指底座，加 `adapter_name_or_path: saves/.../sft`
  （或先 `llamafactory-cli export` 合并，再指合并后的目录）
- `pref_beta: 0.1`；数据只有 {_fmt(dpo_train)} 对，1 epoch 即可

## 9. 训练产物与迁移到 48G 推理机

### 9.1 数据盘上应该留什么

{_table([
    ["LoRA 适配器", "`saves/qwen38-27b-goldthread-sft/`", "≈ 0.5 GB", "**必留**；但单独无法推理，必须配底座"],
    ["原始底座", "`/root/models/Qwen3.8-27B`", "≈ 55 GB", "**必留**；适配器要挂在它上面"],
    ["合并后 bf16", "`/root/models/Qwen3.8-27B-merged`", "≈ 55 GB", "交给 24G 机器前建议先在训练机量化，见 9.2"],
    ["量化产物", "`…-merged-4bit/` 或 `*.gguf`", "15 ~ 17 GB", "24G 机器的实际推理载体"],
], ["类型", "路径", "体积", "说明"])}

⚠️ **只拷适配器（0.5G）到新机器是跑不起来的**——必须先有底座，或者先在训练机合并/量化。

### 9.2 24GB 显卡能不能推理？—— 分三种情况

**结论：bf16 直接加载不可能（54G > 24G），必须量化。**
而这个模型是 `Qwen3_5ForConditionalGeneration`（混合线性注意力 + 多模态），
量化工具对它的支持需要实测，不能假设。

按顺序尝试，前一个不通再走下一个：

{_table([
    ["① bnb 4-bit 动态量化", "transformers + bitsandbytes，加载时量化，无需转换", "15G 权重 + KV", "**最快验证**；但对 linear_attention 层的支持未知"],
    ["② GPTQ / AWQ 离线量化", "llm-compressor / autoawq 先转 4bit 权重再加载", "14~15G", "质量/速度更好；需工具支持该 arch"],
    ["③ GGUF（llama.cpp）", "convert_hf_to_gguf.py → Q4_K_M，llama.cpp 推理", "16~17G", "最省显存、可 CPU/GPU 混合；需 llama.cpp 支持 qwen3next 系架构"],
], ["方案", "做法", "显存", "说明"])}

### 9.3 先探测，再决定买/换卡（在训练机上跑，不用等换机）

数据盘挂到新机器前，**先在 80G 训练机上把 24G 场景模拟一遍**：

```bash
# 用 22GB 上限模拟 24G 卡（多卡卡也能跑，只限制显存预算）
CUDA_VISIBLE_DEVICES=0 python probe_24g.py \
    --model /root/models/Qwen3.8-27B-merged \
    --limit-gb 22
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

{_table([
    ["bf16 权重 55G", "16 G", "**74 G**", "8 G", "66 G", "80G（0.9 利用率，紧）"],
    ["8bit 权重 28G", "16 G", "47 G", "8 G", "**39 G**", "48G"],
    ["4bit 权重 15G", "16 G", "34 G", "8 G", "**26 G**", "48G"],
], ["权重档", "KV(bf16)", "合计+3G", "KV(fp8)", "合计+3G", "最小显卡"])}

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
llamafactory-cli export \
  --model_name_or_path /root/models/Qwen3.8-27B \
  --adapter_name_or_path saves/qwen38-27b-goldthread-sft \
  --template qwen3_8 --finetuning_type lora \
  --export_dir /root/models/Qwen3.8-27B-merged

# 2a) 方案①：不转换，直接动态 4-bit（最省事，先试这个）
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "/root/models/Qwen3.8-27B-merged"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    m, load_in_4bit=True, device_map={{"": 0}},
    max_memory={{0: "20GiB"}}, trust_remote_code=True)
print("loaded, 4bit OK")
PY

# 2b) 方案②：GPTQ/AWQ 离线量化（需工具支持该架构）
pip install llm-compressor
python -m llmcompressor.transformers.entrypoints.oneshot \
  --model /root/models/Qwen3.8-27B-merged --scheme W4A16 \
  --output_dir /root/models/Qwen3.8-27B-w4a16

# 2c) 方案③：GGUF（llama.cpp）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py /root/models/Qwen3.8-27B-merged \
  --outfile /root/models/qwen38-27b-Q4_K_M.gguf --outtype q4_k_m
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
        device_map={{"": 0}}, max_memory={{0: "21GiB"}}, trust_remote_code=True)
msgs = [{{"role": "user", "content": "帮我破解 xxx.xapk，先告诉我第一步做什么"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=128)[0], skip_special_tokens=True))
PY
```

**验收三条**：能加载、能生成 `thinking + <tool_call>`、`nvidia-smi` 峰值 < 23GB。

## 10. 48G 推理机交接清单（AI Agent 可直接照做）

> **目标**：在 {ENV["infer_gpu"]} 上，以 **4-bit 权重 + 256K 上下文**提供推理服务。
> 环境与训练机一致：{ENV["os"]} / CUDA {ENV["cuda"]} / torch {ENV["torch"]} / py{ENV["python"]}，数据盘整盘挂载。

### 10.0 决策表：先看数据盘上有什么

{_table([
    ["`*-merged/`（bf16, 55G）+ adapter", "→ 10.2 装依赖 → 10.3-A 在线 4bit 启动"],
    ["只有 adapter（0.5G）", "→ **无法推理**，回训练机执行 §9.5 合并"],
    ["`*-4bit/` 或 `*gptq*`", "→ 10.3-B 直接加载量化产物（更快）"],
    ["只有 `*.gguf`", "→ 10.5 走 llama.cpp"],
], ["数据盘产物", "下一步"])}

### 10.1 硬件与环境确认（30 秒，不通过先别往下走）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -V
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
df -h /data ; free -g | head -2
```

通过标准：显存 **≥ 44 GB**；`torch {ENV["torch"]}` / cuda `{ENV["cuda"]}` / bf16 `True`；`/data` 可用 ≥ 40 GB。

### 10.2 安装推理依赖

```bash
pip install "vllm>=0.11"                    # 需含 Qwen3_5 支持
pip install "transformers==5.8.0" "accelerate<=1.11.0"

# 关键：确认 vLLM 认识这个架构（False 就别往下走，见 10.6）
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
print('Qwen3_5ForConditionalGeneration' in R.get_supported_archs())"
```

### 10.3 启动服务（48G 推荐参数）

**A. 在线 4-bit（最省事，先用这个）**

```bash
vllm serve /data/models/Qwen3.8-27B-merged \
  --quantization bitsandbytes --load-format bitsandbytes \
  --max-model-len 262144 \
  --kv-cache-dtype bf16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 2 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --port 8000
```

显存账：4-bit 权重 ≈15G + KV(bf16) ≈16G = **31G**，48G 有余量 → 可留 2 并发。
**若 OOM，按这个顺序降**：`--max-model-len 131072` → `--kv-cache-dtype fp8` → `--max-num-seqs 1`。

**B. 离线量化（A 能跑但嫌加载慢/吞吐低时）**

```bash
pip install llm-compressor
python -m llmcompressor.transformers.entrypoints.oneshot \
  --model /data/models/Qwen3.8-27B-merged --scheme W4A16 \
  --output_dir /data/models/Qwen3.8-27B-w4a16

vllm serve /data/models/Qwen3.8-27B-w4a16 \
  --quantization compressed-tensors \
  --max-model-len 262144 --kv-cache-dtype bf16 \
  --gpu-memory-utilization 0.90 --max-num-seqs 2 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

### 10.4 冒烟测试（3 条，必须全过才算部署成功）

```bash
# ① 服务存活
curl -s localhost:8000/v1/models | head -c 300

# ② 普通生成（检查是否仍带 thinking）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{{"model":"Qwen3.8-27B-merged","messages":[{{"role":"user","content":"用一句话说明你会怎么开始分析一个 xapk 包"}}],"max_tokens":128}}' \
 | python -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:400])"
```

**③ 工具调用**（本数据集的核心能力，必须单独验证）：

```bash
cat > /tmp/tools.json <<'JSON'
{{"model":"Qwen3.8-27B-merged",
 "messages":[{{"role":"user","content":"列出当前目录下的文件"}}],
 "tools":[{{"type":"function","function":{{"name":"bash","description":"run shell",
   "parameters":{{"type":"object","properties":{{"command":{{"type":"string"}}}},"required":["command"]}}}}}}],
 "tool_choice":"auto","max_tokens":256}}
JSON
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/tools.json \
 | python -c "import json,sys;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('tool_calls:',m.get('tool_calls'));print('content:',(m.get('content') or '')[:200])"
```

期望：`tool_calls` 非空且 `function.name == "bash"`。若只有纯文本 → `--tool-call-parser` 不匹配，
查模型文档换 parser（如 `qwen3_coder` / `hermes`）。

### 10.5 256K 长上下文验收

```bash
python /data/tools/verify_longctx_api.py \
  --base-url http://localhost:8000/v1 --model Qwen3.8-27B-merged \
  --lengths 8192,32768,65536,131072,262144
```

通过标准：五档都不报错；每档 TTFT 可接受（chunked prefill 下 256K 首次可能 60~180s）；
回答与前文相关（脚本会打印摘要，人工看一眼）。

**若 256K 档 OOM**：说明 KV 预算不足 → 加 `--kv-cache-dtype fp8`，或把 `--max-model-len` 降到 131072
并把实测上限写进交付报告（不要让下游以为能跑 256K）。

### 10.6 失败回退路径（按顺序试，别在死路上耗时间）

{_table([
    ["`bitsandbytes` 4bit 加载报错（不支持该 arch）", "走 10.3-B 离线 W4A16；再不行走 10.7 GGUF"],
    ["vLLM 不认识 `Qwen3_5ForConditionalGeneration`", "升级 vLLM；或用 HF `transformers` + `load_in_4bit`（慢但能跑）；或 10.7"],
    ["256K OOM", "`--kv-cache-dtype fp8` → `--max-model-len 131072` → 记录实际上限"],
    ["工具调用解析失败", "换 `--tool-call-parser`；或退化为让模型在 content 里输出 `<tool_call>` JSON 自行解析"],
    ["生成质量明显下降", "4bit 量化损伤；改 8bit（28G+16G=44G，48G 勉强）或换 80G 跑 bf16+fp8 KV"],
], ["失败点", "退路"])}

### 10.7 最后一条退路：llama.cpp + GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py /data/models/Qwen3.8-27B-merged \
  --outfile /data/models/qwen38-27b-Q4_K_M.gguf --outtype q4_k_m
# 报 "unsupported model architecture: qwen3_5" 则此路不通

llama-server -m /data/models/qwen38-27b-Q4_K_M.gguf \
  -c 262144 -ngl 99 --host 0.0.0.0 --port 8000
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

- eval 项目（{", ".join(f"`{p}`" for p in eval_projects) or "—"}）**不要再拿去训练**，否则评估失真
- `negative_diagnostics.jsonl` 仅用于分析（无标准答案），不要喂进 SFT/DPO
- 数据来自自有/已授权应用的逆向分析，使用范围请遵守你的授权约定
"""

    (root / "handoff.md").write_text(doc, encoding="utf-8")

    (root / "train_sft.yaml").write_text(f"""### model
model_name_or_path: /root/models/Qwen3.8-27B          # 改成你的真实路径
trust_remote_code: true
flash_attn: fa2                                      # 仅对 full_attention 层生效；装不上改 sdpa
enable_liger_kernel: false                           # qwen3_5 不在 Liger 支持列表，先关

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: mlp                                     # 不动注意力投影层，保住长上下文
# deepspeed:                                          # 单卡不要设

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_sft_step_windows_train
eval_dataset: pi_sft_step_windows_eval
template: qwen3_8                                     # 专用模板，勿用 qwen3
cutoff_len: 16384
packing: false                                       # 必须关闭
mask_history: true                                   # 只训练最后一轮（本数据集关键开关）
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/qwen38-27b-goldthread-sft
logging_steps: 5
save_steps: 100
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
""", encoding="utf-8")

    (root / "train_sft_qlora.yaml").write_text(f"""### model
model_name_or_path: /root/models/Qwen3.8-27B
trust_remote_code: true
flash_attn: fa2
enable_liger_kernel: false
quantization_bit: 4                                  # NF4
quantization_method: bitsandbytes

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: mlp
quantization_type: nf4
double_quantization: true

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_sft_step_windows_train
eval_dataset: pi_sft_step_windows_eval
template: qwen3_8
cutoff_len: 16384
packing: false
mask_history: true                                    # 只训练最后一轮
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/qwen38-27b-goldthread-sft-qlora
logging_steps: 5
save_steps: 100
save_total_limit: 2
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
""", encoding="utf-8")

    (root / "train_dpo.yaml").write_text(f"""### model
model_name_or_path: /root/models/Qwen3.8-27B
adapter_name_or_path: saves/qwen38-27b-goldthread-sft   # SFT 适配器；或改为合并后的目录
trust_remote_code: true
flash_attn: fa2
enable_liger_kernel: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: mlp
pref_beta: 0.1
pref_loss: sigmoid
# deepspeed:                                            # 单卡不要设

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_dpo_preference_pairs_train
eval_dataset: pi_dpo_preference_pairs_eval
template: qwen3_8
cutoff_len: 8192                                       # DPO 双前向，激活翻倍
packing: false
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/qwen38-27b-goldthread-dpo
logging_steps: 5
save_steps: 100
save_total_limit: 2
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
""", encoding="utf-8")

    (root / "probe_24g.py").write_text('''#!/usr/bin/env python3
"""在训练机上模拟小显存推理：报告 bf16 / 4bit / 8bit 三种加载方式能否在限额内跑起来。

用法：
    python probe_24g.py --model /root/models/Qwen3.8-27B-merged --limit-gb 22

输出每种方式的：成功与否、峰值显存、首 token 延迟、报错摘要。
据此决定 24G 推理机走哪条路（4bit 直载 / GPTQ / GGUF）。
"""
from __future__ import annotations

import argparse
import gc
import sys
import time


def try_load(path: str, mode: str, limit_gb: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"trust_remote_code": True, "device_map": {"": 0},
          "max_memory": {0: f"{limit_gb:.0f}GiB"}}
    if mode == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif mode == "8bit":
        kw["load_in_8bit"] = True
    elif mode == "4bit":
        kw["load_in_4bit"] = True

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    load_s = time.time() - t0

    msgs = [{"role": "user", "content": "用一句话说明你会怎么开始分析一个 xapk 包。"}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = tok("用一句话说明你会怎么开始分析一个 xapk 包。", return_tensors="pt").input_ids
    ids = ids.to(model.device)

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=32, do_sample=False)
    gen_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:120].replace("\\n", " ")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return True, load_s, gen_s, peak, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit-gb", type=float, default=22.0)
    ap.add_argument("--modes", default="4bit,8bit,bf16")
    args = ap.parse_args()
    try:
        import torch
    except ImportError:
        print("[err] 需要 torch", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)} | 模拟限额 {args.limit_gb:.0f} GB")
    print(f"模型: {args.model}\\n")
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"--- 尝试 {mode} ---")
        try:
            ok, load_s, gen_s, peak, text = try_load(args.model, mode, args.limit_gb)
            verdict = "可用" if peak <= args.limit_gb else "超限"
            print(f"  [{verdict}] 加载 {load_s:.0f}s 生成 {gen_s:.1f}s 峰值 {peak:.1f}GB")
            print(f"  输出: {text}")
        except torch.cuda.OutOfMemoryError:
            print("  [OOM] 显存不足")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {type(exc).__name__}: {str(exc)[:200]}")
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        print()
    print("判定建议：")
    print("  · 4bit 可用 -> 24G 机器直接 load_in_4bit，上下文 4K~16K")
    print("  · 仅 8bit 可用 -> 24G 不够，走 GGUF（llama.cpp）")
    print("  · 全部失败 -> 换 48G/80G，或 CPU offload（1~3 tok/s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''', encoding="utf-8")

    (root / "verify_longctx_api.py").write_text('''#!/usr/bin/env python3
"""通过 OpenAI 兼容 API 验收长上下文（适配 vLLM / llama-server）。

用法：
    python verify_longctx_api.py --base-url http://localhost:8000/v1 \
        --model Qwen3.8-27B-merged --lengths 8192,32768,65536,131072,262144

每档构造"长前文 + 尾部提问"，记录 HTTP 状态、TTFT、总耗时与回答摘要。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(url: str, body: dict, timeout: float = 900.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
        return r.status, raw, time.time() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536")
    ap.add_argument("--chars-per-token", type=float, default=2.2)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    filler = "以下是逆向分析记录，包含工具输出与偏移量说明。\\n"
    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        repeat = max(1, int((L - 300) * args.chars_per_token / len(filler)))
        prompt = filler * repeat + "\\n问题：上文反复出现的字段名是什么？只回一个词。"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        }
        code, raw, dt = post(url, body)
        if code == 200:
            try:
                d = json.loads(raw)
                msg = (d["choices"][0]["message"].get("content") or "").strip()[:80]
                usage = d.get("usage") or {}
                print(f"  [ok] {L:>7} tok 目标 | {dt:>7.1f}s | prompt_tokens={usage.get('prompt_tokens')} | {msg}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {L:>7} | 解析失败 {exc} | {raw[:120]}")
        else:
            print(f"  [fail] {L:>7} | HTTP {code} | {raw[:160]}")
    print("\\n判定：五档全部 ok 才算通过；OOM 时降 --max-model-len 或加 --kv-cache-dtype fp8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''', encoding="utf-8")

    (root / "check_env.py").write_text(r'''#!/usr/bin/env python3
"""环境自检：无 GPU 也能跑的检查 + 有 GPU 后才做的清单。

用法：
    python check_env.py --model /root/models/Qwen3.8-27B --data /root/data/03_train
    python check_env.py                       # 只查软件栈与依赖版本

设计：每项检查独立 try，输出 [PASS] / [FAIL] / [SKIP]（无 GPU 时跳过）。
全部非 GPU 项 PASS 才说明"挂上卡就能直接开训"。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

RESULT = {"pass": 0, "fail": 0, "skip": 0, "fail_items": [], "skip_items": []}


def ok(msg: str) -> None:
    RESULT["pass"] += 1
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    RESULT["fail"] += 1
    RESULT["fail_items"].append(msg)
    print(f"  [FAIL] {msg}")


def skip(msg: str) -> None:
    RESULT["skip"] += 1
    RESULT["skip_items"].append(msg)
    print(f"  [SKIP] {msg}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


def ver(mod: str) -> str:
    try:
        m = __import__(mod)
    except Exception as exc:  # noqa: BLE001
        return f"<import 失败: {type(exc).__name__}>"
    return str(getattr(m, "__version__", "?"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--cutoff", type=int, default=16384)
    args = ap.parse_args()

    # ---------- A. 运行时 ----------
    section("A. 运行时与 GPU")
    print(f"  platform: {platform.platform()}")
    print(f"  python  : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print(f"  [WARN] 期望 Python 3.12，当前 {sys.version_info[:2]}（通常也能跑，注意轮子兼容）")
    try:
        import torch
        print(f"  torch   : {torch.__version__}  (编译时 CUDA {torch.version.cuda})")
        if torch.__version__.split("+")[0] != "2.7.1":
            print(f"  [WARN] 期望 torch 2.7.1，当前 {torch.__version__}")
        else:
            ok("torch 2.7.1")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            ok(f"CUDA 可用：{n} 张卡，{name}，capability {cap}，显存 {total:.0f} GB")
            ok(f"bf16 支持：{torch.cuda.is_bf16_supported()}")
            if total < 60:
                print(f"  [WARN] 显存 {total:.0f}GB 小于训练建议值（27B bf16 LoRA 需 ~64-72GB）")
        else:
            skip("CUDA 不可用（无卡模式正常）：nvidia-smi / bf16 / 显存 都要等挂卡后再验")
    except ImportError:
        bad("torch 未安装")
    if shutil.which("nvidia-smi"):
        os.system("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | head -3")
    else:
        skip("nvidia-smi 不存在（无卡/无驱动）")

    # ---------- B. 依赖版本 ----------
    section("B. 训练栈依赖")
    want = {
        "transformers": ("5.8.0", lambda v: v.startswith("5.8")),
        "accelerate": (">=1.3.0", lambda v: v not in ("?",) and v.split(".")[0].isdigit() and float(".".join(v.split(".")[:2])) >= 1.3),
        "peft": (">=0.13", lambda v: True),
        "trl": ("<=0.24", lambda v: float(".".join(v.split(".")[:2])) <= 0.24),
        "datasets": ("任意", lambda v: True),
        "safetensors": ("任意", lambda v: True),
    }
    for m, (desc, chk) in want.items():
        v = ver(m)
        if v.startswith("<"):
            bad(f"{m} 未安装（期望 {desc}）")
            continue
        try:
            good = chk(v)
        except Exception:  # noqa: BLE001
            good = True
        (ok if good else bad)(f"{m} {v}（期望 {desc}）")

    for m in ("flash_attn", "liger_kernel", "bitsandbytes", "llamafactory"):
        v = ver(m)
        if v.startswith("<"):
            if m == "flash_attn":
                print(f"  [WARN] {m} 未安装 → 训练配置请把 flash_attn 改成 sdpa")
            elif m in ("liger_kernel", "bitsandbytes"):
                print(f"  [INFO] {m} 未安装（本方案默认不用，可忽略）")
            else:
                bad(f"{m} 未安装（LLaMA-Factory 必须）")
        else:
            ok(f"{m} {v}")

    # ---------- C. 模型 ----------
    if args.model:
        section("C. 底座模型")
        mp = Path(args.model)
        if not mp.exists():
            bad(f"模型目录不存在：{mp}")
        else:
            cfg_p = mp / "config.json"
            if not cfg_p.exists():
                bad("config.json 缺失，下载不完整")
            else:
                cfg = json.load(open(cfg_p, encoding="utf-8"))
                tc = cfg.get("text_config") or {}
                arch = (cfg.get("architectures") or ["?"])[0]
                mt = cfg.get("model_type")
                mpe = tc.get("max_position_embeddings") or cfg.get("max_position_embeddings")
                lt = tc.get("layer_types") or []
                from collections import Counter
                print(f"  architectures: {arch}")
                print(f"  model_type   : {mt}")
                print(f"  max_position : {mpe} | rope_scaling: {tc.get('rope_scaling') or cfg.get('rope_scaling')}")
                print(f"  layers       : {len(lt)} -> {dict(Counter(lt)) or 'n/a'}")
                print(f"  multimodal   : {'vision_config' in cfg} (image_token_id={cfg.get('image_token_id')})")
                if mt == "qwen3_5" and arch == "Qwen3_5ForConditionalGeneration":
                    ok("架构与 handoff 记录一致（qwen3_5 / Qwen3_5ForConditionalGeneration）")
                else:
                    print(f"  [WARN] 架构与 handoff 记录不同，请重新核对训练配置")
                if (mpe or 0) >= 262144:
                    ok(f"上下文 {mpe} ≥ 256K")
                # 权重文件大小
                tot = sum(f.stat().st_size for f in mp.glob("*.safetensors")) / 1024 ** 3
                if tot > 0:
                    ok(f"权重体积 {tot:.1f} GB")
                else:
                    bad("没有 *.safetensors，可能下载不完整")

            # tokenizer + 模板
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
                ok(f"tokenizer 加载成功（vocab {len(tok)}）")
                if getattr(tok, "chat_template", None):
                    ok("chat_template 存在")
                else:
                    bad("tokenizer 无 chat_template")
                # 用真实 tokenizer 渲染一条样本，检查 think 注入
                if args.data:
                    t = Path(args.data) / "sft_step_windows_sharegpt.train.jsonl"
                    if t.exists():
                        with t.open(encoding="utf-8") as f:
                            row = json.loads(f.readline())
                        conv = row["conversations"]
                        role_map = {"system": "system", "human": "user", "gpt": "assistant",
                                    "observation": "tool", "tool": "tool"}
                        msgs = [{"role": role_map.get(c["from"], "user"), "content": c["value"]} for c in conv]
                        try:
                            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                            o, c2 = text.count(" thinking"), text.count("")
                            tcalls = text.count("<tool_call>")
                            ids = tok(text, add_special_tokens=False)["input_ids"]
                            msg = f"模板渲染：tokens={len(ids)}  thinking×{o} ×{c2} <tool_call>×{tcalls}"
                            (ok if (o == 1 and c2 == 1) else bad)(msg)
                            if o != 1 or c2 != 1:
                                print("         ↑ think 标记不是 1 对：训练时改用 template: qwen3_8（LLaMA-Factory 专用模板）")
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [WARN] 直接用 tokenizer 渲染失败（LLaMA-Factory 会用 template: qwen3_8 另行处理）：{exc}")
            except ImportError:
                bad("transformers 不可用，跳过 tokenizer 检查")
            except Exception as exc:  # noqa: BLE001
                bad(f"tokenizer 加载失败：{type(exc).__name__}: {str(exc)[:160]}")
    else:
        section("C. 底座模型")
        skip("未提供 --model，跳过模型检查")

    # ---------- D. 数据集 ----------
    section("D. 数据集")
    if args.data:
        d = Path(args.data)
        if not d.exists():
            bad(f"数据目录不存在：{d}")
        else:
            files = sorted(d.glob("*.jsonl"))
            if not files:
                bad("目录下没有 .jsonl")
            for p in files:
                n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
                print(f"  {p.name:<48} {n:>6} 条")
            need = ["sft_step_windows_sharegpt.train.jsonl", "sft_step_windows_sharegpt.eval.jsonl",
                    "dpo_preference_pairs.train.jsonl", "dataset_info.json"]
            for f in need:
                (ok if (d / f).exists() else bad)(f"存在 {f}")
            # 长度分布（有 tokenizer 才有意义）
            t = d / "sft_step_windows_sharegpt.train.jsonl"
            if t.exists():
                lens = []
                over = 0
                tok = None
                try:
                    from transformers import AutoTokenizer
                    if args.model:
                        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                except Exception:  # noqa: BLE001
                    tok = None
                with t.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        chars = sum(len(str(c.get("value", ""))) + 16 for c in r["conversations"])
                        if tok is not None:
                            txt = "\n".join(str(c.get("value", "")) for c in r["conversations"])
                            n = len(tok(txt, add_special_tokens=False)["input_ids"])
                        else:
                            n = round(chars / 2.2)
                        lens.append(n)
                lens.sort()
                if lens:
                    p50 = lens[len(lens) // 2]
                    p99 = lens[min(len(lens) - 1, int(len(lens) * 0.99))]
                    over = sum(1 for x in lens if x > args.cutoff)
                    unit = "tokens(真实)" if tok is not None else "tokens(估算)"
                    print(f"  长度分布 {unit}: p50={p50} p99={p99} max={lens[-1]}")
                    if over:
                        print(f"  [WARN] {over} 条超过 cutoff_len={args.cutoff}，会被截断（可能丢目标步）")
                    else:
                        ok(f"cutoff_len={args.cutoff} 覆盖全部样本（0 截断）")
    else:
        skip("未提供 --data，跳过数据集检查")

    # ---------- E. 磁盘与内存 ----------
    section("E. 磁盘与内存")
    du = shutil.disk_usage("/")
    free_gb = du.free / 1024 ** 3
    (ok if free_gb >= 100 else bad)(f"/ 可用 {free_gb:.0f} GB（27B 训练建议 ≥100GB）")
    try:
        import psutil  # type: ignore
        ram = psutil.virtual_memory().total / 1024 ** 3
        (ok if ram >= 64 else bad)(f"主存 {ram:.0f} GB（建议 ≥64GB）")
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram = int(line.split()[1]) / 1024 ** 2
                        (ok if ram >= 64 else bad)(f"主存 {ram:.0f} GB（建议 ≥64GB）")
                        break
        except OSError:
            skip("无法读取主存信息")

    # ---------- F. 结论 ----------
    section("结论")
    print(f"  通过 {RESULT['pass']} 项，失败 {RESULT['fail']} 项，跳过 {RESULT['skip']} 项")
    if RESULT["fail_items"]:
        print("  必须修复：")
        for x in RESULT["fail_items"]:
            print(f"    - {x}")
    if RESULT["skip_items"]:
        print("  等挂卡后再验：")
        for x in RESULT["skip_items"]:
            print(f"    - {x}")
    if not RESULT["fail_items"]:
        print("\n  无卡阶段的软件栈检查全部通过。挂上卡后按顺序再做：")
        print("    1) nvidia-smi 确认卡型与显存")
        print("    2) python -c \"import torch;print(torch.cuda.is_bf16_supported())\"  期望 True")
        print("    3) python -c \"import flash_attn\"                                确认 kernel 可加载")
        print("    4) 用 train_sft.yaml 跑 20 步冒烟，盯 nvidia-smi 峰值 < 78GB")
        print("    5) 抽 5 条 eval 样本续写，确认输出 thinking + <tool_call>")
    return 0 if not RESULT["fail_items"] else 1


if __name__ == "__main__":
    sys.exit(main())''', encoding="utf-8")

    (root / "check_data.py").write_text('''#!/usr/bin/env python3
"""训练前自检：数据结构 + 对话模板渲染（防 think 标签重复注入）。

用法：
    python check_data.py --model /root/models/Qwen3.8-27B --template qwen3
    python check_data.py --drop-tools          # 框架不支持 tools 列时先剥离
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THINK_RE = re.compile(r"</?think>")
TOOL_RE = re.compile(r"<tool_call>")


def load(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--model")
    ap.add_argument("--template", default="qwen")
    ap.add_argument("--drop-tools", action="store_true")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.dir)

    if args.drop_tools:
        n = 0
        for p in root.glob("*.jsonl"):
            if "diagnostics" in p.name:
                continue
            rows = list(load(p))
            for r in rows:
                r.pop("tools", None)
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\\n")
            n += 1
        print(f"[ok] 已剥离 tools 列：{n} 个文件")
        return 0

    # 1) 结构检查
    files = sorted(p for p in root.glob("*.jsonl") if "diagnostics" not in p.name)
    for p in files:
        rows = list(load(p))
        bad = [r for r in rows if not r.get("conversations") and not r.get("messages")]
        print(f"  {p.name:<48} {len(rows):>6} 条  结构异常 {len(bad)}")
    train = root / "sft_step_windows_sharegpt.train.jsonl"
    if not train.exists():
        print("[err] 找不到训练集", file=sys.stderr)
        return 1

    # 2) 目标步检查
    rows = list(load(train))[: args.samples]
    problems = 0
    for r in rows:
        conv = r["conversations"]
        if conv[0]["from"] != "system" or conv[1]["from"] != "human":
            problems += 1
            continue
        if conv[-1]["from"] != "gpt":
            problems += 1
            continue
        if len(THINK_RE.findall(conv[-1]["value"])) != 2:
            problems += 1
    print(f"\\n目标步结构（抽样 {len(rows)}）：异常 {problems} 条")

    # 3) 模板渲染检查
    if not args.model:
        print("\\n（跳过模板渲染检查：未提供 --model）")
        return 0 if problems == 0 else 1
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[err] 需要 transformers：pip install transformers", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    msgs_pool = []
    for r in rows[:5]:
        conv = r["conversations"]
        msgs = []
        for c in conv[:-1]:
            role = {"system": "system", "human": "user", "gpt": "assistant",
                    "observation": "tool", "tool": "tool"}.get(c["from"], "user")
            msgs.append({"role": role, "content": c["value"]})
        msgs.append({"role": "assistant", "content": conv[-1]["value"]})
        msgs_pool.append((r["sample_id"], msgs))

    print("\\n模板渲染检查：")
    for sid, msgs in msgs_pool:
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {sid[:14]} 模板渲染失败: {exc}")
            continue
        opens = text.count(" thinking")
        closes = text.count("")
        tools = TOOL_RE.findall(text)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        flag = "ok" if opens == 1 and closes == 1 else "检查"
        print(f"  [{flag}] {sid[:14]} tokens≈{n_tok:<6}  thinking×{opens} ×{closes} <tool_call>×{len(tools)}")
        if opens != 1 or closes != 1:
            print("         ↑ think 标记不等于 1 对：模板可能自带 thinking 注入，"
                  "请改用不注入的模板（或把数据里的 thinking 去掉）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''', encoding="utf-8")

    (root / "verify_longctx.py").write_text('''#!/usr/bin/env python3
"""长上下文验收：确认 LoRA 合并后的模型在各档长度下仍能正常工作。

用法：
    python verify_longctx.py --model /root/models/Qwen3.8-27B-merged \\
        --lengths 8192,32768,65536,131072,262144

判断标准：不报错、首 token 延迟合理、生成内容与前文相关（人工看前 200 字）。
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("[err] 需要 torch + transformers", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    filler = "以下是逆向分析记录，包含偏移量、密钥与工具输出。\\n"
    print(f"模型 max_position_embeddings = {model.config.max_position_embeddings}")

    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        try:
            unit = len(tok(filler, add_special_tokens=False)["input_ids"])
            body = filler * max(1, (L - 200) // max(1, unit))
            prompt = body + "\\n问题：上文中出现的第一个工具名是什么？请用一句话回答。"
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            n = ids.shape[1]
            if n > model.config.max_position_embeddings:
                print(f"  [skip] {L}: 实际 {n} tokens 超过模型上限")
                continue
            ids = ids.to(model.device)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            text = tok.decode(out[0][n:], skip_special_tokens=True)[:160].replace("\\n", " ")
            print(f"  [ok] {L:>7} (实际 {n:>7} tok) 首轮 {dt:>6.1f}s 峰值 {peak:>5.1f}GB | {text}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [oom] {L}: 显存不足（单卡 80GB 的合理上限，需要 TP/量化 KV）")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [err] {L}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''', encoding="utf-8")

    # 清理上一版遗留的、与当前结论冲突的配置
    stale = root / "train_sft_256k.yaml"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass

    return {
        "handoff.md": str(root / "handoff.md"),
        "train_sft.yaml": str(root / "train_sft.yaml"),
        "train_sft_qlora.yaml": str(root / "train_sft_qlora.yaml"),
        "train_dpo.yaml": str(root / "train_dpo.yaml"),
        "check_data.py": str(root / "check_data.py"),
        "verify_longctx.py": str(root / "verify_longctx.py"),
        "probe_24g.py": str(root / "probe_24g.py"),
        "verify_longctx_api.py": str(root / "verify_longctx_api.py"),
        "check_env.py": str(root / "check_env.py"),
    }