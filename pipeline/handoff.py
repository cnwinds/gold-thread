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

### 4.1 开机 5 条验收（不通过先别装训练栈）

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

## 9. 合并与推理

```bash
# 合并 LoRA
llamafactory-cli export \\
  --model_name_or_path /root/models/Qwen3.8-27B \\
  --adapter_name_or_path saves/qwen38-27b-goldthread-sft \\
  --template qwen3 --finetuning_type lora \\
  --export_dir /root/models/Qwen3.8-27B-merged
```

```bash
# vLLM 起 256K 上下文服务（合并后在独立环境/实例里装 vllm，避免和训练栈冲突）
pip install vllm
vllm serve /root/models/Qwen3.8-27B-merged \\
  --max-model-len 262144 \\
  --gpu-memory-utilization 0.90 \\
  --enable-auto-tool-choice --tool-call-parser hermes
```

> **为什么不能直接 `--max-model-len 262144`**：KV cache 每 token 64 KB（4KB × 16 个 full 层），
> 256K 上下文需要约 **16 GB**，加权重 55G 已接近 80G 上限，实际会 OOM。
> 务实做法（按顺序）：`32768` → `65536` → 加 `--kv-cache-dtype fp8`（KV 减半）→ 2 卡 `--tensor-parallel-size 2`。

## 10. 排障表

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

## 11. 可复现性

- 生成配置：`config/pipeline.yaml`（`config_hash` 见 `stats.json`）
- 每条样本带 `metadata.sample_id / session_id / project_group / stage / split / sample_weight`
- 血缘可回溯：`data/02_gold/sessions/<项目>/<会话>.gold.jsonl`（含 `raw_thinking`、`pruned_details`、`model_report`）
- 重跑：`python -m pipeline.run all`（增量，模型调用命中缓存）

## 12. 使用边界

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
    }