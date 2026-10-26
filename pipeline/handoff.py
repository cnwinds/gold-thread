"""handoff 文档生成：把训练集的用法写成可直接照做的说明。

产出（与数据集同目录，随 stage3 一起刷新）：
    handoff.md        从环境搭建到训练/验证/推理的全流程
    train_sft.yaml    LLaMA-Factory SFT 配置（已填入真实数据集名与样本数）
    train_dpo.yaml    LLaMA-Factory DPO 配置

内容全部由本次运行的真实产物推导（数据集名、条数、切分、来源模型），
不做任何硬编码假设，避免文档与实际数据脱节。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _table(rows: List[List[str]], head: List[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(":---" for _ in head) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


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
    learner = (cfg.get("llm") or {}).get("model", "—")

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

    files = _table([
        ["**SFT 训练（主）**", f"`sft_step_windows_sharegpt.train.jsonl`", f"`pi_sft_step_windows_train`", _fmt(win_train)],
        ["SFT 验证", "`sft_step_windows_sharegpt.eval.jsonl`", "`pi_sft_step_windows_eval`", _fmt(win_eval)],
        ["完整轨迹训练（可选）", "`sft_gold_trajectories_sharegpt.train.jsonl`", "`pi_sft_gold_trajectories_train`", _fmt(traj_train)],
        ["完整轨迹验证", "`sft_gold_trajectories_sharegpt.eval.jsonl`", "`pi_sft_gold_trajectories_eval`", _fmt(traj_eval)],
        ["**DPO 训练**", "`dpo_preference_pairs.train.jsonl`", "`pi_dpo_preference_pairs_train`", _fmt(dpo_train)],
        ["DPO 验证", "`dpo_preference_pairs.eval.jsonl`", "`pi_dpo_preference_pairs_eval`", _fmt(dpo_eval)],
        ["OpenAI 格式备选", "`sft_step_windows_openai.train.jsonl`", "未注册（同数据另一种格式）", _fmt(cnt("sft_step_windows_openai.train.jsonl"))],
        ["病态诊断（不进训练）", "`negative_diagnostics.jsonl`", "—", _fmt(diag)],
    ], ["用途", "文件", "dataset_info 里的名字", "条数"])

    proj_rows = [[f"`{p}`", _fmt(n)] for p, n in sorted(by_project.items(), key=lambda kv: -kv[1])]
    model_rows = [[f"`{m}`", _fmt(n)] for m, n in sorted(by_model.items(), key=lambda kv: -kv[1])]
    stage_rows = [[f"`{k}`", _fmt(v)] for k, v in sorted(stage_counts.items(), key=lambda kv: -kv[1])]
    dpo_rows = [[f"`{k}`", _fmt(v)] for k, v in sorted(dpo_src.items(), key=lambda kv: -kv[1])]

    sft_yaml = f"""### 训练配置

# LLaMA-Factory 会读取 dataset_dir 下的 dataset_info.json
### train_sft.yaml（已生成到同目录，可直接用）
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct   # 换成你要微调的底座
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_sft_step_windows_train
eval_dataset: pi_sft_step_windows_eval
template: qwen
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/goldthread-sft
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
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
```

### train_dpo.yaml（SFT 之后再跑，可选）
```yaml
### model
model_name_or_path: saves/goldthread-sft   # 用上一步的产物
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_dpo_preference_pairs_train
eval_dataset: pi_dpo_preference_pairs_eval
template: qwen
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/goldthread-dpo
logging_steps: 5
save_steps: 100
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 1.0
bf16: true
gradient_checkpointing: true
```
"""

    doc = f"""# 数据交接说明（handoff）

> 本文件由 GoldThread `stage3` 自动生成，内容取自本次运行的真实产物。
> 数据集目录：`{root.as_posix()}`

## 0. 这批数据是什么

从 pi 会话日志中提纯出来的 **Agent 工程分析轨迹**：解包 / 反编译 / 资源与配置表还原 /
二进制格式破解等任务的真实执行过程。每条样本都经过：

- **程序**：机械去噪、静默执行、轮询命令规范化、窗口拼装、血缘记录
- **模型**：判断删哪些步、接缝是否平滑、整条会话值不值得训练、偏好对是否成立
- **人工**：逐条复核改写与问题步，最后定稿

训练来源模型分布：{", ".join(f"`{m}` × {_fmt(n)}" for m, n in sorted(by_model.items(), key=lambda kv: -kv[1])[:5]) or "—"}

## 1. 数据集清单

{files}

补充统计：

- 阶段权重分布（训练集）：{", ".join(f"{k} {_fmt(v)}" for k, v in stage_counts.items()) or "—"}
  （`highlight` 是模型判定的关键决策步，权重最高）
- 涉及项目：{", ".join(f"`{p}` {_fmt(n)}" for p, n in sorted(by_project.items(), key=lambda kv: -kv[1])) or "—"}
- 训练/验证切分：**按项目整包**切，防相邻步泄漏
  - train：{", ".join(f"`{p}`" for p in train_projects) or "—"}
  - eval：{", ".join(f"`{p}`" for p in eval_projects) or "—"}
- 会话标签：{", ".join(f"{k} {_fmt(v)}" for k, v in label_counts.items()) or "—"}
- 复核状态：{", ".join(f"{k} {_fmt(v)}" for k, v in review_states.items()) or "—"}
- DPO 来源：{", ".join(f"{k} {_fmt(v)}" for k, v in dpo_src.items()) or "—"}
- DPO 复核：检查 {_fmt(dpo_review.get("checked", 0))} 对，
  调换 {_fmt(dpo_review.get("swapped", 0))}、丢弃 {_fmt(dpo_review.get("dropped", 0))}

## 2. 环境搭建

### 2.1 硬件

- 7B 级模型 + LoRA：单卡 **24GB** 可跑（`cutoff_len 16384`，bs 1 + 梯度累积 8 + 梯度检查点）
- 显存不足时：`cutoff_len` 降到 8192、`per_device_train_batch_size` 保持 1、`gradient_checkpointing: true`

### 2.2 软件

```bash
# 1) Python 3.10+ 与 CUDA 驱动（建议 CUDA 12.1+）
python -V

# 2) 训练框架 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 3) 可选：加速与显存优化
pip install flash-attn --no-build-isolation

# 4) 验证安装
llamafactory-cli version
```

### 2.3 数据放置

两种方式任选其一：

```bash
# 方式 A（推荐）：把 dataset_dir 指到本目录，LLaMA-Factory 直接读 dataset_info.json
llamafactory-cli train {root.as_posix()}/train_sft.yaml

# 方式 B：拷到 LLaMA-Factory 的数据目录
cp {root.as_posix()}/*.jsonl  LLaMA-Factory/data/goldthread/
cp {root.as_posix()}/dataset_info.json LLaMA-Factory/data/goldthread/
```

## 3. 数据格式说明（重要）

- 采用 **ShareGPT** 结构：`conversations: [{{from, value}}]`
  - `system` 工程纪律提示词 / `human` 根任务 / `gpt` 助手步 / `observation` 工具输出
- 助手步内容形如：
  ```
   thinking…推理…
  <tool_call>{{"name":"bash","arguments":{{"command":"…"}}}}</tool_call>
  ```
- `tools` 列是 JSON 字符串（bash/read/write/edit/codemode 五个函数声明）。
  **若你的训练框架不支持 function calling 列**，训练前删掉该字段即可：
  ```bash
  python - <<'PY'
  import json, pathlib
  for p in pathlib.Path("{root.as_posix()}").glob("*.jsonl"):
      if "diagnostics" in p.name: continue
      rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
      for r in rows: r.pop("tools", None)
      with p.open("w", encoding="utf-8") as f:
          for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
  PY
  ```
- 单步窗口的构造：`[system, 根任务] + 最近若干步历史 + 目标步`；
  **历史步的 thinking 已剥离，只有目标步带 thinking 并计算 loss**（贴合真实推理时的上下文分布）。
- `metadata.sample_weight`：开局/收尾 3、高光与关键决策 2、常规 1。
  若你的 collator 支持逐样本权重可直接用，否则忽略即可。

## 4. 训练

{sft_yaml}
### 启动

```bash
# SFT
llamafactory-cli train {root.as_posix()}/train_sft.yaml

# 训练完 → DPO（可选，用 SFT 产物作底座）
llamafactory-cli train {root.as_posix()}/train_dpo.yaml
```

### 训练前自检

```bash
python - <<'PY'
import json
from collections import Counter
p = "{root.as_posix()}/sft_step_windows_sharegpt.train.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
print("样本数:", len(rows))
bad = [r for r in rows if not r.get("conversations") or r["conversations"][-1]["from"] != "gpt"]
print("结构异常:", len(bad))
print("阶段分布:", dict(Counter(r["metadata"]["stage"] for r in rows)))
print("项目分布:", dict(Counter(r["metadata"]["project_group"] for r in rows)))
PY
```

## 5. 评估

- **损失**：训练日志里看 `eval_loss`；本数据集的 eval 是**整项目留出**，
  因此它衡量的是"换一个游戏/程序还能不能干活"，比随机切分严格得多。
- **人工抽样**：建议从 eval 集抽 20 条，让模型只看上下文（历史步 + 工具输出）
  续写目标步，对比它生成的 `thinking` 与工具调用是否与参考一致。
- **能力检查表**（人工打分即可）：
  1. 是否先观察再动手（不盲调工具）
  2. 报错后是否查证再重试（不是反复同一个错误命令）
  3. 是否避免心算偏移/密钥并声称已验证
  4. 是否遵守"只在当前工作目录内作业"

## 6. 推理验证

```bash
# 合并 LoRA 后推理
llamafactory-cli export \\
  --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \\
  --adapter_name_or_path saves/goldthread-sft \\
  --template qwen \\
  --finetuning_type lora \\
  --export_dir saves/goldthread-merged

python - <<'PY'
# 或直接用 transformers 加载合并后的模型做一次冒烟
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "saves/goldthread-merged"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, device_map="auto", trust_remote_code=True)
msgs = [{{"role": "user", "content": "帮我破解 E:\\\\Download\\\\com.oakever.<game>@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有文件保存在本文件夹"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=512)[0], skip_special_tokens=True))
PY
```

## 7. 常见问题

| 现象 | 处理 |
| :--- | :--- |
| 报错不认识 `tools` 字段 | 按 §3 去掉 `tools` 列，或升级 LLaMA-Factory |
| OOM | `cutoff_len` 减半；`gradient_checkpointing: true`；bs 保持 1 |
| `eval_loss` 远高于 `loss` | 正常：eval 是整项目留出，属于跨任务泛化评估 |
| 训练后模型变得啰嗦/爱解释 | 说明 SFT 权重过高或样本偏 `highlight`；可按 `metadata.stage` 重采样 |
| DPO 训练不收敛 | `pref_beta` 调小（0.05）；确认 `dataset` 用的是 `*_train` 而非全量 |

## 8. 可复现性

- 生成这批数据的配置：`config/pipeline.yaml`（`config_hash` 见 `stats.json`）
- 每条样本都带 `metadata.sample_id / session_id / project_group / stage / split`
- 上游血缘：`data/02_gold/sessions/<项目>/<会话>.gold.jsonl`（含 `raw_thinking`、`pruned_details`、
  `model_report`），可逐条回溯到原始 pi 会话
- 重跑：`python -m pipeline.run all`（增量，模型调用命中缓存）

## 9. 数据使用边界

- eval 集对应的项目（{", ".join(f"`{p}`" for p in eval_projects) or "—"}）**不要**再拿去训练，
  否则评估会失真
- `negative_diagnostics.jsonl` 只用于分析（无标准答案），不要喂进 SFT/DPO
- 数据来自自有/已授权应用的逆向分析，使用范围请遵守你的授权约定
"""

    (root / "handoff.md").write_text(doc, encoding="utf-8")

    (root / "train_sft.yaml").write_text(f"""### model
model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_sft_step_windows_train
eval_dataset: pi_sft_step_windows_eval
template: qwen
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/goldthread-sft
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
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
model_name_or_path: saves/goldthread-sft
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid

### dataset
dataset_dir: {root.as_posix()}
dataset: pi_dpo_preference_pairs_train
eval_dataset: pi_dpo_preference_pairs_eval
template: qwen
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/goldthread-dpo
logging_steps: 5
save_steps: 100
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 1.0
bf16: true
gradient_checkpointing: true
""", encoding="utf-8")

    return {"handoff.md": str(root / "handoff.md"),
            "train_sft.yaml": str(root / "train_sft.yaml"),
            "train_dpo.yaml": str(root / "train_dpo.yaml")}