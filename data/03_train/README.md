# 03_train — GoldThread 训练数据集（最终版）

> 由 GoldThread 管线 stage1 → stage2 → stage3 生成，经过一次人工复核 + v2 修复，已在 48G A800 上实机跑通训练。
> 最短启动路径见 `START_HERE.md`，完整交接说明见 `handoff.md`，训练实跑记录与踩坑经验见 `RESULT.md`。

## 数据版本

原始版（`.jsonl`）是 first pass 产物；v2 版（`.v2.jsonl`）经过修复后成为**实际训练数据**。dataset_info.json 同时注册了所有版本。

| 用途 | dataset_info 名字 | 文件 | 条数 |
| :--- | :--- | :--- | ---: |
| **SFT 训练（主）** | `pi_sft_step_windows_train_v2` | `sft_step_windows_sharegpt.train.v2.jsonl` | **548** |
| SFT 验证 | `pi_sft_step_windows_eval_v2` | `sft_step_windows_sharegpt.eval.v2.jsonl` | **221** |
| 完整轨迹训练（可选） | `pi_sft_gold_trajectories_train_v2` | `sft_gold_trajectories_sharegpt.train.v2.jsonl` | 2 |
| 完整轨迹验证 | `pi_sft_gold_trajectories_eval_v2` | `sft_gold_trajectories_sharegpt.eval.v2.jsonl` | 1 |
| **DPO 训练** | `pi_dpo_preference_pairs_train_v2w` | `dpo_preference_pairs.train.v2w.jsonl` | **94**（窗口裁剪后；原 100 对） |
| DPO 验证 | `pi_dpo_preference_pairs_eval_v2w` | `dpo_preference_pairs.eval.v2w.jsonl` | 126（原 143） |
| 原始 DPO（全窗口） | `pi_dpo_preference_pairs_train_v2` | `dpo_preference_pairs.train.v2.jsonl` | 100 |
| 病态诊断（**不进训练**） | – | `negative_diagnostics.jsonl` | 814 |

**stage 权重分布**（train）：opening 9, key_decision 418, routine 279, highlight 54, closing 9。
highlight = 模型判定的关键决策步，权重最高。

## 训练 / 验证切分（按项目整包）

| | 项目 |
| :--- | :--- |
| train | `game::altosodyssey`, `game::cookingmadness`, `game::jigsawcard`, `game::mahjong` |
| eval | `game::meowdoku`, `game::riddle` |

eval 是**整项目留出**，所以 eval_loss 高于 train_loss 是正常的，它衡量换一个项目还能不能干活。

## 格式

- **sharegpt**：`conversations: [{from, value}]`
  - `system` → 工程纪律系统提示词
  - `human` → 根任务指令
  - `gpt` → 助手步，格式 ` thinking…` + `<tool_call>{"name","arguments"}</tool_call>` + ``
  - `observation` → 工具输出
- **openai**：`messages: [{role, content}]`，对应 `*_openai.jsonl`
- `tools` 列：JSON 字符串（bash / read / write / edit / codemode 五个函数声明）
- 单步窗口：`[system, 根任务] + 最近 8 步历史 + 目标步`；
  **历史步 thinking 已剥离，只有目标步带 thinking**
- `metadata.sample_weight`：开局 3 / 关键步 2 / 高光步 1.5 / 常规 1

## 实际训练配置（8B + 48G A800）

关键点：
- `mask_history: true` — 只训练最后一轮（本数据集唯一必须手动开的开关）
- `enable_liger_kernel: true` — 48G 卡上必须开，融合 CE 省掉 fp32 logits（省 ~10GB）
- `eval_strategy: "no"` — 训练时 eval 不走 Liger 融合 CE，长样本会 OOM；训完单独跑 eval
- `cutoff_len: 32768`（SFT）/ `4096`（DPO，policy+ref 同时前向占显存翻倍）
- DPO `ref_model` **必须指向 SFT merged 模型**，不能默认用 base（会把模型推崩）
- DPO `learning_rate: 1.0e-5`，预算显存时 ref 需 8bit 量化

> 完整 YAML 见 `train_sft_qwen3_8b.yaml` / `train_dpo_qwen3_8b.yaml`，每条注释都写了原因。

## 快速训练（推荐用监控器）

```bash
# 冒烟（200 条 × 1 epoch，~20 分钟）
python train_monitor.py --timeout-hours 1 --log logs/smoke_monitored.log -- \
  llamafactory-cli train train_sft_smoke_qwen3_8b.yaml

# 全量（548 条 × 3 epoch，~1 小时 15 分）
python train_monitor.py --timeout-hours 3 --log logs/full_monitored.log -- \
  llamafactory-cli train train_sft_qwen3_8b.yaml

# DPO（~10 分钟）
python train_monitor.py --timeout-hours 1 --log logs/dpo_monitored.log -- \
  llamafactory-cli train train_dpo_qwen3_8b.yaml
```

用 `train_monitor.py` 的好处：终端实时输出 loss / 步数；进程 OOM 或被信号 kill 后**立刻**给出诊断（不用你盯着 `nvidia-smi`）；超时可自动发送 SIGTERM/SIGKILL；Ctrl-C 优雅清理整个进程组。

如果你习惯裸跑：

```bash
llamafactory-cli train train_sft_qwen3_8b.yaml
```

## 换 27B 时的三处修改

```diff
- model_name_or_path: /data/models/Qwen3-8B
+ model_name_or_path: /data/models/Qwen3.8-27B
- template: qwen3
+ template: qwen3_8                    # 专用模板，不是 qwen3
- output_dir: /data/saves/qwen3-8b-goldthread-sft
+ output_dir: /data/saves/qwen38-27b-goldthread-sft
- enable_liger_kernel: true
+ enable_liger_kernel: false           # qwen3_5 线性注意力不在 Liger 支持列表
- lora_target: all
+ lora_target: mlp                      # 不动注意力投影层，保住长上下文
```
另需：不设 deepspeed、单卡直接跑。

## API 服务部署（合并后）

```bash
# 一键起 OpenAI 兼容 API
bash serve_api.sh 8000 gt-2026
# 后台：nohup bash serve_api.sh 8000 gt-2026 > logs/api.log 2>&1 &
```

| 端点 | 说明 |
| :--- | :--- |
| `POST /v1/chat/completions` | 标准 OpenAI 格式，支持 `stream:true` + `tools` + `tool_choice` → 结构化 `tool_calls` |
| `GET /v1/models` | 列出可用模型 |
| `GET /health` / `/healthz` / `/v1/health` | 无鉴权健康检查（容器平台探针用） |

完整服务经验（流式 tools、SDK 兼容补丁、公网映射、健康检查路由、LLaMA-Factory 层 5 处 patch）
见 `RESULT.md` §"OpenAI 兼容服务"。

## 重新生成

```bash
python -m pipeline.run all        # 全流程（增量）
python -m pipeline.run stage3     # 仅重算训练集
```

或在 Web 控制台「运行」页点「运行全流程（增量）」。
