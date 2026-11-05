# GoldThread（金线）

Agent 训练数据维护工具：把 pi / Cursor 会话轨迹增量研磨成可审计的 SFT / DPO 数据集，附带本地 Web 控制台。

**产品与术语**以根目录 [`CONTEXT.md`](CONTEXT.md) 为准（控制台行为、采集/分析、血缘）；本 README 侧重管线结构与命令。

```
pi / Cursor 会话
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage1  采集与清洗   01_raw/{positive,negative,pending}            │
   │   源目录预览勾选 → ingest；程序：准入 / 去重 / 回合流 + meta.json   │
   │   模型：回合语义分类（root/task/setup/noise/continuation/correction）│
   │   程序：剥噪音、折叠催促、纠偏转 system、根任务标准化（yaml）       │
   └────┬───────────────────────────────────────────────────────────────┘
        │  单会话或批量「分析」→ stage2（only=sids）
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage2  黄金提纯   02_gold/sessions + dpo_pairs + annotations    │
   │   程序：机械静默执行、轮询命令规范化、执行模型决定 + 血缘           │
   │   模型：步级删/折/改/排、接缝审核、质量报告（判断权在模型）         │
   └────┬───────────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage3  训练集合成   03_train/*.jsonl + dataset_info + stats       │
   │   程序：门控、单步窗口（根任务锚定 + 历史 think 剥离）、切分        │
   │   模型：DPO 逐对复核 good/swap/drop；高光步权重                     │
   │   交付：`04_release/delivery/` zip（见 handoff）                    │
   └───────────────────────────────────────────────────────────────────┘
```

> **Legacy**：旧版曾在控制台提供独立「人工复核」队列（通过/驳回/定稿）。该 UI 已移除；`02_gold/annotations` 里仍可能有历史人工结论，Stage 3 门控规则见 `CONTEXT.md` 与 ADR-0001。

分工原则（详见 `DATASET_BEST_PRACTICES.md` 三·五节）：**判断权归模型**（删什么、接缝顺不顺、
值不值得训、偏好对成不成立），**程序只做执行器与校验器**（执行决定、留血缘、守不变量、
机械提取接缝）。模型调用失败时程序保持原样并降级，不因漏审而静默放行。

## 快速开始

```bash
cd gold-thread          # 仓库根目录，下文命令均在此执行

python serve.py                 # 打开 http://127.0.0.1:8765
python serve.py --port 9000 --no-browser

# 或纯命令行
python -m pipeline.run stage1            # 增量
python -m pipeline.run stage2 --force    # 强制重建
python -m pipeline.run all               # 三阶段顺序跑
python -m pipeline.run review            # 模型评审（正例 + 待定）
```

模型调用走 pi 自己的 `~/.pi/agent/models.json`（provider / baseUrl / apiKey），
默认模型见 `config/pipeline.yaml` 的 `llm.model`（通过 pi `models.json` 调用）。

## 2.0 数据事实链

控制台与管线共用同一套 **sid 血缘**（采集 → 分析 → 黄金 → 训练样本），见 `CONTEXT.md` §6：

1. **Stage 1 · 采集与回合清洗**
   - 从 **pi / Cursor** 预览勾选；**采集所选** 只 ingest，**采集并分析** 再跑 stage2；
   - 已加入列表里可对单会话 **分析 / 重新分析**、改数据集角色（positive/negative/pending）；
   - 原始 `.jsonl` 为不可变副本；回合 `kind / action / source / reason` 写在 `.meta.json`；
   - 可改逐回合分类；`review_key` 变化会使旧模型标注过期。
2. **Stage 2 · 黄金提纯**
   - 有用步 → 该 sid 的 `.gold.jsonl`；弯路 → 该 sid 的 DPO / rejects；
   - `raw_thinking` / 脚本 `thinking` / 模型改写三态可对照；质量报告由模型产出。
3. **Stage 3 · 训练样本**
   - 门控 + 单步窗口 + 按项目整包切分；训练页默认只看将要生成的数据集，样本可回到 sid。

## 目录

```
gold-thread/
├── CONTEXT.md                    # 产品/领域词汇与控制台需求（Agent 先读）
├── config/pipeline.yaml          # 全部规则（准入、标签、回合、提纯、合成）
├── pipeline/
│   ├── collect.py                # pi / Cursor 采集入口
│   ├── common/                   # pilog 解析 · quality 检测 · llm 调用 · format 转换
│   ├── stage1_extract.py         # 原始提取（正/负例分目录）
│   ├── stage2_gold.py            # 黄金提纯（折叠/剪枝/去噪）
│   ├── stage3_synth.py           # 训练数据合成
│   ├── turnclassify.py           # 模型：回合语义分类
│   ├── review.py                 # 模型：黄金裁决 + 长 CoT 重写
│   └── run.py                    # CLI
├── server/app.py                 # FastAPI 控制台
├── web/index.html                # 单页前端（无构建）
└── data/
    ├── 00_state/                 # 增量状态、标签覆盖、运行历史、LLM 缓存
    ├── 01_raw/                   # positive / negative / pending
    ├── 02_gold/                  # sessions / annotations / rejects / dpo_pairs
    └── 03_train/                 # SFT / DPO / 诊断 / dataset_info / stats
    └── 04_release/               # 冻结发布 / delivery zip / 训练 packs（与 03_train 工作区分离）
```

`01_raw` 里的 `.jsonl` 是源文件**逐字节副本**，所有加工信息都在同名 `.meta.json`
（回合 kind/drop、根任务、质量分布、指纹）。因此 raw 永远可以复核、可以重放。

## 脚本 vs 模型

| 环节 | 脚本（确定性、可复算） | 模型（判断/生成，落盘为标注） |
|---|---|---|
| 准入 | 目录前缀、排除名单、同源去重 | — |
| 标签 | 主导模型匹配（`deepseek*`/`glm-*` → 正，`qwen*` → 负） | — |
| 回合 | 空回合/重发检测、折叠、纠偏内化 | 根任务识别、setup/noise/task 语义判定 |
| 提纯 | write→edit 折叠、错误步剪枝、空 think 处理、CoT 启发式压缩、静默执行 | 长 CoT 重写、黄金存废裁决 |
| 合成 | 单步滑窗、根任务锚定、历史 think 剥离、权重、按项目切分、DPO 组装 | 重写后的 thinking、负例润色（可扩展） |

模型只输出 JSON 标注（`02_gold/annotations/*.json`），由脚本校验后应用；
同一个 prompt + 同样输入会命中 `00_state/llm_cache/`，重跑不重复计费。

## 数据语义

**stage1 标签**（`config.labeling`）：

- `positive`：主导模型匹配 `^deepseek` / `^glm-`（teacher 源）
- `negative`：主导模型匹配 `qwen`（student 诊断 / DPO 源）
- `pending`：模型占比低于 `dominant_ratio_min` 或未匹配任何规则 → 人工在 UI 里改
- 手动改标签写入 `00_state/labels.json`，重跑 stage1 后生效，优先级最高

**stage2 产出**：

- `sessions/<project>/<sid>.gold.jsonl`：一条会话一行，含完整 messages、`quality`、
  `flags`（`state_folded` / `detour_pruned` / `thinking_compressed` / `thinking_synthesized`）
- `rejects/<sid>.pruned.jsonl`：被剪掉的弯路
- `rejects/<sid>.diagnostics.jsonl`：原始流中的全部病态步
- `dpo_pairs.jsonl`：折叠/剪枝产生的天然 chosen/rejected 对
- `annotations/<sid>.json`：脚本判定 + 模型评审结果（`review_key` 保证源变了自动失效）

`review_key` 同时绑定原始文件指纹和回合流签名；人工调整根任务或回合分类后，旧模型评审会被标为过期，避免把旧结论错误套到新轨迹上。

**stage3 产出**：

| 文件 | 说明 |
|---|---|
| `sft_gold_trajectories_sharegpt.jsonl` | 完整黄金轨迹（保留全部思考） |
| `sft_step_windows_sharegpt.jsonl` | 单步决策窗口（历史 think 已剥离，根任务永久锚定） |
| `sft_step_windows_openai.jsonl` | 同上，OpenAI messages 格式 |
| `dpo_preference_pairs.jsonl` | 剪枝对 + 反刍压缩对 |
| `negative_diagnostics.jsonl` | 病态行为库（心算 hex、反刍、跨目录…） |
| `dataset_info.json` | LLaMA-Factory 数据集注册 |
| `stats.json` / `index.json` | 统计与切分信息 |

**训练版本（v3 · 冻结发布）**
：上机训练只认 `releases/<版本>/` 解包出来的这一套（格式契约校验通过）。
`dataset_info` 键为 `gt_sft_train/eval`、`gt_dpo_train/eval`，文件为 `sft.{train,eval}.jsonl`、`dpo.{train,eval}.jsonl`。

**`*.lf.jsonl` = 规整版（必须先跑 `normalize_sharegpt.py`）**
：发布包把「一条 assistant 消息里的多个 `<tool_call>`」拆成了多条连续 `observation`，
LLaMA-Factory 的 `SharegptDatasetConverter` 要求严格 `user/assistant` 交替，会**静默丢弃**这些样本：
实测 `sft.train` 丢 67/548、`sft.eval` 丢 107/221（训练只跑 363 步而不是 411）、`dpo` 也有 22/100 不合法。
`normalize_sharegpt.py` 合并连续同角色消息、把 `chosen/rejected` 从 dict 取 `value`，产出 `*.lf.jsonl`，
`dataset_info` 注册为 `gt_sft_train_lf` 等，yaml 全部指向 `_lf` 版本。

**v2 时代的数据（保留，不再被 yaml 引用）**
- `sft_step_windows_sharegpt.train.v2.jsonl`（548 条）/ `.eval.v2.jsonl`（221 条）—— SFT 主数据集
- `dpo_preference_pairs.train.v2.jsonl`（100 对）/ `.eval.v2.jsonl`（143 对）—— DPO 原始对
- `dpo_preference_pairs.train.v2w.jsonl`（窗口裁剪版）/ `.eval.v2w.jsonl`—— 48G 显存 DPO 专用
- `sft_gold_trajectories_sharegpt.*.v2.jsonl` —— 完整长轨迹（可选混训）

**训练配置与脚本**
：
- **`run_train.py` / `chunked_ce_patch.py`** —— **27B 正式启动入口**：lm_head + CE 按 1024 位置分块，
  避免 `16384 × 248320 × 4B ≈ 16.3 GB` 的 fp32 logits（数值等价，实测逐步 loss 差 ≤ 2e-3）
- **`train_lowmem.py` / `lowmem_dpo.py`** —— 低显存启动器：只算监督位置的 logits（DPO 走 `disable_adapter()` 当 ref，不再加载第二份权重）
- `train_sft_qwen3_8b.yaml` / `train_dpo_qwen3_8b.yaml` —— 8B 正式训练配置（含完整调试注释）
- `train_sft_smoke_qwen3_8b.yaml` / `train_dpo_smoke_qwen3_8b.yaml` —— 冒烟测试（200 条 SFT + 50 对 DPO）
- `train_sft.yaml` / `train_dpo.yaml` / `train_sft_qlora.yaml` —— 27B 配置与 QLoRA 备选
- `train_dpo_qlora_27b.yaml` / `_b.yaml` / `train_sft_27b_smoke.yaml` —— 27B DPO 的 QLoRA 尝试（80G 未跑通，见 RESULT_27b_run.md §4）
- `merge_lora.yaml` / `merge_dpo.yaml` —— 合并 SFT / 叠 DPO adapter（`export_device: cpu`，不占显存）
- `eval_sft_qwen3_8b.yaml` —— 单独 eval（训练时关掉 eval，训完用此跑）
- `normalize_sharegpt.py` —— 发布包数据规整（不做会静默丢样本，见上）
- `check_env.py` / `check_data.py` —— 上机前自检（含 tokenizer 渲染验证）
- `smoke_infer_merged.py` —— 合并后推理冒烟（验证 `thinking` + `tool_call`）
- `smoke_ab_dpo.py` / `verify_dpo_adapter.py` / `verify_merged_dpo.py` —— DPO 前后 A/B、adapter 与合并产物验收
- `verify_longctx.py` / `verify_longctx_api.py` —— 长上下文档级验收
- `probe_24g.py` —— 模拟小显存加载（bf16 / 8bit / 4bit 可行性探测）
- `prep_dpo_window.py` —— DPO 窗口裁剪（从头部丢旧步，保留 system + 根任务 + 最近步）
- `serve_api.sh` —— 合并模型后一键起 OpenAI 兼容 API（LLaMA-Factory hf 后端）
- **`serve_vllm.sh`** —— **推荐**：vLLM 规范推理服务（含 `--reasoning-parser deepseek_r1` 自动拆分 thinking）
- `server/reasoning_proxy.py` —— ReasoningProxy 中间件（给 LLaMA-Factory API 补上非流式 reasoning 拆分）
- **`train_monitor.py`** —— **训练进程监控器**：实时 tail 日志 + OOM 秒级诊断 + 进程退出一秒通知（推荐用它替代裸跑 `llamafactory-cli`）

> 训练实跑记录：8B 见 `data/03_train/RESULT.md`，**27B 见 `data/03_train/RESULT_27b_run.md`**
> （单卡 A800 80G：2h21m / 411 步 / 峰值 76 GB；含 fla 必装、`lora_target` 必须点名投影层、
> cutoff 上限定为 12288、分块 CE、DPO 在 80G 上三种配置全部 OOM 的实测结论）。
> 完整交接说明见 `data/03_train/handoff.md`。

切分按 `project_group` 整包留出（默认 `eval_ratio` 取项目数比例），
避免同一游戏的相邻步泄漏到验证集。

## 控制台

- **概览**：三阶段状态、标签分布、项目与健康度
- **原始数据**：采集（pi/Cursor）· 已加入会话库 · 分析/重新分析 · 改标签 · 回合清洗
- **黄金数据**：每会话 SFT 窗口 vs DPO 对去向 · 质量报告 · 血缘
- **训练数据**：将要生成的数据集 · 样本回到 sid
- **梳理与交付**：运行管线 · 选 export profile 打包 · 历史 zip
- **配置**：在线编辑 `pipeline.yaml`

（无独立「审核」侧栏；历史 `review.py` CLI 仍可跑模型评审，见 `python -m pipeline.run review`。）

## 配置要点

```yaml
paths.data_dir: "data"                          # 相对仓库根目录，可改绝对路径
paths.sessions_dir: "~/.pi/agent/sessions"       # 支持 ~ 与绝对路径
admission.workspace_prefixes: ["--d--crack-"]   # 新项目自动准入
labeling.positive_model_patterns: ["^deepseek", "^glm-"]
labeling.negative_model_patterns: ["qwen"]
turns.use_model: true            # 关掉则纯正则（更快但易误判）
gold.min_gold_steps: 10          # 低于此步数不进黄金
review.enabled: false            # 打开后 run_review 默认处理正例
synth.window_turns: 8            # 单步窗口历史长度
synth.eval_ratio: 0.25           # 按项目整包留出验证集
```

## 已知限制

- 工具输出里的乱码（Windows GBK 输出被记为 UTF-8）当前原样保留，未做编码修复；
- 模型回合分类/评审对长会话会做均匀抽样，极端情况下可能漏掉局部问题；
- 训练数据仍以「会话内可观测的成功证据」为准，没有执行级回放验证；
- 旧的单文件脚本 `sync_pi_datasets.py` 与 `output/` 保留作为对照，不再演进。