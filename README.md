# GoldThread（金线）

Agent 训练数据维护工具：把 pi 会话轨迹增量研磨成可审计的 SFT / DPO 数据集，附带本地 Web 控制台。

```
pi 会话日志 (.jsonl)
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage1  提取   01_raw/{positive,negative,pending}                  │
   │   程序：目录准入 / 同源去重 / 分目录落不可变副本 + meta.json        │
   │   模型：回合语义分类（root/task/setup/noise/continuation/correction）│
   │   程序：按分类执行——剥噪音、折叠催促、纠偏转 system、根任务标准化   │
   └────┬───────────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage2  提纯   02_gold/sessions + dpo_pairs + annotations          │
   │   程序：机械静默执行（手册强制）、轮询命令规范化                    │
   │   模型 decide_steps   ：哪些步剪除/折叠/改写/排除（判断权在模型）   │
   │   程序            ：确定性执行 + by/reason 血缘 + 不变量校验        │
   │   程序 extract_seams  ：机械提取"哪里被切断"                        │
   │   模型 audit_seams    ：接缝是否平滑 + 最小补丁（禁新增事实）       │
   │   模型 audit_quality  ：质量报告 + 高光步 + 是否需整条重写          │
   │                        （需整条重写 -> 报告并放弃，不做重写救回）    │
   └────┬───────────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ 人工复核（黄金数据页）逐条浏览 → 采纳/不采纳/手改/排除 → 定稿       │
   │   数据一变（流程签名变化）旧结论自动过期，需重新确认                │
   └────┬───────────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage3  合成   03_train/*.jsonl + dataset_info + stats             │
   │   程序：门控（discard 拦截）、窗口拼装（根任务锚定 + 历史 think 剥离）│
   │   模型：高光步 -> 过采样权重；DPO 逐对复核 -> good/swap/drop        │
   │   程序：全局打乱、按项目整包切 train/eval、写 dataset_info          │
   └───────────────────────────────────────────────────────────────────┘
```

分工原则（详见 `DATASET_BEST_PRACTICES.md` 三·五节）：**判断权归模型**（删什么、接缝顺不顺、
值不值得训、偏好对成不成立），**程序只做执行器与校验器**（执行决定、留血缘、守不变量、
机械提取接缝）。模型调用失败时程序保持原样并降级为待人工，绝不因漏审而放行。

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
默认模型 `deepseek-v4.1-flash-1017`，改 `config/pipeline.yaml` 的 `llm.model` 即可。

## 2.0 数据事实链

控制台和管线现在使用同一套可追踪事实链，不再把「采集、清洗、复核、生成」混成一次黑盒转换：

1. **Stage 1 · 采集与回合清洗**
   - 原始 `.jsonl` 仍是不可变副本；
   - 每个用户回合都有 `kind / action / source / reason`：明确区分直接保留、续接折叠、纠偏内化和彻底剥离；
   - 控制台可人工修改根任务与逐回合分类。人工覆盖优先于模型和正则，保存后自动重算下游；
   - 若源会话目录暂时不可达，可从 `01_raw` 副本离线重放。
2. **Stage 2 · 黄金提纯与复核**
   - 每一步同时保留 `raw_thinking`（原始）、`thinking`（脚本提纯）与模型/人工改写，控制台提供三态血缘对照；
   - 状态折叠、弯路剪枝、思考压缩、静默执行、轮询规范化分别计数，不再只展示笼统的「原长 → 现长」；
   - 评审基于未经截断的原始 thinking，人工可对任意一步改写、排除或强制保留。
3. **Stage 3 · 训练样本生成**
   - 应用模型/人工改写后会重新计算质量；已修复的长反刍或缺失思考步骤可以被挽救，不再沿用旧的病态标记误丢；
   - `sft_allow_subsegments` 支持从指定反例/混合会话抽取已验证的成功子段；
   - 复核改写会额外生成 `review_rewrite` DPO 对；
   - 每个会话和步骤都有入库漏斗：`raw → gold → admitted / dropped`，控制台可查看最终去向与原因。

## 目录

```
gold-thread/
├── config/pipeline.yaml          # 全部规则（准入、标签、回合、提纯、合成）
├── pipeline/
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

**训练版本（`<file>.v2.jsonl`）**
：经过二次修复后的最终裁剪版，是实际上机训练的数据集：
- `sft_step_windows_sharegpt.train.v2.jsonl`（548 条）/ `.eval.v2.jsonl`（221 条）—— SFT 主数据集
- `dpo_preference_pairs.train.v2.jsonl`（100 对）/ `.eval.v2.jsonl`（143 对）—— DPO 原始对
- `dpo_preference_pairs.train.v2w.jsonl`（窗口裁剪版）/ `.eval.v2w.jsonl`—— 48G 显存 DPO 专用
- `sft_gold_trajectories_sharegpt.*.v2.jsonl` —— 完整长轨迹（可选混训）

**训练配置与脚本**
：
- `train_sft_qwen3_8b.yaml` / `train_dpo_qwen3_8b.yaml` —— 8B 正式训练配置（含完整调试注释）
- `train_sft_smoke_qwen3_8b.yaml` / `train_dpo_smoke_qwen3_8b.yaml` —— 冒烟测试（200 条 SFT + 50 对 DPO）
- `train_sft.yaml` / `train_dpo.yaml` / `train_sft_qlora.yaml` —— 27B 配置与 QLoRA 备选
- `eval_sft_qwen3_8b.yaml` —— 单独 eval（训练时关掉 eval，训完用此跑）
- `check_env.py` / `check_data.py` —— 上机前自检（含 tokenizer 渲染验证）
- `smoke_infer_merged.py` —— 合并后推理冒烟（验证 `thinking` + `tool_call`）
- `verify_longctx.py` / `verify_longctx_api.py` —— 长上下文档级验收
- `probe_24g.py` —— 模拟小显存加载（bf16 / 8bit / 4bit 可行性探测）
- `prep_dpo_window.py` —— DPO 窗口裁剪（从头部丢旧步，保留 system + 根任务 + 最近步）
- `serve_api.sh` —— 合并模型后一键起 OpenAI 兼容 API（LLaMA-Factory hf 后端）

> 训练实跑记录与踩坑经验见 `data/03_train/RESULT.md`（含 Liger 必开原因、DPO ref_model 修正、
> eval OOM 绕过、流式+tools 补丁、SDK 兼容修复、脚本 bug 修复等硬经验）完整交接说明见 `data/03_train/handoff.md`。

切分按 `project_group` 整包留出（默认 `eval_ratio` 取项目数比例），
避免同一游戏的相邻步泄漏到验证集。

## 控制台

- **概览**：三阶段状态、正/负/待定分布、项目表、病态分布
- **原始数据**：准入审计、逐回合清洗决策、根任务编辑、消息流、即时改标签
- **黄金数据**：原始/脚本/模型三态对照、逐步人工处置、Stage 3 入库漏斗
- **训练数据**：SFT / 完整轨迹 / DPO / 诊断分型检视、项目与切分过滤、样本血缘
- **配置**：在线编辑 YAML（校验/保存），测试模型连通性
- **运行**：增量/全量、逐阶段或全流程、实时日志、历史记录

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