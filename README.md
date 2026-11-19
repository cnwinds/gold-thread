# GoldThread（金线）

Agent 训练数据维护工具：把 pi / Cursor 会话轨迹增量研磨成可审计的 SFT / DPO 数据集，附带本地 Web 控制台。

**产品与术语**以根目录 [`CONTEXT.md`](CONTEXT.md) 为准（控制台行为、采集/分析、血缘）；本 README 侧重管线结构与命令。

```
pi / Cursor 会话
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage1  采集与清洗   01_raw/{positive,negative,pending}            │
   │   源目录预览勾选 → ingest；准入 / 去重 / 剥离外壳标签 + meta.json   │
   │   回合流：剥前置噪音、折叠催促续跑、纠偏内化（支持 gt curate turns） │
   └────┬───────────────────────────────────────────────────────────────┘
        │  单会话分析或 gt curate steps 落盘决策
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage2  黄金提纯   02_gold/sessions + dpo_pairs + annotations    │
   │   纯确定性执行：静默执行、状态折叠、弯路剪枝、步级决策应用 + 血缘   │
   │   弯路 / 假收尾折叠 / 思维链压缩 → 自动沉淀为同会话 DPO 偏好对      │
   └────┬───────────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ stage3  训练集合成与归档   03_train/ + 04_release/{builds,cas}     │
   │   门控、单步窗口（根任务锚定 + 截断保留历史 think）、按项目整包切分 │
   │   固化不可变 Build + CAS 内容寻址去重快照 → 8 项硬门禁核验          │
   │   打包交付：`04_release/delivery/goldthread-train-<hash>.zip`      │
   └───────────────────────────────────────────────────────────────────┘
```

> **Legacy**：旧版曾在控制台提供独立「人工复核」队列（通过/驳回/定稿）。该 UI 已移除；`02_gold/annotations` 里仍可能有历史人工结论，Stage 3 门控规则见 `CONTEXT.md` 与 ADR-0001。

分工原则（详见 `DATASET_BEST_PRACTICES.md` 三·五节）：**判断权归模型**（删什么、接缝顺不顺、
值不值得训、偏好对成不成立），**程序只做执行器与校验器**（执行决定、留血缘、守不变量、
机械提取接缝）。模型调用失败时程序保持原样并降级，不因漏审而静默放行。

## 快速开始

```bash
cd gold-thread          # 仓库根目录，下文命令均在此执行

# 1. 启动本地留痕与血缘可观测看板
python serve.py                 # 打开 http://127.0.0.1:8765
python serve.py --port 9000 --no-browser

# 2. Agent-First 统一确定性 CLI（详见 skills/goldthread/SKILL.md）
python gt.py inspect graph --json                                        # 虚拟数据集演进 DAG 与工作空间状态
python gt.py inspect corpus --json                                       # 语料库全景摘要
python gt.py inspect source --dir "D:\train-project\teacher" --unadopted # 扫描外部未入库轨迹
python gt.py eval pack latest --json                                     # 训练包全身体检
python gt.py eval diff d9dbbbd1 308e6497                                 # 两版交付包指标对比
python gt.py curate adopt --from "..." --select <sids> --reason "..."    # 采纳会话并记账本
python gt.py curate steps --file decisions.json --reason "..."           # 写入步级决策并重算黄金层
python gt.py build --reason "构建新版训练集"                              # 纯确定性构建 + CAS 快照 + 硬门禁
python gt.py gate check latest                                           # 单独校验 8 项硬性门禁
python gt.py pack --profile qwen-xml --reason "打包交付 zip"              # 自动记录 parent_pack + 软压缩出包
python gt.py report publish --title "..." --md report.md --after <hash> --reason "归档评估报告"
python gt.py ledger list --limit 50                                      # 查看全链路操作审计账本

# 3. 兼容底层管线入口（纯确定性运行；可选小模型初筛走 gt assist turns|judge|dpo）
python -m pipeline.run stage1
python -m pipeline.run stage2 --force
python -m pipeline.run stage3
```

模型调用走 pi 自己的 `~/.pi/agent/models.json`（provider / baseUrl / apiKey），
默认模型见 `config/pipeline.yaml` 的 `llm.model`（通过 pi `models.json` 调用）。

## 2.0 数据事实链

控制台与管线共用同一套 **sid 血缘**（采集 → 分析 → 黄金 → 训练样本），见 `CONTEXT.md` §6：

1. **Stage 1 · 采集与回合清洗**
   - 从 **pi / Cursor** 预览勾选；**采集所选** 只 ingest，**采集并分析** 再跑 stage2；
   - 已加入列表里可对单会话 **分析 / 重新分析**，展示主导模型与 SFT / DPO 实际产物（不再手工设置数据集角色）；
   - 原始 `.jsonl` 为不可变副本；回合 `kind / action / source / reason` 写在 `.meta.json`；
   - 可改逐回合分类；`review_key` 变化会使旧模型标注过期。
2. **Stage 2 · 黄金提纯**
   - 有用步 → 该 sid 的 `.gold.jsonl`；弯路 / 假收尾折叠 / 病态改写 → 该 sid 的 DPO / rejects；
   - `raw_thinking` / 脚本 `thinking` / 改写思考三态可对照。
3. **Stage 3 · 训练样本与虚拟数据集归档**
   - 门控 + 单步窗口 + 按项目整包切分；固化不可变 `builds/` 并按 SHA256 去重归档会话产物至 `04_release/cas/`；
   - 打包产出带 `package_hash` 与 `parent_pack` 的交付 zip，在控制台聚合成可四层穿透检视的**虚拟数据集条目**。

## 目录

```
gold-thread/
├── CONTEXT.md                    # 产品/领域词汇与控制台需求（Agent 先读）
├── gt.py                         # Agent-First CLI 快捷入口（等价 python -m goldthread）
├── skills/goldthread/SKILL.md    # 给模型看的标准化工具手册与闭环 SOP
├── config/pipeline.yaml          # 全部规则（准入、标签、回合、提纯、合成、导出 profile）
├── goldthread/                   # Agent-First 工具原语层（零隐式 LLM + 自动写 ledger）
│   ├── cli.py                    # 统一命令行路由与 --json 输出
│   ├── ledger.py                 # 统一操作审计账本（00_state/ledger.jsonl）
│   ├── lineage_graph.py          # 虚拟数据集实体、演进 DAG 与 CAS 四层时间旅行引擎
│   ├── inspect_tools.py          # 源目录 / 语料库 / 会话 / 单步分级摘要检查器
│   ├── eval_tools.py             # 训练包契约体检、学生实测归因、包间 Diff 引擎
│   ├── curate_tools.py           # 采纳、删会话、回合/步级/DPO 决策落盘执行器
│   ├── build_tools.py            # 纯确定性构建、8 项硬性门禁 (Gate)、打包器
│   ├── report_tools.py           # 评估报告归档与版本演进关联
│   └── assist_tools.py           # 显式可选的小模型初筛子命令
├── pipeline/
│   ├── cas_store.py              # 按 SHA256 去重的内容寻址快照池（04_release/cas）
│   ├── dataset_build.py          # 不可变 Build 固化与配置校验
│   ├── package.py                # 交付 zip 打包、parent_pack 血缘与软压缩长度闸门
│   ├── collect.py                # pi / Cursor 采集入口
│   ├── common/                   # pilog 解析 · quality 检测 · llm 调用 · format 转换
│   ├── stage1_extract.py         # 原始提取与规范 game::<slug> 归组
│   ├── stage2_gold.py            # 黄金提纯（折叠/剪枝/结构化决策执行）
│   ├── stage3_synth.py           # 训练数据合成与不可变 Build 固化
│   ├── train_template/           # 内置 autotrain.py 自适应单/多卡训练包模板
│   └── run.py                    # 底层阶段执行器
├── server/app.py                 # FastAPI 演进图谱、虚拟数据集与数据血缘服务
├── web/index.html                # 单页可观测看板前端（Variant A 演进图谱 + 四层钻取）
└── data/
    ├── 00_state/                 # 操作账本 ledger.jsonl、决策、草稿、增量状态与缓存
    ├── 01_raw/                   # 源会话不可变副本与 .meta.json
    ├── 02_gold/                  # sessions / annotations / rejects / dpo_pairs
    ├── 03_train/                 # 当前工作区 SFT / DPO / 诊断 / dataset_info / stats
    └── 04_release/               # 不可变 builds/ · cas/ 去重快照池 · delivery/ zip · reports/
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
| 合成 | 单步滑窗、根任务锚定、历史 think 截断保留（可配剥离）、权重、按项目切分、DPO 组装 | 重写后的 thinking、负例润色（可扩展） |

模型只输出 JSON 标注（`02_gold/annotations/*.json`），由脚本校验后应用；
同一个 prompt + 同样输入会命中 `00_state/llm_cache/`，重跑不重复计费。

## 数据语义

**stage1 来源血缘标识**（`config.labeling`，仅用于血缘统计，不再限制 SFT 去向）：

- `positive`：主导模型匹配 `^deepseek` / `^glm-` / `^minimax`（Teacher 源）
- `negative`：主导模型匹配 `qwen`（Student 源）
- `pending`：模型占比低于 `dominant_ratio_min` 或未匹配规则（来源未知，无需手工干预）

**stage2 产出**：

- `sessions/<project>/<sid>.gold.jsonl`：一条会话一行，含完整 messages、`quality`、
  `flags`（`state_folded` / `detour_pruned` / `thinking_compressed` / `thinking_synthesized`）
- `rejects/<sid>.pruned.jsonl`：被剪掉的弯路
- `rejects/<sid>.diagnostics.jsonl`：原始流中的全部病态步
- `dpo_pairs.jsonl`：折叠/剪枝产生的天然 chosen/rejected 对
- `annotations/<sid>.json` 与 `00_state/step_decisions.json`：结构化步级决策与质量报告

`review_key` 同时绑定原始文件指纹和回合流签名；人工调整回合分类后，旧模型评审会被标为过期。

**stage3 构建与标准交付包（`goldthread-train-<hash>.zip`）**：

| 产物层 | 关键文件 / 目录 | 说明 |
|---|---|---|
| 工作区与 Build | `sft_step_windows_sharegpt.{train,eval}.jsonl`、`dpo_preference_pairs.{train,eval}.jsonl`、`BUILD.json` | Stage 3 合成窗口与不可变 Build 清单（含文件哈希与 `cas_manifest`） |
| CAS 快照池 | `04_release/cas/<sha[:2]>/<sha>` | 按 SHA256 去重归档原始与黄金会话文件：新建 Build（`cas_fidelity=full`）保障 100% 精确时间旅行回放；旧 Build（`backfilled`）按历史哈希校验回填；无 Build 的早期独立包仅冻结训练切片 |
| 冻结交付包 | `04_release/delivery/goldthread-train-<hash>.zip` | 内置 `autotrain.py`、合并连续同角色后的 `data/*.lf.jsonl`、本地分词软压缩 `length_gate`、`MANIFEST.json`（含 `parent_pack` 与 `build_id`） |

上机训练直接解压 `goldthread-train-<hash>.zip`，在包内执行 `python autotrain.py sft` / `python autotrain.py dpo` 即可自动完成硬件探测、低显存 `train_lowmem.py` 启动、权重合并与冒烟验收（详见 `pipeline/train_template/README.md`）。切分按 `project_group`（`game::<slug>`）整包留出，避免同一游戏的相邻步泄漏到验证集。

## 控制台（极简三栏留痕审阅中心）

- **数据集图谱 `#overview`（Variant A 主工作台）**：
  - 横向 **数据集演进 DAG 画布**：以带 `package_hash` 的交付包为唯一虚拟条目，连线挂载评估报告、包间指标 Diff 抽屉与区间操作账本；工作区同步时自动隐藏草稿，存在未打包改动时浮现虚线 `Workspace` 节点并提供构建/打包动作；顶栏支持打开外部 pi/Cursor 源会话**采集**抽屉；
  - **单数据集四层内聚钻取**：点击任意数据集切换 **1. 概览与门禁体检（含项目分布、阶段构成、包说明、长度统计、下载、删除包） / 2. 训练数据 / 3. 黄金数据 (CAS 快照 · 完整对话回复 + 三态血缘 + 停用 SFT 开关) / 4. 原始数据**
- **账本与任务 `#tasks`（`#ledger`）**：统一检索 **CLI + Web 全量操作审计账本（`ledger.jsonl`）** 与后台任务实时日志
- **配置 `#config`**：在线检视与编辑 `pipeline.yaml`

（无独立「审核」侧栏；策展判断统一由主模型经 `gt curate` 写入账本，控制台仅保留应急停用开关。）

## 配置要点

```yaml
paths.data_dir: "data"                          # 相对仓库根目录，可改绝对路径
paths.sessions_dir: "~/.pi/agent/sessions"       # 支持 ~ 与绝对路径
admission.workspace_prefixes: ["--d--crack-"]   # 新项目自动准入
labeling.positive_model_patterns: ["^deepseek", "^glm-", "^minimax"]
labeling.negative_model_patterns: ["qwen"]
turns.use_model: false           # 默认纯确定性规则，需要小模型初筛用 gt assist turns
gold.min_gold_steps: 10          # 低于此步数不进黄金
judge.enabled: false             # 默认纯确定性，主模型经 gt curate steps 写决策
synth.window_turns: 8            # 单步窗口历史长度
synth.eval_ratio: 0.25           # 按项目整包留出验证集
```

## 已知限制

- 工具输出里的乱码（Windows GBK 输出被记为 UTF-8）当前原样保留，未做编码修复；
- 模型回合分类/评审对长会话会做均匀抽样，极端情况下可能漏掉局部问题；
- 训练数据仍以「会话内可观测的成功证据」为准，没有执行级回放验证。