# GoldThread · 项目上下文

本文件是 **领域词汇与产品/工程需求的单一事实源**。Agent 写 issue、改管线或控制台前应先读这里；术语以本文为准，勿自造同义词。细节规范见 `DATASET_BEST_PRACTICES.md` 与 `README.md`；训练上机见 `data/03_train/handoff.md`。开发约束见 `agents.md`。

**最后整理**：2026-11-19（已完成 ADR-0008 Agent-First `gt` CLI 工具化、纯确定性构建门禁，以及 ADR-0009 虚拟数据集条目、CAS 去重快照池与演进 DAG 图谱看板）。

---

## 1. 我们在做什么

**GoldThread（金线）** 是以 **模型 Agent 为决策主体、确定性原语为执行底座、全链路留痕看板供人类审阅** 的训练数据精炼平台：把 **pi / Cursor Agent 会话轨迹** 增量研磨成 **可审计的 SFT / DPO 训练数据**，提供统一命令行入口 **`python gt.py`（`python -m goldthread`）**、模型技能书 [`skills/goldthread/SKILL.md`](skills/goldthread/SKILL.md) 与本地 **Web 留痕看板**（`serve.py` → 默认 `http://127.0.0.1:8765`）。

**目标能力**：教会模型在 **移动游戏逆向分析** 场景下自主工程化拆解（Unity IL2CPP / Godot 等），强调工具调用、实测验证、长轨迹连贯执行，而不是半途停工等用户说「继续」。

**目标底座模型**：**Qwen3.8-27B**（Dense，长上下文）；8B 用于冒烟与链路验证。交付物是 **`04_release/delivery/`** 下的 zip + `03_train/` 内 YAML / `train_lowmem.py` 等（见 handoff）。

**远程仓库**：`cnwinds/gold-thread`（本地路径常为 `dataset_sync`）。

---

## 2. 核心分工（不可违背）

| 谁 | 做什么 |
|----|--------|
| **程序** | 准入、去重、执行模型决定、留 **血缘**、不变量校验、机械接缝提取、窗口拼装、切分、打包 |
| **模型** | 回合语义分类、步级删/折/改/排、接缝是否顺、质量报告与高光、DPO 偏好对是否成立 |

模型只输出 **JSON 标注**；调用失败时 **保持原样并降级**，不因漏审而静默放行。详见 `DATASET_BEST_PRACTICES.md` §三·五。

**判断权归模型，程序是执行器**——写需求或测试时不要用「脚本自动删掉所有弯路」这类表述替代 `decide_steps`。

---

## 3. 数据阶段（管线词汇）

磁盘布局根目录为 `config/pipeline.yaml` 的 `paths.data_dir`（默认 `data/`）。

| 阶段 | 目录 | 含义 |
|------|------|------|
| **Stage 1 · 采集与清洗** | `01_raw/` | 源 `.jsonl` **逐字节不可变副本** + 同名 `.meta.json`（回合 kind、根任务、标签） |
| **Stage 2 · 黄金提纯** | `02_gold/` | 每条会话的 `.gold.jsonl`、剪枝/DPO、结构化决策与 `annotations/` |
| **Stage 3 · 训练集合成** | `03_train/` | ShareGPT / OpenAI 窗口、DPO、切分、`dataset_info.json` |
| **发布与归档** | `04_release/` | 不可变 `builds/`、`cas/`（按 SHA256 去重的原始/黄金历史快照池）、`delivery/` zip、`reports/` 历次评估与迭代对比报告 |

**增量状态与全链路账本**：`00_state/`（统一操作审计账本 `ledger.jsonl`、步级决策 `step_decisions.json`、DPO 裁决 `dpo_verdicts.json`、数据集草稿 `dataset_draft.json`、标签覆盖 `labels.json`、LLM 缓存、运行历史等）。

**构建确定性原则（ADR-0008）**：`gt curate` / `gt build` / `gt gate check` / `gt pack` 默认 **100% 纯确定性运行（零隐式外部 LLM 请求）**；主模型通过 CLI 直接写入结构化 JSON 决策并附带 `--reason` 留痕，小模型辅助初筛仅通过显式 `gt assist turns|judge|dpo` 触发。

流水线步骤的 **唯一代码事实源**：`pipeline/steps.py`（界面「数据梳理」三阶段：**采集与清洗 → 黄金提纯 → 训练集合成**；已无独立「人工复核」阶段）。

---

## 4. 会话与标签（Stage 1）

### 4.1 准入（Session-Level Gate）

只收 **`--d--crack-`** 等工作区前缀下的会话（见 `admission.workspace_prefixes`），排除 crack 根目录泛化问答、迁移重复前缀等。非目标目录会话 **不得** 进入训练口径。

### 4.2 来源血缘（不再设置数据集角色）

| 旧字段 | 现在的含义 | 典型来源 |
|------|------|----------|
| **positive** | Teacher 来源标识，仅用于血缘 | 主导模型匹配 `^deepseek` / `^glm-` / `^minimax` |
| **negative** | Student 来源标识，仅用于血缘 | 主导模型含 `qwen` |
| **pending** | 来源未知，无需人工处理 | 规则未命中或主导模型歧义 |

**SFT 与 DPO 不互斥**：同一会话可产生两种产物。控制台不再显示或编辑「正例 / 反例 / 待定」，改为展示主导模型、分析状态与实际 SFT / DPO 产物。旧 `label` 字段与 **`00_state/labels.json`** 仅为历史数据兼容保留；Stage 3 不再用 `sft_labels` 或这些标签限制 SFT。

### 4.3 回合级清洗

- **根任务（Root task）**：首条工程分析指令；之前回合视为 setup/noise 剥离。
- **折叠（Turn folding）**：「继续」「可以」等推进语折叠，避免训出「干几步就停」。
- **纠偏内化（Constraint internalization）**：跨目录偷看类用户纠偏 → 写入 system 铁律 + 坏步进 DPO。
- **根任务标准化**：仍可由 `pipeline.yaml` 的 `normalize_root_prompt` **自动**执行；**已移除**控制台人工覆盖「根任务锚点」与 `root_prompt` API。

### 4.4 采集（Collect）

从 **pi** 或 **cursor** 会话目录 **预览 → 勾选 → 采集**（`pipeline/collect.py` + `/api/collect/*`）。采集写入 `01_raw` 并保留 **`collected.json`** 等手工会话来源信息。

**产品口径**：

- 控制台 **Stage 1 页**：左侧按 **项目分组**（可折叠），筛选 **全部 / 未采集 / 已加入**；右侧 **对话视图** 与入库后一致。
- **批量「采集所选」**：ingest 到 `01_raw`，**不**自动 stage2。
- **批量「采集并分析」**：ingest 后对所选 sid 跑 stage2（`only=sids`）。
- **原始库列表**：按 **分析状态 / 项目 / 主导模型** 筛选，保留 **分析 / 重新分析**（抽屉内进度）。

---

## 5. 黄金与 DPO（Stage 2）

### 5.1 产物

- **黄金会话**：`02_gold/sessions/<project>/<sid>.gold.jsonl` — 提纯后的 **有用步** 连续流（理想「完美会话过程」）。
- **弯路 / 反面步**：`rejects/`、`dpo_pairs.jsonl` — 与 **同一会话 sid** 绑定的偏好对来源。
- **血缘**：每步保留 `raw_thinking` / 脚本 `thinking` / 模型改写及 `by` / `reason`；剪枝、折叠、压缩等 **分别计数**。

### 5.2 单会话分析

**分析** / **重新分析**（`/api/raw/session/{sid}/analyze`）：仅重算 **该 sid** — 有用步 → 黄金，弯路 → **该会话的 DPO**，其它会话不动。

### 5.3 人工复核（产品态 vs 数据态）

**产品决策（2026-11）**：

- **去掉**独立「审核」侧栏、工作台待办、黄金页 **通过 / 驳回 / 定稿** UI；梳理流程 **不再** 以人工复核为卡点。
- **保留**模型质量报告、黄金浏览、训练去向预览；**步级排除**在黄金抽屉（无通过/驳回/定稿）。

**遗留**：历史 `annotations` 里的人工通过/驳回 **仍可能被 Stage 3 门控读取**；新结论应依赖 **模型报告 + 用户在黄金/训练页的手工剔除**（见 §6）。README 流程图若仍写「人工复核页」，以 **本文与 `pipeline/steps.py`** 为准。

---

## 6. 训练集与血缘（Stage 3 + 控制台）

### 6.1 合成规则（摘要）

- **单步窗口**：`[system, 根任务] + 最近 W 步历史 + 目标步`；历史步 `<think>` 按 `strip_history_thinking` / `history_thinking_max_chars`（默认保留并截断至 600 字符）对齐推理期分布，仅目标步算 loss（`mask_history: true`）。
- **切分**：按 **`project_group` 整包** 留 eval，禁止逐步随机切分。
- **门控**：discard、模型否决、（遗留）复核态与步级质量决定样本去向，**不再按会话来源标签限制 SFT**。
- **DPO**：Stage 2 剪枝对 + 来源会话病态步 +（可选）模型逐对复核 good/swap/drop；与 SFT 可来自同一会话。

### 6.2 控制台页面与血缘（极简三栏导航）

左侧菜单已去除冗余的全局「原始/黄金/训练/交付」旧分页与「运行全流程」按钮，收敛为 **3 个正交入口**（旧 hash 如 `#raw`/`#gold`/`#train`/`#runs` 自动重定向至 `#overview`）：

| 顶层页面 | 用户应一眼看到什么 |
|------|-------------------|
| **数据集图谱 `#overview`（Variant A）** | 横向 **数据集演进 DAG 画布**（以带 `package_hash` 的交付包为唯一虚拟条目，连线挂载评估报告、包间 Diff 抽屉与区间 `ledger` 因果；工作区 Clean 时隐藏草稿、Dirty 时浮现虚线 `Workspace` 节点并提供一键构建/打包；右上角提供外部 pi/Cursor 会话采集抽屉）+ 下钻检视单数据集 **1. 概览体检（含项目切分表/样本构成/说明/长度统计/下载/删包） / 2. 训练切片 / 3. 黄金 CAS 快照（对话回复 + 三态血缘 + SFT 停用开关） / 4. 原始范围** |
| **账本与任务 `#tasks`（别名 `#ledger`）** | 统一检索 **Agent CLI + Web 全量操作审计账本（`ledger.jsonl`）**（按工具/主体/sid/关键词过滤，点击 sid 直达血缘），并可切换查看后台任务实时日志 |
| **配置 `#config`** | 按分组调整 `pipeline.yaml` 常用参数；高级参数与 YAML 源码保留在同一工作区 |

**异步作业 UX**：提交采集/分析/stage/打包后 **留在当前 tab**，toast 提示「任务中心可查看」；侧栏 **任务中心** 显示 **进行中 N** 角标。打包进度同时显示在 `#runs` 当前包状态区，完整日志统一在任务中心查看。

训练页：`*.lf.jsonl` 为 LLaMA-Factory 合法 ShareGPT（须先 `normalize_sharegpt.py`）；交付 zip 内工具调用格式与 **export profile** 锁定（Qwen XML 等）。

### 6.3 用户仍可做的手工编辑

- 黄金步：**排除 / 改写**（若 UI 仍暴露）— 删除应 **保留血缘**，不抹历史记录。
- 训练样本：从将要生成的集合中 **去掉不想要的行**（与「只从已分析黄金步生成」一致）。
- 会话删除：原始 / 黄金抽屉均可发起，后台任务联动移除当前工作区的 raw、gold、DPO / 诊断、训练样本及标签与人工覆盖；**不删除 pi / Cursor 源目录，不改冻结交付包**，运行与任务审计历史保留。

**与 ADR-0002 对齐**：`POST /api/collect` 的 `analyze` 默认 **false**；原始库单会话 **分析/重新分析** 不变。

### 6.4 数据集草稿与构建

- **数据集草稿**是用户调整将要生成的数据时使用的可变工作区：候选样本、自动准入结果、人工排除/恢复和项目切分计划共同决定草稿内容。
- 自动质量门禁仍是默认准入方式，不恢复全量人工审批。人工可以排除不想要的样本或撤销排除；排除必须跨重建保持，不能直接编辑某一次输出文件代替决策。
- **切分计划**按 `project_group` 持久保存。已有项目不因新增项目或普通重建而被动换组；比例/seed 的全量重算必须是带差异预览的显式操作。
- **层级决策**：训练数据页按 分组 → 黄金会话 → 轨迹 → SFT / DPO 节点 → 切片样本 展开。项目层或会话层标记停用时，下面的轨迹与切片一并停用（黄金页可直接标记某条会话不使用）；切分可在项目层设定、会话层覆盖，下面的切片一并变更。轨迹样本与单条切片仍可单独停用。
- **数据集构建**是草稿某次执行后的确定性结果。预览与实际构建必须共用同一准入和切分决策；历史构建与冻结 release 不随后续草稿编辑改变。
- **DPO 评估**在候选入集前完成并持久保存：待评估或评估失败的偏好对不进入训练集；`good/swap` 可以进入，`drop` 由质量门禁排除。非严格构建在模型评估关闭时可能放行 `bypassed`，交付 SOP 使用 `gt build --require-reviewed-dpo`。裁决针对当前上下文和两侧正文，不是对同一 ID 的永久授权。预览不临时调用模型，也不是黄金层已就绪或实际构建已过门禁的证明。
- **不可变 Build** 是一次构建的固定输入、样本、配置、切分、决策和文件哈希清单。交付打包必须读取明确 Build，不能从工作区历史文件中另选一份数据；最新 Build 与激活 release 是不同概念。
- **Stage 3 只构建、不自动打包。** 构建成功后到交付页显式选格式打 zip；换格式重打不必重跑 Stage 3。
- **交付入口以 zip 历史包为准**：用户从交付页下载/删除历史包，或新建包上机训练；控制台不再引导发布/激活/回滚。底层 `releases/` 与 `ACTIVE_RELEASE.txt` API 仍可被脚本调用，但不构成产品主路径。
- 采集、分析、草稿、构建、打包是不同事实：采集不等于入集，分析不等于构建，构建不等于打包。

---

## 7. 术语表（Glossary）

| 术语 | 定义 |
|------|------|
| **sid** | 会话 ID，贯穿 raw / gold / train metadata |
| **采用（adopted）** | 会话已进入 `01_raw` 索引，参与分析管线 |
| **采集** | 从 pi/cursor **源目录** ingest 到 `01_raw`，不等于仅只读同步 |
| **分析** | Stage 2（+ 依赖的 stage1 重放）对 **单会话** 的提纯与 DPO 构造 |
| **黄金步** | 进入 `.gold.jsonl` 的保留助手步 |
| **弯路** | 被 prune/fold 等移除或标记为反面的步，进 DPO / rejects |
| **接缝（seam）** | 删改造成的上下文断点，由模型 audit 是否需最小补丁 |
| **review_key** | 绑定源指纹 + 回合流签名；源或回合人工变更 → 旧模型评审 **过期** |
| **project_group** | 如 `game::meowdoku`，用于 eval 整包留出 |
| **Teacher / Student** | 会话的主导模型来源，用于血缘与数据分布分析，不代表整条会话质量或固定训练去向 |
| **delivery / release** | 可上机训练的冻结 zip（`04_release/delivery/`）；`releases/` / `ACTIVE_RELEASE` 为遗留脚本能力，非控制台主路径 |
| **lf 数据集** | `*.lf.jsonl` + `gt_*_lf` 键，避免 ShareGPT 交替角色静默丢样本 |
| **候选样本** | 黄金数据经自动合成与质量门禁后、可进入数据集草稿的 SFT / DPO 逻辑样本；候选不等于已发布 |
| **数据集草稿（Dataset Draft）** | 当前将要生成的可变集合，由候选样本、准入结果、人工排除/恢复和切分计划共同决定 |
| **决策账本（Decision Ledger）** | 按逻辑 `sample_id` 保存排除或恢复自动决策的记录；排除跨重建生效，不依赖浏览器状态 |
| **切分计划（Split Plan）** | 按 `project_group` 持久保存的 train/eval 归属及显式固定项，会话层可覆盖项目归属，避免普通重建造成项目漂移 |
| **数据集构建（Dataset Build）** | 使用同一草稿决策生成的确定性结果与统计快照；不等于冻结 release |
| **DPO 评估** | 在构建前按偏好对内容和评审模型缓存 `pending/good/swap/drop/review_failed`；`bypassed` 表示模型判断被显式关闭 |
| **不可变 Build** | 已固定输入指纹、样本、配置、切分、决策和文件哈希的构建版本；交付打包的来源，不被后续草稿或会话删除修改 |
| **虚拟数据集条目** | 控制台对外的一级实体（以交付包 `package_hash` 为标识，内部收敛 `Build` 元数据），点击可穿透查看其概览体检、训练切片、黄金 CAS 快照与原始范围四层数据 |
| **CAS 快照池** | `04_release/cas/<sha[:2]>/<sha>` 内容寻址去重对象池：新建 Build（`fidelity=full`）按 SHA256 完整保存原始/黄金/标注文件并按冻结配置回放，免疫工作区后续删改；旧 Build 按已记录哈希校验回填（`backfilled`）；无关联 Build 的早期独立包仅冻结训练切片（`none`） |
| **parent_pack** | 交付包 `MANIFEST.json` 中的父包 `package_hash`（默认自动继承打包前最新包，可显式指定），与评估报告共同构成演进 DAG |
| **工作空间 Clean / Dirty** | 比对当前工作区实质数据哈希与最新正式包绑定 Build：一致（Clean）时隐藏草稿卡片；存在未打包演进变更（Dirty）或 0 包时在图谱末端浮现虚线 `Workspace` 节点 |

避免说法：把旧 **negative 字段**叫「垃圾会话全删」— 任何来源都可能同时提供 SFT 与 DPO；把旧 **pending 字段**叫「未采集」— 它只表示来源未知，未采集则指源目录尚未 ingest。

---

## 8. 数据质量与已知限制

- **工具输出乱码**：Windows 控制台 GBK → UTF-8 误解码导致 observation 内 **U+FFFD**（约一成量级）；`system`/`human`/`gpt` 基本无。训练可接受但应知悉；清洗策略未默认开启。
- **无执行级回放**：训练依据会话内可观测证据，不验证磁盘产物是否真跑通。
- **长会话**：模型分类/评审可能均匀抽样，极端情况漏局部问题。
- **文档漂移**：`START_HERE.md` §6 与默认 export profile 的 parser 组合需对照 `FORMAT_CONTRACT` / 交付页说明（`README.md` 已与 `CONTEXT` 对齐）。

---

## 9. 工程约束（Agent 必守）

- 每项行为变更 **配套 pytest**；完成前 **`python -m pytest tests/ -q`**（见 `agents.md`）。
- 用户 **未要求** 时不 `git commit` / `push`；不提交 `_tmp_*` 与本地训练 dump。
- 本地服务：**Python 3.12** 与 `python serve.py --port 8765 --no-browser`。
- Matt 工程流：Issue 见 `docs/agents/issue-tracker.md`（GitHub + `gh`；离线用 `.scratch/`）。

---

## 10. 已决策事项（ADR）

| 主题 | 结论摘要 | 文档 |
|------|----------|------|
| Stage 3 门控 | 以 **model_report + 步级 exclude/手改** 为主；**`needs_human` → auto_gold**（无 `hold_needs_review`）；历史 **rejected** 仍阻断直至显式重处理 | [ADR-0001](docs/adr/ADR-0001-stage3-gate-without-human-review-ui.md) |
| 采集 vs 分析 | 目标 **两个动作**：仅采集 / 采集并分析；原始库保留单会话 **分析·重新分析** | [ADR-0002](docs/adr/ADR-0002-collect-vs-analyze-actions.md) |
| observation 乱码 | **默认不修**；若做则 stage1 转码 + 可选 stage3 阈值门控 | [ADR-0003](docs/adr/ADR-0003-observation-encoding-deferred.md) |
| Unsloth | **非目标**，除非显式立项 | [ADR-0004](docs/adr/ADR-0004-no-unsloth-migration.md) |
| 数据集草稿与切分计划 | 训练页操作持久化到草稿决策账本；项目归属跨重建稳定，显式重规划才允许批量移动 | [ADR-0006](docs/adr/ADR-0006-dataset-draft-and-split-plan.md) |
| DPO 评估与不可变 Build | 评估先于构建；Build 固定输入与文件哈希，产品打包只读指定 Build，不自动激活 release | [ADR-0007](docs/adr/ADR-0007-dpo-evaluation-and-immutable-builds.md) |
| Agent-First 工具集与操作账本 | 统一 `gt` CLI + `SKILL.md`；主构建默认 100% 纯确定性（小模型初筛收至 `gt assist`）；CLI/Web 写操作统一留痕 `00_state/ledger.jsonl`，评估报告归档至 `04_release/reports/` 并呈现在可观测看板 | [ADR-0008](docs/adr/ADR-0008-agent-first-cli-and-operation-ledger.md) |
| 虚拟数据集、CAS 快照与演进 DAG | 以 `package_hash` 包为虚拟数据集条目（内聚概览/训练/黄金/原始四层）；`04_release/cas/` 按 SHA256 去重归档会话产物实现 100% 历史回放；工作空间按 Clean/Dirty 动态显隐；首页展示演进关系图 | [ADR-0009](docs/adr/ADR-0009-virtual-datasets-cas-and-lineage-dag.md) |

**本地 backlog**（`.scratch/goldthread-backlog/issues/`）：

| Ticket | 状态 |
|--------|------|
| 001 hold_needs_review | 已完成 |
| 002 采集/分析拆分 | 已完成 |
| 003 README 对齐 | 已完成 |
| 004 黄金步级排除 | 已完成 |

新冲突以 **ADR + 本表** 为准；变更决策时先改 ADR 再改代码。
