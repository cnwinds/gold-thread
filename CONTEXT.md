# GoldThread · 项目上下文

本文件是 **领域词汇与产品/工程需求的单一事实源**。Agent 写 issue、改管线或控制台前应先读这里；术语以本文为准，勿自造同义词。细节规范见 `DATASET_BEST_PRACTICES.md` 与 `README.md`；训练上机见 `data/03_train/handoff.md`。开发约束见 `agents.md`。

**最后整理**：2026-11-05（自仓库文档、配置、控制台文案与近期产品决策归纳）。

---

## 1. 我们在做什么

**GoldThread（金线）** 是把 **pi / Cursor Agent 会话轨迹** 增量研磨成 **可审计的 SFT / DPO 训练数据** 的工具，附带本地 **Web 控制台**（`serve.py` → 默认 `http://127.0.0.1:8765`）。

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
| **Stage 2 · 黄金提纯** | `02_gold/` | 每条会话的 `.gold.jsonl`、剪枝/DPO、模型 `annotations/` |
| **Stage 3 · 训练集合成** | `03_train/` | ShareGPT / OpenAI 窗口、DPO、切分、`dataset_info.json` |
| **发布** | `04_release/` | 冻结版本、`delivery/` zip；工作区与 `03_train` 分离 |

**增量状态**：`00_state/`（标签覆盖 `labels.json`、LLM 缓存、运行历史等）。

流水线步骤的 **唯一代码事实源**：`pipeline/steps.py`（界面「数据梳理」三阶段：**采集与清洗 → 黄金提纯 → 训练集合成**；已无独立「人工复核」阶段）。

---

## 4. 会话与标签（Stage 1）

### 4.1 准入（Session-Level Gate）

只收 **`--d--crack-`** 等工作区前缀下的会话（见 `admission.workspace_prefixes`），排除 crack 根目录泛化问答、迁移重复前缀等。非目标目录会话 **不得** 进入训练口径。

### 4.2 数据集角色标签（不要用「好坏会话」混称）

| 标签 | 含义 | 典型来源 |
|------|------|----------|
| **positive** | Teacher，SFT 正样本源 | 主导模型匹配 `^deepseek` / `^glm-` |
| **negative** | Student，DPO / 诊断源 | 主导模型含 `qwen` |
| **pending** | 待定，需人工定标签 | 规则未命中或主导模型歧义 |

手动标签在 **`00_state/labels.json`**，优先级最高。控制台 **改标签**（`relabel`）**不**在保存时同步跑 Stage 3（避免长时间阻塞像报错）。

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
- **原始库列表**：已去掉「全部/已采用/未采用」分段筛选；保留 **采用** 列与 **分析 / 重新分析**（抽屉内进度）。

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

- **单步窗口**：`[system, 根任务] + 最近 W 步历史 + 目标步`；**历史步 strip thinking**，仅目标步算 loss（`mask_history` 等，见 handoff）。
- **切分**：按 **`project_group` 整包** 留 eval，禁止逐步随机切分。
- **门控**：discard、模型否决、（遗留）复核态等决定 sid 是否产样本。
- **DPO**：Stage 2 剪枝对 + 反例病态步 +（可选）模型逐对复核 good/swap/drop。

### 6.2 控制台四页血缘（2026-11 需求）

| 页面 | 用户应一眼看到什么 |
|------|-------------------|
| **原始数据 `#raw`** | 会话是否 **已采用/已分析**；抽屉内 **分析 / 重新分析** + 改 **数据集角色** |
| **黄金数据 `#gold`** | 每会话 **SFT 窗口数 vs DPO 对数** 及来源关系 |
| **训练数据 `#train`** | 默认 **将要生成** 的数据集；样本可 **回到 sid** |
| **梳理与交付 `#runs`** | 选 **导出 profile** 打包；**历史 zip** 列表与下载 |

训练页：`*.lf.jsonl` 为 LLaMA-Factory 合法 ShareGPT（须先 `normalize_sharegpt.py`）；交付 zip 内工具调用格式与 **export profile** 锁定（Qwen XML 等）。

### 6.3 用户仍可做的手工编辑

- 黄金步：**排除 / 改写**（若 UI 仍暴露）— 删除应 **保留血缘**，不抹历史记录。
- 训练样本：从将要生成的集合中 **去掉不想要的行**（与「只从已分析黄金步生成」一致）。

**与 ADR-0002 对齐**：`POST /api/collect` 的 `analyze` 默认 **false**；原始库单会话 **分析/重新分析** 不变。

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
| **Teacher / Student** | 正例标签源 vs 反例标签源（见 §4.2），不是两个独立产品 |
| **delivery / release** | 可上机训练的冻结包；`ACTIVE_RELEASE.txt` 指向当前版本 |
| **lf 数据集** | `*.lf.jsonl` + `gt_*_lf` 键，避免 ShareGPT 交替角色静默丢样本 |

避免说法：把 **negative 标签** 叫「垃圾会话全删」— 负例会话是 **DPO 与诊断的金矿**；把 **pending** 叫「未采集」— 未采集指 **源目录尚未 ingest**。

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

**本地 backlog**（`.scratch/goldthread-backlog/issues/`）：

| Ticket | 状态 |
|--------|------|
| 001 hold_needs_review | 已完成 |
| 002 采集/分析拆分 | 已完成 |
| 003 README 对齐 | 已完成 |
| 004 黄金步级排除 | 已完成 |

新冲突以 **ADR + 本表** 为准；变更决策时先改 ADR 再改代码。
