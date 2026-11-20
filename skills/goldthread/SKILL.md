---
name: goldthread
description: 从教师会话提纯首版训练集，或依据学生实测诊断迭代下一版数据集。覆盖 gt 命令行使用与动态自查、证据定位、黄金步与合并接缝改写、DPO 因果裁决、门禁打包与交接验收。操作训练语料、评估学生模型、修补黄金步/DPO、构建或打包时使用。
---

# GoldThread 数据集策展与迭代

## 目标

- 以**可核验的目标行为提升**驱动数据修改，不以增加样本数、缩短步数或重打新包为目标。
- 开轮先立一行任务卡：**目标行为｜基线（首版写 `无父包`，迭代写父包哈希与学生实测目录）｜证据来源｜禁入训的留出集与成功标准**。基线不明、日志失配或无改进必要时，只做诊断或归档“不改”。
- 严格区分**数据可交付**与**学生已提升**：未在相同训练配置、工具权限、预算与独立留出集上完成重训盲测前，固定标注 **“数据包已验收，学生效果未验证”**。

## 工作场景流程

1. **首版教师数据提纯**：`inspect graph/corpus/source` 摸底 → `curate adopt` 采纳任务完整且产物可验的教师会话 → `curate turns/steps` 剥离噪声回合、修剪失败弯路、审校合并接缝并统合最终回复 → 逐条审完 DPO → 严格构建、打包与门禁验收。
2. **学生实测反馈迭代**：`eval student` 扫描实测目录 → 对照实际交付物与原始会话日志定性病灶（先排除环境、工具或预算差异；如三步已验收属高效，表层拆包漏解加密资源或抛选择题停下才属早停） → 仅采纳 `train` 集失败会话作 `negative` 源并用 `session_sft:<sid>` 阻断其进入 SFT → 针对病灶构建反早停/纠错 DPO 并重校黄金层 → 严格构建、`--parent` 绑定父包打包、下发包门禁校验与 `eval diff` 对账 → `report publish` 归档交接。

## 关键步骤

### 1. 获取最新命令行用法（执行前自查）

统一入口为 `python gt.py`（等价于 `python -m goldthread`），所有子命令均支持 `--json` 与 `--config <yaml>`。**参数或子命令不确定、或怀疑 CLI 已演进时，按三级自省获取实时定义（文档与 CLI 不一致时一律以实时 `--help` 与 `goldthread/cli.py` 为准）**：

```bash
python gt.py --help                      # 1) 查看顶层命令组：inspect | eval | curate | build | gate | pack | report | ledger | assist
python gt.py <group> --help              # 2) 查看组内动作（如 python gt.py curate --help）
python gt.py <group> <action> --help     # 3) 查看精确参数签名（如 python gt.py curate steps --help）
python gt.py inspect config --json       # 4) 查看当前合并生效的配置叶值、来源与作用阶段
```

### 2. 标准命令与操作序列

1. **摸底与定位证据（只读）**
   - 演进图与语料全景：`python gt.py inspect graph --json`、`python gt.py inspect corpus [--project game::xxx] --json`、`python gt.py ledger list --limit 20 --json`
   - 源会话与学生实测：`python gt.py inspect source --dir <sessions_dir> --unadopted --json`、`python gt.py eval student --student-dir <dir> [--teacher-dir <dir>] --json`
   - 会话步骤与三态血缘：`python gt.py inspect session <sid> --steps --json`、`python gt.py inspect step <sid> <msg_id|step_idx> --json`
2. **确定性策展改写（必带 `--reason "..." --json` 留痕 `ledger.jsonl`）**
   - 采纳 / 级联删除：`python gt.py curate adopt --from <dir> --select <sid前缀> --origin <pi|cursor> --reason "..." --json`；`python gt.py curate drop-session <sid> --reason "..." --json`
   - 回合覆盖（`root|task|setup|noise|continuation|correction`）：`python gt.py curate turns <sid> --file <patch.json> --reason "..." --json`
   - 步级决策与重算黄金层：`python gt.py curate steps --file <decisions.json> --reason "..." --json`
     - JSON 结构：`{"sessions": {"<sid>": {"steps": {"<mid>": {"action": "keep|prune|fold|rewrite|exclude|clear_text", "rewritten": "<新thinking>", "rewritten_text": "<无工具收尾步新正文>", "reason": "..."}}, "seam_patches": {"<mid>": {"patched_thinking": "..."}}, "model_report": {"verdict": "trainable|discard|needs_human", "reason": "..."}}}}`
   - 作用域 / 样本停用与切分锁定：`python gt.py curate exclude --scope <project:x|session:sid|session_sft:sid|session_dpo:sid> [--sample <id>] [--restore] --reason "..." --json`；`python gt.py curate split --pin game::xxx=eval --reason "..." --json`
   - DPO 偏好裁决（`good|swap|drop`）：`python gt.py curate dpo --verdicts @<verdicts.json> --reason "..." --json`（或通过 `/api/dataset/draft?kind=dpo` 与 `/api/dataset/draft/sample` 核查完整上下文）
3. **严格构建、打包、验收与交接**
   - 严格构建：`python gt.py build --require-reviewed-dpo [--dry-run] --reason "..." --json`（阻于待审 DPO 时逐条复核后重试，不取消 `--require-reviewed-dpo`）
   - 交付打包：`python gt.py pack --build <build_id> --profile <Qwen3.8-27B|hermes-json|qwen-xml> --parent <parent_hash> [--copy-to <dir>] --reason "..." --json`（CAS 快照存于 `04_release/cas/`）
   - 实体门禁与差异对账：`python gt.py gate check "<实际下发zip路径>" --json`（不用 `latest` 代替）、`python gt.py eval pack <target> --json`、`python gt.py eval diff <before> <after> --json`
   - 归档交接报告：`python gt.py report publish --title "..." --md <report.md> --before <old> --after <new> --build <build_id> --summary "..." --reason "..." --json`（查询用 `report list` / `report show <id>`）

## 必备常识

- **唯一写入口与零隐式 LLM**：严禁直接增删改 `data/`、原始会话日志或已冻结发布包；所有写操作必须经 `gt` 原语写入账本。主链纯确定性运行，`gt assist turns|judge|dpo` 仅作显式辅助初筛，不可替代人工/Agent 证据裁决。
- **留出集（Eval）红线**：`eval` 项目及其近重复任务严禁参与策展或提取负例；训练会话中若混入用户误贴的留出项目提示词，必须将该回合标为 `noise` 彻底剥离；迭代前后 `sft_eval` 与 `dpo_eval` 规模无故不得变动。
- **`negative` 标签不会自动拦截 SFT**：`label=negative`（如 `^goldthread`、`qwen` 等学生模型）仅标识血缘；未跑模型评审的会话默认 `unreviewed` 仍会进入 SFT。采纳学生实测会话提取 DPO 后，**必须**立即执行 `gt curate exclude --scope session_sft:<sid>`，确保学生会话 SFT 入训条数为 `0`，防止自举污染。
- **步级安全阀与 8 项硬门禁**：首步禁 `prune`，单会话累计 `prune` 不得超过助手步数的 `35%`；`rewrite` / `rewritten_text` 禁空壳、禁跨语种翻译、禁伪造 `<think>`/`<tool_call>`，且 `rewritten_text` 仅限无工具调用的收尾步。交付包必须 100% 通过 8 项门禁（SFT 非空、中途早停为 0、Prompt 标签零泄漏、工具契约 100% 合法、目标思考完整、项目组规范 `game::*`、DPO 对有效、长度门禁生效）。

## 经验

1. **先小样试审（含反例）再批处理**：大改前先抽 2~3 条样本（至少含 1 条“不应改”的反例），写清 `sid / msg_id|sample_id → 原始回执 → 改或不改理由`；不能把后文才得知的结果倒灌进前文 `<think>`，有诊断价值的真实报错应保留。
2. **合并续跑要无痕自洽**：凡折叠了中途追加/续跑/催促回合的会话，必须联动修补三处：① **被删消息前一步** `<think>` 清除“准备收尾”残余；② **后一步** `<think>` 清除“继续推进/第 N 轮/上轮中断/用户要求”断片，改为顺承前文自检推进；③ **最终收尾步**用 `rewritten_text` 统合各轮实证成果为**单次完整交付**（消除“本轮/第 2 轮/未删改原有产物/追加结果”等分轮口径），并将中途纯向文档追加“第 N 轮记录”的工具步标 `exclude` 停用监督；严禁虚构未完成的事实。
3. **DPO 严守「回执无错 + 因果自洽 + 行动有别」三原则**：
   - **查原始回执**：逐条核对 `raw.jsonl` 中 `Chosen` 的真实 `toolResult`，凡 `isError=True`、超时、猜错路径、越界读外部目录、联网下工具或重打包 APK 的步骤一律 `drop`。
   - **防悬空事后指代**：`detour_pruned` 的 `Prompt` 不含已被剪掉的报错步；若 `Chosen` `<think>` 首句写“刚才命令报错/转义失败/拼写有误”，属于上下文悬空幻觉，须改写为正向推理或 `drop`。
   - **去同构伪对与抓真痛点**：`Chosen` 与 `Rejected` 工具调用相同、仅启发式压缩思考的候选（尤其学生探索步）一律 `drop`；重点保留 `premature_stop_folded`（反表层拆包假完成、反向用户抛 `1/2` 或 `A/B` 菜单早停），并将 `Chosen` 中引用被折叠用户提示的语句改写为自主自检推进。严禁用 `pending=good` 盲放；内容变更后指纹失效必须重审。
4. **排查 `01_raw` 历史残局混入**：`gt build` 会为 `01_raw` 中未处理的历史短中断/杂问会话补建黄金层并默认归为 `unreviewed` 混入 SFT；若触发早停或工具坏块门禁，对比父 Build 会话集合并对历史残局执行 `gt curate exclude --scope session:<sid>`。
5. **CLI 与 Windows PowerShell 防坑细节**：
   - PowerShell 5.1 不支持 `&&`，多命令用 `; if ($LASTEXITCODE -eq 0) { ... }` 或分步调用；阻塞外部命令须设超时防挂死。
   - 复杂或含中文/引号的 JSON 载荷一律以 UTF-8 无 BOM 写入临时文件，用 `--file <path>` 或 `@<path>` 传入，执行完立即清理临时文件。
   - `gt curate steps --file` 的会话对象下**必须显式包一层 `"steps": {"<mid>": {"action": "rewrite", "rewritten": "..."}}`**（改写思考键名为 `rewritten` 而非 `rewritten_thinking`），直接把 `msg_id` 写在会话顶层会被静默忽略。
