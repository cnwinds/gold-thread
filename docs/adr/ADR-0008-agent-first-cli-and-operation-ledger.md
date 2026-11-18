# ADR-0008：Agent-First 确定性工具集（`gt`）与全链路操作账本架构

- **状态**：已接受（已落地）
- **日期**：2026-11-18

## 背景

在早期版本中，GoldThread 以「人工点击 Web 工作台按钮 + 管线内部隐式调用小模型（`turns.use_model` / `judge.enabled` / `review.py`）」为主流程。在 `d9dbbbd1 → 308e6497` 的真实学生模型评估与训练集调优迭代中暴露出三个核心痛点：

1. **跨语料对账与精准修补依赖临时脚本**：主模型 Agent 在执行外部教师库扫描、学生实测日志归因、残缺桩清理、定向步级思维链修补与包间指标对比时，缺乏标准化 CLI 原语，不得不编写一次性脚本。
2. **终端操作与 Web 控制台割裂**：命令行执行的语料增删与策展修改不会在 Web 界面留痕，人类审阅者难以追溯「谁在什么时间、出于什么原因改动了哪些会话并产出了哪个训练包」。
3. **管线暗箱 LLM 调用破坏确定性与速度**：构建链路夹带隐式远程小模型请求，既拖慢重建速度，又可能因网络波动或旧 `review_rounds_stale` 状态造成静默拦截。

## 决策

1. **确立三层解耦的 Agent-First 架构**：
   - **模型大脑层**：主模型 Agent（配合仓库内置 `skills/goldthread/SKILL.md`）负责阅读分级摘要、诊断实测根因并产出结构化 JSON 决策；
   - **确定性工具原语层（`goldthread/` 包 + `gt.py` / `python -m goldthread`）**：固化 `inspect`、`eval`、`curate`、`build`、`gate`、`pack`、`report`、`ledger` 及显式可选的 `assist` 子命令，全部支持 `--json` 高密度输出，零守护进程依赖；
   - **数据血缘与留痕看板层**：`data/` 四层存储 + 操作账本 + Web 可观测审阅中心。
2. **主构建管线默认 100% 纯确定性运行**：
   - `turns.use_model` 与 `judge.enabled` 默认关闭，`gt curate` / `gt build` / `gt pack` 在执行期强制禁用隐式外部 LLM 请求；
   - Stage 2 优先读取主模型通过 `gt curate steps` 落盘的 `00_state/step_decisions.json`（及兼容保留的历史 `annotations`），重算时保护已有 `model_report` 不丢失；
   - 原小模型初筛能力（`turnclassify.py`、`judge.py`）收拢至显式可选的 `gt assist turns|judge|dpo` 子命令；移除已废弃的 `pipeline/review.py` 及 6 个旧人工审核 HTTP 路由。
3. **内置硬性质量门禁（`gt gate check`）**：
   - 固化 8 项硬断言：SFT Train/Eval 非空、中途无工具假收尾 `premature_stop_targets == 0`、Prompt 采集外壳标签泄漏 `== 0`、工具契约 100% 合法、目标步 `<think>` 完整且历史思考形态一致、项目组 100% 规范 `game::*`、DPO 对有效、`length_gate` 软压缩生效。
4. **统一操作审计账本（`00_state/ledger.jsonl`）与评估报告归档（`04_release/reports/`）**：
   - CLI 与 Web 应急开关（样本/会话/步级排除）共用底层原语，每次写操作自动追加不可篡改的账本事件（含 `op_id`、`timestamp`、`actor`、`tool`、`reason`、`affected`、`before_summary`、`after_summary`）；
   - `gt report publish` 将历次 Markdown 评估报告与包间 `diff_summary` 原子归档至 `04_release/reports/`。
5. **Web 控制台收敛为留痕与血缘可观测中心**：
   - 首页升级为「演进评估报告流 + 最近操作账本 + 最新 Build/交付包快照 + 三阶段精炼漏斗」；
   - 黄金抽屉新增一站式「三态血缘 Diff」（回合折叠清洗对比、步骤 `kept/rewritten/pruned/folded/excluded` 过滤与原始/黄金思考对照、会话级账本记录）；
   - 交付页内建「硬性门禁体检」与「包间指标 Diff 对比」；任务页融合「操作审计账本」全文检索与反向追踪。

## 后果

- 任何支持 Shell 的模型 Agent 均可通过 `python gt.py ... --json` 零配置完成「学生实测评估 → 语料策展修补 → 纯确定性构建与硬门禁验收 → 打包交付 → 发布报告」完整闭环，无需再编写临时补丁脚本。
- 现有 `data/` 四层目录、33 条入库会话及 `goldthread-train-308e6497.zip` 100% 无缝兼容。
- 终端 CLI 操作与 Web 应急调整统一汇入同一份 `ledger.jsonl`，人类可在浏览器一站式审阅演进脉络、包间指标变化与单步三态血缘。
