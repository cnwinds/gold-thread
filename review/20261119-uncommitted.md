# 代码审查报告

## 基本信息

**审查范围**: `当前工作区未提交的变更（ADR-0009 虚拟数据集、CAS 快照池、演进 DAG、极简三栏控制台与交付门禁加固）`

---

## 审查结论

**审查结果**: `通过（审查识别的 9 项 P1/P2 问题及文档契约偏差均已当场修复并通过 246 项自动化回归测试）`

**综合评分**: `9.6/10`

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | `9.6/10` | 打包暂存受控发布、逐行监督目标门禁、DPO 真实候选校验、重打包幂等与 DAG 无环性、Build 冻结配置回放及逻辑窗口去重全部闭环验证 |
| 安全性 | `9.6/10` | CAS 严格校验 64 位十六进制 SHA256，防止目录穿越；`inspect` 域保持纯只读无写盘副作用；前端动态事件参数统一经 `jsArg` 转义 |
| 可维护性 | `9.6/10` | 控制台左侧导航收敛为「数据集图谱 / 账本与任务 / 配置」三栏，清除约 2,000 行废弃旧分页死代码，功能内聚于图谱与抽屉 |
| 性能 | `9.4/10` | 虚拟数据集概览默认轻量读取清单，按需触发全量门禁体检；CAS 按内容哈希跨版本去重存储 |
| 代码风格 | `9.6/10` | 原子文件替换（`replace_atomic` / `write_json_atomic`）、类型注解、锁保护与仓库既有约定一致 |

**审查意见**: 本次重构完成了以交付包 `package_hash` 为唯一虚拟数据集条目、按 SHA256 去重的 CAS 快照池、横向演进 DAG 关系图与三栏极简控制台架构升级。审查阶段对交付门禁顺序、空目标放行、DPO 审核契约、重打旧包成环、Web 构建路由、步级开关会话绑定、冻结配置回放、加权副本去重及模板 `.pyc` 过滤共 9 项问题完成了逐一修复与独立正负例测试覆盖，达到提交推送标准。

---

## 问题汇总（均已在本次提交前修复闭环）

### 严重 / P1 问题（已修复）

1. **不合格交付包越过门禁发布与复制**（`goldthread/build_tools.py:245`, `server/app.py:398`）
   - **问题**：原 `pack_delivery` 先直接写入正式 `04_release/delivery/` 并执行 `--copy-to`，随后才跑 `gate_check`；Web 端打包作业未经过同一套硬门禁。
   - **修复**：CLI 与 Web 统一调用暂存发布流——先在 `.staging-*` 生成 ZIP 并执行 `gate_check`，仅当 `passed=True` 时原子发布至 `delivery/` 及 `copy_to`；失败立即销毁暂存区并记 `gate_failed` 账本，绝不污染正式目录或覆盖已有合格包。
2. **缺少末尾 assistant 监督目标或缺失项目归组的样本放行**（`goldthread/eval_tools.py:595, 720`, `goldthread/build_tools.py:50`）
   - **问题**：仅含 `human` 提问无 `assistant` 回复的行被跳过未计错；缺失 `project_group` 映射为 `unknown` 后未计入 `non_canonical_projects`。
   - **修复**：逐行强制要求有效末尾 `assistant/gpt` 监督目标（否则计入 `missing_assistant_targets` 触发门禁失败），并将缺失项目的 `unknown` 严格计入 `non_canonical_projects`。
3. **DPO 未审绕过不可见及未知 `sample_id` 静默计数**（`goldthread/curate_tools.py:555`, `goldthread/build_tools.py:155`）
   - **问题**：`curate dpo` 未核对候选池存在性，臆造 ID 仍报 `updated=1`；关闭隐式 judge 时未审项走 `bypassed` 不体现在 `pending_evaluations` 中。
   - **修复**：`curate_dpo` 校验真实候选 ID 集合并拒绝未知 ID；`build --dry-run` 显式输出 `dpo_audit`（区分 `explicit_verdicts` / `cached_reviews` / `bypassed_unreviewed` / `pending_evaluations`），并新增 `--require-reviewed-dpo` 严格门禁选项。
4. **重打旧包刷新时间戳导致 DAG 成环与根节点丢失**（`pipeline/package.py:330`, `goldthread/lineage_graph.py:380`）
   - **问题**：重打已存在的同哈希包重写了 `generated_at`；图谱未区分显式 `parent_pack: null` 根与旧包无字段，按时间反推形成 `A↔B` 环路。
   - **修复**：同哈希已发布包幂等保留原 `generated_at`、`parent_pack` 与文件字节；图谱识别 `has_explicit_parent` 根并增加通用环路检测与断环保护，前端支持多根森林完整渲染。
5. **Web 草稿“1. 构建 Build”请求未注册路径**（`web/index.html:2470`）
   - **问题**：前端调用了不存在的 `POST /api/dataset/draft/build`（404）。
   - **修复**：对接真实 `POST /api/run`（`{"stage": "stage3", "force": false}`），并通过 `TestClient` 端到端测试验证。

### 中等 / P2 问题（已修复）

6. **虚拟数据集抽屉步级排除开关丢失或串用 `session_id`**（`web/index.html:3372`）：按钮新增显式 `data-gold-sid` 绑定，`goldSetStepGate` 显式接收目标 `sid` 并经 Node VM 覆盖首次打开与跨会话切换测试。
7. **历史包对话回放受当前配置影响且修改配置不触发 Dirty**（`goldthread/lineage_graph.py:218, 1085`）：历史包回放优先加载对应 Build 冻结的 `build_config.json`；Clean/Dirty 判定纳入确定性合成配置摘要（`synth_config_changed`）。
8. **黄金列表将 `#w2` 权重展开副本计入唯一窗口数**（`goldthread/lineage_graph.py:850`）：统一按 `origin_sample_id` / 剥离 `#w` 后缀去重统计逻辑窗口数 `sft_windows`，与展开行数 `sft_lf_rows` 明确区分。
9. **打包模板收集器混入 `__pycache__/*.pyc` 字节码缓存**（`pipeline/package.py:78`）：`_template_static_files()` 过滤 `__pycache__`、隐藏目录及 `.pyc/.pyo/.pyd` 后缀，保证跨环境 `package_hash` 一致。
10. **文档与 `SKILL.md` 契约校准**：`inspect graph` 改为 `readonly=True` 零写盘副作用；按 `full / backfilled / none` 如实分级说明 CAS 快照精度；校准 `SKILL.md` 真实 JSON 路径、DPO ID 获取方式与退出码 `2` 异常处理规程。

---

## 验证记录

- `python -m py_compile` 全涉及模块：**通过**
- `web/index.html` 提取 `<script>` 执行 `node --check`：**通过**
- `python -m pytest tests/ -q`：**246 passed**
