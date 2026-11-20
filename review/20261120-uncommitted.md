# 代码审查报告（合并续跑接缝统合、控制台停用徽章/DPO 查看器与 893fd9f2 语料迭代）

## 基本信息

**基准提交**: `7685dbe` (`feat(console): show dataset branches and causal bridges on a shared timeline`)
**审查范围**:
1. 多轮合并续跑会话无工具收尾步 `rewritten_text` 确定性改写、`premature_stop_folded` 顺延有效 `chosen` 修复与 `^goldthread` 学生模型默认负例归类（`pipeline/stage2_gold.py`, `goldthread/curate_tools.py`, `config/pipeline.yaml`, `pipeline/common/config.py`）；
2. 虚拟数据集工作区实时停用徽章 `workspace_usage` 与会话抽屉只读单对翻页「DPO 偏好对」查看器（`goldthread/lineage_graph.py`, `server/app.py`, `web/index.html`）；
3. `skills/goldthread/SKILL.md` 五节结构精炼重组与 `893fd9f2` 训练集统计索引刷新（`data/03_train/{index,stats}.json`）；
4. 配套自动化回归测试（`tests/test_curate_text_rewrite.py`, `tests/test_gold_usage_badges.py`, `tests/test_session_dpo_view_{api,ui}.py`, `tests/test_virtual_datasets.py`, `tests/test_config_http_contract.py`）。

---

## 审查结论

**审查结果**: `通过（审查识别的 2 项跨回合/三态血缘边界隐患已当场修复闭环，全量 571 项自动化测试、py_compile 与 node --check 全部通过）`

**综合评分**: `9.8/10`

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | `9.8/10` | `rewritten_text` 严格限定无工具收尾步并保留 `raw_text`；`premature_stop_folded` 跳过首步报错且不跨未折叠用户回合串台；历史包冻结计数与 `workspace_usage` 实时草稿解耦 |
| 安全性 | `9.9/10` | `_text_rewrite_ok` 阻断 `<tool_call>` / `<think>` 伪标签注入与跨语种误翻；前端新增 DPO 渲染器与停用徽章全程经 `esc()` / `jsArg()` 转义，无 XSS 或路径穿越风险 |
| 可维护性 | `9.7/10` | `_prose_only` 剥离代码块避免英文路径密集的中文技术报告被误判；`SKILL.md` 按固定五节收敛并固化三级 `--help` 动态自省契约 |
| 性能与并发 | `9.7/10` | `/api/datasets/{id}/train` 按 `kind` 单路读取并做 `session_id` 精确匹配；前端 `DPO_VIEW_REQ_SEQ` 防止抽屉异步快速切换造成旧响应覆盖 |
| 测试覆盖 | `9.9/10` | 新增 4 个专项测试模块并扩展 2 个既有测试套件，总测试数由 518 增至 571 项，覆盖后端管线、CAS 血缘、HTTP API 与前端脚本契约 |

---

## 审查发现并当场闭环的边界加固

1. **防 `pending_popped_await` 跨未折叠用户回合串台（`pipeline/stage2_gold.py`）**：
   - **发现问题**：`purify` 与 `apply_step_decisions` 引入 `pending_popped_await` 以在折叠早停步后首步报错被剪除时顺延绑定首个无错工具步；但若被折叠回合后的唯一助手步因报错被剪除、紧接着遇到未折叠的新 `user` 消息单元时，`pending_popped_await` 未清空，存在跨入新用户回合错误配对的极端边界风险。
   - **闭环修复**：在 `purify` 与 `apply_step_decisions` 的 `if cur["type"] != "assistant_turn":` 分支均补入 `pending_popped_await = None`，并在 `tests/test_curate_text_rewrite.py::test_premature_stop_folded_clears_across_unfolded_user_boundary` 增加回归断言。

2. **三态血缘对「仅改写 `rewritten_text`、未改 `thinking`」收尾步的状态与 Diff 对齐（`goldthread/lineage_graph.py`, `server/app.py`, `web/index.html`）**：
   - **发现问题**：三态血缘原先仅依据 `raw_thinking != gold_thinking` 计算 `changed`，且 `server/app.py` 的 `flags` 列表未纳入 `"text_rewritten"`，前端展开收尾步时也仅渲染最终 `text` 而无 `raw_text` vs `text` 对照。若某收尾步仅统合改写了对外正文而未改动思考（且回放无 `sd_step` 快照），状态会退化为 `kept` 且无法直观查看正文去轮次化前后的差异。
   - **闭环修复**：在 `goldthread/lineage_graph.py` 与 `server/app.py` 同步输出 `raw_text`、`text_changed` 并将 `"text_rewritten"` 纳入 `state = "rewritten"` 判据（保持 `changed` 专表思考变更，避免无思考差异时渲染冗余双栏思考框）；前端 `web/index.html` 在 `st.text_changed` 为真时自动展示「原始回复 (Raw) vs 黄金统合回复 (Gold)」Diff。新增 `test_lineage_marks_text_only_rewrite_as_rewritten_with_raw_text` 覆盖模块与 HTTP 路由双链路。

---

## 各模块重点核查记录

### 1. 黄金收尾正文改写与早停折叠配对（`pipeline/stage2_gold.py`, `goldthread/curate_tools.py`）
- `curate_steps` 支持 `action="rewrite"` 传入 `rewritten_text`（可与 `rewritten` 思考改写独立或同时使用），当场校验目标步必须位于无 `tool_calls` 的最终/收尾步，并通过 `_text_rewrite_ok` 校验长度、标签安全与语种一致性。
- `stage2_gold` 在 `purify`、`apply_step_decisions`、`purify_by_judge` 全链路保留 `raw_text` 并打标 `text_rewritten`，下游 `stage3_synth` 直接消费统合后的 `text` 生成无轮次割裂感的单次完整交付 SFT 样本。

### 2. 学生模型标签规则与生产配置契约（`config/pipeline.yaml`, `pipeline/common/config.py`）
- `labeling.negative_model_patterns` 新增 `"^goldthread"`，确保采集学生自测会话（如 `goldthread-27b`）时自动归入 `negative` 仅作 DPO 负例源，默认不混入正向黄金候选；`PRODUCTION_CONFIG_SHA256` 同步更新并锁定契约。

### 3. 控制台实时停用徽章与会话级 DPO 偏好对查看器（`goldthread/lineage_graph.py`, `server/app.py`, `web/index.html`）
- **实时停用徽章**：`get_virtual_dataset_sessions` 与 `get_virtual_dataset_session_lineage` 计算 `workspace_usage`（区分项目级 `project_sft`、会话级 `session_sft` 与全排除 `exclude_session`），冻结包列表顶部计数严格保持发布时快照不变，仅在行内提示当前工作区停用状态；被上层规则停用时隐藏无效的会话级 SFT 开关。
- **DPO 单对查看器**：`/api/datasets/{dataset_id}/train` 支持 `session_id` 精确匹配（兼容 `metadata.session_id` 与顶层回退），严禁前缀误匹配；前端按需懒加载、单对翻页展示完整未截断的上下文与 `Chosen` / `Rejected` 对照。

### 4. Agent 技能规范收敛（`skills/goldthread/SKILL.md`）
- 全文精炼为 73 行，严格按「目标 → 工作场景流程 → 关键步骤 → 必备常识 → 经验」五节组织，明确三级 `--help` 动态探查机制、合并续跑无痕自洽要求、`session_sft:<sid>` 正负隔离方法与“数据包已验收，学生效果未验证”诚实交付边界。

---

## 提交前验证记录

- `python -m py_compile` 全涉及与新增 Python 模块：**通过**
- `web/index.html` 提取 `<script>` 执行 `node --check`：**通过**
- `python -m pytest tests/ -q`：**571 passed**
