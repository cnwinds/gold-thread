# ADR-0001：Stage 3 门控在无人工复核 UI 下的有效结论

- **状态**：已接受（部分落地，代码待对齐）
- **日期**：2026-11-05

## 背景

产品已移除独立「审核」页与会话级通过/驳回/定稿 UI，流程改为 **全模型化 + 按会话分析 + 血缘浏览**（见 `CONTEXT.md` §5.3、§6）。  
Stage 3 仍通过 `pipeline/curation.py` 的 `resolve()` 计算 `effective_state`，并据此门控是否产出 SFT（`IN_SFT`）。

## 决策

1. **新流程的权威门控**（无 UI 时代）：
   - **会话级**：以 Stage 2 **`model_report`**（`trainable` / `discard` / `needs_wholesale_rewrite`）及 `judge` 置信度为准，映射为 `auto_gold` / `auto_reject`。
   - **步级**：以黄金数据中的 **排除 / 手改**（`human_review.items` 的 `step` / `edited`）及 judge 的 `excluded` 为准；**不**再依赖新的会话级 `decision: approved|rejected`。

2. **`needs_human`**：`model_report` 或评审轮次里的该 verdict **不再扣住队列**；`resolve()` 固定映射为 **`auto_gold`**（报告仍可在控制台阅读）。已删除配置项 `review.hold_needs_review`。

3. **历史数据**：已写入且 **未 stale** 的 `human_review.decision === "rejected"` **继续阻断** SFT，避免 silently 把旧驳回会话灌进新训练集。若要重训，须重分析或清标注（显式操作，非默认）。

4. **`unreviewed`**：在 `model_report` 尚未产出时，允许按 **positive 标签 + 步级剔除** 走默认入库（与当前 `IN_SFT` 一致）。

## 后果

- 已删除 `hold_needs_review`；`held` 仅作历史 `effective_state` 文案兼容，新 resolve 不再产出。
- `curation.py` 模块 docstring 仍写「人工复核层」— 语义改为 **「有效结论层（模型 + 步级编辑）」**，API 可保留供脚本/旧标注。
- README 流程图「人工复核页」应删或标注 legacy。

## 未决（实现时确认）

- 黄金页 **步级 exclude** 是否覆盖全部「用户手工去掉不想要的内容」场景；训练页剔除与 `excluded` 账本是否需统一。
