# ADR-0002：采集与分析的显式动作

- **状态**：已接受（UI/API 部分落地）
- **日期**：2026-11-05

## 背景

用户期望：**人工从源同步/勾选会话 → 再按会话分析或重新分析**；批量操作不应隐含「全库 stage2」。  
当前实现：`POST /api/collect` job 在 `ingest_selected` 后 **立即** `stage2_gold.run(..., only=sids)`；单会话 `POST /api/raw/session/{sid}/analyze` 已存在。

## 决策

1. **两个用户可见动作**（目标 UX）：
   - **仅采集**：源 → `01_raw`（+ stage1 回合流），**不**跑 stage2。
   - **采集并分析**：采集后对所选 sid 跑 stage2（与现 collect job 相同）。

2. **已入库列表**（原始库）：只保留 **分析 / 重新分析**（单 sid），与「是否已采用」列一致；不在此页重复「分段筛选已采用」。

3. **术语**：界面文案 **采集** = ingest；**分析** = stage2 提纯（有用步 → 黄金，弯路 → 该 sid 的 DPO）。避免用「同步」指 stage2。

## 后果

- 实现：`CollectBody` 增加 `analyze: bool`（默认 `true` 保兼容，或默认 `false` 若产品优先「先采后析」— **推荐默认 false**，批量按钮两个）。
- 前端：`pickGo` 拆为「采集所选」「采集并分析所选」或主/次按钮。
- `CONTEXT.md` 与 `PAGES.raw` 文案与 ADR 对齐。

## 当前代码事实（2026-11-05）

- **`POST /api/collect`** 接受 `analyze`（默认 **false**）：false 仅 ingest；true 再跑 stage2（`only=sids`）。
- 前端：**「采集所选」**（primary）与 **「采集并分析」** 两个按钮。
