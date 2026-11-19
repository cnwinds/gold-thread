# ADR-0009：虚拟数据集条目、CAS 去重快照与演进 DAG 关系图（Variant A）

- **状态**：已接受（已落地）
- **日期**：2026-11-19

## 背景

完成 ADR-0008 后，训练集主要由主模型 Agent 通过 `gt` CLI 自动闭环产出，人类审阅重点转向「数据集包之间的版本因果演进」与「单包内部四层数据回溯」。原有按管线加工车间（Stage 1/2/3）平铺并同时暴露 `不可变 Build` 与 `交付包` 双卡片的信息架构存在三项痛点：

1. **一级实体割裂**：用户需要同时理解 Build ID 与交付包 `package_hash`，心智冗余。
2. **工作空间常态重复**：当 Agent 完成打包后，当前工作区与最新正式包完全一致，常驻展示独立「工作空间」只产生重复信息；但若彻底移除工作空间，又无法检视未打包的演进草稿。
3. **历史包回放受可变工作区影响**：`01_raw` 与 `02_gold` 在工作区共享存储，若后续版本改写或删除了某条会话，点开旧包无法精确还原当时的黄金步与原始回合态。

## 决策

1. **以带 `package_hash` 的交付包作为唯一一级「虚拟数据集条目」**：
   - `Build` 收敛为包内部溯源属性；点击任意数据集条目，在同一上下文内切换查看 **1. 概览与门禁体检、2. 训练数据（冻结切片）、3. 黄金数据（CAS 快照）、4. 原始数据（归属范围）** 四层视图。
2. **工作空间（Workspace）采用 Git Working Tree 式按需动态显隐**：
   - 比对当前工作区实质性数据输入（`sources`、非空 `stage3_inputs`、`decisions`、`scope_decisions`、`split_plan` 及 `LATEST_BUILD`）与最新正式包绑定 Build 是否一致；
   - **Clean 态**自动隐藏草稿节点，在最新正式包标注 `HEAD · 与工作区同步`；**Dirty 态**（或 0 包冷启动）在图谱末端自动浮现虚线 `Workspace` 演进草稿节点。
3. **引入 CAS 内容寻址去重快照池（`04_release/cas/<sha[:2]>/<sha>`）**：
   - `dataset_build.create_build()` 自动将入选会话的 `raw_file`、`meta_file`、`gold_file`、`annotation_file`、`reject_file`、`dpo_file`、`diag_file` 按 SHA256 去重归档，写入 `BUILD.json` 的 `cas_manifest`；
   - 对既有历史 Build 通过 `cas_backfill.json` 侧车文件无损回填（不篡改历史 `BUILD.json`）；
   - 历史包查询黄金/原始列表与单会话三态血缘（`/api/datasets/{id}/sessions/{sid}`）严格走 CAS 只读还原，并标注 `exact` / `workspace_modified` / `workspace_deleted`。
4. **首页升级为横向演进关系图（Variant A · Lineage DAG）**：
   - 打包 `MANIFEST.json` 记录 `parent_pack`（默认继承打包前最新包，支持 `gt pack --parent` 显式指定），结合评估报告 `before_pack -> after_pack` 构建 DAG；
   - 演进连线自动挂载关联评估报告与基于 `ledger.jsonl` 追加日志物理序号切分的区间操作账本。

## 后果

- 首页直观呈现「包 A ──(评估报告 + AI 策展账本)──► 包 B ──(Dirty 时虚线)──► Workspace」完整演进谱系。
- 携带 `cas_manifest` 的新建 Build（`fidelity=full`）在工作区后续改写、删除会话或调整配置后，仍能使用冻结配置与 CAS 对象 100% 精确回放当时的训练切片、黄金步与原始对话血缘；历史遗留 Build 按已保存哈希尽力校验回填（`fidelity=backfilled`），无关联 Build 的早期独立包诚实标注仅训练切片冻结（`fidelity=none`）。
