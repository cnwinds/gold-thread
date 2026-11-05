# 代码审查报告

## 基本信息

**审查范围**: 工作区未提交变更（Git: 已暂存 + 未暂存）

---

## 审查结论

**审查结果**: 有条件通过

**综合评分**: 8.4/10

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | 8/10 | 会话删除、来源中立门控、步骤日志与 UI 重构有配套测试；抽屉加载失败已补错误提示 |
| 安全性 | 9/10 | 删除路径有 sid 校验与目录越界检查，冻结 delivery/releases 不删 |
| 可维护性 | 8/10 | `steps.begin/finish` 成为可见日志唯一入口，任务元数据集中在 task_registry |
| 性能 | 8/10 | 日志与进度分离；删除走事务暂存，规模随会话文件量线性 |
| 代码风格 | 8/10 | 与现有 pipeline/server/web 风格一致 |

**审查意见**: 本次变更可一并提交。核心行为（来源中立 SFT、会话删除、步骤开始/结束日志、控制台去标签化与黄金抽屉多视图）方向正确，测试全绿。提交前已修抽屉 404 卡在「加载中」的问题。数据产物与本地训练脚本不纳入本次提交。

---

## 问题汇总

### 严重问题 (必须修复)

无。

### 中等问题 (建议修复)

```
UX / 可观测性
任务持久化 tasks.json 原先不存 logs；完成后打开历史任务会「暂无日志」。
本次未改持久化模型，仅保证 live 任务日志完整。若产品要求历史可回看日志，需后续把 logs 落盘或裁剪落盘。
[pipeline/task_registry.py / server/app.py]
```

```
可维护性
stage3 在会话循环前同时 begin s3.gate / s3.windows / s3.traj，日志会出现三条「开始」挤在一起。
功能正确，但可读性一般；后续可改为串行语义或合并为一条总步骤。
[pipeline/stage3_synth.py]
```

### 轻微问题 (可选修复)

```
测试隔离
此前 collect 单测未 mock sync_job，污染了真实 data/00_state/tasks.json。
已 mock；本地脏任务已清理。建议 CI 使用独立 data_dir。
[tests/test_collect_api.py]
```

### 问题统计

| 问题类型 | 数量 |
|----------|------|
| 接口变更未同步 | 0 |
| 未解决问题 | 0 |
| 资源泄露 | 0 |
| 空指针风险 | 0 |
| 死循环风险 | 0 |
| 性能问题 | 0 |
| 非最优解 | 1 |
| 安全漏洞 | 0 |
| 代码风格不一致 | 0 |
| UX/可观测性遗留 | 1 |
| 测试隔离遗留说明 | 1 |
| **总计** | **3** |

---

# 详细分析

## 1. 变更信息

### 主要改动面

1. **来源中立**：去掉 `sft_labels` 门控；label 仅作血缘；CONTEXT / config / stage3 行为对齐。
2. **会话删除**：新增 `pipeline/session_delete.py`、`DELETE /api/session/{sid}`、任务 kind=`delete`、UI 确认删除。
3. **步骤日志**：`pipeline/steps.py` 增加 `begin`/`finish`（工作说明 + 耗时），stage1/2/3 接线。
4. **控制台**：去数据集角色编辑；黄金抽屉支持黄金/SFT/DPO 浏览；任务行与 runs 职责收敛；抽屉加载失败显式报错。
5. **测试**：`test_session_delete` / `test_source_neutral` / `test_steps_logging` / `test_console_ui` 等；`test_collect_api` 隔离 sync_job。

### 验证

- `python -m pytest tests/ -q` → 通过
- `node --check` 对 `web/index.html` 内联脚本 → 通过

## 2. 调用链（摘要）

- UI 删除 → `DELETE /api/session/{sid}` → `_start("delete")` → `session_delete.delete_session`
- 管线步骤 → `stepcat.begin/finish` → `Job.log`（可见）+ `@@STEP`（progress）
- Stage3 门控 → 不再读 `sft_labels`；质量/复核决定是否产 SFT

## 3. 接口变更

| 接口 | 变更 | 兼容性 |
|------|------|--------|
| `DELETE /api/session/{sid}` | 新增异步删除任务 | 新增，向前兼容 |
| 配置 `synth.sft_labels` | 废弃不用 | 旧配置忽略，不报错 |
| 控制台 relabel UI | 移除 | 行为变更，已有 CONTEXT 说明 |

## 4. 安全

- sid 正则校验；文件必须落在 data 根内；`releases`/`delivery` 冻结目录不改。
- 删除与写接口互斥（`_assert_no_delete_running` / job clash）。

## 5. 提交建议纳入 / 排除

**纳入**：源码、配置、CONTEXT、相关测试、本审查报告。

**排除**：`data/03_train/**` 训练产物与本地脚本、`.scratch/`、`tools/_*` 一次性补丁、其它无关 review 草稿。
