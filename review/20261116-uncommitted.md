# 代码审查报告

## 基本信息

**审查范围**: 当前工作区未提交的变更（Git: 已暂存 + 未暂存）

---

## 审查结论

**审查结果**: 有条件通过

**综合评分**: 8.2/10

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | 8/10 | DPO `tools` 漏导已从 Stage2/3 + 规整/打包多层补齐；任务排队与评估前置逻辑闭环 |
| 安全性 | 9/10 | 未见密钥硬编码；删除任务与写接口互斥合理 |
| 可维护性 | 8/10 | 模板同步与测试覆盖到位；`web/index.html` 体量大但仍集中 |
| 性能 | 8/10 | 原子写重试/最长优先探测合理；`shared_sft_tools` 会扫一遍 SFT，可接受 |
| 代码风格 | 8/10 | 与现有 pipeline/server 风格一致 |

**审查意见**: 核心缺陷（DPO 缺 `tools`）与模板探测/卸载链、任务排队可一并合入推送。提交时排除 `.scratch/`、临时脚本与本地训练产物；`ensure_tools_column` 对字面量 `"[]"` 仍视为「已有值」属边角，当前真实故障路径已覆盖。

---

## 问题汇总

### 严重问题 (必须修复)

无。

### 中等问题 (建议修复)

```
空 tools 哨兵未回填
ensure_tools_column 只把 None/"" 当缺失；若行内已有 "[]" 或 []，规整层不会换成 DEFAULT_TOOLS。
当前线上问题是字段缺失（已修）；模板侧 fill_missing_tools_from_sft 可兜底打包副本。
[pipeline/common/format.py: ensure_tools_column]
```

```
replace_atomic 回退路径非原子
占用重试失败后 unlink + rename，窗口期内读者可能短暂看不到文件。
注释已说明权衡，Windows 下可接受；无需阻塞合入。
[pipeline/common/store.py: replace_atomic]
```

### 轻微问题 (可选修复)

```
lf_normalize 函数内延迟 import
normalize_sharegpt_row 内 import ensure_tools_column，与文件顶部导入风格略不一致。
[pipeline/lf_normalize.py]
```

```
工作区噪音文件勿入库
.scratch/、mj*_tmp.txt、data/03_train 下训练烟雾脚本/candidates/exports、tools/_* 一次性补丁脚本不应进入本提交。
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
| 非最优解 | 2 |
| 安全漏洞 | 0 |
| 代码风格不一致 | 1 |
| **总计** | **3** |

---

# 详细分析

## 1. 变更摘要

### 修改文件列表（拟合入）
- pipeline：`format`/`store`/`stage2`/`stage3`/`lf_normalize`/`package`/`collect`/`judge`/`dataset_draft`/`task_registry`/`length_stats`/`train_template/*`
- server/`app.py`：任务排队、Stage3 先评估再构建、长度统计与交付包
- web/`index.html`：黄金页/任务中心/长度图等 UX
- tests：配套用例（含 `test_dpo_tools_column`、`test_job_queue`、`test_train_template_sync` 等）
- docs / CONTEXT / README / ADR

### 变更统计（代码相关，不含 data 噪音）
- 约 +3385 / -696（pipeline+server+web+tests+docs）

---

## 2. 代码变更分析

### 新增/关键函数
- `default_tools_json` / `ensure_tools_column` — 统一 DPO/SFT tools 列
- `replace_atomic` — Windows 占用下原子替换重试
- `_pump_queue` / `_active_blocker` — 作业排队与冲突检测
- `_evaluate_dpo_or_raise` — 构建前强制完成 DPO 评估
- 模板：`shared_sft_tools`、探测选档、`fill_tools`、`dpo_single_offload_*`

### 接口变更
- 任务列表 API 增加 `queued` 计数与 `queued` 状态（向前兼容）
- Stage3 作业语义变为「评估并合成」；失败时 pending/failed 可见

---

## 3. 调用链分析

```
_dpo_from_pruned / _negative_dpo / _review_rewrite_dpo
├── 写出 tools ← default_tools_json / _tools_json
├── synthesize_dpo_candidates → ensure_tools_column（读旧 gold）
└── stage3 run → lf_normalize.normalize_sharegpt_row → ensure_tools_column
    └── package.prepare_datasets → 交付 zip dataset_info 声明 tools

_start(stage)
├── 冲突 → status=queued
└── 无冲突 → running + thread
    └── _run_job finally → _pump_queue → 启动下一批不冲突作业
```

---

## 4. 接口变更评估

**检测结果**: 是（任务状态机扩展；行为兼容）
- 旧客户端忽略 `queued` 字段仍可用
- Stage3 失败条件更严（评估未完成即失败）——符合产品意图

---

## 5. 问题解决验证（对照会话目标）

| 目标 | 状态 |
|------|------|
| DPO 导出漏 `tools` | 已解决（源写出 + 规整/合成回填 + 模板过滤副本补齐） |
| 模板 20261116 集成 | 已解决（不含 data/） |
| 任务排队 / Stage3 先评估再构建 | 已解决，有测试 |

---

## 6–8. 潜在问题 / 安全 / 风格

- 资源泄露 / SQL / XSS / 硬编码密钥：未发现
- `_pump_queue` 在锁内改 status、锁外起线程，模式正确
- `shared_sft_tools` 全表扫描 SFT：仅 DPO 准备数据时一次，可接受

---

## 9. 主要优点

1. DPO `tools` 从产生到打包多层防御，并与 SFT 同契约
2. 训练模板探测 + 单卡 offload 链与真实 OOM 经验对齐
3. 任务排队与 DPO 评估门禁减少控制台误操作
4. 测试覆盖核心回归点（203 passed）

---

## 附录

| 配置项 | 值 |
|--------|-----|
| 调用链深度 | 2 |
| 跳过风格检查 | false |
| 跳过安全检测 | false |

**报告生成时间**: 2026-11-16  
**审查工具版本**: Code Review Skill v1.3
