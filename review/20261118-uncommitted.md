# 代码审查报告

## 基本信息

**审查范围**: `当前工作区未提交的变更（Git: 已暂存 + 未暂存）`

---

## 审查结论

**审查结果**: `通过（审查中发现的 3 项问题均已在本次提交前当场修复并通过自动化回归测试）`

**综合评分**: `9.5/10`

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | `9.5/10` | 纯确定性管线（Stage 1/2/3）、硬性质量门禁（8 项断言）、增量采纳切分并集及步级 `null` 重置覆盖逻辑严密，全量 246 项 pytest 用例通过 |
| 安全性 | `9.5/10` | 无硬编码密钥或命令拼接，路径遍历与 `session_id` 格式均受校验，前端动态事件参数统一经 `jsArg`（`JSON.stringify` + HTML 转义）包裹 |
| 可维护性 | `9.5/10` | `goldthread/` 工具层、底层 `pipeline/` 算法层与 `server/`/`web/` 可观测层三层解耦清晰，移除废弃 `pipeline/review.py` 与 `sync_pi_datasets.py` 死代码逾 2,000 行 |
| 性能 | `9.3/10` | `server/app.py` 对 `/api/eval/pack` 与 `/api/eval/diff` 引入基于文件 `mtime_ns` + `size` 指纹的线程安全 LRU 缓存；`gt inspect` 分级截断避免上下文膨胀 |
| 代码风格 | `9.6/10` | 类型注解、原子文件替换（`write_json_atomic` / `replace_atomic`）、锁粒度与既有仓库规范高度一致 |

**审查意见**: 本次变更完成了从「重人工按钮 + 隐式小模型调用」向「Agent-First 确定性工具链（`gt` CLI + `SKILL.md`）+ 全链路操作账本（`ledger.jsonl`）+ Web 可观测看板」的架构演进，同时落实了 `d9dbbbd1 → 308e6497` 评估中的早停拦截、`<file>` 标签剥离、规范 `game::*` 归组与 `review_rewrite` DPO 激活。审查期间识别出的 3 项潜在问题（步级 `null` 清除被历史 `.gold.jsonl` 反向回填、部分 Web 写路由漏记账本、内联事件参数单引号逃逸与 `curate dpo` 快捷语法对齐）均已全部修复并补充回归测试，达到合入标准。

---

## 问题汇总

> 以下是本次审查发现的所有问题，按严重程度排序（均已在本次审查闭环内修复）。

### 中等问题 (已修复)

```
[状态机回退缺陷]（已修复）
curate_steps 支持通过 {"msg_id": null} 移除单步显式决策，但 stage2_gold.apply_step_decisions 先执行 _extract_prev_gold_decisions(prev_gold_rec) 从上一版 .gold.jsonl 提取历史 judge_reason / excluded，导致被置 null 的步骤在重跑 Stage 2 时被旧黄金文件反向恢复。
修复：在 step_decisions.json 中记录 reset_steps 集合并在 apply_step_decisions 中优先剔除。
[goldthread/curate_tools.py:377, pipeline/stage2_gold.py:725]
```

```
[审计留痕覆盖缺口]（已修复）
Web 端的样本/作用域/步级排除已接入 ledger.jsonl，但切分修改（/api/dataset/draft/split、/replan）、回合覆盖（/api/raw/session/{sid}/turns）及异步作业（采集/删会话/打包）尚未记录 human 操作账本。
修复：在对应路由与作业完成分支补齐 ledger.record_operation(..., actor="human")。
[server/app.py:308, 345, 389, 1480, 2550]
```

### 轻微问题 (已修复)

```
[前端属性转义与 CLI 快捷语法一致性]（已修复）
1. web/index.html 新增视图的部分内联 onclick 使用单引号拼接 '${esc(x)}'，当包名或标识含单引号时 HTML 解码 &#39; 后会截断 JS 字面量；已改为 jsArg(x)（esc(JSON.stringify(...))）。
2. curate_tools.curate_dpo 已补齐 all=good|swap|drop 与 pending=good|swap|drop 快捷语法，与 SKILL.md 完全对齐。
[web/index.html:1540, goldthread/curate_tools.py:553]
```

### 问题统计

| 问题类型 | 数量 |
|----------|------|
| 接口变更未同步 | `0` |
| 未解决问题 | `0`（发现 3 项，均已当场修复） |
| 资源泄露 | `0` |
| 空指针风险 | `0` |
| 死循环风险 | `0` |
| 性能问题 | `0` |
| 非最优解 | `1`（已修复） |
| 安全漏洞 | `1`（轻微属性转义隐患，已修复） |
| 代码风格不一致 | `1`（账本留痕一致性，已修复） |
| **总计** | **`3`（全部已修复）** |

---

# 详细分析

## 1. 变更摘要

### 核心修改与新增文件
- `gt.py`、`goldthread/{__init__,__main__,cli,ledger,inspect_tools,eval_tools,curate_tools,build_tools,report_tools,assist_tools}.py`（新增：Agent-First CLI 与确定性工具原语层）
- `skills/goldthread/SKILL.md`、`docs/adr/ADR-0008-agent-first-cli-and-operation-ledger.md`（新增：模型标准技能手册与架构决策记录）
- `pipeline/{common/config.py,common/pilog.py,stage1_extract.py,stage2_gold.py,stage3_synth.py,curation.py,dataset_draft.py,package.py,session_delete.py,run.py,judge.py}`（修改：默认零隐式 LLM、`<file>` 剥离、中途假收尾拦截与转 DPO、规范项目归组、`step_decisions.json` 集成）
- `pipeline/review.py`、`sync_pi_datasets.py`（删除：清理已废弃旧模块与历史单文件脚本）
- `server/app.py`、`web/index.html`（修改：新增 `/api/ledger`、`/api/reports`、`/api/eval/pack`、`/api/eval/diff`、`/api/sessions`、`/api/sessions/{sid}/lineage`、`PUT /api/gold/session/{sid}/step-gate`，下线 6 个废弃 `/review/*` 路由，前端升级可观测与三态血缘看板）
- `tests/{test_eval_adjustments.py,test_gt_phase1.py,test_gt_phase2.py,test_gt_phase3.py}`（新增：全链路单元与沙箱集成测试套件）

---

## 2. 调用链与接口变更评估

### 2.1 核心调用关系图（深度 2 层）

```text
gt.py / python -m goldthread (goldthread/cli.py:main)
├── inspect.*  ──► goldthread/inspect_tools.py ──► pilog.parse_session / stage1_extract.infer_project_group
├── eval.*     ──► goldthread/eval_tools.py    ──► check_toolcall_contract / length_gate
├── curate.*   ──► goldthread/curate_tools.py  ──► stage1_extract / stage2_gold / dataset_draft ──► ledger.record_operation
├── build/pack ──► goldthread/build_tools.py   ──► stage2_gold.run / stage3_synth.run / package.build_delivery_zip_from_build ──► gate_check ──► ledger.record_operation
└── report.*   ──► goldthread/report_tools.py  ──► eval_tools.eval_diff ──► 04_release/reports/ ──► ledger.record_operation

server/app.py (FastAPI)
├── GET /api/overview              ──► report_tools.list_reports + ledger.read_ledger
├── GET /api/ledger                ──► ledger.read_ledger
├── GET /api/reports[/{id}]        ──► report_tools.list_reports / get_report
├── GET /api/eval/{pack,diff}      ──► _cached_eval_pack ──► eval_tools.eval_pack / build_tools.gate_check
├── GET /api/sessions/{sid}/lineage──► pilog.build_flow + curation.resolve + ledger.read_ledger
└── PUT /api/gold/session/{sid}/step-gate ──► curate_tools.curate_steps(actor="human")
```

### 2.2 接口变更兼容性评估
- **移除路由**：`/api/gold/session/{sid}/review*` 与 `/finalize` 共 6 个旧人工审核接口已从 `server/app.py` 移除，同时前端 `web/index.html` 中所有对应调用（`revFlush`/`revDecide`/`revFinalize`/`goldSetStepGate` 旧路径）均已同步清理或迁移至 `/step-gate`，全仓检索确认 **0 遗漏调用点**。
- **配置默认值变更**：`turns.use_model` 与 `judge.enabled` 默认改为 `False`，历史已生成的 `02_gold` 质量报告与步级改写通过 `apply_step_decisions` 无损继承，对既有 33 条会话与 `308e6497.zip` 100% 向后兼容。

---

## 3. 主要优点

1. **严格的离线确定性与防呆安全阀**：`curate_steps` 内置首步禁删、单会话 35% 剪除上限与 `_rewrite_ok` 跨语种/空壳拦截；`gate_check` 将中途假收尾 `stop == 0`、`<file>` 标签泄漏 `== 0` 等血泪教训固化为构建硬断言。
2. **CLI 与 Web 账本彻底打通**：所有数据变更统一落入线程安全的 `00_state/ledger.jsonl`，彻底消除终端修数据、网页无痕迹的割裂感。
3. **高覆盖率隔离沙箱测试**：新增 4 套专项测试文件，全程使用 `tmp_path` 隔离沙箱并 `patch` 阻断隐式外部 LLM 请求，全量 246 条测试均在本地快速稳定通过。

---

**报告生成时间**: `2026-11-18`  
**审查工具版本**: Code Review Skill v1.3
