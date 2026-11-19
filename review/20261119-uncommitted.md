# 代码审查报告（5f4052a 后复审加固与 6d76c072 语料迭代）

## 基本信息

**基准提交**: `5f4052a` (`feat: upgrade virtual dataset lineage graph, CAS store, and gate safeguards`)
**审查范围**: `5f4052a` 之后的复审边界加固（对照 `review/20261119-followup-5f4052a.md`）、`curation.resolve` 重算回退修复、三款游戏教师会话替换配置及配套回归测试。

---

## 审查结论

**审查结果**: `通过（复审识别的 6 项深层边界问题、curation 死锁及 README 锚点偏差均已修复闭环，全量 518 项自动化测试通过）`

**综合评分**: `9.7/10`

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | `9.8/10` | 交付包复用/暂存门禁绑定实际文件、Stage 2 后严格 DPO 核验、DPO 全内容指纹绑定、原子精确 span 折叠、不可变 `parent_pack` 及 `curation` 过期回退均验证闭环 |
| 安全性 | `9.7/10` | `--copy-to` 采用目标目录临时文件原子替换（`NamedTemporaryFile` + `replace`）并禁止覆盖 `delivery/` 下其他交付包或自拷贝；异常路径确保清理 `.staging-*` 与临时副本 |
| 可维护性 | `9.6/10` | 自动规则与 judge/agent `fold` 共用 `_do_fold` 单一实现；`PURIFY_VERSION` 控制旧黄金缓存按需无损升级；`skills/goldthread/SKILL.md` 精炼收敛 |
| 并发与队列 | `9.7/10` | Web 新增确定性 `build` 阶段与 `stage1`、`package`、`all`、`delete` 及黄金写作业正确互斥排队，步级门禁编辑在 `all` 运行期返回 HTTP 409 |
| 测试覆盖 | `9.8/10` | 新增 `test_followup_{delivery,dpo,folding,web}.py` 四组专项回归套件并扩展 `test_eval_adjustments.py` 等，总用例达 518 项 |

---

## 重点核查项与修复记录

### 1. 交付与不可变血缘加固（`goldthread/build_tools.py`, `pipeline/package.py`）
- **复用包门禁与内容一致性**：当目标 `delivery/goldthread-train-<hash>.zip` 已存在时，直接对实际将复用的已发布文件执行 `gate_check` 并调用 `require_matching_delivery_zip` 逐成员比对内容（忽略时间戳与 `MANIFEST.generated_at` 差异），防止历史 `--no-gate` 包借同名复用绕过门禁。
- **原子复制与防误覆写**：`--copy-to` 先写入目标同目录 `.tmp` 临时文件再原子 `replace`，拷贝或替换异常均保留原目标文件并清理临时文件。
- **不可变 `parent_pack`**：已发布同哈希包若传入不一致的 `--parent` 或自指 `parent_pack == package_hash`，立即抛出 `ValueError` 并记 `refused` 账本，杜绝篡改历史谱系。

### 2. 严格 DPO 审核时机与全内容指纹绑定（`goldthread/build_tools.py`, `goldthread/curate_tools.py`, `pipeline/judge.py`, `pipeline/stage3_synth.py`）
- **生成后核验**：`build --require-reviewed-dpo` 在 Stage 2 刷新黄金层后、Stage 3 写入正式 Build 前校验本次实际入集候选；未审项触发 `DPOReviewRequired` 并记 `dpo_review_required` 账本，不推进 `LATEST_BUILD.json`。黄金层未就绪时 `--dry-run` 返回 `readiness="undetermined"` 而非谎报零待审。
- **内容指纹绑定**：`dpo_content_fingerprint` 对完整 `conversations`、`chosen`、`rejected`、`tools` 计算 SHA256；`curate_dpo` 与 `dpo_review_cache` 均绑定该指纹。正负例或上下文任一处改动后旧裁决自动转为 `pending`，杜绝旧 ID 裁决跨版本穿透。

### 3. 确定性 `write → edit` 折叠与弯路剪枝边界（`pipeline/stage2_gold.py`）
- **严格回执与原始 span 原子重放**：`_do_fold` 要求同 `turn_index`、单工具调用、`_success_receipt` 完整成功回执（核对写入字节数或替换块数及内联镜像一致性），并在 LF 归一化原文上按无重叠唯一精确 span 一次性重建（结合 `_fuzzy_compare` 排除引号/破折号多义匹配，保留 BOM 与 CRLF 风格）。失败或超上限时完整保留原步骤，绝不半途丢弃部分编辑。
- **防失败编辑误吞**：`_fold_participant` 与 `_unverified_edit` 防止失败 `edit` 被弯路剪枝误删或误判为修正正例。

### 4. Web 工作区构建与 `curation.resolve` 死锁修复（`server/app.py`, `web/index.html`, `pipeline/curation.py`, `README.md`）
- **Web 图谱一键构建**：`buildWorkspaceFromGraph()` 改为提交 `stage="build"`，离线串联已采纳会话的 Stage 2 提纯、Stage 3 合成与硬门禁校验，解决纯 raw 会话直接跑 `stage3` 失败的问题。
- **历史人工通过会话重算回退**：`pipeline/curation.py` 中当 `PURIFY_VERSION` 升级导致 `review_key` 变化时，历史 `decision="rejected"` 继续严格阻断，而历史 `decision="approved"` 若具备有效 `model_report` 则按 `model_report` 裁决放行，避免全量历史黄金会话死锁于 `stale`。
- **文档链接修正**：审查中发现 `README.md` 两处指向精简前 `skills/goldthread/SKILL.md` 旧章节标题的死锚点，已当场修正为有效相对链接；同步更新 `config/pipeline.yaml` 追加续跑折叠正则与 `PRODUCTION_CONFIG_SHA256`。

---

## 提交前验证记录

- `python -m py_compile` 全涉及 Python 模块：**通过**
- `web/index.html` 提取 `<script>` 执行 `node --check`：**通过**
- `python -m pytest tests/ -q`：**518 passed**
