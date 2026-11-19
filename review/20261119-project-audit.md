# GoldThread 项目审核报告 · 2026-11-19

**结论：现有测试全绿，但不建议把当前工作树用于无人值守交付。** 最优先的两处问题是：不含训练目标的样本能通过硬门禁；即使门禁明确失败，ZIP 也已经发布并复制到指定目的地。其次，当前 Skill 不能可靠地告诉 Agent「所有 DPO 是否真的审过」，版本图也不能始终作为审计依据。

**范围**：`main` / `6609319` 加当前工作树（含未提交的 CAS、图谱、前端和测试），不是仅审上一次提交。重点走查 `gt` 策展 → Stage 2/3 → 门禁 → 交付、Web 操作与历史回放；对照 README、CONTEXT、最佳实践、ADR-0007/0008/0009 及仓库内 `skills/goldthread/SKILL.md`。复现用临时目录与合成会话，不在真实语料上执行策展、构建或打包。未跑 GPU 训练、真实学生测评或线上服务；以下不是对所有训练脚本或语料内容的认证。

## 一、优先修复的代码问题

标记「实测」表示在当前工作树复现了行为；仅沿调用链得出的结论单独注明。

| 优先级 | 问题及证据 | 建议修复与验收条件 |
|---|---|---|
| **P1** | **不合格包仍会交付。** `goldthread/build_tools.py:241–264` 先调用打包器写正式 ZIP，再检查门禁；`--copy-to` 的复制不判断 `passed`。实测模拟本地分词器不可用：`gt pack` 返回 **2**、`gate.passed=false`，但 `delivery/` 和模拟学生目录里都有 ZIP。另据静态检查，Web 的 package job 直接调用底层打包器并标记完成，不运行同一套交付门禁（`server/app.py:397–424`）。 | CLI 和 Web 共用同一发布函数：在暂存区生成并检查 ZIP，通过才原子发布及复制。失败要留账，但不能成为正式包、HEAD 或学生输入；已有同名合格包也不能被覆盖。对两个入口分别测失败路径。 |
| **P1** | **没有 assistant 目标也能通过全部八项门禁。** 沙箱 ZIP 的 train/eval 各有一条只有用户问题、没有项目元数据的 SFT 行，带长度门禁记录。实测 `gt gate check` 退出 **0**，`passed=true`、`violations=[]`。`goldthread/eval_tools.py:605–608,720–721` 把无项目映射为 `unknown` 而不计违规，找不到 assistant 直接跳过；`goldthread/build_tools.py:66–80` 再将零计数解释为通过。 | 逐行要求有效监督目标，并区分「检查通过」和「缺少可检查的数据」。兼容旧格式若需推断项目，应有可靠来源；未知就不能宣称项目校验通过。用「仅用户」「缺项目」各做独立负例。 |
| **P1（无人值守场景）** | **DPO 的“待审为零”不代表审完。** `gt build` 强制关闭 judge（`goldthread/build_tools.py:16–20`）；裁决缺失记为 `bypassed`（`pipeline/judge.py:697–741`），不计入 `pending_evaluations`，仍可入集。沙箱中未下发任何有效裁决，dry-run 就报 **1 个 DPO、0 待审**。再提交 Skill 形式的不存在 ID `dpo::01a15341::0` 的 `drop`，CLI 仍报 `updated=1`；随后 Build 仍纳入那 1 对且过门禁。`goldthread/curate_tools.py:580–600` 只写键值，没有验证候选存在。 | ADR-0007 明确允许关闭 judge 时 bypass，**不应简单重新开启隐式模型调用**。应分别暴露「显式裁决、缓存命中、未审绕过」数量；严格交付模式要求未审为零。拒绝或显式报告未知 ID，使 `updated` 代表真实命中。 |
| **P1** | **重打旧包能把 DAG 变成环，首页失去根节点。** 两种 profile 生成通过门禁的 A、B，再重打 A；三次门禁均通过，第三次复用 A 的包哈希却重写 `generated_at`。实测 API 同时返回 `A→B`、`B→A`，根节点数 **0**。`pipeline/package.py:329–330,357–367,398` 刷新旧包；`goldthread/lineage_graph.py:343,379–387` 把显式 `parent_pack:null` 的根也按时间推断父包；`web/index.html:2283–2309` 从根开始绘制。 | 已发布包重打应幂等，不改变创建顺序、父包或历史字节。区分「明确是根」和「旧清单没有 parent 字段」，解析后检查无环。用 A、B、重打 A 的序列同时断言根、边和可见卡片。 |
| **P1（Web 构建入口）** | **“1. 构建 Build”按钮不可用。** `web/index.html:2471–2476` 发送 `POST /api/dataset/draft/build`；FastAPI 注册的是 `POST /api/run`（`server/app.py:949–969`）。用 TestClient 对按钮的真实路径发请求，得到 **404**。CLI 可用不等于看板可用。 | 明确按钮应跑哪个受门禁约束的构建流程，接到实际的任务接口。集成测试从按钮请求断言取得 `job_id`，并检查结果对应预期 Build；不是只断言 HTML 中有函数名。 |
| **P2** | **工作区的步级排除开关丢失当前会话。** 在虚拟数据集抽屉中当前 sid 写到 `_VDS_SESSION_CACHE`，但 `goldSetStepGate` 从旧的 `STATE.drawerData / GOLD_ACTIVE_SID` 取 sid（`web/index.html:2724–2728,3371–3388`）。Node 直接执行前端函数：新开抽屉时产生 **0 个请求**；模拟先看过另一会话时，请求发往 **上一会话**。若两会话有相同 `msg_id`，可能改错决策。 | 将当前抽屉的 `session_id` 和 `msg_id` 一同显式传给写操作。覆盖首次打开和 A→B 切换，断言请求目标、账本 sid 均是 B。 |
| **P2** | **历史对话展示会随今天的配置改变，工作区仍报 Clean。** 在同一冻结包和同一 sid 上，仅更改当前 `synth.max_tool_output_chars`，两次返回的 `raw_messages` 不同，`snapshot_status` 仍为 `exact`，图谱 `dirty=false`。CAS 文件本身没变；变化发生在重新解析时（`goldthread/lineage_graph.py:1091–1100`）。工作区比对也未检查配置，尽管配置属于 Build 指纹（同文件 `:214–243`；`pipeline/dataset_draft.py:189–225`）。 | 历史解析使用 Build 保存的配置；当前配置只用于工作区视图。将影响输出的配置纳入 Clean/Dirty 判定。验收：仅改配置，旧包展示不变、工作区可见。 |
| **P2** | **加权副本被当成不同 SFT 窗口。** 实测同一会话 **4 个逻辑窗口 → 12 条 LF 训练行**，历史黄金列表返回 `sft_windows=12, sft_lf_rows=12`。副本带 `#w2` 等后缀和 `metadata.origin_sample_id`（`pipeline/lf_normalize.py:233–243`），列表却按副本 ID 去重（`goldthread/lineage_graph.py:838–850`）。会高估训练覆盖面。 | 与 `eval_pack` 共用逻辑窗口标识，同时展示唯一窗口与实际训练行。加权样例应断言两个不同的数，而非只断言大于零。 |
| **P2** | **交付包内容和标识受本机缓存影响。** `pipeline/package.py:75–90,317–326,389–390` 递归打包训练模板目录，除极少数项外全部参与哈希和入包。实测当前模板收集器选中 **7 个 `.pyc`**，含 `__pycache__/autotrain.cpython-312.pyc`。同一 Build 在干净检出和已编译目录打包可得到不同 `package_hash`。 | 为交付模板使用文件清单或允许列表，排除解释器缓存和运行产物。测试在模板副本中加入 `.pyc`，断言包标识不变、ZIP 不含该文件。 |

## 二、文档与实现是否一致

| 文档承诺 | 核对结果 |
|---|---|
| `README.md:163`、`CONTEXT.md:182`、ADR-0009:23–32：「任意历史包 100% 精确回放」 | **保证过宽。** 新建 Build 的 CAS 抗工作区删改已有正向测试。旧 Build 只记录部分历史哈希；回填时 raw 直接取现存文件，部分辅助数据取当前工作区，若文件早已删除且没有 CAS 对象就无法还原（`pipeline/cas_store.py:198–250`）。无关联 Build 的独立 ZIP 只能反推会话，`files` 为空（`goldthread/lineage_graph.py:725–762`）。应分别说明「已验证快照」「尽力回填」「无法还原」，不要统一显示 100%。 |
| Skill 将 `inspect graph` 列为只读扫描 | **对旧 Build 不成立。** 查询图谱会加载 CAS 清单；清单缺失自动调用回填，写 CAS 对象和 `cas_backfill.json`（`goldthread/lineage_graph.py:143–149`；`pipeline/cas_store.py:261–285`）。建议把迁移改成显式、带原因的命令；否则至少标出首次读取的副作用和磁盘需求。此项为静态调用链核查，未在真实历史库触发迁移。 |
| README 对历史 `<think>` 的处理 | `README.md:24` 写截断保留，`:134` 写剥离。仓库实际配置是 `strip_history_thinking:false`、上限 600 字符（`config/pipeline.yaml:218–227`），与 Stage 3 分支和 `DATASET_BEST_PRACTICES.md` 页首的现行说明一致。应区分可选模式与当前默认；最佳实践已注明早期方案属历史，不应把整篇历史记录视为缺陷。 |
| 快速开始与写操作原因 | `README.md:56` 的 `report publish` 示例没有 `--reason`；Skill 第 21 行则要求所有写操作提供原因。CLI 的写命令普遍默认空串（如 `goldthread/cli.py:306,334,354,384,401`），账本可保存空原因，部分操作会生成默认原因。这是 **Agent 操作纪律**，不是 **CLI 强制校验**。建议统一示例并明确两者；若作为审计要求，写入前拒绝空原因，保留 dry-run 例外。 |

## 三、仓库 Skill 质量评价

**作为人工陪同的操作手册：可用；作为 Agent 无人值守的执行契约：不合格。** 值得保留的是触发场景清楚，明确禁止绕过账本，先看摘要再钻到单步，给了结构化输入和从实测到归档的顺序。问题集中在最需要精确的地方：取哪个 ID、读哪个返回字段，以及何时必须停手。

1. **给可获取的真实标识，而不是仿真的 ID。** `SKILL.md:127–133` 的 DPO 示例不能照会话前缀加序号推算。应展示如何从候选或 `inspect step` 返回的 `dpo_pairs[].sample_id` 取完整 ID，怎样检查实际命中数；明显标注 `all=good` 会覆盖已有裁决，不是逐对复核的替代品。
2. **把放行条件写成真实 JSON 路径。** `SKILL.md:175` 要读 `pending_dpo`；这个名字出现在人类可读文本里，`--json` 实际字段是 `gate_preview.pending_evaluations`。而且修正字段名仍不能解决 `bypassed` 被当成审完的问题。应一起写明字段含义、异常退出码、未审策略。`eval pack latest` 指向最新 **Build**，验收学生实际拿到的 ZIP 应传 `pack` 返回的哈希或路径（`goldthread/eval_tools.py:82–101`）。
3. **准确写保护措施的边界。** `SKILL.md:92` 称 `prune/fold` 都受 `max_drop_ratio` 保护。实际策展校验只对 `prune` 计数，上限 `max(1, floor(助手步数 × 0.35))`；`fold` 另按相邻步骤能否合并执行（`goldthread/curate_tools.py:413–421`；`pipeline/stage2_gold.py:792–833`）。不要让 Agent 以为可以依赖一个实际不存在的联合限额。
4. **补一条失败路径，比再加一页命令更有价值。** 写清打包退出 **2** 后停止交付、检查 `violations`，不能凭 ZIP 已存在继续下发；在发布顺序修复前，先不传 `--copy-to`，确认具体 ZIP 门禁通过再复制。用沙箱把 Skill 中的示例连跑，并断言产物、账本、状态，而非只比对命令名称。

## 四、验证记录与建议顺序

- `python -m pytest tests/ -q`：**244 passed, 1 warning**（48.77 s）。
- `python -m pytest tests/test_virtual_datasets.py tests/test_readme_alignment.py tests/test_gt_phase3.py -q`：**15 passed, 1 warning**（31.40 s）。两次均为 Starlette TestClient 使用 AnyIO 弃用别名的警告。
- `python -m py_compile` 检查 `goldthread/{build_tools,cli,lineage_graph,eval_tools,curate_tools}.py`、`pipeline/{cas_store,dataset_build,package}.py`、`server/app.py`：通过。提取 `web/index.html` 内联脚本运行 `node --check`：通过。对上述核心 Markdown 和 ADR 的相对 Markdown 文件链接检查：未发现缺失目标。
- 全绿与上面的问题不矛盾：`tests/test_virtual_datasets.py` 已验证新建包在改写/删除会话后的回读；其前端测试主要查函数名和 JS 语法。`tests/test_readme_alignment.py` 校验文档标记，不执行 Skill 示例。这两类测试均不能证明失败包没发出、按钮能工作、旧包重打后无环。
- **建议顺序**：先修交付顺序、逐行门禁和 DPO 审核语义；再修 Web 构建/步级操作及图谱无环性；最后校准快照与统计口径，并让文档示例成为可执行的契约测试。

本次仅新增此审核报告，没有改业务代码、文档原文或测试文件；沙箱复现不等于已修复，也未计入上述 pytest 用例数。
