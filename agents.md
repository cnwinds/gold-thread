# GoldThread Agent 开发约定

本文档约束在本仓库内执行开发任务的 Agent（含 Cursor Agent）。**任务完成前必须满足测试要求，不得跳过。**

## 1. 测试与功能配套

- **每一项开发任务**（新功能、行为变更、缺陷修复、重构、API/配置语义变更）都必须有**配套的自动化测试**。
- 测试应覆盖本次改动的**核心行为与回归点**，而不是只测实现细节或常量。
- 优先在 `tests/` 下新增或扩展用例，命名与现有风格一致（`test_*.py`）。
- 若某次改动确实无法写自动化测试（例如纯文案、静态资源），须在任务说明中**写明原因**；其余情况不允许以「改动小」为由省略测试。

## 2. 任务完成前的验证（必做）

标记任务完成前，**必须本地执行测试并通过**：

```bash
python -m pytest tests/ -q
```

针对本次改动相关的子集至少跑一遍，例如：

```bash
python -m pytest tests/test_pipeline_flow.py -q
```

除 pytest 外，若改动涉及以下内容，还需做对应检查：

| 改动类型 | 额外验证 |
|----------|----------|
| Python 管线 / `server/` | `python -m py_compile` 涉及模块 |
| `web/index.html` 内联脚本 | 提取 `<script>` 后 `node --check` |
| HTTP 接口行为 | 对相关路径做一次请求或已有测试覆盖 |

**未运行测试或未通过测试，不得向用户宣称「已完成」。**

## 3. 交付说明

向用户汇报完成时，应简要说明：

1. 改了什么（行为层面）
2. **新增/更新了哪些测试文件**
3. **实际执行的测试命令与结果**（通过 / 失败）

## 4. 与本项目相关的注意点

- 默认 Python 环境与本地 `python` 一致；跑测试前确认依赖已安装。
- 不要为通过测试而削弱断言；不要提交仅用于本地调试的 `_tmp_*` 脚本作为正式测试。
- 用户未要求时，不要 git commit / push。

---

**摘要：有开发必有测试；做完必跑 `pytest` 验证，通过后再交付。**

## Agent skills（Matt 工程流）

本仓库已按 [Matt Pocock 工程 skills](https://github.com/mattpocock/skills) 约定完成初始化。Agent 在执行 `to-tickets`、`triage`、`to-spec`、`implement` 等流程前，先读本节引用的文件。

### Issue tracker

工作在 **GitHub Issues**（`cnwinds/gold-thread`），操作用 `gh` CLI。详见 [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md)。

### Triage labels

五个标准分拣角色与标签名一致（`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`）。详见 [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md)。

### Domain docs

**Single-context**：根目录 `CONTEXT.md`（按需由 `/grill-with-docs` 沉淀）+ `docs/adr/`。详见 [`docs/agents/domain.md`](docs/agents/domain.md)。

### 推荐工作流（idea → ship）

1. **`/grill-with-docs`** — 在本仓库里把想法问清楚，写入 `CONTEXT.md` / ADR。  
2. 需要可运行验证时 → **`/handoff`** + **`/prototype`** + 再 **`/handoff`** 回来。  
3. 多会话交付 → **`/to-spec`** → **`/to-tickets`** → 每个 ticket 单独 **`/implement`**（内部走 **`/tdd`** + **`/code-review`**）。  
4. 小改动可在同一会话直接 **`/implement`**。

上下文接近 smart zone 上限时，用 **`/handoff`** 开新会话，不要硬撑 compaction。

