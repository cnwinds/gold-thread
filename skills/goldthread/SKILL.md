---
name: goldthread
description: Operate the GoldThread Agent-First dataset refinery via the deterministic `gt` (`python -m goldthread`) CLI — inspect teacher/student sessions, audit & diff training packs, curate step/turn/DPO decisions with full causal ledger (`ledger.jsonl`), run zero-LLM deterministic builds with hard quality gates, pack delivery zips, and publish evaluation reports to the Web observatory.
---

# GoldThread Agent-First 操作技能手册 (`gt`)

当用户要求你**扫描教师/学生轨迹、评估学生模型实测效果、清洗修补黄金步/DPO、构建新版训练集、打包交付 `.zip` 或归档评估报告**时，**严禁手写绕过账本的一次性临时脚本直接篡改 `data/`**。请全程调用仓库内置的确定性工具集：

```bash
python gt.py <domain> <action> [options] --json
# 等价入口：
python -m goldthread <domain> <action> [options] --json
```

---

## 一、三条核心铁律

1. **先摘要后下钻（Token 友好）**：始终带 `--json` 先读 `inspect corpus` / `inspect source` / `inspect session <sid> --steps` 紧凑摘要，仅对需要改写的具体步骤调用 `inspect step <sid> <msg_id>`，切勿直接将数兆 `.jsonl` 倾倒进上下文。
2. **写操作强制留痕（`--reason`）**：所有策展、构建、打包、发布报告命令必须传入清晰的 `--reason "因果说明"`（默认 `--actor agent`），工具会自动记录前后快照并追加到 `data/00_state/ledger.jsonl`，实时呈现在 Web 看板。
3. **主构建链路 100% 纯确定性**：`gt curate`、`gt build`、`gt gate check`、`gt pack` **零隐式外部 LLM 请求**。判断权归主模型（你），程序只负责校验不变量、语言一致性与原子落盘；若需小模型预筛，只能显式调用 `gt assist ...`。

---

## 二、五大标准命令域速查

### 1. 只读扫描与下钻 (`inspect`)

```bash
# 扫描外部教师/学生源目录，对比已入库情况与未覆盖新项目
python gt.py inspect source --dir "D:\train-project\teacher" --unadopted --json

# 查看当前语料库全景（项目切分、SFT/DPO 规模、最新 Build）
python gt.py inspect corpus [--project game::meowdoku] --json

# 查看虚拟数据集演进关系图（纯只读无写盘副作用：DAG 节点/边、HEAD 包、Clean/Dirty 状态与 CAS 对象数）
python gt.py inspect graph --json

# 查看单条会话元信息、回合标记与紧凑步骤列表（支持唯一前缀如 01a15341）
python gt.py inspect session 01a15341 --steps --json

# 下钻查看单步三态血缘（raw_thinking / gold thinking / tool_calls / 关联 DPO）
python gt.py inspect step 01a15341 <msg_id_or_0_based_index> --json
```

### 2. 训练包体检与学生实测归因 (`eval`)

```bash
# 对交付 zip / 8位哈希 / build_id / latest 做全维度契约与分布体检
python gt.py eval pack latest --json
python gt.py eval pack 308e6497 --style coder-xml --json

# 对比两版交付包或 Build 的全维度指标增减
python gt.py eval diff d9dbbbd1 308e6497 --json

# 自动解析学生模型实测工作目录与 pi 会话日志，横向对账教师基准
python gt.py eval student \
  --student-dir "D:\train-project\student\goldthread-308e6497" \
  --teacher-dir "D:\train-project\teacher" --json
```

### 3. 确定性语料策展 (`curate`)

```bash
# 批量采纳源会话（自动剥离 <file> 外壳、规范化 game::<slug>、去重）
python gt.py curate adopt --from "D:\train-project\teacher" --select 01a1d2ec,01a1bf57 --reason "补齐新版教师轨迹" --json
python gt.py curate adopt --from "D:\train-project\teacher" --all-unadopted --reason "采纳全部新增验证通过教师会话" --json

# 安全级联删除残缺旧桩（不触碰外部源目录与冻结历史 zip）
python gt.py curate drop-session 01a16770 --reason "剔除早期 37 步残缺旧桩，由完整版替代" --json

# 覆盖会话回合分类（root / task / continuation / correction / setup / noise）
python gt.py curate turns <sid> --patch '{"1": "continuation"}' --reason "折叠中途催促回合" --json

# 写入主模型步级结构化决策（keep / prune / fold / rewrite / exclude）并立即重算黄金层
python gt.py curate steps --file step_decisions_patch.json --reason "压缩长思维链反刍并剪除报错弯路" --json

# 写入 DPO 偏好对裁决（good / swap / drop；支持 all=good 快捷串或 JSON 文件）
python gt.py curate dpo --verdicts dpo_verdicts_patch.json --reason "复核新增 DPO 候选对" --json

# 锁定项目/会话的 train/eval 切分计划
python gt.py curate split --pin game::altosodyssey=eval --reason "锁定留出验证集项目" --json

# 应急停用或恢复单条样本 / 会话 / 项目作用域
python gt.py curate exclude --sample <sample_id> [--restore] --reason "停用异常切片" --json
python gt.py curate exclude --scope session_sft:<sid> [--restore] --reason "仅保留该会话 DPO，停用 SFT" --json
```

#### `gt curate steps --file <json>` 输入结构规范

支持单会话简写或多会话批量结构（安全阀边界：`rewrite` 文本须与原思考保持中文/英文主语言一致；单会话 `prune` 数量上限为 `max(1, floor(assistant_steps * 0.35))` 且首个助手步禁止 `prune`；`fold` 独立按相邻同路径 `write+edit` 能否合并校验）：

```json
{
  "sessions": {
    "01a15341": {
      "steps": {
        "msg_0012": {
          "action": "rewrite",
          "rewritten": "定位 global-metadata.dat 自定义段表偏移后，直接编写脚本验证解密。",
          "reason": "消除重复自我怀疑反刍"
        },
        "msg_0015": {
          "action": "prune",
          "reason": "参数拼写错误导致报错的无效弯路"
        },
        "msg_0020": {
          "action": "exclude",
          "reason": "保留上下文连贯性但不作为 SFT 监督目标"
        }
      },
      "model_report": {
        "verdict": "trainable",
        "confidence": 0.9,
        "reason": "完整逆向闭环，工具验证扎实",
        "highlights": [
          {"msg_id": "msg_0012", "weight": 1.5, "why": "关键加密结构突破"}
        ]
      }
    }
  }
}
```

#### `gt curate dpo --verdicts <json>` 输入结构规范

**重要**：请勿凭会话前缀与序号臆造 ID！必须先通过 `gt inspect step <sid> <step_ref> --json` 的 `dpo_pairs[].sample_id`（或读取 `data/02_gold/dpo_pairs.jsonl`）获取真实完整 `sample_id`；`curate dpo` 会严格校验候选池存在性，未知 ID 将直接报错拦截。
> 注意：`--verdicts pending=good` 仅放行尚未裁决的新增项；`--verdicts all=good` 会**全量覆盖已有裁决**，仅用于测试或明确确认全放行场景，不可替代逐对复核。

```json
{
  "<从 dpo_pairs[].sample_id 获取的完整真实 ID>": {"verdict": "good", "reason": "正负例对比鲜明，惩罚中途早停"},
  "<真实 ID 2>": {"verdict": "swap", "reason": "原 rejected 更简洁准确，对调纳入"},
  "<真实 ID 3>": {"verdict": "drop", "reason": "差异过小，无偏好教学价值"}
}
```

### 4. 纯确定性构建、硬门禁与打包 (`build` / `gate` / `pack`)

```bash
# 先预览草稿规模与 DPO 审核明细（不落盘；检查 JSON 中的 gate_preview.pending_evaluations 与 gate_preview.bypassed_unreviewed）
python gt.py build --dry-run [--require-reviewed-dpo] --json

# 正式确定性构建（Stage 2 -> Stage 3 -> 固化不可变 Build + 自动归档 04_release/cas/ 快照池 -> 自动跑 gate check）
# 加 --require-reviewed-dpo 可在无人值守模式下强制要求 bypassed_unreviewed == 0
python gt.py build [--require-reviewed-dpo] --reason "构建新一版训练集" --json

# 单独校验指定 Build 或交付 zip 的硬性质量门禁（违例返回退出码 2）
# 注意：`latest` 指向最新 Build；验收学生训练将使用的最终 ZIP 包时，请传入 pack 返回的 package_hash 或 zip 路径！
python gt.py gate check <pack_hash_or_zip_path> --json

# 从不可变 Build 打包交付 zip（先在暂存区生成并执行 gate check：仅门禁通过才原子发布至 delivery/ 及 --copy-to；失败返回退出码 2 且不产出污染包）
python gt.py pack --profile qwen-xml [--parent d9dbbbd1] --copy-to "D:\train-project\student" --reason "导出 Qwen3.8 训练包" --json
```

### 5. 评估报告归档与账本查询 (`report` / `ledger`)

```bash
# 将评估与迭代报告归档至 data/04_release/reports/（自动计算包间 diff 并展示在 Web 首页）
python gt.py report publish \
  --title "308e6497 学生实测评估与下一版训练集优化报告" \
  --md evaluation_report.md \
  --before 308e6497 --after <new_pack_hash> \
  --reason "归档本轮实测归因与构建对比" --json

# 查询报告与操作审计账本
python gt.py report list --json
python gt.py ledger list --limit 50 [--sid 01a15341] [--tool curate.steps] --json
```

---

## 三、新一轮迭代标准闭环 SOP

1. **实测对账**：运行 `gt eval student` 与 `gt inspect source --unadopted --json`，定位学生模型的早停、盲区、上下文长度问题及未入库教师轨迹。
2. **精准策展**：用 `gt curate adopt` / `drop-session` / `turns` / `steps` / `dpo` / `split` 下发结构化决策，每步均写明 `--reason`（DPO 必须取真实候选 `sample_id` 并确认返回 `updated` 命中数）。
3. **门禁构建与受控出包**：
   - 先 `gt build --dry-run --require-reviewed-dpo --json`，确认 `gate_preview.pending_evaluations == 0` 且 `gate_preview.bypassed_unreviewed == 0`；
   - 再运行 `gt build --require-reviewed-dpo --reason "..." --json`，确认退出码为 `0` 且 `gate.passed == true`；
   - 最后运行 `gt pack --profile qwen-xml --copy-to ... --reason "..." --json`：**若退出码为 `2`（`ok == false`），必须立即停手排查 `gate.violations`**（暂存包已自动销毁，切勿拿旧包冒充新包下发）；确认 `ok == true` 后可再用 `gt eval pack <package_hash> --json` 复核最终 ZIP。
4. **对比归档**：运行 `gt eval diff <old_hash> <new_hash> --json` 核查全维度指标提升，调用 `gt report publish --reason "..."` 将完整分析报告落盘，供人类在 `http://127.0.0.1:8765` 审阅。
