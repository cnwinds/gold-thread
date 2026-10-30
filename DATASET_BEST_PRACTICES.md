> **工程实现说明（2026-10 更新）**：本文档保留领域知识与提纯规范。实际执行已重构为三阶段管线
> （`01_raw` → `02_gold` → `03_train`），并附带本地 Web 控制台；命令、目录与脚本/模型分工见 `README.md`。
> 旧版单文件脚本 `sync_pi_datasets.py` 与 `output/` 仅作对照保留。

# GoldThread · Agent 工程分析训练数据集整理与黄金轨迹蒸馏最佳实践手册

> **核心定位**：本手册结合 `jigsawcard`（Unity IL2CPP + ANort 加固 + OLLVM）与 `meowdoku`（Godot 4.6 + GDScript 字节码 + ZSTD/XOR/Smaz 混合加密）两个实战工程分析课题中 `DeepSeek-v4.1-flash`、`GLM-5.3`、`Qwen3.8-27B` 的逐帧对比复盘，确立从 `.pi` 原始会话日志（`.jsonl`）中**精准筛选项目会话、剥离无关杂音、提纯黄金轨迹并构造 SFT / DPO / GRPO 数据集**的完整工程指南。

---

## 一、 数据源准入与“三层除噪网关”（最关键的第一道防线）

在 `.pi/agent/sessions/` 目录下混杂着大量日常使用记录（106 个会话文件中仅有 **11 个**位于 `--D--crack-<project>--` 下）。如果直接全量扫描或简单按 `role == "user"` 切分，会把配置模型、测试 `hi`、生图提问、甚至中途催促/纠偏语（如 `"不要参考其他的目录，自己研究"`）当成独立训练样本，严重污染模型分布。

整理数据集时，必须首先通过 **三层除噪网关（Three-Layer Domain & Turn Filtering）**：

### 第 1 层：会话级领域准入网关（Session-Level Gate）
1. **严格限定工作区目录**：
   - 仅准入 `--D--crack-<project>--`（对应 `D:\crack\<project>`）下的目标项目会话；
   - **100% 剔除**全部非目标目录会话（如 `--C--Users-lvan--` 下的 `zhipu` 模型配置、`--D--ai_projects*--` 下的 `gpt-image-2` 与 `批、` 误触、`--E--Download*--` 与 `--Y--stock-projects*--` 下的 AI 新闻生图与量化策略）；
   - **剔除根目录泛化问答**：排除 `--D--crack--` 根目录下仅询问 `"qwen3.8 拒绝回答的问题，我怎么通过微调让他回答"` 的非实战会话（`01a15cd1`）；
   - **剔除跨目录迁移产生的重复前缀副本**：如 `--C--Users-lvan-.pi-agent--\01a1523a` 是会话迁移到 `--D--crack-jigsawcard--\01a15375` 之前的前 20 轮完全重复副本，必须排除以防样本重复。
2. **严格区分 Teacher 会话（SFT 正样本源）与 Student 会话（DPO 负样本/诊断源）**：

| 会话分类 | 会话文件（共 11 个项目会话） | 模型 | 角色定位与数据集流向 |
| :--- | :--- | :--- | :--- |
| **A. 黄金 Teacher 会话（4 个）** | `--D--crack-meowdoku--\01a1589b`<br>`--D--crack-meowdoku-glm--\01a158b3`<br>`--D--crack-jigsawcard--\01a15375`<br>`--D--crack-jigsawcard-glm--\01a15341` | `deepseek-v4.1-flash-1017`<br>`glm-5.3` / `glm-5.3-flash` | **SFT 主力正样本源**：经状态折叠与弯路剪枝后生成 `sft_gold_trajectories`（4 条完整长轨迹）与 `sft_step_windows`（高纯度单步窗口），被剪除的弯路同步转入 `dpo_preference_pairs` |
| **B. 局部白名单混合会话（1 个）** | `--D--crack-meowdoku-ds-code-v2--\01a15c80` | `qwen3.8-27b` | **仅抽取 `#050~#106` 成功子段**（用 `OpenGDS` 反编译 789 个 `.gdc` + 读 `marshalls.cpp` 解密 58 个关卡表）补充进 `sft_step_windows`；其余中途放弃（`#046`）与末尾 `Godot headless` 死锁（`#118+`）转入负例库 |
| **C. Student 诊断会话（6 个）** | `--D--crack-jigsawcard-qwen3.8--\01a153ca`<br>`--D--crack-jigsawcard-qwen3.8--\01a1577a`<br>`--D--crack-jigsawcard-qwen3.8--\01a15816`<br>`--D--crack-meowdoku-qwen3.8--\01a158b9`<br>`--D--crack-meowdoku-qwen3.8-27b--\01a1535e`<br>`--D--crack-meowdoku-qwen3.8-27b--\01a1599c` | `Qwen3.8-27B-Uncensored-Q4_K_M`<br>`Qwen3.8-27B-Uncensored-Q8_0`<br>`qwen3.8-27b` | **严禁混入 SFT 正样本！** 专门用于收割心算十六进制打勾 `✓`、万字反刍死循环、脑补 C++ 源码、跨目录偷看等生成 `dpo_preference_pairs` 与 `negative_diagnostics` |

---

### 第 2 层：回合级除噪网关（Turn-Level Noise Stripping）
即使在准入的 11 个项目会话内部，也经常包含大量**任务开始前或中途穿插的环境配置与闲聊回合**，必须在解析时逐轮剥离（共自动剥离 **45 个无关 Turn**）：
1. **前置模型配置与连通性测试（Pre-Task Setup Turns）**：
   - 在首个包含工程分析目标（`.xapk` / `.apk` / `破解`）的 **根任务指令（Root Crack Prompt）** 出现之前，丢弃一切配置 `models.json`、`sub2api`、`openwebui`、`ollama`、切换推理强度、`/modle`、以及发送 `hi` 测试 API 连通性的回合（例如 `01a15375` 的 Turn 0~6、`01a15c80` 的 Turn 0~15、`01a15816` 的 Turn 0~4、`01a1599c` 的 Turn 0~4）。
2. **中途穿插的报错排查与硬件/磁盘闲聊（Mid-Session Admin & Hardware Chat）**：
   - 剔除中途粘贴 `Error: 400 / 502 / 503`、`隔壁的会话报错了`、`要下载 fd linux版`、`gh_proxy`（如 `01a153ca` 的 Turn 2~9）；
   - 剔除与目标包分析逻辑无关的个人硬件咨询与 C 盘清理回合（如 `01a15375` 中的 `"我目前只有一台小米平板1代，root过了，这个设备可以吗"`、`"华为mate50pro的android系统可以吗"`、`"frida-server 是干什么用的"`、`"先清理"`、`"目前界面没有uac弹窗"`）。

---

### 第 3 层：过程追问折叠与纠偏指令内化（eliminating Headless Prompts）
为什么不能按每次 `role == "user"` 把会话切成孤立样本？
- **问题 1（产生无头 Prompt）**：如果按每次用户发言切分，像 `"不要参考其他的目录，自己研究"`、`"继续"`、`"可以"`、`"补全结构"` 就会变成没有前文背景的独立样本（模型根本不知道要研究哪个包）。
- **问题 2（训练出“半途停工等指令”的坏毛病）**：在交互式会话中，模型汇报阶段进展后停下等用户说 `"继续"` 或 `"可以"`。如果原样保留，微调后的模型也会干几步就停下来等人类催促，失去全自主（Autonomous）一镜到底的能力。
- **标准处理规范**：
  1. **根任务指令标准化**：将会话开启时的首条破解指令统一规范化为明确的全量目标（如 `"帮我破解 E:\Download\com.oakever.<game>@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有解密文件保存在本文件夹"`）；
  2. **过程催促/确认折叠（Turn Folding）**：对于 `"继续"`、`"继续A"`、`"可以"`、`"补全结构"`、`"所有的文件都要还原成明文"` 等过程推进指令，将其从 User 消息中折叠隐藏，并删去上一轮末尾模型用于“停下来请示用户”的纯文本消息，使前后工具调用无缝拼接为一条**全自主连续执行流**；
  3. **纪律纠偏内化（Constraint Internalization）**：对于 `"不要参考其他的目录，自己研究"` / `"你不要看其他的目录，专心干自己的目录"`，将其内化为 `System Prompt` 的第 1 条铁律（*“严格在当前工作目录内独立完成研究与产出，严禁读取、参考或拷贝同级其他实验目录的文件”*），并将触发该纠偏的跨目录偷看步收割进 DPO 负样本库。

经过以上三层网关后，整个数据集中**只存在以下 6 条纯粹的工程分析指令**：
1. `帮我破解 E:\Download\com.oakever.jigsawcard@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有解密文件保存在本文件夹`
2. `帮我破解 E:\Download\com.oakever.meowdoku@arm64-v8a.xapk 游戏，获取 资源，配置，代码 所有文件保存在本文件夹`
3. `你现在看看他的代码里面，有没有什么千人千面的设计， 玩法上数值是怎么设计的，是纯通过配置表，还是有动态的计算调整`
4. `帮我看看他的千人千面是怎么实现的`
5. `SelectStageByUserTag 的完整逻辑逆出来`
6. `你破解出的代码，总结一份说明给我`

---

## 二、 黄金轨迹提纯的“四阶段流水线”（Four-Stage Distillation Pipeline）

即使在筛选出的 4 个黄金 Teacher 会话中，也不能原封不动直接拿去 SFT，必须经过以下四道提纯工序：

### 阶段 1：轨迹图剪枝与状态折叠（Graph Pruning & State Folding）

| 弯路/瑕疵类型 | 实例定位 | 黄金 SFT 处理方式 | DPO 转化方式（变废为宝） |
| :--- | :--- | :--- | :--- |
| **A. 状态折叠（State Folding）** | `DS meowdoku #25 -> #26`：`#25` 写 `decrypt_levels.py` 时三元表达式 `_skip` 求值顺序写反导致错位，`#26` 调 `edit` 修复 | 将 `#26` 的 `edit` 直接合并回 `#25` 的 `write` 内容中，**删除 `#26`**，让 SFT 模型一步写出无 Bug 代码 | 以 `#25` 上下文为 `prompt`，合并后的正确代码为 `chosen`，初版带 Bug 代码为 `rejected` |
| **B. 手滑与环境报错剪枝（Detour Pruning）** | `DS meowdoku #12`（传错父目录报 `Not a Godot directory`）、`#49`（`cp *.apk` 手滑覆盖刚打好补丁的包）、`#53`（调不存在的 `zip -r`） | **直接剪除报错回合**，清理下一步思考开头的懊恼词（`Oops...`），直接衔接正确调用 | 以报错前上下文为 `prompt`，修正后的调用为 `chosen`，手滑报错调用为 `rejected` |
| **C. 假阴性死胡同切除（Dead-End Slicing）** | `DS meowdoku #36~#41`：借用内置了 `gdre_tools.pck` 的 `gdre_tools.exe --path` 跑测试脚本得到假阴性 `rel=false`，又调未暴露的 `has_custom_feature()` 报错 | 切除中间无果的试错回合，仅保留**查证引擎源码确认机制**（`#38~#39`）与**全工程检索影响面**（`#42`） | 将调用不存在 API 的步骤收入负样本库 |
| **D. 真实业务探针排错（必须保留！）** | `DS meowdoku #27 -> #30`：批量解密遇到未知 `variant type 65539` $\rightarrow$ 打印偏移 52 字节流 $\rightarrow$ 识别出 `0x00010003` 的 `ENCODE_FLAG_64` 并修复 | **完整保留该“遇未知格式异常 $\rightarrow$ 写探针看字节 $\rightarrow$ 修脚本重跑”的科学排错闭环**（仅精简 `#30` 的发散思考） | 将 `#30` 压缩前后的 `<think>` 配对为反刍压缩 DPO 样本 |

---

### 阶段 2：思维链（`<think>`）去噪与静默执行规范化

1. **严禁“空 `<think>`”进入 SFT 库**：
   - 如 `GLM` 在 `meowdoku` 中有 20 步 `th_len == 0` 直接盲调工具。对于空思考且报错的步直接剪除；空思考但执行成功的步，将其对外文本内化为 `<think>` 或从单步窗口中剔除，确保模型“先思考、后出招”。
2. **超长发散 `<think>` 压缩（控制在 `300 ~ 1,600` 字符）**：
   - 对超过 `2,400` 字符或含多次 `Wait / Actually` 的 `<think>`，剔除中间盲猜偏移的段落和默写的大段代码，保留“观测事实 $\rightarrow$ 源码/格式假设 $\rightarrow$ 并发验证动作”三段式结构。
3. **中间步骤强制“静默执行（Silent Execution）”**：
   - 在最终总结前的所有工具调用步，**清空对外输出的 `text`（`SAY`）字段**，只保留 `<think>` + `<tool_call>`。

---

### 阶段 3：跨模型优势嫁接（“DeepSeek 骨架 + GLM 插件”）

- **以清洗后的 `DeepSeek` 为主干骨架**：继承其高并发工具调用（单步 2~3 个并行命令）、先看 Hex 再写解析器、4 大密钥与 165 项配置挖掘、重签名作弊包构建与全量自检。
- **嫁接 `GLM` 的两大独门突破**：
  1. **`meowdoku` 75 国语言包 Smaz 还原**：嫁接 GLM 用 `gdre_tools --bin-to-txt` 剥离 RSRC 外壳 + 拉取 Godot `optimized_translation.cpp` / `smaz.c` 源码还原 75 国文本的子轨迹（修正其误砍 `"MEOWDOKU\0"` 9 字节与漏 `.rstrip('\x00')` 的小瑕疵），并合入第 58 个题库文件 `levels-7x7.json` 的通配与 SHA256 断言；
  2. **`jigsawcard` 模拟器动态内存脱壳**：面对 ANort 加密 `global-metadata.dat` 与 `libil2cpp.so` 代码段，收入通过 `adb` 冻结进程、`dd /proc/pid/mem` 导出运行时内存并扫描恢复结构体的实战范式。

---

### 阶段 4：单步滑动窗口展开、历史 `<think>` 剥离与根任务锚定

在生成主力训练集 `sft_step_windows.jsonl` 时，严格执行三项构造规范：
1. **永久锚定根任务指令（Root Task Anchoring）**：
   - 无论当前处于第几百步，滑动窗口的开头永远固定为 `[System Prompt, Root Crack User Prompt]` + 最近 $W=8$ 步交互历史，彻底杜绝无上下文的孤立窗口。
2. **剥离历史步骤的 `<think>`（History CoT Stripping）**：
   - 窗口内前序步骤的 Assistant 消息强制设置 `strip_thinking=True`（仅保留 `tool_calls` 与 `toolResult`），**只在当前预测的目标步保留清洗后的 `<think>` + `<tool_call>` 并计算 Loss**，完全对齐真实推理时的上下文分布。
3. **阶段均衡过采样权重（Stage Balancing）**：
   - `opening`（前 3 步侦察立项）：`sample_weight = 3`
   - `key_decision`（并发调用、写解析脚本、`python -c` 验证）：`sample_weight = 2`
   - `closing`（最后 3 步全盘核验与写报告）：`sample_weight = 3`
   - `routine`（常规查看步）：`sample_weight = 1`

---

## 三、 多游戏混合训练的三大铁律

1. **跨游戏全局打乱（Global Shuffle）**：必须将 `jigsawcard`、`meowdoku` 及后续新游戏的单步样本全局打乱混合训练，严禁按游戏顺序先后训练，防止优化器动量偏向单一引擎导致灾难性遗忘。
2. **按“完整游戏（`project_group`）”切分验证集（Held-out Games）**：严禁按步骤随机抽样切分 Train/Eval（同一游戏相邻步共享上下文历史，随机切分会导致严重的数据泄漏）。必须按 `metadata.project_group`（如 `game::jigsawcard` vs `game::meowdoku`）整包留出验证集。
3. **开启长度分桶（`group_by_length: true`）**：将相近长度的跨游戏步骤拼入同一 Batch，消除 Padding 浪费。

---

## 三·五、 程序与模型的职责边界（判断权归属）

本节把"谁有权做决定"写死，避免实现漂移。

| 环节 | 程序（确定性） | 模型（判断） |
| :--- | :--- | :--- |
| 解析与切块 | 解析 `.jsonl`、按 user 切块、组窗口、拼 payload | — |
| 会话准入 | 目录前缀匹配、同源副本去重 | 回合语义分类（root/task/setup/noise/continuation/correction） |
| 去噪与剪枝 | — | **删哪些**：弯路 B/C 类、噪音、假阴性死胡同；D 类科学排错必须保留 |
| 状态折叠 | 执行合并（write+edit 同路径） | **要不要折叠**该步 |
| 空 thinking | — | **剪除 / 用播报文本回填 / 依据工具参数合成** |
| CoT 压缩 | 执行替换并留 `raw_thinking` | **压到多少、删哪些自我推翻**（300~1600 字、三段式） |
| 静默执行 | 机械清空中间步对外文本（手册强制规则） | — |
| 轮询规范化 | 正则改写 `nohup+sleep+tail` | — |
| 接缝 | 机械提取"哪里被切断"（被删内容 + 前后相邻步） | **接缝是否平滑**；不平滑给最小补丁（只改接缝后那步，禁新增事实） |
| 质量报告 | 抽样、计数、写盘 | **值不值得训练**；需要整条重写 -> 报告并放弃，**不许重写救回** |
| 高光 | 按模型给的高光步设置过采样权重 | **哪些是关键决策步**及理由 |
| DPO | 组装 chosen/rejected、执行 swap/drop | **这对是否成立**（风格/长短差异不算正误） |
| 校验与门控 | 不变量（首步不可删、剪除量 ≤60%）、血缘、计数、切分 | 质量否决（`discard` / `needs_wholesale_rewrite` 直接不产样本） |

两条铁律：
1. **模型只输出"决定 + 最小补丁"，程序确定性执行并记录 `by/reason/patch` 血缘**；模型调用失败时程序保持原样（失败只退化为保守，绝不误删）。
2. **"全量重写"不是可用手段**。若一条会话需要整条重写才合格，说明它质量不足，正确动作是由模型报告并放弃训练，而不是改写救回。

---

## 四、 当前纯净版工程分析数据集产出统计（`output/`）

运行 `python sync_pi_datasets.py --force-rebuild` 后：
- **已扫描并准入项目会话**：`11 个`（自动忽略其余 `95 个` 非目标杂音会话，并在 11 个项目会话内部自动剥离了 `45 个` 模型配置/`hi`/硬件闲聊回合）；
- **`sft_gold_trajectories_sharegpt.jsonl`（4 条完整黄金长轨迹）**：
  - `gold_traj::01a15375`（`jigsawcard` DeepSeek，257 步纯净自主轨迹）
  - `gold_traj::01a15341`（`jigsawcard` GLM，209 步纯净自主轨迹）
  - `gold_traj::01a1589b`（`meowdoku` DeepSeek，163 步纯净自主轨迹）
  - `gold_traj::01a158b3`（`meowdoku` GLM，54 步纯净自主轨迹）
- **`sft_step_windows.jsonl`（620 条高纯度单步窗口样本）**：100% 锚定根破解指令、100% 剥离历史 `<think>`、不含任何空思考或无关提问；
- **`dpo_preference_pairs.jsonl`（270 对工程分析专属偏好对）**：涵盖状态折叠代码修复对、手滑命令剪枝对、心算十六进制打勾反例、万字反刍压缩对与跨目录违规负例；
- **`negative_diagnostics.jsonl`（449 条工程分析病态诊断记录）**。

---

## 五、 工具调用格式契约（Tool-Call Format Contract，2026-10 27B 事故复盘）

> **事故**：`Qwen3.8-27B` 用 `template: qwen3_8` 训练本数据集后，工具调用格式混乱——
> 同一会话里时而输出 `<tool_call>{"name":"bash",...}</tool_call>`（能被服务端 `hermes` parser 解析，正常干活），
> 时而输出 `<tool_call><function=bash><parameter=command>…</parameter></function></tool_call>`（parser 解析不了；
> 流式下这段被服务端直接丢弃，`finish_reason=stop` + 正文只剩 `\n\n` → pi 认为回合结束 → **答一句就停工**）。
> 而 8B 用 `template: qwen3` 训练同样数据，**没有**出现该问题。

### 5.1 根因：数据与模板互相打架

| 环节 | 实际值 | 与数据集是否一致 |
| :--- | :--- | :--- |
| 数据集 `gpt` 步 / `tool_calls` | `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>`（hermes JSON） | — |
| 8B 训练 `template: qwen3` | system 注入「return a **json object** … within `<tool_call></tool_call>`」 | ✔ 一致 |
| **27B 训练 `template: qwen3_8`** | system 注入「ONLY reply in the following format：`<function=…><parameter=…>`」 | ✘ **冲突** |
| 推理 `tools/serve_vllm.sh` | `TOOL_PARSER=hermes`（认 JSON） | ✔ 与数据一致 |

同一条训练样本里，system 要求写 XML、示范答案却是 JSON。模型学到的是「两种都行」，
推理时随机二选一 —— 恰好一半机会触发服务端解析失败。**这不是模型质量问题，是协议没统一。**

### 5.2 三条铁律

1. **三方一致（SSoT）**：`数据渲染格式` == `训练 chat template 注入的格式` == `推理端 --tool-call-parser`。
   换任何一个，另外两个必须同步改；不允许「数据是 JSON、模板教 XML、服务端解析 XML」这种错位组合。
   - hermes JSON ⇄ `--tool-call-parser hermes` ⇄ LLaMA-Factory `template: qwen3`
   - coder XML ⇄ `--tool-call-parser qwen3_coder` ⇄ `template: qwen3_8`（及 Qwen3-Coder 系）
2. **优先用 openai 结构化格式训练，不要在 sharegpt 的 `gpt` 文本里硬编码 `<tool_call>` 文本。**
   `messages` + `tool_calls`（`sft_step_windows_openai.*.jsonl`）由 chat template 负责渲染，
   换模型/换模板时数据一行都不用改，天然不会与模板冲突；sharegpt 的文本写法则必须随模板重渲染。
3. **训练后必做格式回归**：固定 prompt 跑 20~50 次工具调用，要求输出格式 100% 单一、可解析率 100%；
   同时用 `tools/check_toolcall_contract.py` 在训练前卡住不一致（会同时校验 JSON 合法性、`name` 是否在 `tools` 内、
   `arguments` 是否为 object、是否混入裸 `<function=…>`）。

### 5.3 其它容易踩的格式细节

- **`` 后不要插空行**：数据前缀目前是 `…\n\n` + 调用块，模型因此学会在正文开头先吐 `\n\n`
  （推理实测正是 `content="\n\n"`）。改成 `…\n` + 调用块，和推理端 reasoning parser 的切分点对齐，
  避免「块不在文本首位」导致的严格正则失配。
- **`tools` 列必须和推理端工具箱完全同构**：名字（`bash`/`read`/`write`/`edit`/`codemode`）、必填项、参数名、
  description 都要一致。训练里教了推理端没有的工具 → 模型幻觉调用不存在的函数。
- **DPO 对里只允许质量差异，不允许格式差异**：若 `chosen` 不带 tool_call 而 `rejected` 带，
  模型学到的是「少调工具」而不是「调得对」。当前 `dpo.*.jsonl` 有 **2 对**属于此类，应剔除或补齐。
- **模板版本同源**：`merge_lora` 之后确认 merged 模型的 `tokenizer_config.json.chat_template`
  与训练时用的是同一份，否则训练/推理看到的注入指令又不一样。
- **推理端 fallback**：服务端解析失败时应退化为「原文落到 content」，而不是在流式路径静默丢弃 token
  （27B 这次就是丢弃，客户端只看到 `\n\n`，问题被伪装成「模型不干活」）。

### 5.4 27B 的正确重训姿势

```yaml
# data/03_train/train_sft_qlora.yaml / train_dpo_qlora_27b*.yaml
template: qwen3        # ← 由 qwen3_8 改回 qwen3（与 8B、与 hermes 数据、与 serve_vllm.sh 对齐）
```

或反向：保留 `qwen3_8`（coder XML），把 `pipeline/common/format.py` 的 `to_sharegpt` 改为渲染
`<function=…><parameter=…>`，同时把 `serve_vllm.sh` 的 `TOOL_PARSER` 改成 `qwen3_coder`——
但改数据要重跑整条管线，不如改模板一行。**已训崩的 merged 模型无法修复，只能按统一协议重训。**

---

## 六、 长度预算与 cutoff_len（2026-10 实测口径）

长度不能再按「字符数 / 2.2」粗估：用真实 tokenizer 校准过一条渲染后的真实样本（18,996 字符 = 8,911 tokens，
2.13 字符/token），粗估会**低估 3~5%**，正好卡在 12K 边界时会误判「没超」。而 LLaMA-Factory 是
`truncation_side='right'`，超长被切掉的恰恰是最后一个 assistant 目标步；叠加 `mask_history: true`
就是「标签全被切光、loss=0」的假训练——loss 曲线好看，能力没学到。

### 6.1 口径（`pipeline/common/seqlength.py`，handoff 与 CLI 共用）

- 按 ChatML 逐轮渲染：`system` / `user` / `<tool_response>` 包装的 observation / `assistant`，末尾 `assistant\n`
  生成位；**不是**把各段 `\n` 拼起来。
- 加上模板注入的固定开销（`Reasoning effort` 行 + `# Tools` + tools JSON + 调用格式指令），
  27B/qwen3_8 实测 **686 tokens/条**。前缀从训练日志的 `inputs:` / `chosen_inputs:` 里抠，跟实际训练完全一致。
- tokenizer 来源优先级：`--tokenizer` → `GT_TOKENIZER` 环境变量 → `/data/models/<模型>/tokenizer.json`。
  找不到时退化为「字符 / 2.13」估算，并在文档里明确标注「估算」，不会冒充实测。

### 6.2 当前数据包实测（548 条 SFT / 94 对 DPO）

| | p50 | p90 | p99 | max | 超 12K |
|---|---|---|---|---|---|
| SFT train | 4,535 | 7,850 | 12,840 | **14,980** | **6 条** |
| SFT eval | 4,620 | 9,187 | 13,222 | 13,495 | 6 条 |

| DPO 文件 | 对数 | max | 超 8K |
|---|---|---|---|
| `*_train.v2w.jsonl`（窗口裁剪） | 94 | 4,190 | 0 |
| `dpo.train.jsonl` / `*_train.v2.jsonl`（全窗口） | 100 | 17,069 | 28 |

超长样本的成因很集中：**单条工具输出太大**（十六进制 dump / 偏移熵统计表），不是历史步数多。
`synth.max_tool_output_chars`（默认 12000 **字符**）就是那个旋钮：降到 4000 后 SFT 最长 12,212（0 条超 12K），
降到 3000 后最长 10,769。

### 6.3 命令

```bash
# SFT 12K 体检（自动找 tokenizer + 从日志抠模板前缀）
python tools/check_seq_len.py --kind sft --limit 12288 --log-dir data/03_train/logs \
  --files sft.train.jsonl sft_step_windows_sharegpt.eval.v2.jsonl

# DPO 8K 体检（prompt+chosen 与 prompt+rejected 取大者）
python tools/check_seq_len.py --kind dpo --limit 8192 --log-dir data/03_train/logs \
  --files dpo_preference_pairs.train.v2w.jsonl dpo_preference_pairs.eval.v2w.jsonl

# 训练机上直接用模型自己的 tokenizer（顺便验证 find_tokenizer 是否命中）
GT_TOKENIZER=/data/models/Qwen3.8-27B/tokenizer.json python tools/check_seq_len.py ...
```

`handoff.md` 的第 2 节现在由 `pipeline/handoff.py::_data_profile` 自动生成，会逐文件列出
v1 / v2 / v2w 各自的长度与超限数，并按最长样本给出 `cutoff_len` 建议值；生成出来的 `check_env.py`
也改成了同样的 ChatML 口径（多了 `--prefix-tokens` 参数，默认 700）。

### 6.4 怎么把 tokenizer 换成「真」的（不用上训练机）

两条路，都接进工具了：

**A. 你给文件（推荐，确定性强）**

```
D:\ai_projects\dataset_sync\.tokenizer\qwen3.8-27b\tokenizer.json     # 首选，只这一个文件就够
# 或：vocab.json + merges.txt (+ tokenizer_config.json)
```

`find_tokenizer()` 会**优先**扫描这个目录（在 `/data/models/...` 之前），说明见 `.tokenizer/README.md`。
来源：本地模型目录直接拷 / `modelscope download --include 'tokenizer*'` / HF 单文件下载 / 找有模型的人要一个。

**B. 你给我一个端点（能服务同一份模型即可）**

```bash
python tools/check_seq_len.py --kind sft --limit 12288 \
  --via-api http://106.75.44.136:8000/v1 --api-model qwen38-dpo --api-key gt-2026 \
  --log-dir data/03_train/logs --verify --files sft.train.jsonl
```

- 优先走 vLLM 原生 `POST /tokenize`（直接数原始文本，不受 chat template 包装影响）；
  没有该接口时退化为 `/chat/completions` 的 `usage.prompt_tokens` 差值法（先量出包装开销）；
- 结果按文本 sha1 缓存在 `.cache/api_token_counts.json`，重跑零成本。

**验收标准（关键）**：用训练日志里的 `input_ids` 当 ground truth（那是当时训练真 tokenizer 数出来的）：

```
ground truth 校验 27b_probe16k.log: 日志 8,911 vs 本口径 8,923 tokens（偏差 +12 / +0.135%）→ ✅ 同一份分词器
```

- 偏差 ≤ 0.5% → 可用（8B 代理 tokenizer 就在这个水平）
- 偏差 > 2% → 拿错文件/端点不是同一模型，长度数字别用

**实测反例（不要再踩）**：

| 口径 | 同一段渲染文本 | 相对真实值 |
|---|---|---|
| 训练日志（真 27B tokenizer） | 8,911 | 基准 |
| Qwen3-8B 代理 tokenizer | 8,923 | +0.135%（可用） |
| sub2api `/v1` 的 `qwen3.8-27b` | 9,075 | **+1.84%（不可用）** |
| 同上，另一条样本 | 10,341 | **+0.99%（不可用）** |

所以「随便找个 qwen3.8 端点」不行——`--via-api` 会自动用 ground truth 打这个分，偏大直接标 ⚠️。

### 6.5 并行工具调用（一步多调用）的供给与验证

vLLM 侧支持并行输出（`--enable-auto-tool-choice --tool-call-parser hermes`，
一个 assistant 消息里的多个 `<tool_call>` 块会被拆成 `tool_calls[0..n]`；流式按 `index` 递增下发），
前提是：parser 与训练模板一致、**没有**把 `</tool_call>` 配成 stop、`max_tokens` 足够。

真正会掉链子的是数据侧。实测（`tools/check_parallel_calls.py --data`）：

| 文件 | assistant 步 | 每步调用数分布 | ≥2 并行 |
|---|---|---|---|
| `sft.train.jsonl`（jigsawcard） | 2,606 | 0个=43, 1个=2,424, 2个=139 | **5.3%** |
| `sft.eval.jsonl`（meowdoku） | 940 | 0个=5, 1个=689, 2个=243, 3个=3 | **26.2%** |

两个坑：

1. **分布倒挂**：eval 依赖并行的程度是 train 的 5 倍——并行步几乎全在被「按项目整包留出」的 meowdoku 里，
   训练时模型几乎看不到这个模式，训完会退化成「一次一个调用」，而 `eval_loss` 上看不出来。
2. **train 完全没覆盖「3 个并行调用」**（eval 有 3 例）。

对策（已实现）：`synth.parallel_weight`（默认 `1.5`）——对 `tool_calls >= 2` 的步做权重加成。

```yaml
# config/pipeline.yaml
synth:
  parallel_weight: 1.5      # 1.0 = 关闭；只乘不降；highlight 不参与（避免双重计分）
```

- 落点是 `pipeline/stage3_synth.py::apply_parallel_boost`，四舍五入用 `int(x+0.5)`（避开 `round(4.5)==4` 的银行家舍入）；
- 权重以整数写回 `metadata.sample_weight`，同时在 `metadata.parallel_calls` / `metadata.weight_boost`
  与 `step_audits[].weight_before_parallel_boost` 留血缘；
- 单测见 `tests/test_parallel_weight.py`；
- 训练后再用 `tools/check_parallel_calls.py --probe` 验证服务端真能吐 `tool_calls >= 2`（支持 `--stream`）。

若加权还不够（比如想把 3 并行也覆盖到），可以用同一条开关临时调到 `2.0`，或按项目补样本进 train。

### 6.6 历史步要不要保留思考（2026-10 实测推翻旧假设）

旧假设：推理时历史里没有思考，所以 SFT 窗口把历史步的 ` thinking` 剥掉（`strip_thinking=True`）。
实测（pi + vLLM，用 `usage.prompt_tokens` 逐次调用做差分，分量用真实 tokenizer 核）：

| 场景 | Δ prompt_tokens | 含思考估计 | 不含估计 | 判定 |
|---|---|---|---|---|
| 轮内（assistant 带 tool_calls） | 289 | 301 | 171 | **思考带上** |
| 轮内 | 448 | 465 | 426 | **思考带上** |
| 轮内 | 768 | 721 | 597 | **思考带上** |
| 跨轮（上轮有真实文本） | 94（user 57 + text 10） | 29+57 | 10+57 | **思考带上**（残差 8） |
| 跨轮（上轮文本是 `"\n\n"`） | 11 | 98 | 9 | **整条 assistant 消息被丢弃** |

第三条的原因是 pi 的 `convertMessages`：

```js
// content 只保留 trim 后非空的 text block；空文本 + 无 tool_calls 的 assistant 会被整条跳过
if (!hasContent && !assistantMsg.tool_calls) continue;
```

所以真相是：**轮内多步调用时思考 100% 带进上下文；跨轮也带**，除非那一轮的回复是空白
（这正是 qwen38-dpo "只回 `\n\n` 就停" 的副作用——下一轮连它做过什么都看不见）。

结论：**剥离历史思考 = 训练/推理分布失配**。开关已加：

```yaml
synth:
  strip_history_thinking: true      # true=剥离（旧行为）；false=保留（对齐推理）
  history_thinking_max_chars: 600   # 保留时每步思考压缩到多少字符（0=不压缩）
```

翻开关前先算账（不重跑管线、不调 LLM，按 `metadata.msg_id` 回 `02_gold/sessions/*.gold.jsonl` 取思考）：

```bash
python tools/simulate_history_thinking.py --limit 12288 --caps 3000 --history-caps 0 1200 600 300
```

实测结果（`max_tool_output_chars=3000`）：

| 形态 | p50 | p99 | max | 超 12K | 增量 |
|---|---|---|---|---|---|
| 历史剥离（现状） | 4,330 | 9,203 | 10,769 | 0 | — |
| 保留、不压缩 | 6,111 | 14,895 | 17,898 | **20** | +1,911 |
| 保留 + 压到 1200 字符 | 5,227 | 10,444 | 12,388 | 1 | +778 |
| **保留 + 压到 600 字符** | 4,898 | 9,816 | 11,631 | **0** | **+468** |
| 保留 + 压到 300 字符 | 4,644 | 9,620 | 11,230 | 0 | +270 |

推荐组合：`strip_history_thinking: false` + `history_thinking_max_chars: 600` +
`max_tool_output_chars: 3000` —— 守住 12K，同时让历史里"有 ` thinking`"这件事与推理一致
（内容压缩到 600 字符，位置/形态对齐，代价仅 +468 tokens/条）。

改完必须验收形态一致性（同一批训练集不能"有的带有的不带"）：

```bash
python tools/check_thinking_shape.py                       # 默认扫三个文件
python tools/check_thinking_shape.py --expect keep         # 改了开关后按 keep 校验
```

判据：历史带思考比例要么 ≈0%（剥离）要么 ≈100%（保留）；目标步带思考应 100%，
空/极短思考占比 <1%。
