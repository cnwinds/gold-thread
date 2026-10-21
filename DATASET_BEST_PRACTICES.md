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

## 四、 当前纯净版工程分析数据集产出统计（`D:\ai_projects\dataset_sync\output\`）

运行 `python "D:\ai_projects\dataset_sync\sync_pi_datasets.py" --force-rebuild` 后：
- **已扫描并准入项目会话**：`11 个`（自动忽略其余 `95 个` 非目标杂音会话，并在 11 个项目会话内部自动剥离了 `45 个` 模型配置/`hi`/硬件闲聊回合）；
- **`sft_gold_trajectories_sharegpt.jsonl`（4 条完整黄金长轨迹）**：
  - `gold_traj::01a15375`（`jigsawcard` DeepSeek，257 步纯净自主轨迹）
  - `gold_traj::01a15341`（`jigsawcard` GLM，209 步纯净自主轨迹）
  - `gold_traj::01a1589b`（`meowdoku` DeepSeek，163 步纯净自主轨迹）
  - `gold_traj::01a158b3`（`meowdoku` GLM，54 步纯净自主轨迹）
- **`sft_step_windows.jsonl`（620 条高纯度单步窗口样本）**：100% 锚定根破解指令、100% 剥离历史 `<think>`、不含任何空思考或无关提问；
- **`dpo_preference_pairs.jsonl`（270 对工程分析专属偏好对）**：涵盖状态折叠代码修复对、手滑命令剪枝对、心算十六进制打勾反例、万字反刍压缩对与跨目录违规负例；
- **`negative_diagnostics.jsonl`（449 条工程分析病态诊断记录）**。
