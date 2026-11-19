# GoldThread 配置逐项审核 · 2026-11-19

**结论：保留确定性提纯、数据切分和严格交付的能力；项目特例移出全局默认，辅助模型设置下沉，旧评审选项有计划地退役。优先修复的是配置的可信度，而不是再加旋钮。**

基线：本地 `5f4052a` 加当前未提交修复。A＝教师示范→首版数据；B＝学生反馈→新版数据。审核的是“值在什么入口生效、是否适合这两个流程”，不是用现有样本量反推训练最优值。未读取真实密钥、未请求远端模型、未跑 GPU 训练，未改配置或历史数据。

## 1. 应先处理的六件事

|顺序|实际问题与依据|建议及验收|
|---|---|---|
|1 · 写入前校验|Web 的 `validate_config` 基本只解析 YAML。TestClient 实测：拼错的 `synth.window_turn`、`eval_ratio: 1.7`、字符串布尔值均返回 200/ok；非空列表根节点返回 500。`load_config(不存在的显式路径)` 也静默加载代码默认值。|同一份版本化 schema 用于加载、原文保存、结构化保存、UI 和 CLI；检查未知字段、类型（不把字符串当布尔）、范围、正则、跨字段约束。保存前校验**合并后的配置**并检查敏感字段变更；失败返回具体路径，不动旧文件。显式指定不存在的文件应报错。|
|2 · 工作区隔离|显式指定配置后，并非所有下游都使用它。`pipeline/common/llm.py:40–95` 又调用无参 `load_config()`，影响模型目录、缓存、默认调用参数；`pipeline/run.py:_record` 也按默认工作区写日志，且 force 固定记 false。此项是静态调用链核对，不是实际发起 LLM 调用。|端到端传同一个 cfg；用两个临时工作区和假的模型服务测：B 的调用只能读 B 的配置、写 B 的缓存和账本。UI 展示每个有效值的来源（内建/文件/操作覆盖/已锁定）。|
|3 · 改了值是否真的生效|沙箱 3 个黄金步，阈值从 2 改为 99，普通 Stage 2 仍 reused=1、status=gold；强制重跑才变 rejected，但草稿仍有 3 条 SFT。前者对应 `stage2_gold.run` 在 judge 关闭时回退旧会话状态，后者是 Stage 3 依据 curation/步质量而非该 status 准入。|把影响产物的输入拆成每会话指纹；仅打开/关闭远端评审时可复用已审结论，不应因此忽略真正改变提纯的配置。把“短轨迹提醒”和“SFT 硬排除”分开，不建议简单删除所有短而有证据的示范。|
|4 · 教师事实与学生归因|现有根任务归一化会按任务文本**或 cwd**做子串匹配并**整段替换**。实测在命中历史项目名的目录里，一个普通统计请求也被换掉。按模型名推断来源，且 negative 标签参与 Stage 3 的一个 DPO 候选来源。|A 默认保留原任务；项目替换和特殊白名单迁入对应数据版本的策展决策。B 显式关联学生模型、训练包和任务，不凭模型品牌定优劣；未独立复测不得把出新包称为效果提升。|
|5 · 界面优先级倒置|有效管线字段 93 个（列表按一个参数计），其中 YAML 显式 77 个，另外 16 个来自内建。Web 分项表单只有 44 个，其中 **20 个 judge、0 个 gold**；分项未显示的字段并不等于不能经 YAML 设置。GT 主链强制关闭隐式模型，但 UI 仍称关掉 judge 仅用于对比。|首屏只放输入工作区、证据规则、切分与监督门禁，明确显示“这个命令的有效值”；远端辅助单独一栏，模型路由、并发等默认折叠。别把剩余 49 个全塞进主屏。|
|6 · 交付训练与复测|训练包的 settings 默认是某台机器的绝对路径；更换导出 profile 会改模板契约，但不等于换好了底座路径、合并前缀和显存计划。训练入口的最长优先也会覆盖管线阶段的洗牌意图。|把格式契约、训练配方、机器路径分层；发布时校验模型/模板/解析器/tokenizer 配套，训练前检查实际路径、截断率和保留评估样本。两版模型复测固定同一套独立任务与预算。|

## 2. 主配置：全部有效叶字段

清单逐键从 `config/pipeline.yaml`、`pipeline/common/config.py:DEFAULTS/load_config` 和 `web/index.html:CFG_SECS` 对齐。列“来源/界面”：**件**＝当前 YAML 明写，**默**＝仅内建，**表**＝有结构化表单；仅写“件/默”的项可通过 YAML 管理，不一定适合公开。**保留**不等于数值已被实测证明最优；**迁移/退役**均指兼容迁移，不改旧 Build 快照。长规则列表仅列条数，精确内容以当前 YAML 为准。

### paths · 输入和工作区

依据：pipeline/common/config.py:resolve_path/stage_dirs；pipeline/collect.py；pipeline/common/llm.py。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`paths.sessions_dir`|`~/.pi/agent/sessions`|件|保留|自动扫描/学生日志的后备根目录。教师和学生显式目录继续走单次操作参数，不混成一个全局目录。|
|`paths.data_dir`|`data`|件|保留|工作区边界；在首页显示解析后的绝对路径、写权限；切换需确认，不把仓库 data 当所有实验的默认写目标。|
|`paths.pi_models_json`|`~/.pi/agent/models.json`|件|保留·受限|仅辅助调用需要，配置的是路径不是密钥。无鉴权时应只读；不要把凭据复制进配置或报告。|

### llm · 辅助模型

依据：pipeline/common/llm.py:resolve_model/chat；pipeline/turnclassify.py:classify_session；pipeline/judge.py:_call。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`llm.model`|`glm-5.3-flash`|件·表|调整|留作显式辅助模型的兜底，不是负责策展的主 Agent。当前两套默认不同；支持显示实际 provider/model，缺模型时给可操作错误。|
|`llm.temperature`|`0.2`|件·表|保留·高级|这是未显式传温度时的值，judge 的 _call 默认传 0.0。不要标成所有评审的统一温度，也不能承诺输出完全确定。|
|`llm.max_tokens`|`16384`|件·表|保留·高级|输出预算，不是输入上下文；按模型/provider 限额校验。未测调用失败率前不盲调大。|
|`llm.timeout_seconds`|`300`|件·表|保留·高级|保留正值、加超时类型和等待量统计；无需把慢模型等待时长放教师提纯首页。|
|`llm.cache`|`true`|件|保留·高级|模型响应缓存不是 DPO 裁决缓存。建议在工作区隔离、按 provider/模型版本区分后保持开启；不同设置不保证“重跑零成本”。|

### admission · 扫描准入

依据：pipeline/stage1_extract.py:_admitted/run；goldthread/curate_tools.py:adopt_sessions。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`admission.workspace_prefixes`|`["--d--crack-"]`|件·表|迁移|保留自动扫描白名单能力；当前规则针对特定 Windows 目录，移到本地/项目 profile。显式 adopt 不是同一过滤入口。|
|`admission.exclude_folders`|`["--d--crack--"]`|件·表|迁移|保留排除优先级，项目名称本地化；预览中说明命中的规则，防止合法任务被静默漏采。|
|`admission.dedupe_duplicate_prefix`|`true`|件·表|调整|名字误导：不是按 sid 前缀，而是同 cwd + 首条用户文字前 200 字，副本行数小于最长的一半才剔除。建议改名、默认标候选而非替用户判断短学生失败就是副本。|

### labeling · 来源识别

依据：pipeline/stage1_extract.py:label_for；pipeline/stage3_synth.py:synthesize_dpo_candidates。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`labeling.positive_model_patterns`|`["^deepseek","^glm-","^minimax"]`|件·表|迁移|作为历史教师来源的推断规则保留，不是质量准入；同一系列可既当教师又当学生，应由采纳来源优先。|
|`labeling.negative_model_patterns`|`["qwen"]`|件·表|迁移|历史学生来源推断。当前会影响 negative-step DPO 分支；界面“不决定 DPO 去向”须限定为不直接判偏好。|
|`labeling.dominant_ratio_min`|`0.6`|件·表|保留|实际是第一模型占比最低要求，不是“第二模型占比超过”。范围 0–1；模式重叠时当前先匹配 positive，应告警。|
|`labeling.unknown_label`|`pending`|件|收口|固定兼容枚举 pending 即可，无需用户任填标签/目录名；未知是来源不明，不是训练质量差。|

### turns · 回合清洗

依据：pipeline/common/pilog.py:Flows/detect_root_regex/build_flow；pipeline/turnclassify.py；goldthread/assist_tools.py。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`turns.use_model`|`false`|件·表|保留·高级|GT 主链有效值为 false；显式 assist turns 可调用，旧 stage1/all 可遵循 YAML。不要把它当全项目联网总开关。|
|`turns.model_max_turn_chars`|`800`|件|保留·高级|仅回合分类输入截断；长任务约束可能被漏读。保留 800 起点，统计实际截断率。|
|`turns.root_task_patterns`|8 条规则|件·表|迁移|保留规则机制，将领域词表放入领域 profile，通用匹配不要把无关对话当任务。正则须编译校验。|
|`turns.fallback_first_turn_as_root`|`true`|件·表|保留|回退要求非噪音且至少 3 个助手步等条件，并非无条件取第一轮；短而成功的示范应可显式指定根任务。|
|`turns.normalize_root_prompt`|`true`|件|调整默认|通用/新项目建议 false。整段替换会丢用户约束；旧数据保留快照与 raw_root_prompt。|
|`turns.root_normalizations.jigsawcard`|历史任务模板（含本机路径）|件|迁出|具体任务和绝对路径，非产品默认。保留为该历史数据集可追踪的映射；移出 YAML 的同时也要处理内建映射。|
|`turns.root_normalizations.meowdoku`|历史任务模板（含本机路径）|件|迁出|同为单项目整段替换，优先按确切会话/项目确认，不单凭 cwd 子串套话术。|
|`turns.irrelevant_patterns`|15 条规则|件·表|迁移|通用噪音规则和项目词分开；预览命中样例。合法的环境排错也可能命中，不宜无条件丢弃。|
|`turns.irrelevant_whitelist_patterns`|`["破解","\\.apk","\\.xapk"]`|件|保留·高级|保留词表例外机制，具体词迁入领域 profile；显示与噪音/根任务规则的优先级。|
|`turns.continuation_patterns`|21 条规则|件·表|调整|区分纯催促与新增目标/验收条件；后者保留原消息。短答要结合上下文，历史单项目长句移出通用表。|
|`turns.correction_patterns`|`["不要参考其他的目录","你不要看其他的目录","不要看别的目录","专心干自己的目录"]`|件·表|迁移|保留约束纠正的语义和原文；项目特定句不做通用纪律，涉及行为改变时需能审阅。|
|`turns.fold_await_user_messages`|`true`|件·表|保留|有上下文证据的无效等待可折；合理求确认不能仅因无工具调用被删。保留前后对照。|

### gold · 黄金提纯

依据：pipeline/stage2_gold.py:purify/apply_step_decisions/run；pipeline/stage3_synth.py:evaluate_session_synthesis。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`gold.state_folding`|`true`|件|保留·首屏|保留只折有成功回执、可无歧义完整重建的 write/edit；开关描述注明规则范围，不把显式策展动作与自动规则混为一谈。|
|`gold.max_fold_edits`|`8`|件|保留·高级|8 是一次 edit 内的替换块上限，不是可丢的尾部块数；超过上限整对不折，不放宽证据条件。|
|`gold.detour_pruning`|`true`|件|调整|纯手滑与提供诊断信息的失败分开。保留剪枝能力；通用基线建议保守/先出候选，有审核数据后再决定是否默认自动剪。|
|`gold.fix_missing_thinking`|`true`|件|调整语义|不是所有补写的总闸；即使关掉，成功工具步可能仍从已有对外文本回填。区分原有文本迁移与合成意图，后者默认不作为未经确认的 SFT 目标。|
|`gold.silent_intermediate_steps`|`true`|件|按配方|纯工具助手可启用；需要提示风险、征求确认或汇报进度的助手不宜全清。保留原文，把决策写入配方。|
|`gold.max_thinking_chars`|`2400`|件|保留·高级|这是字符不是 token；与下一项联验，分开报告触发了多少压缩以及是否误伤关键信息。|
|`gold.target_thinking_chars`|`1600`|件|保留·高级|与上限联动，0 < target ≤ max。先保留当前基线，依据截断与质量复核再调。|
|`gold.min_gold_steps`|`10`|件|调整定义|改名为短轨迹标记或明确成为准入政策；当前设 10 并不阻止短会话写 gold 或产生 SFT。推荐默认提醒、允许短且成功示范。|

### judge · 显式/旧入口的模型判断

依据：pipeline/judge.py:_jcfg/enabled/_model/_call/decide_steps/audit_quality；pipeline/stage2_gold.py:purify_by_judge；goldthread/assist_tools.py。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`judge.enabled`|`false`|件·表|保留·高级|主链保持 false；对比展示显式 assist、Web/底层兼容入口的有效行为，删除“关闭质量必降”的 UI 断言。|
|`judge.model`|`""`|件|合并兼容|真实存在的中间兜底，表单却不展示。若去掉先迁移非空旧值；保留时按真实继承树显示，不再和 llm.model 重复填。|
|`judge.decide_model`|`""`|件·表|保留·高级|步级调用多，分开选模型有意义。空值是继承，不是禁用。|
|`judge.audit_model`|`""`|件·表|保留·高级|接缝、质量、DPO 的公共兜底。空值先回 judge.model，再到 llm.model。|
|`judge.task_models.seams`|`""`|件·表|保留·高级|仅有特殊成本/质量需求才覆盖，否则继承审计模型。|
|`judge.task_models.quality`|`""`|件·表|保留·高级|同上；质量报告模型与最终训练准入规则不是同一个开关。|
|`judge.task_models.dpo`|`""`|件·表|保留·高级|可指定不同的辅助评审模型；展示实际解析结果及缓存来源，不能替代内容绑定的裁决。|
|`judge.seam_min_silent_chars`|`80`|件·表|保留·高级|控制送审接缝的最小字符量，只优化辅助调用；不是核心数据质量阈值。|
|`judge.step_batch`|`4`|件·表|保留·高级|当前 4，不能仅因注释提到 8 就上调；以漏判、失败率和成本校准。|
|`judge.seam_batch`|`3`|件·表|保留·高级|仅接缝批量；有效范围应由 schema 约束而不是只靠前端滑条。|
|`judge.dpo_batch`|`2`|件·表|保留·高级|影响小模型复核批量，不是训练 batch；缺项/失败要保持待审。|
|`judge.concurrency`|`2`|件·表|保留·高级|当前 2，按 provider 限额设置；SEM 会限制在飞请求，不是与 workers 简单相乘。|
|`judge.session_workers`|`4`|件·表|保留·高级|会话预热并行数。与 cache 联动；缓存关闭时预热后主循环可能重复调用。|
|`judge.max_steps_per_audit`|`30`|件·表|保留·高级|质量审核抽样上限，不等同于逐步全审；展示未抽到的步数。|
|`judge.quality_batch`|`8`|件·表|保留·高级|质量抽样分块；和上项、请求上下文预算一同验证。|
|`judge.item_think_chars`|`1100`|件·表|保留·高级|只截送审文本，不能与 gold 的训练文本长度混称。保留 1100 起点。|
|`judge.fallback_to_rules`|`true`|件·表|调整|仅无可用判断且有失败批时才走整段规则回退，不是所有失败自动重试。质量敏感任务建议失败显示待复核。|
|`judge.decisions.prune`|`true`|件·表|保留·高级|限制辅助模型返回的 prune，不是关闭自动规则或人工/Agent 决策。|
|`judge.decisions.fold`|`true`|件·表|保留·高级|限制辅助模型 fold，成功回执及完整匹配仍是底线。|
|`judge.decisions.rewrite`|`true`|件·表|保留·高级|仅辅助模型改写能力；关闭不等于仓库不再存在其他来源的改写。|
|`judge.decisions.exclude`|`true`|件·表|保留·高级|只过滤辅助模型动作；不应禁用显式的紧急排除。|
|`judge.decisions.clear_text`|`true`|件|退役|_jcfg 收了该字段，但 decide_steps 的合法动作不含 clear_text，提示也明确不让模型返回它。保留显式 curate 的 clear_text 和 gold 的静默规则，别展示无效开关。|

### review · 历史评审兼容

依据：pipeline/common/config.py:DEFAULTS；pipeline/curation.py:resolve；server/app.py:overview。未见下列旧调参项在当前评审执行链消费；仍可能影响整段配置哈希。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`review.enabled`|`false`|默|退役兼容|旧 workflow 开关；看板摘要仍读此值，不代表能启动旧复核队列。|
|`review.labels`|`["positive","pending"]`|默|退役兼容|旧评审选标签，不是 judge.labels；保留旧 Build 的读取能力即可。|
|`review.compress_thinking_over`|`1200`|默|退役兼容|当前压缩由 gold 及显式步级决策负责。|
|`review.max_steps_per_session`|`40`|默|退役兼容|旧复核采样；现行辅助质量审核有独立抽样上限。|
|`review.judge_max_think_chars`|`3000`|默|退役兼容|不是当前 judge.item_think_chars；别同时暴露两套控制。|
|`review.temperature`|`0`|默|退役兼容|旧评审温度不控制现行 judge._call。|
|`review.rewrite_batch_size`|`4`|默|退役兼容|当前辅助步级判断使用 judge.step_batch。|
|`review.max_rewrite_batches`|`6`|默|退役兼容|未见当前执行链消费；历史快照不删除。|
|`review.judge_gold`|`true`|默|退役兼容|不是当前模型报告/黄金准入的开关。|
|`review.judge_min_score`|`60`|默|退役兼容|未见当前准入比较此阈值，不能向用户宣称 60 分为训练门槛。|
|`review.judge_confidence_min`|`0.7`|默|迁移保留|resolve 实际读取！影响结论中默认接受改写，不是“低于阈值整条不训”。迁到 curation 命名、旧键作别名。|

### synth · 训练合成与切分

依据：pipeline/stage3_synth.py:evaluate_session_synthesis/synthesize_dpo_candidates/run；pipeline/dataset_draft.py:resolve_split_plan；pipeline/lf_normalize.py。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`synth.window_turns`|`8`|件·表|保留·常用|实际切的是历史消息索引窗口并重新锚定根任务，不是完整用户轮数。用准确单位/示例，保证调用与结果对齐。|
|`synth.max_tool_output_chars`|`4000`|件|保留·常用|采集解析/回放等也读取，非纯 Stage 3 参数。主配置 4000、内建 12000；按截断率与证据完整性确定基线。|
|`synth.weights.opening`|`3`|件·表|保留·配方|开局 3；按阶段统计逻辑窗口和加权 LF 行，基于学生结果再调。|
|`synth.weights.key_decision`|`2`|件·表|保留·配方|关键步 2；是启发式识别，不直接等同于真实关键决策，和并行加权有叠加。|
|`synth.weights.closing`|`3`|件·表|保留·配方|收尾 3；学生有早停问题时先查目标质量，不要把加大收尾权重当作修复。|
|`synth.weights.routine`|`1`|件·表|保留·配方|普通步 1 为基准；权重须是有效正整数。|
|`synth.parallel_weight`|`1.5`|件|保留·配方|非高光步的多调用权重倍数，经取整；不是训练时要求强行并发。按可比任务做消融，保留 1.0 关闭方式。|
|`synth.strip_history_thinking`|`false`|件|调整默认来源|仓库 YAML 为 false，代码兜底 true；由目标推理模板选择，历史形态在同一配方内一致，勿静默切换。|
|`synth.history_thinking_max_chars`|`600`|件|保留·配方|仅保留历史思考时生效。600 是字符截断，0 表示不截；长度/超限率与目标监督分别验。|
|`synth.sft_allow_subsegments`|1 条：01a15c80 / 50–106|件|移到账本|单 sid 前缀 + 第 50–106 步是历史例外，不是产品默认。改为明确样本/会话版本、原因和范围；旧 Build 继续按快照解释。|
|`synth.shuffle`|`true`|件·表|保留|当前打乱 SFT 窗口/轨迹，不是重新划分项目，也不打乱所有 DPO；训练端可能再覆盖次序。|
|`synth.seed`|`42`|件|保留|参与窗口洗牌和初始切分；已存 Split Plan 不因改 YAML 自动重新分配。分开显示洗牌 seed 与计划 seed。|
|`synth.eval_projects`|`[]`|件|迁移|仅初始草稿的显式留出列表；运行后改走带理由的 curate split。不要形成两套同时生效的真值。|
|`synth.eval_ratio`|`0.25`|件·表|保留·常用|0.25 是初始项目比例，不是精确样本行比例；已有计划要显式 replan。检查非空分片、项目/任务泄漏。|
|`synth.formats.sharegpt`|`true`|件|保留·契约|当前标准交付依赖此格式。选择标准 ZIP 时不应允许关掉再到门禁才报空训练集。|
|`synth.formats.openai`|`true`|件|保留·可选|有消费方时保留；只需 LF 交付时可作为额外导出而非必写。被禁格式不要仍在清单中宣称可用。|
|`synth.dpo.from_pruned_steps`|`true`|件|保留·常用|有真实修复证据才是候选，不是天然合格；交付使用严格内容裁决。|
|`synth.dpo.from_negative_steps`|`true`|件|调整|仅 negative 标签的会话才进入该候选分支。对学生反馈不要因“来自学生”就学作反例；应显式选来源、验证更优一侧。|
|`synth.dpo.from_review_rewrites`|`true`|件|保留|改写前后形成候选；必须同一上下文、真实差异且内容指纹仍有效。|
|`synth.dpo.min_thinking_chars`|`80`|件|保留·高级|80 用于 negative-step 来源筛选，不是所有 DPO 对的质量阈值；建议改名/限定作用域。|

### domain · 领域覆盖（没有写进当前 YAML）

依据：pipeline/common/pilog.py:resolve_system_prompt/resolve_tools；pipeline/common/format.py；pipeline/stage2_gold.py；pipeline/stage3_synth.py。

|字段|当前有效值|来源/界面|结论|理由、动作或边界|
|---|---|---|---|---|
|`domain.profile`|`reverse_engineering`|默|退役或接线|内建 profile 名未见用作实际分发开关。若要引入 profile，必须真能解析到一份可冻结的规则集；否则改为只读说明。|
|`domain.system_prompt`|`""`|默|保留·配方|空值会回退到内建领域 prompt，不代表无 system。通用/跨项目样本应显式设置合适的中性任务上下文。|
|`domain.use_session_system_prompt`|`false`|默|保留·高级|优先尝试源会话 system 消息，找不到再回退；应记录来源，使用前审核是否含私有信息或不适合训练的指令。|
|`domain.tools`|`[]`|默|补全接线|SFT 用带 cfg 的 schema，但 Stage 2 剪枝 DPO 及 Stage 3 的部分 DPO helper 调用未传 cfg，可能回退内建工具表。验收同会话 SFT/DPO 的工具契约一致。|
|`domain.enable_re_pathologies`|`true`|默|保留·领域|Stage 2/3 可关领域病态检查；Stage 1 部分统计仍用质量函数默认值。通用 profile 建议按域启用，页面标明各阶段口径。|

### 配置来源、隐藏键与补全项

- 当前 YAML 与内建有 **11 个已有叶值不同**：`llm.model/max_tokens/timeout_seconds`、`labeling.positive_model_patterns`、`turns.root_task_patterns/irrelevant_patterns/irrelevant_whitelist_patterns/continuation_patterns/correction_patterns`、`synth.max_tool_output_chars/strip_history_thinking`。另有三个 `judge.task_models.*` 是文件加到内建空映射里的键。建议内建只保留安全通用基线，仓库 profile 可以合理不同，但 CLI 应能输出合并后的值和来源；字典深合并，列表整体替换，空字典不能清掉内建映射。
- 文件和内建都没列，但**已有代码支持**：`paths.cursor_sessions_dir`（`pipeline/collect.py`，供 Cursor 采集；建议补注释化示例，不要求每个人填）；`judge.labels`（可选限定旧 judge 对会话来源的适用面，不是全局 SFT/DPO 过滤）；`judge.task_models.steps`（优先于 `decide_model`，建议只留一处主入口，冲突时告警）。`synth.sft_labels` 仅是旧注释，不要恢复成按模型品牌限制 SFT 的开关。
- **建议新增的是政策或元数据，非大量全局 YAML 字段**：① 教师/学生/未知的显式来源及相应会话或任务标识；② 本次运行的严格 DPO 要求、选定 Build、导出 profile 和验收 ZIP 的可查询记录，复用已有 CLI 和账本，不新增第二套独立开关；③ 学生实测的 `model_run_id → 训练包 → 留出任务集版本` 关联、任务成功条件、时间/工具预算（保存于评估运行或报告）；④ 规则与有效配置的版本，以及修改前后会影响哪些阶段/样本。以上是**建议**，不是声称已存在这些新字段。
- 形成一个不可弱化的发布策略：首版和迭代版共用 `build --require-reviewed-dpo` 与成品 ZIP 门禁；基础门禁、原始副本不可覆盖、冻结包不可改父指针都不是适合让新手全局关闭的调参项。

## 3. 交付包与训练机配置（独立于管线 YAML）

### 3.1 `pipeline/train_template/settings.yaml` 的 29 项

消费方：`autotrain.py:stage_keys/choose_plan/build_yaml/find_safe_plan`、解释器重启入口、`setup_env.sh`、`tools/check_data.py`；打包时 `pipeline/package.py:_patch_settings` 会改模板与可用 DPO 数据项。不要从 Web 管线配置直接改学生机器的路径。

|字段|模板值|去留/调整|
|---|---|---|
|`project.name`|`GoldThread`|降为显示元信息；不以这个名字确定数据或底座，未见 autotrain 据此选型。|
|`project.merged_prefix`|`Qwen3.8-27B-goldthread`|保留；应从实际底座/包版本派生，避免不同学生输出都叫 27B。|
|`project.template`|`qwen3_8`|保留但由导出 profile 写入；必须与数据工具编码、推理解析器配套。|
|`project.model_type`|`qwen3_5`|保留作兼容检查；当前不符主要告警，生产可设为开训前需确认。|
|`project.datasets.sft.train`|`gt_sft_train_lf`|保留为包内数据契约；由 manifest 生成/验证，不给全局页手填。|
|`project.datasets.sft.eval`|`gt_sft_eval_lf`|保留；开训前确认与 train 无项目/任务泄漏。|
|`project.datasets.dpo.train`|`gt_dpo_train_lf`|按包是否有 DPO 启用；不能用 SFT 字段代替。|
|`project.datasets.dpo.eval`|`gt_dpo_eval_lf`|保留，明确一个分片缺失时是否跳过 DPO 阶段，而不是显示可以无条件开训。|
|`project.datasets.eval.eval`|`gt_sft_eval_lf`|保留用于包内验证/冒烟；不能代替独立学生实测任务。|
|`python`|`/usr/local/miniconda3/envs/py312/bin/python`|迁到机器配置；autotrain 会尝试按此解释器重新执行，不存在时继续用系统 Python，应回显实际版本。|
|`base_model`|`/model/ModelScope/Qwen/Qwen3.8-27B`|必留，改成部署时明确选择并记录模型版本；仅选导出风格不等于已选对底座。|
|`sft_merged`|`""`|保留；空值可自动查最近 SFT 合并物。正式 DPO 建议显式锁定证据明确的 SFT 运行版本。|
|`work_dir`|`/data/work/_template`|迁到机器配置，按包/运行隔离；预检可写、空间和清理边界。|
|`saves_dir`|`/cloud/cloud-ssd1/packages/_template/saves`|迁到机器配置；确保 checkpoint 路径按版本隔离。|
|`merged_dir`|`/cloud/cloud-ssd1/packages/_template/models`|迁到机器配置；检查磁盘和同名覆盖。|
|`length_tokenizer`|`/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct`|保留并锁版本；训练机不存在时可以回退到底座。与打包的软长度统计区分，记录实际采用者。|
|`llamafactory_dir`|`/opt/LLaMA-Factory`|接线或弃用。模板里有键，但安装脚本用 LF_DIR，未见 autotrain 读取该 YAML 键传给安装脚本。|
|`min_free_gb_work`|`20`|保留；按数据/缓存估算而非永久固定 GB。|
|`min_free_gb_saves`|`2`|调整；2 GB 不能证明可以保存特定 adapter/checkpoint，结合 save_total_limit 与实际磁盘检查。|
|`min_free_gb_merge`|`53`|调整；53 GB 是当前机器/模型假设，按底座大小、精度和临时空间派生。|
|`clean_work_on_success`|`true`|保留；只可清当前运行的工作目录，复现所需配置和过滤摘要应保留。|
|`max_attempts`|`4`|保留作重试上限；区分 OOM 降级与确定性配置错误。|
|`auto_fix_install`|`true`|建议通用默认 false、显式授权才允许安装；可重复训练需要锁依赖与来源。|
|`gpu_busy_mb`|`2048`|保留为本机策略；2048 是阈值不是 GPU 可用性保证。|
|`timeout_hours`|`8`|保留为资源预算；依数据规模可覆盖，应记录实际值和超时原因。|
|`probe_before_train`|`true`|保留 true；最长样本探测不等于整轮训练一定成功。|
|`probe_max_peak_gb`|`70`|按 GPU profile 配；固定 70 GB 只适合目标显存档，建议相对余量并显示阈值。|
|`train_longest_first`|`true`|拆开“探测最长优先”和“全量训练顺序”。当前 true 会在正式训练写 disable_shuffling；评估两版时务必固定，而非以为 synth.shuffle 决定最终顺序。|
|`fill_missing_tools_from_sft`|`true`|条件保留；仅同一份已验证共享 schema 时补，混合契约不猜。|

### 3.2 `configs/{sft,dpo,eval_sft}.yaml`（按同名键对照）

`—` 表示未写入对应模板；`null` 槽位由 `autotrain.py:build_yaml` 等运行逻辑填充。这些是训练机上的配置建议，未用真实训练验证超参数优劣。

|字段|SFT|DPO|Eval SFT|去留/调整|
|---|---|---|---|---|
|`model_name_or_path`|`null`|`null`|`null`|运行时派生：锁定实际底座或指定 SFT 合并物。|
|`trust_remote_code`|`true`|`true`|`true`|按模型且仅对已审核、锁定的版本启用；不作通用默认。|
|`flash_attn`|`fa2`|`fa2`|`fa2`|按机器和软件栈选择并记录，非所有设备支持 fa2。|
|`enable_liger_kernel`|`false`|`false`|`false`|保持已验证基线，变更前独立验兼容与数值一致性。|
|`stage`|`sft`|`dpo`|`sft`|保留框架契约；eval 使用 sft 阶段做评估，不是训练新模型。|
|`do_train`|`true`|`true`|`false`|保留阶段模板值，不做顶级用户开关。|
|`finetuning_type`|`lora`|`lora`|`lora`|保留当前 LoRA 配方；更换训练方式属于新配方。|
|`lora_rank`|`32`|`32`|`32`|保留 32 作基线；依据容量与实测再改。|
|`lora_target`|`gate_proj,up_proj,down_proj`|`gate_proj,up_proj,down_proj`|`gate_proj,up_proj,down_proj`|绑定模型架构；验证目标层确实存在，不建议跨底座照搬。|
|`dataset_dir`|`null`|`null`|`null`|运行时派生为 cutoff 过滤后的版本目录，并留过滤摘要。|
|`dataset`|`null`|`null`|—|派生自 project.datasets 对应的 train 键。|
|`eval_dataset`|`null`|`null`|`null`|派生自对应 eval 键；不是独立实测任务集。|
|`template`|`null`|`null`|`null`|运行时从交付契约派生，勿由多个文件分别维护。|
|`cutoff_len`|`null`|`null`|`null`|由 plan 决定，开训后记录实际降档 cutoff 和被丢样本数量。|
|`packing`|`false`|`false`|`false`|建议保留 false；改变拼接方式需重验上下文边界。|
|`mask_history`|`true`|—|`true`|保留 true，保护单步监督口径；和是否在历史中保留思考是两件事。|
|`overwrite_cache`|`true`|`true`|`true`|可保留开发默认；生产按输入指纹复用，防陈旧缓存亦避免无谓重建。|
|`preprocessing_num_workers`|`8`|`8`|`8`|保留为机器资源项，8 不是所有主机基线。|
|`output_dir`|`null`|`null`|`null`|派生为 run/attempt 唯一路径；不写到共享默认目录。|
|`logging_steps`|`5`|`5`|—|保留；短数据集不应一个阶段都没有有效观测。|
|`save_steps`|`100`|`25`|—|保留分阶段默认；检查与总步数和磁盘是否匹配。|
|`save_total_limit`|`2`|`2`|—|保留保留数；与最佳模型及恢复策略一并说明。|
|`plot_loss`|`true`|—|—|可选诊断，不属于交付验收硬条件。|
|`overwrite_output_dir`|`true`|`true`|`true`|仅限明确隔离的新运行目录，恢复/复训前检查已有产物。|
|`per_device_train_batch_size`|`1`|`1`|—|保留；与计划中的 global_batch、进程数联合验算。|
|`gradient_accumulation_steps`|`4`|`4`|—|正式训练由 grad_accum(plan,nproc) 重算，模板值不是真实固定值；记录取整后的有效全局 batch。|
|`learning_rate`|`0.0001`|`0.00001`|—|保留分阶段配方；比较 V1/V2 数据效果时固定。|
|`num_train_epochs`|`3`|`3`|—|保留，可能被探测模式或命令行覆盖。|
|`lr_scheduler_type`|`cosine`|`cosine`|—|保留并固定在比较实验中。|
|`warmup_ratio`|`0.05`|`0.05`|—|保留；检查对应的实际 warmup 步数。|
|`bf16`|`true`|`true`|`true`|保留为当前硬件配方；执行前检查支持。|
|`gradient_checkpointing`|`true`|`true`|—|保留为显存/速度取舍，和最长样本探测联动。|
|`per_device_eval_batch_size`|`1`|`1`|`1`|保留；不可用训练 batch 推断评估峰值。|
|`eval_strategy`|`no`|`no`|—|当前训练阶段不自动定期评估；保留独立 eval，但有 eval_dataset 不等于训练已验证效果。|
|`pref_beta`|—|`0.1`|—|保留 DPO 训练配方；固定对照后再调，不是数据筛选阈值。|
|`pref_loss`|—|`sigmoid`|—|保留 DPO 配方和实际损失类型。|
|`adapter_name_or_path`|—|—|`null`|评估时派生，必须指向本次评估目标。|
|`do_eval`|—|—|`true`|保留显式 eval 任务开关。|

### 3.3 `configs/plans.yaml`：每个方案是否保留

默认顺序是策略的一部分：取首个 `auto_select=true` 且显存、卡数、RAM 匹配的方案。`verified/estimated` 是配方作者的历史标注，并非本次审核在该机器上的验证。

|方案|自动选择|cutoff / global batch|OOM 后继|建议|
|---|---|---|---|---|
|`sft_ddp`|true|16384 / 8|sft_ddp_10k|保留为对应硬件候选；不要把 75 GB 门槛推广到其他模型。|
|`sft_ddp_10k`|false|10240 / 8|终止|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`sft_single`|true|12288 / 4|sft_single_10k|保留为对应硬件候选；不要把 75 GB 门槛推广到其他模型。|
|`sft_single_10k`|false|10240 / 4|sft_single_8k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`sft_single_8k`|false|8192 / 4|sft_single_6k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`sft_single_6k`|false|6144 / 4|终止|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_z3_10k`|false|10240 / 4|dpo_z3_offload_10k|仅保留手选/历史对照；不提升为自动首选。|
|`dpo_z3`|false|8192 / 4|dpo_z3_offload|仅保留手选/历史对照；不提升为自动首选。|
|`dpo_z3_offload`|false|8192 / 4|dpo_z3_offload_6k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_z3_offload_6k`|false|6144 / 4|dpo_ddp|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_z3_offload_10k`|true|10240 / 4|dpo_z3_offload|保留为对应硬件候选；不要把 75 GB 门槛推广到其他模型。|
|`dpo_ddp`|false|5120 / 4|dpo_ddp_4k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_ddp_4k`|false|4096 / 4|终止|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_single_offload_10k`|true|10240 / 4|dpo_single_offload_8k|保留为对应硬件候选；不要把 75 GB 门槛推广到其他模型。|
|`dpo_single_offload_8k`|false|8192 / 4|dpo_single_offload_6k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_single_offload_6k`|false|6144 / 4|dpo_single_offload_4k|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_single_offload_4k`|false|4096 / 4|终止|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`dpo_single`|true|5120 / 4|dpo_single_4k|保留为对应硬件候选；不要把 75 GB 门槛推广到其他模型。|
|`dpo_single_4k`|false|4096 / 4|终止|保留作明确降档或手选；检查降档后仍有足够训练及验证数据。|
|`eval_single`|true|12288 / 1|终止|保留验证专用方案；不与训练重试合并。|

字段判断：`id` 保留且唯一；`auto_select` 保留但应显示为何命中；`min_gpus`、`min_gpu_mem_gb`、`min_ram_gb` 保留并验证单位；`max_procs` 保留并检查不小于最少卡数；`deepspeed` 引用必须存在；`cutoff_len` 保留且大于零；`global_batch` 要检查除以实际进程数后的有效值；`env`（现有 `LOWMEM_DPO_CHUNK`）仅作为该方案的白名单覆盖；`on_oom` 引用须同阶段、无环；`status`、`note` 保留作已测/估算来源说明，把具体旧数据量、耗时和硬件标成历史基线而非承诺。

### 3.4 `configs/ds_z3*.json`（两个模板的联合叶字段）

|字段|普通 / CPU 卸载|建议|
|---|---|---|
|`train_batch_size`|auto / auto|保留 auto，由训练器与梯度累积推导。|
|`train_micro_batch_size_per_gpu`|auto / auto|保留 auto，避免与 per_device_train_batch_size 分叉。|
|`gradient_accumulation_steps`|auto / auto|保留 auto，服从 plan 派生值。|
|`gradient_clipping`|auto / auto|保留 auto；有实际梯度异常再单独校准。|
|`zero_allow_untested_optimizer`|true / true|现有配方可保留；新优化器要有兼容性验证。|
|`bf16.enabled`|auto / auto|保留 auto，与模型精度配置一致。|
|`zero_optimization.stage`|3 / 3|保留为这两个 ZeRO-3 方案的契约。|
|`zero_optimization.overlap_comm`|true / true|保留当前 profile，性能需按真实互联实测。|
|`zero_optimization.contiguous_gradients`|true / true|保留为显存布局策略。|
|`zero_optimization.sub_group_size`|1000000000 / 1000000000|保留在硬件配方中；不作为用户数据调参项。|
|`zero_optimization.reduce_bucket_size`|auto / auto|保留 auto，若手调需有吞吐和峰值证据。|
|`zero_optimization.stage3_prefetch_bucket_size`|auto / auto|保留 auto；与卸载策略一起测。|
|`zero_optimization.stage3_param_persistence_threshold`|auto / auto|保留 auto，避免两版数据对比混入资源变化。|
|`zero_optimization.stage3_max_live_parameters`|1000000000 / 1000000000|保留硬件配方，超限时先看实际显存。|
|`zero_optimization.stage3_max_reuse_distance`|1000000000 / 1000000000|保留硬件配方，避免在管线 YAML 再复制。|
|`zero_optimization.stage3_gather_16bit_weights_on_model_save`|true / true|保留并验证保存/合并所需磁盘与峰值。|
|`zero_optimization.offload_param.device`|— / cpu|仅卸载方案保留 CPU，需满足 RAM 条件。|
|`zero_optimization.offload_param.pin_memory`|— / true|仅卸载方案保留，观察锁页内存预算。|

### 3.5 不应并入同一个配置页的其他输入

- **服务器与凭据**：保留 `serve.py --host/--port/--no-browser` 为部署参数，默认仅回环；`GOLDTHREAD_API_TOKEN` 用环境秘密，`GOLDTHREAD_ALLOWED_HOSTS` 用部署白名单，不进入语料快照、共享 YAML 或表单回显。弱化本地保护的部署覆盖不应做常用按钮。原文 PUT 和结构化 PATCH 应对**最终完整配置**执行同一权限与路径校验。
- **打包和长度**：保留显式 `--profile`；未指定时还有上次记住的 profile。发布任务应显示具体 profile、模型模板、解析器和数据键，不要再新增一份未接线的 `export_profiles` YAML。`GT_TOKENIZER` 是本地长度工具的候选路径，`GT_ALLOW_REMOTE_TOKENIZER` 是显式联网许可，应分开；严格场景要求本地可用且记录 tokenizer 版本。打包软门限读 Build 中 `train_sft.yaml/train_dpo.yaml:cutoff_len`（缺失时 12288/5120）；训练时硬过滤看 plan，两个口径都应留痕。
- **数据与历史产物**：`data/03_train/*.yaml`、`dataset_info.json`、旧 DeepSpeed 文件、Build 的 `build_config.json`、历史包 MANIFEST/索引/账本均不等于新的全局配置。历史产物应留可回放原貌；当前工作区的训练 YAML 可能影响下一次固化/软长度门限，也不能当“死文件”删除。维护模板与其派生物分开。`tools/ds_z2_config.json`、`tools/ds_z3_config.json` 是旧工具/配方层，不因新模板缺同名项就作删除结论。此报告不清理历史运行、机器环境及未跟踪文件。

## 4. 落地顺序（本轮未实施）

1. **配置可靠性优先**：统一解析/语义校验，显式路径缺失要失败；新增只读“查看有效配置、来源及作用阶段”功能。回归应覆盖原文/表单/CLI 同一份非法输入、写入不生效时不产生半配置、独立工作区隔离和缓存失效。
2. **移走项目特例**：冻结旧实例的等效配置，再将硬编码教师目录、模型族、任务替换、特定 sid 子段映射迁到命名配方/账本。迁移前后对同一沙箱比较 sid、目标步、DPO 指纹、切分和门禁；历史 Build 只读。
3. **重排 UI 和默认政策**：常用栏给 A 的证据型提纯、B 的来源与独立留出、共用的切分及严格交付；模型评审批量/并发归“辅助”；机器路径/凭据归“本机”；兼容键只读告警。A 验收“不伪造修正”，B 验收“改了内容重新评审、未重训不宣布改善”。
4. **最后才调数值**：保持一个可复现基线，每次只变一类参数。先看目标步损失、截断与丢弃数量、项目留出、DPO 未审数，再在独立任务上比较模型表现。没有这些证据，不建议把 0.25、1.5、600 或学习率改成另一个看似更好的常数。

**验证**：叶字段由配置解析后与本报告清单做集合比对；表单键取自 `CFG_SECS`。语义抽查在临时目录合成会话及只校验请求中完成，见 §1。`python -m pytest tests/ -q`：396 passed，1 个 Starlette/AnyIO 弃用警告；`python -m pytest tests/test_readme_alignment.py tests/test_console_ui.py tests/test_train_template_sync.py -q`：18 passed，同一警告。这些测试是当前基线，不表示报告指出的缺陷已经修复。本次只新增审核文档，未新增测试文件或变更运行行为。
