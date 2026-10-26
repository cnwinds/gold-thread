# 本次 8B SFT 实跑记录（2026-10-26）

## 结果

| 项目 | 数值 |
| :--- | :--- |
| 底座 | `/data/models/Qwen3-8B`（16G，5 shard） |
| 数据 | `pi_sft_step_windows_train_v2` 548 条 / eval 221 条（整项目留出） |
| 方法 | LoRA r32 / all-linear / mask_history / bf16 / fa2 + Liger 融合 CE |
| 步数 | 207 步 = 548×3 epoch ÷ (bs2×ga4) |
| 耗时 | 1:15:32（≈21.9 s/step） |
| 显存峰值 | ≈34 GB / 47.4 GB |
| train_loss | 2.04（step5）→ 1.06（最终平均），末段 0.71~0.87 |
| eval_loss | 1.6396（hold-out 项目 meowdoku/riddle，高于 train 属预期） |
| 产物 | adapter `/data/saves/qwen3-8b-goldthread-sft`（350 MB）<br>merged `/data/models/qwen3-8b-merged`（16G） |

## 与 handoff 的两处偏差（都是为了绕开 48G 卡的实际限制）

1. **必须开 `enable_liger_kernel: true`**。
   不开的话第一个 step 就在 `cross_entropy` 处 OOM：8.7K token 样本 × bs2 的
   fp32 logits 一次要 10.1 GB。装的是 `liger-kernel 0.8.5`（`apply_liger_kernel_to_qwen3` 命中），
   融合 CE 后不再实体化 logits，峰值降到 34 GB。

2. **训练中关掉 eval（`eval_strategy: "no"`）**。
   eval 走的是标准 `ForCausalLM` loss 路径（不走 Liger 融合 CE），bs2 时在
   `cross_entropy` 处要 14.5 GB → 冒烟跑到 25/25 步后在 eval 阶段 OOM 挂掉。
   改法：训练只训；训完用 `eval_sft_qwen3_8b.yaml`（bs1 + cutoff 16384）单独跑，
   2 分 40 秒拿到 eval_loss=1.6396，无 OOM。

其他：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 一起带上（减少碎片）。

## 顺带修掉的脚本 bug

`check_env.py:181` / `check_data.py:107` 里统计 `` 的字符串字面量被写成了
空串 `""`（`text.count("")` 恒等于 `len(text)+1`），所以一直报
``think×0 ×16325`` 这种假 FAIL。已改成 `"<" + "/think>"`。
用 LLaMA-Factory 真实模板 `qwen3` 复核：每条样本 `thinking` 与 `` 各 1 次、
`<tool_call>` 数与目标步一致（5/5 抽样）。

## 合并后冒烟（transformers，无 vllm）

`python smoke_infer_merged.py --n 5 --max-new-tokens 1024 --sample`

- 5/5 输出 `thinking…` + `<tool_call>{"name":...,"arguments":{...}}</tool_call>`
- 4/5 JSON 完整可解析（1 条在 1024 token 上限处被截断，非格式问题）
- 未开采样（greedy）时会陷入长思考不收敛，**部署时请用 temp 0.6 / top_p 0.95 / top_k 20**

## DPO（第 3 次才跑对，前两次是配置问题不是卡的问题）

| 尝试 | 配置 | 结果 |
| :--- | :--- | :--- |
| ① 原 yaml | ref = **隐式 base**（LoRA 关 adapter）、lr 5e-5、cutoff 8192 | 训练能跑，但**模型崩了**：输出变成乱码，`think` 不闭合、无 tool_call |
| ② 只改 ref | `ref_model: qwen3-8b-merged`，lr 1e-5 | 显存 OOM（policy 16G + ref 16G + 图） |
| ③ 最终 | ref 量化 8bit + lr 1e-5 + 窗口数据 cutoff 4096 | 23 步 / 7:27（94 对 ÷ bs1×ga4），train_loss 0.704，acc 0.50→0.70，margin +0.50，**格式正常** |

**根因**：LoRA 做 DPO 时，参考模型默认是「关掉 adapter 的 base」，也就是 Qwen3-8B 本身，
而不是我们 SFT 出来的模型。SFT 模型和 base 分布差得远，DPO 就把它往 base 上拽 → 退化。
必须显式给 `ref_model:`（这里用了 merged SFT 权重，8bit 量化省 7G 显存）。

**数据**：原 DPO 数据 prompt 中位 5274 token，LLaMA-Factory 的 `cutoff_len` 截断砍的是
prompt **尾部**（丢掉最近的 observation），所以新写了 `prep_dpo_window.py`：
从头部丢最早的步，保留 system+根任务+最近 4 步，注册成 `pi_dpo_preference_pairs_*_v2w`。

**验证**（eval 前 3 条、采样、1536 token 上限）：

| | #0 | #1 | #2 |
| :--- | :--- | :--- | :--- |
| SFT | 1536 token 内未出 tool_call | 208 token 出 tool_call | 1536 内未出 |
| SFT+DPO | **334 token 出 tool_call** | 650 token 出 tool_call | 1536 内未出 |

两边 think 都正常闭合，DPO 版更简洁、更快进到工具调用。之前的「0/5 全挂」
是**我测试用了 1024 token 上限**，这几条 eval 样本思考本身就长，不是模型坏了。

**顺手修的库 patch**（都在 `/opt/LLaMA-Factory`，另有 `.orig` 备份）：
- `train/trainer_utils.py:get_batch_logps`：log_softmax 按 1024 位置分块 + `checkpoint` 重算，
  峰值从「16k×152k×4B」降到 chunk 级；
- `train/dpo/trainer.py`：`all_logits` 不再整体 `.to(torch.float32)`（那是 9.7 G）。


## OpenAI 兼容服务（远程验收用）

```bash
bash data/03_train/serve_api.sh 8000 gt-2026     # 后台起：nohup bash serve_api.sh 8000 gt-2026 > logs/api.log 2>&1 &
```

| 项 | 值 |
| :--- | :--- |
| Base URL | `http://<本机IP>:8000/v1` |
| 后端 | LLaMA-Factory 自带 API + HF engine（`--infer_backend huggingface`，bf16） |
| model id | `qwen3-8b-goldthread-sft`（`API_MODEL_NAME` 控制） |
| API Key | `gt-2026`（`API_KEY`，用 `Authorization: Bearer gt-2026`；不设则免鉴权） |
| 速度 | 8B bf16 / 4090，约 30~40 token/s，单请求串行 |
| 特性 | `/v1/chat/completions`（含 `stream:true` SSE）、`tools` / `tool_choice` → 返回结构化 `tool_calls`、多轮 tool 结果回填 |

实测（3 轮 agent 循环）：

```
round 1 finish=tool_calls 3.2s  tool: bash | args: {'command': 'mkdir -p work && cd work && apktool d ...'}
round 2 finish=tool_calls 6.0s  tool: bash | args: {'command': 'cd work && mkdir ghidra && cp lib/...'}
round 3 finish=tool_calls 5.0s
```

**踩到的坑（已修，改的是 `/opt/LLaMA-Factory/src/llamafactory/api/chat.py`，有 .orig 备份）**：
LLaMA-Factory 把历史里的 assistant `tool_calls` 直接 `json.dumps` 成 OpenAI 结构后塞进对话，
`FunctionFormatter` 又对 `arguments`（本身是字符串）再 `json.dumps` 一次 → 历史里出现
`"arguments": "{\"command\": ...}"` 双层编码，模型学会模仿，于是第二轮起就把 `<tool_call>`
写成字符串参数，`extract_tool` 解析失败、`tool_calls` 变 null。
做法：入参 `_arguments_object()` 先剥成对象再渲染；出参 `_normalize_arguments()` 兜底剥掉多余编码。

注意：思考过程是**内联在 content 里**的（`thinking...`），没有单独的 `reasoning_content`
（vLLM + `--reasoning-parser` 才会拆出来）。客户端取答案时可直接按 `</`+`think>` 切分。

### 公网映射（2026-10-26 实测）

平台（Compshare / UCloud）用的是**通配符域名 + 端口做第一级标签**：

```
https://<port>-<pod名>-<zone>.pod.compshare.cn  →  pod:<port>
```

Pod 内部只需要**把服务监听到 `0.0.0.0:<port>`**，公网就能用同款域名访问。
安全提示：公网可达 = 谁拿到 pod 名就能访问，靠 `API_KEY` 兜底。

### 客户端兼容补丁（2026-10-26，422 报错修复）

现象：客户端报 `422 {"detail":[{"loc":["body","messages",0,"role"],"msg":"Input should be 'user','assistant','system','function' or 'tool'","input":"developer"…}]}`

原因：LLaMA-Factory 的 `Role` 枚举里没有 `developer`（OpenAI 新版 SDK / o1 系用 `developer` 替代 `system`）。

改动（`/opt/LLaMA-Factory/src/llamafactory/api/protocol.py`、`chat.py`，均有 `.orig` 备份）：

1. `Role` 增加 `DEVELOPER = "developer"`，请求不再 422；
2. `_normalize_messages()`：任意位置的 `system`/`developer` 按序并进开头 system；连续同侧消息
   （`tool+tool`、`user+user`、`assistant+assistant`）合并，满足框架要求 `u/a/u/a` 交替；
   若首条是 assistant 自动补空 user；
3. 支持 `max_completion_tokens`（等同 `max_tokens`）；
4. 其余新字段（`tool_choice` / `parallel_tool_calls` / `response_format` / `reasoning_effort` /
   `seed` / `stream_options` …）pydantic 默认忽略，不会 422。

验证（重启后实测，均 200）：`developer` 角色 → `finish_reason=tool_calls` 且参数正确；
`max_completion_tokens=64` → 64 token 停止；`developer + 连续两条 tool 消息` → 正常；
kitchen-sink（10 个额外字段）→ 正常。

### `/v1` 404 ≠ 端口不通（2026-10-26）

LLaMA-Factory 原本只有 `/v1/models` 和 `/v1/chat/completions`，GET `/v1` 是 404；
而平台 WAF 对**未监听的端口**同样返回 404 —— 两者无法区分，于是会被误判成"公网端口不通"。

已加健康检查路由（无鉴权）：`/`、`/health`、`/healthz`、`/v1`、`/v1/health` → 200，
返回 `{"status":"ok","object":"health","model":"...","endpoints":[...]}`。
改动在 `/opt/LLaMA-Factory/src/llamafactory/api/app.py`（有 `.orig` 备份）。

### 流式 + tools 崩溃 → 已支持（2026-10-26）

原代码里写死了一句 `if tools: raise HTTPException(400, "Cannot stream function calls.")`，
所以客户端只要同时开 `stream=true` 和 `tools`，就必然报错/崩溃。

改法（`/opt/LLaMA-Factory/src/llamafactory/api/{protocol,chat}.py`，均有 `.orig` 备份）：

1. 协议层新增 `FunctionCallDelta`（带 `index`）和 `ChatCompletionStreamMessage`，`delta` 改用后者
   —— OpenAI 流式协议要求 tool_call 分片带 index，原来的模型没有这个字段，序列化就会失败；
2. 流式实现重写：正文照常逐 token 下发，但一旦检测到 `<tool_call>`（含跨 token 被切断的情况，
   用 `len("<tool_call>")-1` 做尾巴缓冲）就**停止把 content 往下发**，避免原始 JSON 混进正文；
3. 生成结束后统一 `extract_tool()` 解析，先发一个带 `tool_calls` 的 delta，再发
   `finish_reason=tool_calls` 的收尾 chunk，最后 `[DONE]`；
4. 如果 tools 在但模型没调工具，则把扣住的正文补发完，正常以 `finish_reason=stop` 收尾。

用官方 OpenAI SDK（openai 3.37.0）实测：

| 场景 | 结果 |
| :--- | :--- |
| 流式 + tools（首轮） | `finish_reason=tool_calls`，参数可 `json.loads` |
| 流式 + 回填 tool 结果（多轮） | 第二轮继续返回 `tool_calls` |
| 流式纯文本（不带 tools） | 正常流完，`stop` 收尾 |

注：工具调用整体放在**一个** delta 里下发（不是逐字符增量），符合协议，SDK 按 `index` 累积即可；
思考仍是内联在 `content` 里。

## Reasoning 标签未被拆分的问题（`防腐剂` 当普通文本）

### 问题

模型输出格式是 `防腐剂推理内容  `，这是 Qwen3 的 thinking 格式。
但推理服务层（LLaMA-Factory API + HF backend）**没有 reasoning parser**——
模型输出被直接塞进 `message.content`，客户端看到的就是原始标签文本。

这不是模型的问题（模型输出格式正确），是**推理服务框架的选择问题**。

### 方案 A（推荐）：迁移到 vLLM

vLLM 是目前最规范的开源推理框架，支持 `--reasoning-parser deepseek_r1`，
可以把 `防腐剂` 标签自动拆分为推理内容。

```bash
bash tools/serve_vllm.sh /data/models/qwen3-8b-merged 8000 gt-2026
```

关键参数：
- `--reasoning-parser deepseek_r1` — 匹配 `防腐剂` 和 ` ` 标签，拆到 reasoning tokens
- `--tool-call-parser hermes` — 结构化 tool_calls
- `--enable-auto-tool-choice` — 自动触发工具调用

vLLM 对 Qwen3-8B 支持完整；**27B（qwen3_5 混合线性注意力）需确认**
vLLM 0.6.x 对该架构的支持状态，若不支持则回退方案 B。

### 方案 B（过渡）：ReasoningProxy 中间件

如果不能/不想迁移 vLLM，在 LLaMA-Factory API 前面加一层代理，
从 content 中切出 `reasoning_content`：

```bash
# Terminal 1：LLaMA-Factory API（跟之前一样）
bash serve_api.sh 8000 gt-2026

# Terminal 2：ReasoningProxy
python server/reasoning_proxy.py \
  --upstream http://localhost:8000/v1 \
  --port 9000 \
  --api-key gt-2026

# 客户端连 :9000，收到规范的 reasoning_content + content 拆分
```

限制：
- 非流式请求 ✅ 完整拆分
- 流式请求 ⚠️ 直接透传（不拆分），因为代理层做流式 reasoning 拆分非常复杂且不可靠
  → 需要流式 reasoning 请用方案 A（vLLM）

### 客户端怎么消费 reasoning_content

拿到规范响应后，客户端按 `message.reasoning_content` 和 `message.content` 分别渲染即可：

```python
# 非流式
msg = resp.choices[0].message
if getattr(msg, "reasoning_content", None):
    print("[思考]", msg.reasoning_content)  # 折叠/灰色渲染
print("[回复]", msg.content)

# 工具调用
if msg.tool_calls:
    print("[工具]", msg.tool_calls[0].function.name, msg.tool_calls[0].function.arguments)
```

## 待办（按建议顺序）

1. **推理服务升级**：当前 LLaMA-Factory API 有 5 个 patch + reasoning 未拆分；建议迁移到 **vLLM**（规范框架，支持 reasoning parser + tool parser）
2. **DPO 是否上线**：DPO 产物只是 adapter，未合并；收益有限
3. 换 27B 时按 handoff §6 改三处，另外 **27B 上 Liger 要关**（qwen3_5 线性注意力不在支持列表）
