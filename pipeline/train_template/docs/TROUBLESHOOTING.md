# 问题库

本文件由 `python autotrain.py issues` 根据 `issues/known_issues.json` 生成，不要手改。
训练失败时 autotrain 会按表中顺序匹配日志，命中后执行「自动处理」；没命中的会记入下方「未识别问题」。

## 训练运行时（自动匹配）

### 低显存 DPO 补丁没有生效

- 编号：`dpo_patch_skipped`；阶段：dpo；命中 0 次
- 日志特征：`跳过 DPO 补丁`、`无法导入 LLaMA-Factory DPO trainer`
- 原因：LLaMA-Factory 的 CustomDPOTrainer.concatenated_forward 结构变了，lowmem_dpo.py 拒绝打补丁，DPO 退回全序列 fp32 logits，必然 OOM。
- 自动处理：停下，人工处理
- 人工处理：把 /opt/LLaMA-Factory 切回已验证版本（0.9.6.dev0 main），或按新源码更新 tools/lowmem_dpo.py 的 expected 关键字与实现。

### 模型不支持张量形式的 logits_to_keep

- 编号：`logits_to_keep_unsupported`；阶段：sft, dpo, eval；命中 0 次
- 日志特征：`没有按 logits_to_keep 裁剪`
- 原因：transformers 版本不对，模型 forward 没有按位置裁剪 logits，低显存路径失效。
- 自动处理：自动安装后重试：`pip install transformers==5.8.0 --no-deps`
- 人工处理：确认 transformers==5.8.0（python autotrain.py env）。

### 缺少线性注意力加速包 flash-linear-attention / causal-conv1d

- 编号：`fla_missing`；阶段：sft, dpo, eval；命中 0 次
- 日志特征：`requires flash-linear-attention`、`Falling back to torch implementation`、`No module named '(fla|causal_conv1d)'`
- 原因：27B 有 48 层 linear_attention，缺这两个包会退回 PyTorch 实现，显存和速度都明显变差，长序列容易 OOM。
- 自动处理：自动安装后重试：`pip install flash-linear-attention==0.4.2 causal-conv1d --no-build-isolation`
- 人工处理：pip install flash-linear-attention==0.4.2 causal-conv1d --no-build-isolation

### 缺少 DeepSpeed

- 编号：`deepspeed_missing`；阶段：sft, dpo；命中 0 次
- 日志特征：`No module named 'deepspeed'`、`requires the deepspeed`、`deepspeed is not installed`
- 原因：ZeRO-3 方案需要 deepspeed。
- 自动处理：自动安装后重试：`pip install deepspeed==0.19.9`
- 人工处理：pip install deepspeed==0.19.9

### flash-attn 不可用

- 编号：`flash_attn_unavailable`；阶段：sft, dpo, eval；命中 0 次
- 日志特征：`No module named 'flash_attn'`、`FlashAttention2 has been toggled on, but it cannot be used`、`flash_attn.*(undefined symbol|is not installed)`
- 原因：flash-attn 没装或与 torch 版本不匹配。它只作用于 16 层 full_attention，换 sdpa 能跑，16K 附近多占约 3-5GB。
- 自动处理：自动带参数重试：flash_attn=sdpa
- 人工处理：YAML 改 flash_attn: sdpa；想装预编译轮子，选 cp312 + torch2.7 + cu12 的 flash-attn release，不要源码编译。

### transformers 版本不对

- 编号：`transformers_version`；阶段：*；命中 0 次
- 日志特征：`cannot import name .* from 'transformers`、`transformers>=?[0-9.]+.*is required`、`Qwen3_5.*(not found|does not exist|Unrecognized)`、`model type `qwen3_5``
- 原因：模型 config 要求 5.8.0.dev0，LLaMA-Factory main 上限 5.8.0，只能用 5.8.0。安装 LLaMA-Factory 时常被覆盖成别的版本。
- 自动处理：自动安装后重试：`pip install transformers==5.8.0 --no-deps`
- 人工处理：pip install transformers==5.8.0 --no-deps；装 LLaMA-Factory 时必须带 --no-deps；不要 pip install -U transformers。

### 多卡通信失败或超时

- 编号：`nccl_failure`；阶段：sft, dpo；命中 0 次
- 日志特征：`NCCL error`、`ncclInternalError`、`ncclUnhandledCudaError`、`Watchdog caught collective operation timeout`、`ProcessGroupNCCL.*[Tt]imeout`
- 原因：本机两卡经 PCIe 主机桥（PHB）互联、像是虚拟机直通，点对点通信可能不可用。
- 自动处理：自动带参数重试：NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1
- 人工处理：带 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 重跑；仍失败就用 --plan 选单卡方案。

### torchrun 端口被占用

- 编号：`port_in_use`；阶段：sft, dpo；命中 0 次
- 日志特征：`EADDRINUSE`、`[Aa]ddress already in use`
- 原因：上一次的进程没退干净或别的任务占了端口。autotrain 每次重试都会换新端口。
- 自动处理：自动带参数重试：
- 人工处理：ps aux | grep train_lowmem 确认没有残留进程。

### GPU 显存不足（CUDA OOM）

- 编号：`cuda_oom`；阶段：sft, dpo, eval；命中 1 次，最近 2026-11-06 03:57:16
- 日志特征：`CUDA out of memory`、`OutOfMemoryError`、`cuMemAlloc failed`、`Allocation on device`
- 原因：当前方案的 cutoff_len 对这张卡太长。梯度累积不影响峰值。
- 自动处理：自动换下一个方案
- 人工处理：按 configs/plans.yaml 的 on_oom 链降级；单卡 DPO 还可设 LOWMEM_DPO_CHUNK=512。

### 进程被系统 OOM Killer 杀掉（主机内存不足）

- 编号：`system_oom_killer`；阶段：*；命中 0 次
- 日志特征：`__exit_sigkill__`、`Killed\s*$`、`exitcode\s*:\s*-9`
- 原因：没有 CUDA OOM 却被 SIGKILL，通常是主机内存不够：ZeRO-3 offload 要把整份权重放内存，合并 27B 峰值约 60GB。
- 自动处理：停下，人工处理
- 人工处理：free -g 看内存；加内存或 swap，或换不带 offload 的方案（--plan）。合并时保证可用内存 ≥ 64GB。

### lora_target 写成了 mlp

- 编号：`lora_target_mlp`；阶段：sft, dpo；命中 0 次
- 日志特征：`Target module Qwen3_5MLP`、`Qwen3_5MLP.*is not supported`
- 原因：qwen3_5 的 mlp 是容器模块，peft 不接受。
- 自动处理：自动带参数重试：lora_target=gate_proj,up_proj,down_proj
- 人工处理：lora_target: gate_proj,up_proj,down_proj

### YAML 里多写了 tool_format

- 编号：`tool_format_conflict`；阶段：sft, dpo, eval；命中 0 次
- 日志特征：`Error.*tool_format`、`tool_format.*(not supported|conflict|invalid)`
- 原因：qwen3_8 模板已内置工具格式，再写 tool_format 会报错。
- 自动处理：停下，人工处理
- 人工处理：删掉 configs/*.yaml 里的 tool_format。

### 数据集键找不到

- 编号：`dataset_undefined`；阶段：sft, dpo, eval；命中 0 次
- 日志特征：`Undefined dataset`、`Cannot find valid samples`、`dataset_info.json.*(not found|No such file)`
- 原因：YAML 的 dataset 键与 data/dataset_info.json 不一致，或过滤后没有样本。
- 自动处理：停下，人工处理
- 人工处理：python tools/check_data.py；不要改 data/ 里的文件名和 dataset_info.json。

### 模型或 tokenizer 路径无效

- 编号：`model_path_invalid`；阶段：*；命中 0 次
- 日志特征：`is not a local folder and is not a valid model identifier`、`does not appear to have a file named`、`Incorrect path_or_model_id`
- 原因：settings.yaml 里的路径不存在，或模型下载不完整。
- 自动处理：停下，人工处理
- 人工处理：检查 settings.yaml 的 base_model / sft_merged / length_tokenizer；重新下载缺分片的模型（autotrain 开训前会核对分片）。

### 磁盘写满

- 编号：`disk_full`；阶段：*；命中 0 次
- 日志特征：`No space left on device`
- 原因：磁盘写满。本地盘 work_dir 放 checkpoint 和过滤数据；挂载盘放 adapter 和合并模型（每个约 52GB）。
- 自动处理：停下，人工处理
- 人工处理：df -h /data /cloud/cloud-ssd1；本地盘清理 /data/goldthread/work 下失败运行的现场；挂载盘清理旧合并模型（部署只需 DPO 合并模型）。

### 输出目录在只读文件系统上

- 编号：`readonly_fs`；阶段：*；命中 0 次
- 日志特征：`Read-only file system`
- 原因：本机 /model（以及指向它的 /data/models）是只读共享存储，只能放下载的底座。
- 自动处理：停下，人工处理
- 人工处理：settings.yaml 的 work_dir 指到本地盘 /data 下，saves_dir / merged_dir 指到挂载盘 /cloud/cloud-ssd1 下；/model 只读。

### 单次尝试超时

- 编号：`timeout`；阶段：*；命中 0 次
- 日志特征：`__timeout__`
- 原因：超过 settings.yaml 的 timeout_hours。可能是正常长跑，也可能卡死在通信上。
- 自动处理：停下，人工处理
- 人工处理：看 runs/<run>/attempt-N.log 末尾的进度；正常长跑就调大 timeout_hours。

## 训练效果（人工判断）

### DPO 后乱码、think 不闭合、不再调工具

- 编号：`dpo_ref_is_base`；阶段：dpo；命中 0 次
- 原因：reference 成了原始底座：LoRA 做 DPO 时参考模型是关掉 adapter 的底座，底座若是原始 27B，DPO 会把模型往原始分布拽。8B 上 lr 5e-5 也出现过同样崩坏。
- 人工处理：DPO 底座必须是 SFT 合并模型（autotrain 会拒绝用 base_model 做 DPO），不设 ref_model，lr 1e-5。丢掉这次 adapter 重训。

### 另加载一份 8bit/4bit reference 仍然 OOM

- 编号：`independent_ref_oom`；阶段：dpo；命中 0 次
- 原因：27B 单卡放不下 policy 与独立 reference 两套权重和 logits；4bit policy + 4bit ref 实测峰值仍 77GB。
- 人工处理：只用「SFT 合并模型 + 关闭 adapter 当 reference」这一种做法（configs/dpo.yaml 已是）。

### loss 不降，或生成只有工具调用没有思考

- 编号：`loss_flat_or_no_think`；阶段：sft；命中 0 次
- 原因：模板错了（think 被注入两次）或 mask_history / packing 设置不对。
- 人工处理：template 必须是 qwen3_8，mask_history: true，packing: false；python tools/check_data.py --model <底座> 看 think 是否恰好一对。

### 冒烟时一直在思考、不出工具调用

- 编号：`smoke_greedy_stall`；阶段：sft, dpo；命中 0 次
- 原因：贪心解码会卡在长思考；部分验证样本本身思考就长，1024 token 上限不够。
- 人工处理：用采样（temp 0.6 / top_p 0.95 / top_k 20，smoke_infer.py 默认如此），max-new-tokens 给到 1536 以上。

### 训练步正常，一进 eval 就 OOM

- 编号：`eval_oom_during_train`；阶段：sft, dpo；命中 0 次
- 原因：训练中 eval 会额外占激活。
- 人工处理：训练 YAML 保持 eval_strategy: "no"，训完用 python autotrain.py eval。

## 合并与部署（人工判断）

### 推理服务：正文里出现 </think> 或 <tool_call> 原文

- 编号：`vllm_think_in_content`；阶段：serve；命中 0 次
- 原因：vLLM 没加 reasoning parser，或 tool-call-parser 与训练格式不一致。
- 人工处理：vllm serve 加 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder（本数据是 Qwen XML，不是 hermes）。

### 推理服务：24GB 卡加载不了合并模型

- 编号：`vllm_small_gpu`；阶段：serve；命中 0 次
- 原因：bf16 权重约 52GB。
- 人工处理：换 48GB/80GB 卡，或先在大卡上量化成该架构支持的 4bit 再搬；不支持 qwen3_5 的量化工具会直接报架构错误，换工具。

### 合并 adapter 时被杀

- 编号：`merge_cpu_oom`；阶段：merge；命中 0 次
- 原因：合并 27B 时 CPU 内存峰值约 60GB。
- 人工处理：保证可用内存 ≥ 64GB，保留 --export_size 5；不要在内存不足时反复重试把盘写满。

## 未识别问题（待解决）

暂无。

