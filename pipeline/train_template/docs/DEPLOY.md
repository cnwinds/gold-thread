# 合并、评测与推理服务

`python autotrain.py sft|dpo` 成功后会自动合并和冒烟，本文只在需要手动处理或上线时看。

## 合并（autotrain 自动执行）

磁盘不够时 autotrain 会跳过合并，腾出空间后补：`python autotrain.py merge --run <运行名>`。等价的手动命令：

```bash
llamafactory-cli export \
  --model_name_or_path <底座> \
  --adapter_name_or_path /cloud/cloud-ssd1/saves/goldthread/<运行名> \
  --template qwen3_8 --finetuning_type lora --trust_remote_code true \
  --export_dir /cloud/cloud-ssd1/models/Qwen3.8-27B-goldthread-<阶段>-<运行名> --export_size 5
```

- SFT 的底座是原始 27B（`/model/ModelScope/Qwen/Qwen3.8-27B`）；DPO 的底座是 SFT 合并模型，不是原始 27B。
- 合并目录不能放在 `/model` 下（只读）。
- 合并时主机内存峰值约 60GB。
- 只拷 adapter 到新机器是推理不了的：adapter 必须配原来的底座，否则就拷合并后的整模。

## 换机器部署

成果都在挂载盘 `/cloud/cloud-ssd1` 上，训练机本地盘 `/data` 里只有中间数据，不用带走。把挂载盘挂到推理机上：

- 部署只需要 `models/Qwen3.8-27B-goldthread-dpo-<运行名>/`，它是完整模型，不依赖 `/model` 里的底座。
- 挂载点最好也是 `/cloud/cloud-ssd1`。`runs/history.jsonl` 里记录的是绝对路径，挂载点不同只影响 `autotrain.py status` 的显示和 DPO 自动找 SFT 底座，不影响推理。
- 最近一次 DPO 合并模型的路径：`python autotrain.py status` 里「合并：」那一行。

## 冒烟（autotrain 自动执行）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/smoke_infer.py --model <合并目录> --n 5
```

判定标准：思考闭合（生成里恰好一个 `</think>`），并且至少有一个合法的 Qwen XML 工具调用。超过一半样本通过就算通过。必须采样解码，贪心解码会卡在长思考里。个别样本撞到 1536 token 上限，多半只是思考太长，不一定是格式问题。

另外，建议人工抽 10 条看行为：先观察再调用工具、报错后会改命令、不声称心算过偏移或密钥、只在当前目录作业。

## 评测

```bash
python autotrain.py eval                       # 用最近一次 SFT adapter
python autotrain.py eval --adapter <adapter目录>
```

eval loss 高于 train loss 是正常的，验证集是整包留出的项目。

## 长上下文验收

```bash
CUDA_VISIBLE_DEVICES=0 python tools/verify_longctx.py --model <合并目录> --lengths 8192,32768,65536
```

模型原生支持 256K（`rope_scaling` 为空，不要再加 YaRN），但这次 LoRA 只训练到 16K。对外报上下文能力时，只报实测通过的档位；某一档 OOM，那一档就是这台机器的上限。

## 推理服务（vLLM）

vLLM 装在单独的 conda 环境里，不要装进训练环境（会弄坏 torch）。首次启动需要系统里有 `g++`（flashinfer 会即时编译）。

```bash
conda create -n vllm python=3.12 -y && conda activate vllm
pip install "vllm>=0.11"

vllm serve /cloud/cloud-ssd1/models/Qwen3.8-27B-goldthread-dpo-<运行名> \
  --served-model-name goldthread-27b --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

两个 parser 都不能少：`--reasoning-parser qwen3` 把思考拆到 reasoning 字段；`--tool-call-parser qwen3_coder` 对应本数据的 Qwen XML 工具格式（不是 hermes）。加载视觉塔失败或因此 OOM 时，加 `--language-model-only`。

验证：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"goldthread-27b","messages":[{"role":"user","content":"列出当前目录文件"}],
       "tools":[{"type":"function","function":{"name":"bash","description":"run shell",
         "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
       "max_tokens":512}' \
  | python -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('reasoning:', (m.get('reasoning_content') or m.get('reasoning') or '')[:80]); print('tool_calls:', m.get('tool_calls'))"
```

通过标准：`tool_calls` 非空且函数名是 `bash`；reasoning 字段有内容，正文里没有裸露的 `</think>`。A800 80GB、32K 上下文、bf16 下实测约 28 token/s。部署类问题见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 的「合并与部署」一节。
