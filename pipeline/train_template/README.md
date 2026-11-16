# GoldThread 27B 训练

> **本目录是轻量模板**（无模型）。建包步骤见 [TEMPLATE.md](TEMPLATE.md)；目录约定见 [docs/LAYOUT.md](docs/LAYOUT.md)。
> 正式 sft/dpo：开训前按 token 最长样本探测（看 `max_memory_allocated`）；全量最长优先并关掉 shuffle；单卡 DPO 默认 ZeRO-3 参数卸载到 CPU。

> AI 助手请先读 [AGENTS.md](AGENTS.md)：里面是完整的操作流程和需要先问用户的操作。

在 GoldThread 数据上给 Qwen3.8-27B 做 LoRA SFT 和 DPO。所有训练都通过 `autotrain.py` 完成：你只说训练什么（`sft` 或 `dpo`），它负责检查环境、按当前硬件选择单卡或多卡方案、准备数据、启动并监控训练；失败时按问题库自动处理，成功后合并权重并做冒烟测试。

## 快速开始

```bash
cd /cloud/cloud-ssd1/goldthread-train
python autotrain.py env --install    # 第一次：检查环境，缺什么装什么
python autotrain.py plans            # 可选：看硬件检测结果和会选哪个方案
python autotrain.py sft              # SFT：训练 → 合并 → 冒烟
python autotrain.py dpo              # DPO：底座自动取最近一次 SFT 合并模型
```

用哪个 `python` 启动都行：autotrain 会自动切换到 `settings.yaml` 里 `python` 指定的训练环境（本机是 `/usr/local/miniconda3/envs/py312/bin/python`）。训练要跑一两个小时，建议放到后台，再用 `status` 看进度（是否在跑、阶段、进度；进程意外退出也会提示）。同一时间只允许一个训练，重复启动会被拒绝：

```bash
nohup python autotrain.py sft > runs/sft.out 2>&1 &
python autotrain.py status
```

正式 `sft` / `dpo` **默认会先探测**：按 token 最长样本试跑，用训练进程打印的 `max_memory_allocated`（不是 nvidia-smi）对照 `probe_max_peak_gb`，沿降级链找到本机能稳住的 cutoff，再全量训练。全量数据也会按 token 从长到短排，并设 `disable_shuffling`，尽早暴露 OOM。只探测：`python autotrain.py sft --probe`；跳过：`--no-probe`。快速试跑：`python autotrain.py dpo --max-samples 20 --epochs 1`（不合并）。

## 训练方式（自动选择）

| 硬件 | SFT | DPO |
| --- | --- | --- |
| 2 张及以上 80GB 卡 | `sft_ddp`：cutoff 16384，548 条全保留，实测约 76 分钟 | `dpo_z3_offload_10k`：ZeRO-3 + 参数卸载，cutoff 10240，双卡实测约 67 分钟 |
| 1 张 80GB 卡（主机内存 ≥128GB） | `sft_single` 链：探测后常用 6k–12k；某次实测 6k 约 236 分钟 | `dpo_single_offload_10k`：单卡也走 ZeRO-3 CPU 卸载，cutoff 10240；某次实测约 168 分钟、冒烟 5/5 |
| 1 张 80GB 卡（主机内存不足） | 同上 | `dpo_single`：不卸载，cutoff 5120（显存更紧） |
| 80GB 以下 | 不支持（27B bf16 权重约 54GB） | 不支持 |

- 已被其他进程占用（显存超过 2GB）的卡不参与训练。用 `--gpus 0,1` 可以只用指定的卡。
- 爆显存时自动降级，长度优先。多卡：`dpo_z3_offload_10k` → `dpo_z3_offload`（8192）→ `dpo_z3_offload_6k` → `dpo_ddp`（5120）→ 4096。单卡：`dpo_single_offload_10k` → 8k → 6k → 4k；主机内存不够才退到不卸载的 `dpo_single`。DeepSpeed（含单卡卸载）一律走 torchrun。完整降级链见 `configs/plans.yaml`。
- DPO 行若缺 `tools` 且 SFT 全部行共用同一份工具定义，过滤到工作副本时会自动补上（`fill_missing_tools_from_sft`，不改 `data/`）。
- 想手动指定方案就加 `--plan <id>`，例如 `--plan dpo_z3`（8192，去掉最长的一对）。
- 选型依据（显存怎么算、卡间带宽对速度的影响、哪些是实测哪些是推算）见 [docs/PLANS.md](docs/PLANS.md)。

## autotrain 每次做什么

1. **检测硬件**：GPU 型号、显存、是否被占用、卡间是否有 NVLink、主机内存。
2. **检查环境**：核对各包版本是否与已验证版本一致。有问题时自动执行 `setup_env.sh`；flash-attn 不可用时自动改用 sdpa。
3. **检查模型和数据**：底座有没有 `config.json`、权重分片是否齐全；运行 `tools/check_data.py`，核对数据集键、角色交替和 think 标签。
4. **选方案**：从 `configs/plans.yaml` 里，按顺序选第一个硬件满足的方案。
5. **准备本次运行**：按方案的 cutoff 把 `data/` 过滤到 `work_dir/<运行名>/data_<cutoff>/`（`data/` 本身只读），再用 `configs/<阶段>.yaml` 加上方案参数，生成 `attempt-N.yaml`。
6. **启动和监控**：无 DeepSpeed 的单卡直接启动；DeepSpeed（含单卡卸载）与多卡用 torchrun 启动 `tools/train_lowmem.py`。日志写到 `attempt-N.log`，进度和显存峰值写到 `status.json`，超时会结束进程。
7. **处理失败**：用 `issues/known_issues.json` 匹配日志，按命中条目的动作处理：换方案、带参数重试、安装依赖后重试，或者停下等人工处理。默认最多尝试 4 次。
8. **收尾**：把最终 adapter 从本地盘发布到挂载盘，合并得到整模（也在挂载盘），清理本地中间数据；用 5 条验证样本做冒烟测试；把结果追加到 `runs/history.jsonl`。

## 失败了怎么办

问题库里已有的问题会自动处理，每条的原因和处理方式见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

问题库没认出来的失败，按下面的流程走，新问题会被记下来：

1. autotrain 停下，把报错特征、日志位置、当时的配置和包版本记进 `issues/pending.jsonl`，同时列在 TROUBLESHOOTING.md 的「未识别问题」里。
2. 你按日志排查并修复：改 `settings.yaml`、装包、换方案等。
3. 重跑同一条命令。这次成功的话，autotrain 会对比失败那次和成功这次在配置、环境变量、包版本上的差异，自动登记成一条新问题（标记为「待复核」）。下次遇到同样的报错，就直接按这个差异修复。
4. 如果自动学到的条目不准确，或者修复方式是改代码这类比对不出来的变化，就手动登记或改正：

```bash
python autotrain.py record-issue --from-pending <待处理编号> --id <英文短名> \
  --diagnosis "原因" --manual "人工处理步骤" \
  --action '{"type":"retry","env":{"NCCL_SHM_DISABLE":"1"}}'
python autotrain.py issues           # 查看问题库，重新生成 TROUBLESHOOTING.md
```

`--action` 的 `type` 有四种：`next_plan`（换到降级方案）、`retry`（带 `yaml` / `env` 覆盖重试）、`install`（执行 `commands` 后重试）、`abort`（停下并显示人工处理步骤）。

## 不能改的约定

这些约定每条都踩过坑，原因写在问题库里。

- **工具调用格式要三处一致**：数据是 Qwen XML（`<function=...><parameter=...>`），训练模板是 `qwen3_8`（`settings.yaml` 的 `project.template`），vLLM 推理用 `--tool-call-parser qwen3_coder`。改其中一处，另外两处也得跟着改。
- **必须用 `tools/train_lowmem.py` 启动**，不要用 `llamafactory-cli train`。词表有 248320，原生实现会对整段序列计算 fp32 logits，长样本在 80GB 卡上会 OOM。`train_lowmem.py` 只在有监督的位置计算 logits，loss 数学上等价。
- **DPO 的底座必须是 SFT 合并模型**，且不设 `ref_model`，学习率 1e-5。拿原始底座做 DPO 会把模型推崩，autotrain 会直接拒绝。
- **`mask_history: true`，`packing: false`**，`lora_target: gate_proj,up_proj,down_proj`，`enable_liger_kernel: false`。
- **训练中不做 eval**（会额外占显存），训完用 `python autotrain.py eval`。
- **`data/` 只读**。不要改文件名，也不要改 `dataset_info.json`。

## 目录

```
goldthread-train/
├── autotrain.py            唯一入口
├── setup_env.sh            安装/修复环境（可重复执行）
├── settings.yaml           项目设置、本机路径与全局开关（换机器、做新包都只改这里）
├── configs/
│   ├── plans.yaml          硬件 → 训练方案表（含降级链）
│   ├── sft.yaml dpo.yaml eval_sft.yaml   基础训练配置
│   └── ds_z3.json ds_z3_offload.json     DeepSpeed 配置
├── tools/
│   ├── train_lowmem.py     训练启动器（只算监督位置的 logits）
│   ├── lowmem_dpo.py       DPO 的同类补丁（由 train_lowmem 自动加载）
│   ├── filter_by_cutoff.py 按 cutoff 过滤和压缩超长样本
│   ├── check_data.py       数据契约检查
│   ├── smoke_infer.py      合并后冒烟（按 Qwen XML 判定工具调用）
│   └── verify_longctx.py   长上下文验收
├── issues/
│   ├── known_issues.json   问题库（机器可读）
│   └── pending.jsonl       未识别问题（自动生成）
├── docs/
│   ├── PLANS.md            方案选型依据与实测数据
│   ├── TROUBLESHOOTING.md  问题库（由 known_issues.json 生成）
│   └── DEPLOY.md           合并、评测、推理服务
├── data/                   原始数据（只读）+ MANIFEST.json
└── runs/                   每次运行的记录（配置、日志、汇总）+ history.jsonl
```

## 输出目录规划

原则：**中间数据放本地盘，成果放挂载盘**。训练完成后把挂载盘整块挂到别的机器就能部署，不依赖这台机器上的任何东西。

| 存储 | 用途 |
| --- | --- |
| `/model`（只读共享存储） | 只放下载好的底座 |
| `/data`（机器本地盘） | 训练中间数据：过滤后的数据、checkpoint。换机器后不再需要 |
| `/cloud/cloud-ssd1`（挂载盘，118GB） | 成果：最终 adapter、合并模型、本项目目录（含运行记录） |

`/data/models`、`/data/saves` 是指向 `/model` 和挂载盘的旧软链接，配置里一律写真实路径（`settings.yaml`）。

| 内容 | 位置 | 盘 | 大小 |
| --- | --- | --- | --- |
| 原始底座 | `/model/ModelScope/Qwen/Qwen3.8-27B/` | 只读 | 约 52GB |
| 过滤后的数据、训练 checkpoint | `/data/goldthread/work/<运行名>/`（`data_<cutoff>/`、`output/`） | 本地 | 每次几 GB |
| 运行记录：配置、日志、汇总 | `/cloud/cloud-ssd1/goldthread-train/runs/<运行名>/` | 挂载 | 几 MB |
| 最终 LoRA adapter（不含 checkpoint） | `/cloud/cloud-ssd1/saves/goldthread/<运行名>/` | 挂载 | 几百 MB |
| 合并后的整模 | `/cloud/cloud-ssd1/models/Qwen3.8-27B-goldthread-<sft或dpo>-<运行名>/` | 挂载 | 每个约 52GB |
| 运行历史 | `runs/history.jsonl`，或 `python autotrain.py status` | 挂载 | — |

运行名的格式是 `<日期>-<时间>-<阶段>`，比如 `20261106-140210-sft`，三处目录用同一个名字，可以互相对上。`runs/<运行名>/` 下有：`attempt-N.yaml` / `attempt-N.log`（每次尝试的配置和日志）、`run.json`（汇总）、`status.json`（实时进度）、`filter_summary_<cutoff>.json`（过滤统计）、`merge.log`、`smoke.log`；eval 运行还有 `eval_results/`。

一次运行的数据流：

1. 过滤数据和训练都在本地盘的 `work/<运行名>/` 里进行，checkpoint 也写在这里。
2. 训练成功后，把 `output/` 里的最终 adapter（排除 `checkpoint-*`）复制到挂载盘的 `saves/goldthread/<运行名>/`。先写到 `.partial` 临时目录，复制完整后再改名。
3. 用挂载盘上的 adapter 合并，整模直接写到挂载盘的 `models/`。合并失败会删掉写了一半的目录。
4. 发布成功后删除本地 `work/<运行名>/`（`clean_work_on_success`）。失败的运行保留本地现场，方便排查，确认不需要后手动 `rm -rf`。

**磁盘预算**：挂载盘要放两个合并模型（SFT 约 52GB，加 DPO 约 52GB）。autotrain 的保护：

- 开训前，本地盘 `work_dir` 不到 20GB（`min_free_gb_work`）或挂载盘 `saves_dir` 不到 2GB（`min_free_gb_saves`）就拒绝开训，并列出盘上最大的几项。
- 合并前，`merged_dir` 不到 53GB（`min_free_gb_merge`，合并模型实测 51GB 加 2GB 余量）就跳过合并，adapter 照样保留。腾出空间后用 `python autotrain.py merge --run <运行名>` 补合并。
- 部署只需要 DPO 合并模型。DPO 合并完成后，可以删掉 SFT 合并模型腾空间；SFT adapter 还在，需要时可以重新合并。

**拿到别的机器部署**：只需要挂载盘上的 `models/Qwen3.8-27B-goldthread-dpo-<运行名>/`，它是完整的模型，不依赖 `/model` 里的底座。部署方法见 [docs/DEPLOY.md](docs/DEPLOY.md)。

DPO 默认使用最近一次成功 SFT 的合并模型作底座。想换别的，就在 `settings.yaml` 的 `sft_merged` 里填路径，或者加 `--model`。在别处训练出来的 adapter，可以用 `python autotrain.py merge --stage sft --adapter <目录>` 导入：合并后会登记进运行历史，之后的 `dpo` 会自动用它作底座。

## 数据

| 数据集键 | 文件 | 条数 |
| --- | --- | --- |
| `gt_sft_train_lf` | `sft.train.lf.jsonl` | 1132 |
| `gt_sft_eval_lf` | `sft.eval.lf.jsonl` | 77 |
| `gt_dpo_train_lf` | `dpo.train.lf.jsonl` | 178 对 |
| `gt_dpo_eval_lf` | `dpo.eval.lf.jsonl` | 4 对 |

数据是 ShareGPT 格式的单步窗口：历史步没有思考，只有最后一轮 assistant 带 `<think>` 和工具调用。验证集按项目整包留出（meowdoku、riddle），所以 eval loss 高于 train loss 是正常的。observation 里偶尔出现的 U+FFFD 是采集时的编码损坏，不影响训练。各文件的 sha256 见 `data/MANIFEST.json`。

## 用这个目录做新的训练包

项目相关的设置都集中在 `settings.yaml` 的 `project` 段：合并目录名前缀、LLaMA-Factory 模板、底座的 `model_type`，以及各阶段用哪个数据集键。训练配置 `configs/*.yaml` 里的 `dataset`、`eval_dataset`、`template` 都由 autotrain 按这一段自动填写，不用改。

做新包的步骤：

1. 复制整个目录，删掉 `runs/` 里的内容和 `issues/pending.jsonl`。`issues/known_issues.json` 是踩坑记录，保留。
2. 换数据：把新的 jsonl 放进 `data/`，在 `data/dataset_info.json` 里登记，更新 `data/MANIFEST.json`。
3. 改 `settings.yaml`：`project` 段（名称、合并目录名前缀、模板、数据集键），以及路径（底座、`work_dir`、`saves_dir`、`merged_dir`）。
4. 运行 `python tools/check_data.py`，确认数据集键、文件、角色交替都通过。
5. 重新确定方案：`configs/plans.yaml` 里的 cutoff 和样本保留数是按这次数据的长度定的。先 `python autotrain.py <阶段> --dry-run` 看过滤后保留多少条，再用 `--probe` 拿最长的样本实测显存，按结果调整方案表。
6. 改 README 和 `AGENTS.md` 里与项目有关的描述（标题、数据一节、方案表）。

换了底座架构（不是 Qwen3.8-27B）时，`configs/*.yaml` 里的 `lora_target`、`docs/PLANS.md` 的显存模型、`tools/train_lowmem.py` 和 `tools/lowmem_dpo.py` 的补丁都要重新验证。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `python autotrain.py sft --dry-run` | 只生成配置和过滤后的数据，打印手动启动命令 |
| `python autotrain.py dpo --plan dpo_single` | 指定方案 |
| `python autotrain.py sft --gpus 1` | 只用 1 号卡 |
| `python autotrain.py eval` | 用最近一次 SFT adapter 跑验证集 |
| `python autotrain.py merge --run <运行名>` | 补合并（磁盘不够时被跳过的那次） |
| `python autotrain.py merge --stage sft --adapter <目录>` | 导入外部 adapter，合并后供 DPO 使用 |
| `python autotrain.py status` | 运行历史和最近一次的进度 |
| `python autotrain.py issues` | 查看问题库，重新生成 TROUBLESHOOTING.md |
| `python tools/check_data.py --model <底座>` | 用真实模板抽查 think 标签 |

合并、评测、长上下文验收和推理服务见 [docs/DEPLOY.md](docs/DEPLOY.md)。
