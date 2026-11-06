# 给 AI 助手的工作约定

本目录用于训练 GoldThread 27B 模型（Qwen3.8-27B 的 LoRA SFT + DPO）。所有训练都通过 `autotrain.py` 完成，它会自己选方案、监控、处理已知问题、合并和冒烟。你的工作是：确认现状、启动、盯进度、处理它处理不了的问题、向用户汇报。

背景知识在 README.md（方案、目录规划、约定）；遇到报错先查 docs/TROUBLESHOOTING.md。

## 用户说「训练模型」时

默认是完整流程：SFT → DPO。只说「训练 SFT」或「训练 DPO」就只做那一步。

1. **看现状**：`python autotrain.py status`。
   - 显示「正在运行」：不要再启动，改为跟进度。
   - 显示「[异常]」：按提示排查，再决定是否重跑。
   - 历史里已有成功的 SFT（有「合并：」路径）：问用户是直接做 DPO，还是重训 SFT。
2. **看方案**：`python autotrain.py plans <阶段>`，告诉用户检测到的硬件、将用的方案、是实测还是推算、预计耗时、保留多少样本。有卡被占用时，告诉用户是哪张卡、哪个进程（`nvidia-smi`），由用户决定等还是用 `--gpus` 指定。
3. **启动**（放后台，不要前台阻塞）：

   ```bash
   cd <本目录> && nohup python autotrain.py <阶段> > runs/<阶段>.out 2>&1 &
   ```

4. **跟进度**：每 5–10 分钟运行一次 `python autotrain.py status`。SFT 双卡约 76 分钟，DPO 约 1–2 小时。不需要盯每一步；想看细节就读 `runs/<阶段>.out` 末尾几十行，不要读整个文件（很长）。
5. **结束判断**：`status` 显示「当前没有训练在跑」，且历史最后一行是本次运行。`ok` 表示成功，`failed` 表示失败（看下面「失败处理」）。
6. **SFT 成功后做 DPO**：直接运行 `python autotrain.py dpo`（同样放后台），它会自动用刚合并的 SFT 模型作底座。
7. **汇报**（见下面「完成后汇报」）。

DPO 报「需要 SFT 合并模型」时，它会列出盘上已有的 adapter。如果其中有用户想用的 SFT 结果，用 `python autotrain.py merge --stage sft --adapter <目录>` 导入，不要重训。拿不准是哪一个时问用户。

## 必须先问用户的操作

- 删除任何模型、adapter、运行记录，或清空回收站。磁盘不够时列出占用最大的几项，让用户选。
- 结束不是 autotrain 启动的 GPU 进程。
- 改「不能改的约定」（README 里那一节）：模板、工具调用格式、`train_lowmem.py`、DPO 底座和学习率、LoRA 目标层。
- 改 `data/` 下的任何文件。
- 升级或降级 torch、transformers、LLaMA-Factory 的版本（`setup_env.sh` 按钉住的版本自动装的除外）。

## 失败处理

1. 先看 autotrain 的诊断输出（`runs/<阶段>.out` 末尾）。它已经自动处理过的问题，不要重复处理。
2. 停在「未识别的失败」时：读 `issues/pending.jsonl` 里对应条目和 `runs/<运行名>/attempt-N.log` 末尾，定位根因，然后修复。
3. 修复后用同一条命令重跑。成功后 autotrain 会自动生成一条 `learned_*` 问题，标记为待复核。
4. 复核：用 `python autotrain.py record-issue --id <learned 条目的 id 或新英文名> ...` 补全准确的原因、日志正则和自动动作。修复方式是改代码这类 autotrain 比对不出来的变化时，必须手动登记。正则要能匹配到这类错误，又不能误伤正常日志；可以拿失败日志验证：`python -c "import re;print(bool(re.search(r'<正则>', open('<日志>').read(), re.I|re.M)))"`。
5. 改了问题库后运行 `python autotrain.py issues`，确认 `docs/TROUBLESHOOTING.md` 已更新。
6. 同一个问题修了两次还不行，停下来把现象、日志位置、试过的办法告诉用户。

## 完成后汇报

告诉用户这几项（都在 `python autotrain.py status` 和 `runs/<运行名>/run.json` 里）：

- 用的方案、cutoff、保留了多少样本（`runs/<运行名>/filter_summary_<cutoff>.json`）、用时、GPU 峰值。
- 中途是否自动降级或重试过，原因是什么。
- adapter 路径、合并模型路径、冒烟结果（如 `4/5`）。冒烟不到一半通过时，读 `smoke.log` 说明哪里不对。
- DPO 完成后：部署用的是 `/cloud/cloud-ssd1/models/Qwen3.8-27B-goldthread-dpo-<运行名>/`，部署方法见 docs/DEPLOY.md。

## 不要做的事

- 不要绕过 autotrain 直接调 `llamafactory-cli train` 或 `tools/train_lowmem.py`。需要手动排查时，用 `--dry-run` 生成配置和启动命令。
- 不要直接改生成出来的 `runs/*/attempt-N.yaml`。参数调整放进 `configs/plans.yaml`（方案）或 `settings.yaml`（项目设置与路径：数据集键、模板、合并目录名前缀都在它的 `project` 段）。
- 存储分工不要混：中间数据（过滤数据、checkpoint）放本地盘 `work_dir`（`/data/goldthread/work`）；成果（最终 adapter、合并模型、运行记录）放挂载盘 `/cloud/cloud-ssd1`，训练完挂载盘会被挂到别的机器部署。

## 实测数据回填

跑完一个状态为 `estimated` 的方案后，按 `docs/PLANS.md` 末尾的「方案跑完后更新这张表」更新 `configs/plans.yaml`、`docs/PLANS.md` 和 README 的方案表。

## 修改代码

- `tools/lowmem_dpo.py` 依赖 LLaMA-Factory 的内部实现，升级 LLaMA-Factory 后要先用 `--max-samples 20` 试跑，确认日志里有 `[lowmem-dpo] 已接管`。
- 版本钉在 `autotrain.py` 的 `PINNED` 和 `setup_env.sh` 的 `PINS` 里，两处要同步修改。
- 新增方案只改 `configs/plans.yaml`，autotrain 会自动识别。
