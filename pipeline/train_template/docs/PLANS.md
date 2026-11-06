# 训练方案选型依据

`configs/plans.yaml` 是方案的唯一来源，本文解释每个方案为什么这样定。标注「实测」的数字来自本机 2×A800-SXM4-80GB；标注「推算」的还没跑过，跑完后请按文末的方法更新。

## 显存怎么算

| 部分 | 大小 | 说明 |
| --- | --- | --- |
| 27B bf16 冻结权重 | 约 54GB | 每张卡一份；只有 ZeRO-3 会把它分到各卡 |
| LoRA 参数、梯度、Adam 状态 | 不到 1GB | rank 32，只挂在 MLP 上 |
| 激活（随序列长度增长） | SFT 约 1.7GB / 1K token；DPO 约 3.7GB / 1K token | DPO 要把 chosen 和 rejected 拼在一起前向，所以约是 SFT 的两倍 |

反推依据（实测）：

- SFT 单卡 cutoff 12288 时峰值约 76GB，16384 第一步就 OOM。
- DPO 单卡 cutoff 5120 时还剩约 5GB 余量，6144 时峰值 77.9GB，8192 OOM。

由此得出几条结论：

- **ZeRO-2 没用**：它只切优化器状态和梯度，而 LoRA 的这两项本来就不到 1GB。
- **DDP 不省显存**：每张卡的显存和单卡一样，只是两张卡分摊样本，所以更快。SFT 最长的样本约 11.3K token，batch 内按最长样本 padding，cutoff 设 16384 和 12288 的实际显存一样，能覆盖全部样本。
- **ZeRO-3 理论上每卡能腾出约 27GB**，但实测纯 ZeRO-3 跑 SFT 时 13312 就 OOM 了（参数聚合缓冲、显存碎片吃掉了大部分收益）。所以 DPO 的纯 ZeRO-3 方案保守定在 8192。
- **ZeRO-3 + 参数卸载到 CPU** 能把权重整体挪出显存，长度余量最大，代价是主机内存要能放下整份权重（≥128GB，本机 472GB），而且更慢。

## 各 cutoff 保留多少样本（实测）

超长样本先压缩 observation，仍然超长就整条剔除，不会截掉要学的那一步。

| cutoff | SFT 训练（548） | DPO 训练（100） | DPO 验证（143） |
| --- | --- | --- | --- |
| 4096 | — | 46 | 103 |
| 5120 | — | 65 | 126 |
| 8192 | 547 | 99 | 143 |
| 10240 | 548 | 100 | 143 |
| 12288 / 16384 | 548 | 100 | 143 |

DPO 是这个项目里受长度限制最大的部分：单卡 5120 会丢掉 35% 的训练对，这是双卡时优先用 ZeRO-3 跑 8192 的原因。

## 速度

### 卡间带宽（实测）

本机两张卡经 PCIe 主机桥互联（`nvidia-smi topo -m` 显示 PHB），没有 NVLink，看起来是虚拟机直通，卡间没有点对点直连。nccl-tests 的 all_reduce 结果：32MB 以上的消息 busbw 约 4.7–5.0 GB/s；1MB 消息延迟约 0.4 毫秒，只有 2.6 GB/s。

### 估算

| 方案 | 每步/每条耗时 | 依据 |
| --- | --- | --- |
| `sft_single` | 约 20 s/步，3 个 epoch 411 步约 2.3 小时 | 实测 |
| `sft_ddp` | 约 22 s/步，207 步约 76 分钟（全局 batch 8） | 实测 |
| `dpo_single` | 每条约 17 s，65 对 × 3 个 epoch 约 1 小时 | 推算 |
| `dpo_z3_10k`（纯 ZeRO-3） | 最长样本反向时 OOM | 实测 |
| `dpo_z3_offload`（8K） | 比 `dpo_z3` 慢 20–40%，约 1.3–2.2 小时 | 推算 |
| `dpo_z3_offload_10k` | 75 步 67 分钟（平均约 48 s/步），峰值 78.2GB | 实测 |

ZeRO-3 的通信量：每条样本要聚合 3 次完整权重（参考模型前向、策略模型前向、带重算的反向），每次每卡收约 27GB，4.7 GB/s 下约 5.7 秒一次，合计约 17 秒，和样本长度无关。计算量随长度增长，所以越长通信占比越低。两张卡同时各跑一条，整体吞吐与单卡相近，ZeRO-3 换来的是能跑更长，不是更快。

卸载方案每次聚合前还要把权重从 CPU 内存拷到显存。这条路和卡间通信共用同一个主机桥，会互相抢带宽。卡空闲时可以先测一下 CPU→GPU 带宽：

```bash
python -c "
import torch,time
x=torch.empty(2<<30,dtype=torch.uint8).pin_memory();torch.cuda.synchronize()
t=time.time()
for _ in range(5): y=x.to('cuda',non_blocking=True)
torch.cuda.synchronize();print('H2D GB/s',5*2/(time.time()-t))"
```

如果只有 5–8 GB/s，卸载方案每条还要再多约 10 秒。

### 可选提速（未启用）

在 `configs/ds_z3.json` 的 `zero_optimization` 里加 `"zero_quantized_weights": true`（ZeRO++ 的 int8 权重聚合），通信量减半，每条约省 8 秒。代价是前向用的权重有轻微的量化误差。建议先跑出不开的基线，再对比。

## 方案跑完后更新这张表

方案表里的数字决定了 autotrain 的选择，跑完一个「推算」方案后请更新：

1. 从 `python autotrain.py status` 或 `runs/<运行名>/run.json` 读出实际的每步耗时和 GPU 峰值。
2. 改 `configs/plans.yaml` 里对应方案：把 `status` 改成 `verified`，在 `note` 里写实测值。
3. 如果显存余量很大（峰值低于 70GB），可以新增一个更长 cutoff 的方案放在前面；如果 OOM 了，autotrain 会自动降级，同时说明这个方案在本机不可用，应该把它的 `auto_select` 改成 `false`。
4. 同步更新本文和 README 里的表格。
