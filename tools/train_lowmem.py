#!/usr/bin/env python3
"""低显存 SFT 启动器：只在有监督的位置计算 logits，替代 `llamafactory-cli train`。

为什么需要：
    Qwen3.8-27B 词表 248320，且 qwen3_5 不在 Liger 支持列表（拿不到融合 CE）。
    原生路径会对全部 token 计算 logits 并升 fp32：
        16K × 248320 × (bf16 8G + fp32 16G + 反向梯度 16G) ≈ 40GB
    再叠加 54G 权重 → 单卡 80G 在 16K 样本上 OOM。
    而 mask_history: true 下只有最后一轮 assistant（通常 < 2K token）参与 loss，
    其余 90% 的 logits 都是白算。

做法：
    替换 transformers.Trainer.compute_loss：
      1) 找出 labels[:, 1:] != -100 的位置（logits[t] 预测 labels[t+1]）
      2) 通过 logits_to_keep=位置张量 让模型只对这些位置过 lm_head
      3) 自己算交叉熵（分块 + checkpoint，fp32 峰值只占一块）
    数学上与原生 loss 等价，归一化 / 梯度累积缩放与 Trainer 原逻辑一致。

用法（参数与 llamafactory-cli train 完全相同，YAML 不用改）：
    单卡：  CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml
    多卡：  torchrun --nproc_per_node 2 train_lowmem.py train_sft.yaml
    配合监控：
        python train_monitor.py --timeout-hours 5 --log logs/27b_sft.log -- \\
          python train_lowmem.py train_sft.yaml

stage: dpo 同样生效：lowmem_dpo.py 替换 CustomDPOTrainer.concatenated_forward，
policy / ref 前向都只算 response 位置的 logits（python train_lowmem.py train_dpo.yaml）。
"""
from __future__ import annotations

import os
import sys

IGNORE_INDEX = -100
CE_CHUNK = int(os.environ.get("LOWMEM_CE_CHUNK", "2048"))
REPORT_STEPS = int(os.environ.get("LOWMEM_REPORT_STEPS", "3"))


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _log(msg: str) -> None:
    if _is_rank0():
        print(f"[lowmem] {msg}", flush=True)


def _preflight() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        print(
            f"[lowmem] 检测到 {torch.cuda.device_count()} 张卡但不是 torchrun 启动，"
            "HF Trainer 会退化成 DataParallel（显存与速度都更差）。\n"
            "  单卡：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py <yaml>\n"
            "  多卡：torchrun --nproc_per_node N train_lowmem.py <yaml>",
            file=sys.stderr,
        )
        sys.exit(2)

    missing = []
    for mod, pkg in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
    if missing:
        _log(
            f"[WARN] 未安装 {' / '.join(missing)}：48 层 linear_attention 会退回 PyTorch 实现，"
            f"显存和速度都更差。建议：pip install {' '.join(missing)} --no-build-isolation"
        )


def _chunked_ce_sum(logits, target):
    """sum 形式的交叉熵；每块单独 checkpoint，反向时重算，fp32 只占一块的显存。"""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def _ce(lg, tg):
        return F.cross_entropy(
            lg.float().reshape(-1, lg.size(-1)), tg.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )

    k = logits.size(1)
    if k <= CE_CHUNK:
        return _ce(logits, target)
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for s in range(0, k, CE_CHUNK):
        total = total + checkpoint(_ce, logits[:, s:s + CE_CHUNK], target[:, s:s + CE_CHUNK], use_reentrant=False)
    return total


def patch_trainer() -> None:
    import torch
    from transformers import Trainer

    original = Trainer.compute_loss
    state = {"reported": 0, "fallback_reported": 0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 注意：Trainer 传来的 inputs 是 BatchEncoding（继承 UserDict），不是 dict 子类，
        # 早先的 isinstance(inputs, dict) 恒为 False，导致补丁静默失效、退回全序列 logits（16K 必 OOM）。
        has_labels = hasattr(inputs, "get") and "labels" in inputs
        labels = inputs.get("labels") if has_labels else None
        # 只接管「默认 CE」路径；DFT/EAFT/ASFT、label smoothing、需要完整 logits 的 eval 指标都走原逻辑
        usable = (
            labels is not None
            and labels.dim() == 2
            and self.compute_loss_func is None
            and self.label_smoother is None
            and (model.training or self.compute_metrics is None)
        )
        if not usable:
            if model.training and state["fallback_reported"] < 1:
                state["fallback_reported"] += 1
                _log(
                    "[WARN] lowmem 未接管本次 loss，退回全序列 logits："
                    f"labels={'有' if labels is not None else '无'} "
                    f"compute_loss_func={self.compute_loss_func!r} "
                    f"label_smoother={self.label_smoother!r} "
                    f"training={model.training} compute_metrics={self.compute_metrics!r}"
                )
            return original(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        inputs = dict(inputs)
        labels = inputs.pop("labels")
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(IGNORE_INDEX)
        pos = valid.any(dim=0).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:  # 整条被截断掉监督部分：保留一个位置，loss 为 0 但计算图完整
            pos = torch.tensor([shift_labels.size(1) - 1], device=labels.device)

        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch
        outputs = model(**inputs, logits_to_keep=pos)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        if logits.size(1) != pos.numel():
            raise RuntimeError(
                f"[lowmem] 模型没有按 logits_to_keep 裁剪（期望 {pos.numel()} 个位置，实际 {logits.size(1)}）。"
                "当前 transformers / 模型实现不支持张量形式的 logits_to_keep，请改回 llamafactory-cli train。"
            )

        loss_sum = _chunked_ce_sum(logits, shift_labels[:, pos])

        # 归一化与 Trainer.compute_loss / training_step 的约定保持一致：
        #   model_accepts_loss_kwargs 且给了 num_items_in_batch → 按整个累积 batch 的 token 数归一，
        #   training_step 不再除以梯度累积步数；否则按本 batch 求均值，由 training_step 除以累积步数。
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            denom = num_items_in_batch.to(loss_sum.device) if torch.is_tensor(num_items_in_batch) else num_items_in_batch
            loss = loss_sum / denom
            if self.args.average_tokens_across_devices:
                loss = loss * (self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)
        else:
            loss = loss_sum / valid.sum().clamp(min=1)

        if model.training and state["reported"] < REPORT_STEPS:
            state["reported"] += 1
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
            _log(
                f"seq_len={labels.size(1)} 监督位置={pos.numel()} "
                f"（logits 缩减 {labels.size(1) / max(pos.numel(), 1):.1f}×）峰值显存={peak:.1f}GB"
            )

        return (loss, outputs) if return_outputs else loss

    Trainer.compute_loss = compute_loss
    _log(f"已接管 Trainer.compute_loss（只算监督位置 logits，CE 分块 {CE_CHUNK}）")


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        print(__doc__)
        sys.exit(1)
    _preflight()
    patch_trainer()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lowmem_dpo import patch_dpo_trainer

    patch_dpo_trainer()
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
