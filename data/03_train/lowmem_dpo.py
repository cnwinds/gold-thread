#!/usr/bin/env python3
"""低显存 DPO 补丁：policy / ref 前向都只在有监督的位置计算 logits。

为什么需要：
    原生 CustomDPOTrainer.concatenated_forward 对 chosen+rejected 拼接后的全部 token 算 logits
    并 .to(float32)：2 × 8192 × 248320 × (bf16 8G + fp32 16G + 反向梯度) ≈ 40GB，
    单卡 80G 放不下。而 DPO loss 只用 response 部分（labels != -100）的 log-prob。

做法（替换 llamafactory.train.dpo.trainer.CustomDPOTrainer.concatenated_forward）：
    1) pos = (labels[:, 1:] != -100).any(0) 的位置（batch 内取并集，padding 行按 mask 置 0）
    2) model(..., logits_to_keep=pos) 只对这些位置过 lm_head（bf16，不整体升 fp32）
    3) 按位置分块算 log-prob 求和：fp32 只存在于当前块；需要梯度时每块套 checkpoint，
       反向时重算，不会把所有 fp32 块留到 backward
    ref 前向（compute_reference_log_probs → disable_adapter → concatenated_forward）自动走同一路径。

与原实现的语义对齐（llamafactory.train.trainer_utils.get_batch_logps）：
    - 返回逐条 logps 之和、valid_length = 监督 token 数
    - ld_alpha（长度脱敏）：前 min(len_c, len_r) 个监督 token 权重 1，其余权重 ld_alpha；ref 不用
    - ipo / orpo / simpo：logps / valid_length
    - chosen / rejected 按 batch 前后一半切分
    唯一差异：日志里的 logits/chosen、logits/rejected 是监督位置 logits 的均值（原来是全部位置），
    只影响监控曲线，不影响 loss。

用法：由 train_lowmem.py 自动调用 patch_dpo_trainer()；YAML 不用改。
    分块大小：环境变量 LOWMEM_DPO_CHUNK（默认 1024 个位置，batch=2 时 fp32 块 ≈ 2GB）
"""
from __future__ import annotations

import os
import sys

IGNORE_INDEX = -100
DPO_CHUNK = int(os.environ.get("LOWMEM_DPO_CHUNK", "1024"))
REPORT_STEPS = int(os.environ.get("LOWMEM_REPORT_STEPS", "3"))


def _log(msg: str) -> None:
    if os.environ.get("RANK", "0") == "0":
        print(f"[lowmem-dpo] {msg}", flush=True)


def select_positions(labels):
    """labels: (B, L)。返回 batch 内任一行有监督的 shift 位置（logits[t] 预测 labels[t+1]）。"""
    import torch

    shift = labels[:, 1:]
    pos = shift.ne(IGNORE_INDEX).any(dim=0).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:  # 监督部分整段被截断：保留一个位置，logps 为 0 但计算图完整
        pos = torch.tensor([shift.size(1) - 1], device=labels.device)
    return pos


def _token_weights(labels, pos, ld_alpha):
    """返回 (targets, weights, valid_length)，形状 (B, K) / (B, K) / (B,)，对齐 get_batch_logps。"""
    import torch

    shift = labels[:, 1:]
    loss_mask = shift.ne(IGNORE_INDEX)
    valid_length = loss_mask.sum(-1)

    mask_k = loss_mask[:, pos]
    targets = shift[:, pos].masked_fill(~mask_k, 0)  # dummy token，与原实现一致
    weights = mask_k.float()
    if ld_alpha is not None:
        num_examples = shift.size(0) // 2
        min_lengths = torch.min(valid_length[:num_examples], valid_length[num_examples:])
        start_positions = torch.argmax(loss_mask.int(), dim=1)
        public_lengths = start_positions + torch.cat([min_lengths, min_lengths], dim=0)
        ld_mask = pos.unsqueeze(0) < public_lengths.unsqueeze(1)
        weights = torch.where(ld_mask, weights, weights * ld_alpha)
    return targets, weights, valid_length


def _chunk_logps(logits, targets, weights):
    import torch.nn.functional as F

    lg = logits.float()
    tok = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets.reshape(-1), reduction="none")
    return (tok.view_as(weights) * weights).sum(-1)


def chunked_logps(logits, targets, weights, chunk: int = DPO_CHUNK):
    """logits: (B, K, V) 任意精度；返回 (B,) fp32 的加权 log-prob 之和。"""
    import torch
    from torch.utils.checkpoint import checkpoint

    k = logits.size(1)
    need_ckpt = torch.is_grad_enabled() and logits.requires_grad
    if k <= chunk and not need_ckpt:
        return _chunk_logps(logits, targets, weights)
    total = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    for s in range(0, k, chunk):
        args = (logits[:, s:s + chunk], targets[:, s:s + chunk], weights[:, s:s + chunk])
        if need_ckpt:
            total = total + checkpoint(_chunk_logps, *args, use_reentrant=False)
        else:
            total = total + _chunk_logps(*args)
    return total


def sliced_batch_logps(logits, labels, pos, ld_alpha=None, chunk: int = DPO_CHUNK):
    """logits 是 model(..., logits_to_keep=pos) 的输出 (B, len(pos), V)。

    返回值与 get_batch_logps(full_logits, labels, ld_alpha=...) 相同：(logps_sum, valid_length)。
    """
    if logits.size(1) != pos.numel():
        raise RuntimeError(
            f"[lowmem-dpo] 模型没有按 logits_to_keep 裁剪（期望 {pos.numel()} 个位置，实际 {logits.size(1)}）。"
            "当前 transformers / 模型实现不支持张量形式的 logits_to_keep，请改回 llamafactory-cli train。"
        )
    targets, weights, valid_length = _token_weights(labels, pos, ld_alpha)
    return chunked_logps(logits, targets, weights, chunk), valid_length


def patch_dpo_trainer() -> bool:
    """替换 CustomDPOTrainer.concatenated_forward。导入失败或源码结构不符时只告警、不打补丁。"""
    try:
        import inspect

        import torch
        from llamafactory.train.dpo.trainer import CustomDPOTrainer
        from llamafactory.train.trainer_utils import nested_detach
    except Exception as e:  # noqa: BLE001
        _log(f"[WARN] 无法导入 LLaMA-Factory DPO trainer，跳过 DPO 补丁：{e!r}")
        return False

    original = CustomDPOTrainer.concatenated_forward
    if getattr(original, "_lowmem_patched", False):
        return True
    try:
        src = inspect.getsource(original)
    except (OSError, TypeError):
        src = ""
    expected = ("get_batch_logps", "chosen_logps_avg", "ld_alpha", "is_ref_model")
    if src and not all(tok in src for tok in expected):
        _log(
            "[WARN] CustomDPOTrainer.concatenated_forward 结构与预期不符（LLaMA-Factory 版本变化？），"
            "跳过 DPO 补丁，回退原生实现（可能 OOM）。"
        )
        return False

    state = {"reported": 0}

    def concatenated_forward(self, model, batch, is_ref_model: bool = False):
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error（与原实现一致）

        labels = batch.pop("labels")
        pos = select_positions(labels)
        logits = model(**batch, return_dict=True, use_cache=False, logits_to_keep=pos).logits
        all_logps, valid_length = sliced_batch_logps(
            logits, labels, pos, ld_alpha=(self.ld_alpha if not is_ref_model else None)
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = logits.detach().split(batch_size, dim=0)  # 只用于日志 .mean()
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = chosen_logps / chosen_length

        if not is_ref_model and model.training and state["reported"] < REPORT_STEPS:
            state["reported"] += 1
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
            _log(
                f"batch={labels.size(0)} seq_len={labels.size(1)} 监督位置={pos.numel()} "
                f"（logits 缩减 {labels.size(1) / max(pos.numel(), 1):.1f}×）峰值显存={peak:.1f}GB"
            )

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
        }

    concatenated_forward._lowmem_patched = True
    CustomDPOTrainer.concatenated_forward = concatenated_forward
    _log(f"已接管 CustomDPOTrainer.concatenated_forward（policy/ref 只算监督位置 logits，分块 {DPO_CHUNK}）")
    return True


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0 if patch_dpo_trainer() else 1)
