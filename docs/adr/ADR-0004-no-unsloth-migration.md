# ADR-0004：不迁移 Unsloth 训练栈

- **状态**：已接受
- **日期**：2026-11-05

## 背景

曾评估用 Unsloth 替代 LLaMA-Factory + 自研 `train_lowmem.py` / 分块 CE / `mask_history` / Qwen3 工具模板。成本高，且不能自动解决超大词表 logits、DPO ref 约定等问题。

## 决策

**非目标**：除非用户显式立项「换训练栈」，否则继续以 **LLaMA-Factory + 本仓库 handoff/YAML/交付 zip 契约** 为唯一上机路径。

## 后果

- 文档与 issue 中 Unsloth 仅作对比说明，不作为 roadmap 默认项。
