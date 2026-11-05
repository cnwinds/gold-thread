# ADR-0003：工具 observation 编码修复 — 默认不进入管线

- **状态**：已接受
- **日期**：2026-11-05

## 背景

约 **12%** 的 observation 含 `U+FFFD`，源于 Windows 控制台 GBK 输出被当作 UTF-8 写入 pi 日志（见 `CONTEXT.md` §8、`handoff.md` 说明）。`system` / `human` / `gpt` 基本不受影响。

## 决策

1. **不在 stage1/stage3 默认开启** 自动转码或丢弃；训练可接受该噪声，handoff 已披露。
2. **若要做**，优先 **Stage 1 落盘前** 对 tool 结果尝试 `utf-8` → 失败则 `gbk` → 仍失败则保留原样；并写 **`stats.json` / meta 计数** 供控制台展示。
3. **可选门控**（后续）：stage3 丢弃 observation 内 `U+FFFD` 密度超过阈值的 **单步窗口**（非整会话删）。

## 后果

- 无立即代码变更；research/implement ticket 可引用本 ADR。
- 新会话若在 Linux 环境采集，问题率应显著下降— 与修复优先级挂钩。
