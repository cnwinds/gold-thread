# 训练包模板

本目录是**轻量模板**：含 `autotrain.py`、配置、工具、问题库和示例 `data/`，**不含** adapter、合并模型与运行记录。

## 当前代码特性（相对早期模板）

- 正式 `sft` / `dpo` 开训前按 **token 最长样本**探测；阈值看训练进程打印的 `max_memory_allocated`（`probe_max_peak_gb`），不是 nvidia-smi。
- 全量训练 **最长优先**：过滤副本按 token 从长到短排序，并设 `disable_shuffling=true`。
- **单卡 DPO** 默认 `dpo_single_offload_*`：ZeRO-3 `offload_param` 到 CPU（需主机内存 ≥128GB），单卡也走 torchrun；内存不够才退回不卸载的 `dpo_single`。
- DPO 行缺 `tools`、且 SFT 全部行共用同一份工具定义时，在过滤副本里自动补上（`fill_missing_tools_from_sft`，不改 `data/`）；计长度时也会把 tools 前缀算进去。

## 用它建新包

```bash
PKG=goldthread-$(date +%Y%m%d)    # 或自定义别名，如 goldthread-d9dbbbd1
cp -a /cloud/cloud-ssd1/goldthread-train /cloud/cloud-ssd1/packages/$PKG
rm -rf /cloud/cloud-ssd1/packages/$PKG/runs/*
rm -f  /cloud/cloud-ssd1/packages/$PKG/issues/pending.jsonl
mkdir -p /cloud/cloud-ssd1/packages/$PKG/{saves,models} /data/work/$PKG
```

然后：

1. 换 `packages/$PKG/data/`（jsonl + `dataset_info.json` + `MANIFEST.json`）
2. 改 `packages/$PKG/settings.yaml`：
   - `work_dir: /data/work/$PKG`
   - `saves_dir: /cloud/cloud-ssd1/packages/$PKG/saves`
   - `merged_dir: /cloud/cloud-ssd1/packages/$PKG/models`
3. `cd packages/$PKG && python tools/check_data.py`
4. `nohup python autotrain.py sft > runs/sft.out 2>&1 &`

完整约定见 [docs/LAYOUT.md](docs/LAYOUT.md)。

已有完整产物包示例：`packages/goldthread-20261106/`、`packages/goldthread-d9dbbbd1/`。
