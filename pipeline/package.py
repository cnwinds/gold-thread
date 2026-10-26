"""训练交付包：把 handoff + 数据集打成 zip，交给下游 AI 直接训练。

产出（默认在 data/03_train/delivery/）：
    goldthread-delivery-<config_hash>.zip
        README.txt          30 秒速览
        handoff.md          完整训练交接文档
        train_sft.yaml / train_dpo.yaml
        dataset_info.json / stats.json / MANIFEST.json
        data/*.jsonl        全部训练/验证数据文件
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.config import config_hash, load_config, stage_dirs
from pipeline.common.store import now_iso, read_json

# 打进 zip 的数据文件（不含 diagnostics 全量，避免包过大且明确标注不进训练）
TRAIN_FILES = [
    "sft_step_windows_sharegpt.train.jsonl",
    "sft_step_windows_sharegpt.eval.jsonl",
    "sft_gold_trajectories_sharegpt.train.jsonl",
    "sft_gold_trajectories_sharegpt.eval.jsonl",
    "dpo_preference_pairs.train.jsonl",
    "dpo_preference_pairs.eval.jsonl",
    "sft_step_windows_openai.train.jsonl",
    "sft_step_windows_openai.eval.jsonl",
]

META_FILES = [
    "handoff.md",
    "train_sft.yaml",
    "train_sft_256k.yaml",
    "train_sft_qlora.yaml",
    "train_dpo.yaml",
    "ds_z2_config.json",
    "ds_z3_config.json",
    "stats.json",
]


def _readme(cfg_hash: str, stats: Dict[str, Any]) -> str:
    counts = stats.get("counts") or {}
    win = counts.get("sft_step_windows_sharegpt.train.jsonl", 0)
    dpo = counts.get("dpo_preference_pairs.train.jsonl", 0)
    return f"""GoldThread 训练交付包（目标模型：Qwen/Qwen3.8-27B Dense）
=========================================================

这是从 pi Agent 会话提纯出的 SFT/DPO 训练数据集，专为微调 Qwen3.8-27B 稠密模型准备。

快速开始（给接手 AI）：
1. 解压本 zip 到任意目录，例如 ./goldthread-delivery/
2. 必读 handoff.md（含 27B 显存核算、环境安装、DeepSpeed 多卡/单卡启动、合并导出全流程）
3. 按 GPU 显存选择配置：
   - 80GB 单卡或多卡：FORCE_TORCHRUN=1 llamafactory-cli train ./train_sft.yaml
   - 48GB 单卡或 2×24GB：llamafactory-cli train ./train_sft_qlora.yaml
4. 可选 DPO：FORCE_TORCHRUN=1 llamafactory-cli train ./train_dpo.yaml
5. 合并 LoRA 并导出最终 27B 模型（见 handoff.md §6 & §10）

包内结构：
  handoff.md            完整交接文档（27B 硬件/环境/训练/验收清单）
  train_sft.yaml        27B BF16 LoRA 主配置（dataset_dir 已指向 ./data）
  train_sft_qlora.yaml  27B 4-bit QLoRA 配置（显存受限时使用）
  train_dpo.yaml        27B DPO 训练配置（SFT 完成后使用）
  ds_z2_config.json     DeepSpeed ZeRO-2 配置
  ds_z3_config.json     DeepSpeed ZeRO-3 配置
  dataset_info.json     LLaMA-Factory 数据集注册表
  stats.json            统计与切分信息
  MANIFEST.json         文件清单与校验
  data/                 全部 *.jsonl 训练与验证文件

数据规模：SFT 训练 {win} 条，DPO 训练 {dpo} 条
配置版本：{cfg_hash}
生成时间：{stats.get('generated_at', '—')}

注意：negative_diagnostics.jsonl 不含在本包内（仅用于分析，不进训练）。
"""


def _patch_yaml_paths(yaml_text: str, data_rel: str) -> str:
    """把 yaml 里的 dataset_dir 改成 zip 内相对路径。"""
    lines = []
    for line in yaml_text.splitlines():
        if line.strip().startswith("dataset_dir:"):
            lines.append(f"dataset_dir: {data_rel}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def build_delivery_zip(
    train_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log=None,
) -> Dict[str, Any]:
    """打包 handoff + 数据集为 zip，返回路径与元信息。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    root = Path(train_root or dirs["train"])
    delivery_dir = root / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    cfg_h = config_hash(cfg)
    zip_name = f"goldthread-delivery-{cfg_h[:8]}.zip"
    zip_path = delivery_dir / zip_name

    stats = read_json(root / "stats.json", default={}) or {}
    handoff_path = root / "handoff.md"
    if not handoff_path.exists():
        raise FileNotFoundError("handoff.md 尚未生成，请先运行 stage3")

    manifest_files: List[Dict[str, Any]] = []
    data_rel = "./data"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # README
        readme = _readme(cfg_h, stats)
        zf.writestr("README.txt", readme)
        manifest_files.append({"path": "README.txt", "size": len(readme.encode("utf-8"))})

        # 元数据文件（根目录）
        for name in META_FILES:
            src = root / name
            if not src.exists():
                continue
            content = src.read_text(encoding="utf-8")
            if name.endswith(".yaml"):
                content = _patch_yaml_paths(content, data_rel)
            zf.writestr(name, content)
            manifest_files.append({"path": name, "size": len(content.encode("utf-8"))})

        # 数据文件（data/ 子目录）
        for name in TRAIN_FILES:
            src = root / name
            if not src.exists():
                continue
            arc = f"data/{name}"
            zf.write(src, arcname=arc)
            manifest_files.append({"path": arc, "size": src.stat().st_size})

        # dataset_info 里的 file_name 指向 data/ 子目录
        info_path = root / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            patched = {}
            for k, v in info.items():
                entry = dict(v)
                fn = entry.get("file_name", "")
                if fn and not fn.startswith("data/"):
                    entry["file_name"] = f"data/{fn}"
                patched[k] = entry
            info_json = json.dumps(patched, ensure_ascii=False, indent=2)
            # 若已在 META_FILES 写入，这里覆盖为 patched 版
            zf.writestr("dataset_info.json", info_json)

        manifest = {
            "package": zip_name,
            "generated_at": now_iso(),
            "config_hash": cfg_h,
            "stats_generated_at": stats.get("generated_at"),
            "files": manifest_files,
            "train_files": TRAIN_FILES,
            "counts": stats.get("counts") or {},
            "split": stats.get("split") or {},
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        zf.writestr("MANIFEST.json", manifest_json)

    result = {
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest_files) + 1,
        "generated_at": manifest["generated_at"],
        "config_hash": cfg_h,
    }
    if log:
        size_mb = result["size_bytes"] / (1024 * 1024)
        log(f"[package] 交付包已生成: {zip_path.name} ({size_mb:.1f} MB, {result['file_count']} 个文件)")
    return result
