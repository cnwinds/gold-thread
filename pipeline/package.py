"""训练交付包：把 handoff + 已规整数据集打成 zip，解压后可直接用 LLaMA-Factory 训练。

产出（默认在 data/04_release/delivery/）：
    goldthread-delivery-<package_hash>.zip
        START_HERE.md / README.txt / handoff.md
        train_*.yaml          dataset_dir=./data，dataset 键为 gt_*_lf
        train_lowmem.py / lowmem_dpo.py / check_data.py / check_env.py
        data/dataset_info.json 键与 YAML 一致，file_name 相对 dataset_dir
        data/sft.train.lf.jsonl 等规范文件（不再打包未重切的 stage3 原名）
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.config import config_hash, load_config, stage_dirs
from pipeline.common.store import now_iso, read_json
from pipeline.lf_normalize import dumps_jsonl, dataset_info_for, prepare_datasets, require_sft
from pipeline import length_gate as length_gate_mod

_REPO_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_DATA_REL = "./data"

# 这些脚本必须以仓库版本打进 zip：它们认识 data/ 与 gt_*_lf，而 03_train 里可能还是旧检查。
_REPO_SCRIPTS = (
    "check_data.py",
    "check_env.py",
    "train_lowmem.py",
    "lowmem_dpo.py",
    "filter_by_cutoff.py",
)

_OPTIONAL_SCRIPTS = (
    "train_monitor.py",
    "verify_longctx.py",
    "probe_24g.py",
    "smoke_infer_merged.py",
)

_YAML_FILES = (
    "train_sft.yaml",
    "train_sft_qlora.yaml",
    "train_dpo.yaml",
    "train_dpo_qlora_27b.yaml",
    "train_dpo_qlora_27b_b.yaml",
    "eval_sft.yaml",
    "train_sft_qwen3_8b.yaml",
    "train_dpo_qwen3_8b.yaml",
    "train_sft_smoke_qwen3_8b.yaml",
    "train_dpo_smoke_qwen3_8b.yaml",
    "eval_sft_qwen3_8b.yaml",
)

_DS_FILES = ("ds_z2_config.json", "ds_z3_config.json")

_HANDOFF_BANNER = """> **本交付包已按可直接训练的布局打包**（不要再改文件名、不要手改 dataset_info）。
>
> - 解压后进入该目录，先跑 `python check_data.py`（退出码 0 再开训）
> - 数据集在 `data/`，YAML 里 `dataset_dir: ./data`（相对当前工作目录）
> - SFT：`gt_sft_train_lf` / `gt_sft_eval_lf` → `data/sft.train.lf.jsonl`、`data/sft.eval.lf.jsonl`
> - DPO：`gt_dpo_train_lf` / `gt_dpo_eval_lf` → `data/dpo.train.lf.jsonl`、`data/dpo.eval.lf.jsonl`
> - 27B 必须在解压目录启动：`CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml`
> - 训练前 `train_lowmem.py` 会按 YAML 的 `cutoff_len` 过滤超长样本（包内保留全集）
>
> 下文若仍写着绝对路径、`pi_*` 键或 stage3 原始文件名，以本包 YAML 与 `data/dataset_info.json` 为准。最短步骤见 `START_HERE.md`。

"""


def _is_dpo_yaml(name: str, text: str) -> bool:
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith("stage:"):
            return "dpo" in code.split(":", 1)[1].lower()
    return "dpo" in name.lower()


def _patch_yaml(text: str, name: str, available: set[str]) -> Optional[str]:
    """把 dataset 键改成已登记的 gt_*_lf，dataset_dir 改成 ./data。

    训练所需的 lf 键不在 available 里时返回 None（调用方不打包这份 YAML）。
    """
    dpo = _is_dpo_yaml(name, text)
    train_key = "gt_dpo_train_lf" if dpo else "gt_sft_train_lf"
    eval_key = "gt_dpo_eval_lf" if dpo else "gt_sft_eval_lf"
    saw_dataset = False
    saw_eval = False
    lines: List[str] = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        stripped = code.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("dataset_dir:"):
            lines.append(f"{indent}dataset_dir: {_DATA_REL}")
        elif stripped.startswith("dataset:"):
            saw_dataset = True
            if train_key not in available:
                return None
            lines.append(f"{indent}dataset: {train_key}")
        elif stripped.startswith("eval_dataset:"):
            saw_eval = True
            if eval_key not in available:
                continue
            lines.append(f"{indent}eval_dataset: {eval_key}")
        else:
            lines.append(line)
    if saw_dataset and train_key not in available:
        return None
    if not saw_dataset and not saw_eval:
        return None
    if not saw_dataset and eval_key not in available:
        return None
    body = "\n".join(lines).rstrip() + "\n"
    return body


def _read_script(name: str, root: Path) -> Optional[bytes]:
    if name in _REPO_SCRIPTS:
        path = _REPO_TOOLS / name
        return path.read_bytes() if path.is_file() else None
    for path in (root / name, _REPO_TOOLS / name):
        if path.is_file():
            return path.read_bytes()
    return None


def _readme(cfg_hash: str, stats: Dict[str, Any], counts: Dict[str, int], sources: Dict[str, str]) -> str:
    sft = counts.get("sft.train.lf.jsonl", 0)
    sft_eval = counts.get("sft.eval.lf.jsonl", 0)
    dpo = counts.get("dpo.train.lf.jsonl", 0)
    dpo_eval = counts.get("dpo.eval.lf.jsonl", 0)
    src = ", ".join(f"{k}←{v}" for k, v in sources.items()) or "—"
    return f"""GoldThread 训练交付包（目标模型：Qwen/Qwen3.8-27B Dense）
=========================================================

解压后按 START_HERE.md 操作即可开训，不需要改文件名、不需要再跑规整脚本、不需要改 dataset_info。

快速开始（在解压目录里执行，dataset_dir 是相对路径 ./data）：
1. python check_data.py
   退出码必须是 0。它会核对每份 YAML 的 dataset 键、data/dataset_info.json 和 jsonl 是否对齐。
2. 27B SFT（不要用 llamafactory-cli train，16K 全词表 logits 在单卡 80G 上会 OOM）：
   CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml
3. 显存不够时：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft_qlora.yaml
4. SFT 合并之后的 DPO：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_dpo.yaml
5. 训完单独评测：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py eval_sft.yaml

包内结构：
  START_HERE.md         解压后的最短路径（以本文件布局为准）
  handoff.md            环境与架构说明（开头有本包布局，冲突时以 YAML 为准）
  train_lowmem.py       27B 低显存启动器（SFT/eval 只算监督位置 logits）
  lowmem_dpo.py         DPO 低显存补丁（train_lowmem.py 自动加载）
  check_data.py         训练前数据契约检查（本包的验收命令）
  check_env.py          环境检查
  train_sft.yaml        dataset: gt_sft_train_lf / gt_sft_eval_lf
  train_dpo.yaml        dataset: gt_dpo_train_lf / gt_dpo_eval_lf
  data/dataset_info.json
  data/sft.train.lf.jsonl  data/sft.eval.lf.jsonl
  data/dpo.train.lf.jsonl  data/dpo.eval.lf.jsonl
  data/sft.train.jsonl 等  规整前的 ShareGPT（键 gt_sft_train / gt_dpo_train，正式训练用 lf）

数据规模（lf，LLaMA-Factory 实际读取的条数）：
  SFT train {sft} / eval {sft_eval}
  DPO train {dpo} / eval {dpo_eval}
来源：{src}
配置版本：{cfg_hash}
统计生成时间：{stats.get("generated_at", "—")}

注意：negative_diagnostics、openai 副本、未重切的 stage3 原名不在本包内。
正式训练读 *.lf.jsonl：连续同角色已合并，DPO 的 chosen/rejected 保持消息字典。
"""


def _start_here(counts: Dict[str, int], has_dpo: bool) -> str:
    sft = counts.get("sft.train.lf.jsonl", 0)
    sft_eval = counts.get("sft.eval.lf.jsonl", 0)
    dpo = counts.get("dpo.train.lf.jsonl", 0)
    dpo_eval = counts.get("dpo.eval.lf.jsonl", 0)
    dpo_block = ""
    if has_dpo:
        dpo_block = f"""
## 4. DPO（可选，SFT 合并之后）

`train_dpo.yaml` 的 `model_name_or_path` 要改成 SFT 合并目录。数据集已是 `gt_dpo_train_lf` / `gt_dpo_eval_lf`（{dpo} / {dpo_eval} 条）。

```bash
cd <解压目录>
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_dpo.yaml
```
"""
    else:
        dpo_block = """
## 4. DPO

这次打包没有可用的 DPO lf 文件，zip 里不会带 `train_dpo.yaml`。不要拿未打包的原始偏好对硬训。
"""
    return f"""# START HERE · GoldThread 交付 zip

在**解压出来的这一层目录**里操作。YAML 的 `dataset_dir` 是 `./data`，相对当前工作目录，不是相对 yaml 文件。

## 0. 解压

```bash
unzip goldthread-delivery-*.zip -d goldthread-delivery
cd goldthread-delivery
python check_data.py
```

包内 `*.lf.jsonl` 是全集（含相对默认 cutoff 仍偏长的样本）。`train_lowmem.py` 开训前会按该次 YAML 的 `cutoff_len` 过滤；跳过过滤可设环境变量 `GT_SKIP_LENGTH_FILTER=1`。

`check_data.py` 退出码必须是 0。它确认：

- 每份 `train_*.yaml` / `eval_*.yaml` 的 `dataset` / `eval_dataset` 都在 `data/dataset_info.json`
- `file_name` 落在 `dataset_dir`（`./data`）下面，文件真实存在
- `*.lf.jsonl` 的角色交替符合 LLaMA-Factory（连续 observation 已合并）

不要改名，不要再跑规整脚本，不要手改 `dataset_info.json`。

## 1. 本包实际布局

```
./
├── START_HERE.md  README.txt  handoff.md
├── train_sft.yaml            # dataset: gt_sft_train_lf
├── train_sft_qlora.yaml
├── train_dpo.yaml            # dataset: gt_dpo_train_lf（有 DPO 数据时才在）
├── eval_sft.yaml
├── train_lowmem.py  lowmem_dpo.py  check_data.py  check_env.py
└── data/
    ├── dataset_info.json
    ├── sft.train.lf.jsonl    # gt_sft_train_lf   {sft} 条
    ├── sft.eval.lf.jsonl     # gt_sft_eval_lf    {sft_eval} 条
    ├── dpo.train.lf.jsonl    # gt_dpo_train_lf   {dpo} 条
    ├── dpo.eval.lf.jsonl     # gt_dpo_eval_lf    {dpo_eval} 条
    └── sft.train.jsonl / dpo.train.jsonl   # 规整前 ShareGPT，正式训练不要指这里
```

环境安装、模型下载、显存说明见 `handoff.md`。下文只给与**这个 zip** 一致的启动命令。

## 2. 训练前自检

```bash
cd <解压目录>
python check_data.py
python check_env.py --data ./data --model /data/models/Qwen3.8-27B
```

没有模型权重时，`check_data.py` 不加 `--model` 也会检查数据结构，退出码 0 即可进入下一步。

## 3. 27B SFT

必须用 `train_lowmem.py`。直接 `llamafactory-cli train` 会对全部 token 算 248320 词表的 fp32 logits，16K 样本在单卡 80G 上会 OOM。

```bash
cd <解压目录>
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml
```

多卡：

```bash
torchrun --nproc_per_node 2 train_lowmem.py train_sft.yaml
```

显存更紧：

```bash
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft_qlora.yaml
```

训完单独跑 eval（训练 YAML 里 eval 通常关掉）：

```bash
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py eval_sft.yaml
```

可选监控（OOM 时区分显存耗尽和被系统杀掉）：

```bash
python train_monitor.py --timeout-hours 5 --log logs/27b_sft.log -- \\
  python train_lowmem.py train_sft.yaml
```

启动后前几步应出现 `[lowmem] seq_len=… 监督位置=…`。没有这行说明补丁没生效。
{dpo_block}
## 5. 8B 冒烟

若包内有 `train_sft_qwen3_8b.yaml` / `train_sft_smoke_qwen3_8b.yaml`，它们的数据集键同样是 `gt_sft_*_lf`，`dataset_dir` 同样是 `./data`。8B 可以直接：

```bash
cd <解压目录>
llamafactory-cli train train_sft_smoke_qwen3_8b.yaml
```

27B 不要走这条命令。
"""


def _prepend_banner(handoff: str) -> str:
    marker = "本交付包已按可直接训练的布局打包"
    if marker in handoff[:800]:
        return handoff
    return _HANDOFF_BANNER + handoff



def compute_package_hash(
    cfg_h: str,
    prepared_files: Dict[str, List[Dict[str, Any]]],
    length_gate_fp: str,
) -> str:
    """8-hex fingerprint of delivery payload (post-gate lf bytes + config + gate version).

    Soft-compress / data rebuilds that change packed jsonl bytes, or a
    LENGTH_GATE_VERSION bump, produce a new zip filename even when
    pipeline.yaml config_hash is unchanged.
    """
    parts: List[str] = [cfg_h]
    for name, rows in sorted(prepared_files.items()):
        content = dumps_jsonl(rows)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        parts.append(f"{name}:{digest}")
    parts.append(length_gate_fp)
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def build_delivery_zip(
    train_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log=None,
) -> Dict[str, Any]:
    """打包 handoff + 规整后的数据集为 zip，返回路径与元信息。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    root = Path(train_root or dirs["train"])
    # 发布产物统一落在 data/04_release/，与 03_train 工作区分离
    release_root = root.parent / "04_release"
    delivery_dir = release_root / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    cfg_h = config_hash(cfg)

    def _emit(msg: str) -> None:
        if log:
            log(msg)

    handoff_path = root / "handoff.md"
    if not handoff_path.exists():
        raise FileNotFoundError("handoff.md 尚未生成，请先运行 stage3")
    if not (root / "train_sft.yaml").is_file():
        raise FileNotFoundError("train_sft.yaml 尚未生成，请先运行 stage3")

    _emit("[package] 正在准备数据集并做长度闸门（软压缩）…")
    prepared = prepare_datasets(root, log=log)
    require_sft(prepared)
    info = dataset_info_for(prepared)
    available = set(info)
    has_dpo = "gt_dpo_train_lf" in available and "gt_dpo_eval_lf" in available

    lg_fp = length_gate_mod.length_gate_fingerprint()
    pkg_h = compute_package_hash(cfg_h, prepared.files, lg_fp)
    zip_name = f"goldthread-delivery-{pkg_h}.zip"
    zip_path = delivery_dir / zip_name

    stats = read_json(root / "stats.json", default={}) or {}
    manifest_files: List[Dict[str, Any]] = []

    def _add(zf: zipfile.ZipFile, arc: str, content: str | bytes) -> None:
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        zf.writestr(arc, data)
        manifest_files.append({"path": arc, "size": len(data)})

    _emit(f"[package] 正在写入 zip：{zip_name}…")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _add(zf, "README.txt", _readme(cfg_h, stats, prepared.counts, prepared.sources))
        _add(zf, "START_HERE.md", _start_here(prepared.counts, has_dpo))
        _add(zf, "handoff.md", _prepend_banner(handoff_path.read_text(encoding="utf-8")))

        if (root / "RESULT.md").is_file():
            _add(zf, "RESULT.md", (root / "RESULT.md").read_text(encoding="utf-8"))
        if (root / "stats.json").is_file():
            _add(zf, "stats.json", (root / "stats.json").read_text(encoding="utf-8"))

        packed_yaml: List[str] = []
        skipped_yaml: List[str] = []
        for name in _YAML_FILES:
            src = root / name
            if not src.is_file():
                continue
            patched = _patch_yaml(src.read_text(encoding="utf-8"), name, available)
            if patched is None:
                skipped_yaml.append(name)
                continue
            _add(zf, name, patched)
            packed_yaml.append(name)

        if "train_sft.yaml" not in packed_yaml:
            raise FileNotFoundError("train_sft.yaml 未能打进交付包（缺少 gt_sft_*_lf 数据）")

        for name in _DS_FILES:
            src = root / name if (root / name).is_file() else _REPO_TOOLS / name
            if src.is_file():
                _add(zf, name, src.read_bytes())

        for name in _REPO_SCRIPTS + _OPTIONAL_SCRIPTS:
            blob = _read_script(name, root)
            if blob is None:
                if name in ("train_lowmem.py", "lowmem_dpo.py", "check_data.py"):
                    raise FileNotFoundError(f"交付包缺少脚本 {name}")
                continue
            _add(zf, name, blob)

        for name, rows in prepared.files.items():
            _add(zf, f"data/{name}", dumps_jsonl(rows))
        info_json = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
        _add(zf, "data/dataset_info.json", info_json)

        if prepared.length_gate:
            detail = length_gate_mod.overlength_report_jsonl_from_summary(prepared.length_gate)
            if detail:
                _add(zf, "overlength_report.jsonl", detail)

        manifest = {
            "package": zip_name,
            "layout": "llamafactory-dataset_dir-data",
            "dataset_dir": _DATA_REL,
            "generated_at": now_iso(),
            "config_hash": cfg_h,
            "package_hash": pkg_h,
            "length_gate_version": lg_fp,
            "stats_generated_at": stats.get("generated_at"),
            "datasets": {key: entry.get("file_name") for key, entry in info.items()},
            "sources": prepared.sources,
            "dropped": prepared.dropped,
            "counts": prepared.counts,
            "length_gate": prepared.length_gate,
            "yaml": packed_yaml,
            "yaml_skipped": skipped_yaml,
            "files": manifest_files,
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        manifest["files"] = manifest_files + [{"path": "MANIFEST.json", "size": len(manifest_json.encode("utf-8"))}]
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        zf.writestr("MANIFEST.json", manifest_json)

    result = {
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest_files) + 1,
        "generated_at": manifest["generated_at"],
        "config_hash": cfg_h,
        "package_hash": pkg_h,
        "counts": prepared.counts,
        "sources": prepared.sources,
        "datasets": manifest["datasets"],
        "length_gate": prepared.length_gate,
    }
    if log:
        size_mb = result["size_bytes"] / (1024 * 1024)
        sft_n = prepared.counts.get("sft.train.lf.jsonl", 0)
        dpo_n = prepared.counts.get("dpo.train.lf.jsonl", 0)
        gate = prepared.length_gate or {}
        salv = sum(int(v.get("salvaged_by_obs_truncate") or 0) for v in gate.values() if isinstance(v, dict))
        over = sum(int(v.get("overlength_at_default_cutoff") or 0) for v in gate.values() if isinstance(v, dict))
        log(
            f"[package] 交付包已生成: {zip_path.name} ({size_mb:.1f} MB, "
            f"SFT lf {sft_n} / DPO lf {dpo_n}, {result['file_count']} 个文件, "
            f"length_gate salvaged={salv} overlength_at_default={over})"
        )
    return result
