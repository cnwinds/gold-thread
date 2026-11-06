"""训练交付包：把 handoff + 已规整数据集打成 zip，解压后可直接用 LLaMA-Factory 训练。

格式在打包时选定（默认 Qwen XML）。assistant 侧的工具调用、YAML 的 template、
推理用的 --tool-call-parser 一起锁定；03_train 里的原文件不动。


产出（默认在 data/04_release/delivery/）：
    goldthread-delivery-<package_hash>.zip
        START_HERE.md / README.txt / handoff.md
        train_*.yaml          dataset_dir=./data，dataset 键为 gt_*_lf
        merge_lora.yaml / merge_dpo.yaml  SFT / DPO 权重合并（llamafactory-cli export）
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
from pipeline.common.format import convert_sharegpt_row, profile_includes_yaml, resolve_export_profile
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

# 合并配置没有 dataset 键，不参与 _patch_yaml；只锁 template 后原样打包，
# 让“训练 → 合并 → DPO → 合并”整条链在包内自洽（不用手敲 export 命令）。
_MERGE_YAML_FILES = (
    "merge_lora.yaml",   # SFT LoRA -> /data/models/Qwen3.8-27B-merged（train_dpo.yaml 的底座）
    "merge_dpo.yaml",    # DPO LoRA -> /data/models/Qwen3.8-27B-dpo-merged
)

_DS_FILES = ("ds_z2_config.json", "ds_z3_config.json")

# 记在 delivery/ 旁边，不进 zip。stage3 再次打包时沿用上次在界面上选的格式。
_PROFILE_CHOICE = "export_profile.json"

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


def _format_banner(contract: Dict[str, Any]) -> str:
    if contract.get("pack_8b_yaml"):
        eight = "8B 的 `template: qwen3` 配置已打进本包，与数据同一套协议。"
    else:
        eight = "8B 的 `*qwen3_8b.yaml` 留在仓库里，没有打进本包（那些配置的 template 是 qwen3）。"
    return f"""> **工具调用格式已锁定为 {contract["label"]}**（{contract["habit"]}）。
>
> - 数据：assistant 侧 `{contract["style"]}`（observation / human / system 原样保留）
> - 训练：YAML `template: {contract["template"]}`
> - 推理：`--tool-call-parser {contract["parser"]}`
>
> {eight} 数据、template、parser 要一起用。

"""


def _remembered_profile(delivery_dir: Path) -> Optional[str]:
    path = delivery_dir / _PROFILE_CHOICE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("profile") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return resolve_export_profile(name)["id"]
    except ValueError:
        return None


def _apply_tool_style(files: Dict[str, List[Dict[str, Any]]], style: str) -> Dict[str, int]:
    """就地改写 assistant 侧工具调用。无法识别的块直接让这次打包失败。"""
    converted = unparsed = 0
    for rows in files.values():
        for row in rows:
            c, u = convert_sharegpt_row(row, style)
            converted += c
            unparsed += u
    if unparsed:
        raise ValueError(
            f"有 {unparsed} 处 assistant 工具调用无法识别，已停止打包。"
            "只接受 hermes JSON 或 Qwen XML 的 <tool_call> 块。"
        )
    return {"blocks_converted": converted, "blocks_unparsed": unparsed}


def _force_template(text: str, template: str) -> str:
    """把 YAML 的 template 锁成与数据同一套。行尾旧注释一并去掉，避免写着「勿用 qwen3」却已经改成 qwen3。"""
    lines = text.splitlines()
    out: List[str] = []
    found = False
    insert_at: Optional[int] = None
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("template:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}template: {template}")
            found = True
            continue
        out.append(line)
        if stripped.startswith("dataset:") or stripped.startswith("eval_dataset:"):
            insert_at = len(out)
    if not found:
        indent = ""
        if insert_at is not None:
            ref = out[insert_at - 1]
            indent = ref[: len(ref) - len(ref.lstrip())]
        added = f"{indent}template: {template}"
        if insert_at is None:
            out.append(added)
        else:
            out.insert(insert_at, added)
    return "\n".join(out).rstrip() + "\n"


def _readme(
    cfg_hash: str,
    stats: Dict[str, Any],
    counts: Dict[str, int],
    sources: Dict[str, str],
    contract: Dict[str, Any],
) -> str:
    sft = counts.get("sft.train.lf.jsonl", 0)
    sft_eval = counts.get("sft.eval.lf.jsonl", 0)
    dpo = counts.get("dpo.train.lf.jsonl", 0)
    dpo_eval = counts.get("dpo.eval.lf.jsonl", 0)
    src = ", ".join(f"{k}←{v}" for k, v in sources.items()) or "—"
    if contract.get("pack_8b_yaml"):
        eight = "8B 配置已打入，template 同为 qwen3。"
    else:
        eight = "8B 的 qwen3 配置没有打入这个包。"
    format_line = (
        f"\n工具调用格式：{contract['label']}（{contract['habit']}）。\n"
        f"数据风格 {contract['style']}，YAML template: {contract['template']}，"
        f"推理 --tool-call-parser {contract['parser']}。{eight}\n"
        "数据、template、parser 是一套。\n"
    )
    return f"""GoldThread 训练交付包（目标模型：Qwen/Qwen3.8-27B Dense）
=========================================================
{format_line}
解压后按 START_HERE.md 操作即可开训，不需要改文件名、不需要再跑规整脚本、不需要改 dataset_info。

快速开始（在解压目录里执行，dataset_dir 是相对路径 ./data）：
1. python check_data.py
   退出码必须是 0。它会核对每份 YAML 的 dataset 键、data/dataset_info.json 和 jsonl 是否对齐。
2. 27B SFT（不要用 llamafactory-cli train，16K 全词表 logits 在单卡 80G 上会 OOM）：
   CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft.yaml
3. 显存不够时：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_sft_qlora.yaml
4. SFT 合并出 DPO 的底座：llamafactory-cli export merge_lora.yaml
   （产出 /data/models/Qwen3.8-27B-merged，就是 train_dpo.yaml 的 model_name_or_path）
5. DPO：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_dpo.yaml
6. DPO 合并：llamafactory-cli export merge_dpo.yaml
   （产出 /data/models/Qwen3.8-27B-dpo-merged，推理直接用这个目录）
7. 训完单独评测：CUDA_VISIBLE_DEVICES=0 python train_lowmem.py eval_sft.yaml

包内结构：
  START_HERE.md         解压后的最短路径（以本文件布局为准）
  FORMAT_CONTRACT.json  本包锁定的工具调用格式（style / template / parser）
  handoff.md            环境与架构说明（开头有本包布局，冲突时以 YAML 为准）
  train_lowmem.py       27B 低显存启动器（SFT/eval 只算监督位置 logits）
  lowmem_dpo.py         DPO 低显存补丁（train_lowmem.py 自动加载）
  check_data.py         训练前数据契约检查（本包的验收命令）
  check_env.py          环境检查
  train_sft.yaml        dataset: gt_sft_train_lf / gt_sft_eval_lf
  train_dpo.yaml        dataset: gt_dpo_train_lf / gt_dpo_eval_lf
  merge_lora.yaml       SFT 合并（LoRA -> /data/models/Qwen3.8-27B-merged）
  merge_dpo.yaml        DPO 合并（LoRA -> /data/models/Qwen3.8-27B-dpo-merged）
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


def _start_here(counts: Dict[str, int], has_dpo: bool, contract: Dict[str, Any]) -> str:
    sft = counts.get("sft.train.lf.jsonl", 0)
    sft_eval = counts.get("sft.eval.lf.jsonl", 0)
    dpo = counts.get("dpo.train.lf.jsonl", 0)
    dpo_eval = counts.get("dpo.eval.lf.jsonl", 0)
    dpo_block = ""
    if has_dpo:
        dpo_block = f"""
## 4. DPO（可选，需先合并 SFT）

`train_dpo.yaml` 的 `model_name_or_path` 已经是 SFT 合并目录 `/data/models/Qwen3.8-27B-merged`，
用包内的 `merge_lora.yaml` 生成它（不用改 YAML 里的路径）：

```bash
cd <解压目录>
llamafactory-cli export merge_lora.yaml          # SFT LoRA -> /data/models/Qwen3.8-27B-merged
CUDA_VISIBLE_DEVICES=0 python train_lowmem.py train_dpo.yaml
llamafactory-cli export merge_dpo.yaml           # DPO LoRA -> /data/models/Qwen3.8-27B-dpo-merged
```

数据集已是 `gt_dpo_train_lf` / `gt_dpo_eval_lf`（{dpo} / {dpo_eval} 条）。
合并走 `export_device: cpu`，不占 GPU；内存需 ≥ 64GB（`export_size: 5`）。
"""
    else:
        dpo_block = """
## 4. DPO

这次打包没有可用的 DPO lf 文件，zip 里不会带 `train_dpo.yaml`。不要拿未打包的原始偏好对硬训。
"""
    parser = contract["parser"]
    contract_block = f"""
## 工具调用格式

本包锁定为 **{contract['label']}**（{contract['habit']}）。

- assistant 工具调用：`{contract['style']}`
- YAML `template: {contract['template']}`
- 推理 `--tool-call-parser {parser}`

改其中一处时，另外两处一起改。03_train 里的原文件保持不动，格式只写进这个 zip。
"""
    if contract.get("pack_8b_yaml"):
        smoke_block = """## 5. 8B 冒烟

包内的 `train_sft_qwen3_8b.yaml` / `train_sft_smoke_qwen3_8b.yaml`（若有）数据集键同样是 `gt_sft_*_lf`，`dataset_dir` 同样是 `./data`，`template` 与本包数据同为 `qwen3`。8B 可以直接：

```bash
cd <解压目录>
llamafactory-cli train train_sft_smoke_qwen3_8b.yaml
```

27B 仍走上面的 `train_lowmem.py train_sft.yaml`。
"""
    else:
        smoke_block = """## 5. 8B

本包是 Qwen XML，没有打入 `*qwen3_8b.yaml`。那些配置的 template 是 qwen3。要改成 Hermes JSON，在导出界面改选格式后重新打包。
"""
    ops_block = f"""
## 6. 训完之后：合并权重的高速推理（速查）

1. 先合并（SFT/DPO 的 LoRA 都走包内的 export 配置，产物目录名与各 YAML 里的路径一致）：

```bash
llamafactory-cli export merge_lora.yaml     # -> /data/models/Qwen3.8-27B-merged
llamafactory-cli export merge_dpo.yaml      # -> /data/models/Qwen3.8-27B-dpo-merged
```

2. 另开 conda 环境 `vllm`（不要装进训练 Python）；首次启动需系统 `g++`（flashinfer JIT）
3. 两个 parser 必须显式给：`--reasoning-parser qwen3`（否则思维链混进 content）+
   `--tool-call-parser {parser}`（与本包的数据和 YAML template 是一套）

```bash
vllm serve <SFT+DPO 合并目录> --served-model-name <交付包同名模型> \\
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \\
  --enable-auto-tool-choice --tool-call-parser {parser} --reasoning-parser qwen3
```

完整命令与排障（CUDA 驱动 vs toolkit、flash-attn/causal-conv1d 轮子选型）见仓库
`DATASET_BEST_PRACTICES.md` §七与仓库 `data/03_train/START_HERE.md` §8。parser 以本包 `FORMAT_CONTRACT.json` 为准。
"""
    return f"""# START HERE · GoldThread 交付 zip

在**解压出来的这一层目录**里操作。YAML 的 `dataset_dir` 是 `./data`，相对当前工作目录，不是相对 yaml 文件。
{contract_block}

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
├── START_HERE.md  README.txt  handoff.md  FORMAT_CONTRACT.json
├── train_sft.yaml            # dataset: gt_sft_train_lf
├── train_sft_qlora.yaml
├── train_dpo.yaml            # dataset: gt_dpo_train_lf（有 DPO 数据时才在）
├── merge_lora.yaml           # SFT LoRA -> /data/models/Qwen3.8-27B-merged
├── merge_dpo.yaml            # DPO LoRA -> /data/models/Qwen3.8-27B-dpo-merged（有 DPO 时才在）
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
{smoke_block}
{ops_block}
"""


def _prepend_banner(handoff: str, contract: Optional[Dict[str, Any]] = None) -> str:
    marker = "本交付包已按可直接训练的布局打包"
    if marker not in handoff[:800]:
        handoff = _HANDOFF_BANNER + handoff
    if contract and "工具调用格式已锁定" not in handoff[:2000]:
        if handoff.startswith(_HANDOFF_BANNER):
            handoff = _HANDOFF_BANNER + _format_banner(contract) + handoff[len(_HANDOFF_BANNER):]
        else:
            handoff = _format_banner(contract) + handoff
    return handoff



def compute_package_hash(
    cfg_h: str,
    prepared_files: Dict[str, List[Dict[str, Any]]],
    length_gate_fp: str,
    profile: str = "",
    extras: Optional[Dict[str, str | bytes]] = None,
) -> str:
    """8-hex fingerprint of delivery payload (post-gate lf bytes + config + gate version).

    Soft-compress / data rebuilds that change packed jsonl bytes, or a
    LENGTH_GATE_VERSION bump, produce a new zip filename even when
    pipeline.yaml config_hash is unchanged. `extras` covers the packed YAML /
    scripts, so editing a config or script also yields a new filename instead of
    silently reusing it under the old name.
    """
    parts: List[str] = [cfg_h]
    if profile:
        parts.append(f"profile:{profile}")
    for name, rows in sorted(prepared_files.items()):
        content = dumps_jsonl(rows)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        parts.append(f"{name}:{digest}")
    for name, content in sorted((extras or {}).items()):
        raw = content if isinstance(content, bytes) else content.encode("utf-8")
        parts.append(f"{name}:{hashlib.sha256(raw).hexdigest()}")
    parts.append(length_gate_fp)
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def build_delivery_zip(
    train_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log=None,
    profile: Optional[str] = None,
    delivery_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """打包 handoff + 规整后的数据集为 zip，返回路径与元信息。"""
    cfg = cfg or load_config()
    dirs = stage_dirs(cfg)
    root = Path(train_root or dirs["train"])
    build_manifest = read_json(root / "BUILD.json", default={}) or {}
    build_id = str(build_manifest.get("build_id") or "")
    # 发布产物统一落在 data/04_release/，与 03_train 工作区分离
    release_root = root.parent / "04_release"
    if delivery_dir:
        delivery_dir = Path(delivery_dir)
    elif build_id:
        delivery_dir = stage_dirs(cfg)["root"] / "04_release" / "delivery"
    else:
        delivery_dir = release_root / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_export_profile(profile or _remembered_profile(delivery_dir))

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

    _emit(
        f"[package] 导出格式 {spec['label']}："
        f"{spec['style']} / template {spec['template']} / parser {spec['parser']}"
    )
    style_stats = _apply_tool_style(prepared.files, spec["style"])
    _emit(f"[package] 已按该格式改写 {style_stats['blocks_converted']} 处工具调用")

    # YAML / 脚本先定下来（既用于指纹，也用于写包）：改了配置或脚本也要换 zip 名，
    # 否则会出现"同名不同内容"的包。
    packed_yaml_texts: Dict[str, str] = {}
    skipped_yaml: List[str] = []
    for name in _YAML_FILES:
        src = root / name
        if not src.is_file():
            continue
        if not profile_includes_yaml(name, spec):
            skipped_yaml.append(name)
            _emit(f"[package] 跳过 {name}：{spec['label']} 只带 template {spec['template']} 的配置")
            continue
        patched = _patch_yaml(src.read_text(encoding="utf-8"), name, available)
        if patched is None:
            skipped_yaml.append(name)
            continue
        packed_yaml_texts[name] = _force_template(patched, spec["template"])

    if "train_sft.yaml" not in packed_yaml_texts:
        raise FileNotFoundError("train_sft.yaml 未能打进交付包（缺少 gt_sft_*_lf 数据）")

    packed_merge_texts: Dict[str, str] = {}
    for name in _MERGE_YAML_FILES:
        src = root / name
        if not src.is_file():
            continue
        # merge_dpo 依赖 DPO 产物，没 DPO 数据时不留悬空指令
        if name == "merge_dpo.yaml" and not has_dpo:
            _emit("[package] 跳过 merge_dpo.yaml：本包没有 DPO 数据")
            continue
        packed_merge_texts[name] = _force_template(src.read_text(encoding="utf-8"), spec["template"])

    script_blobs: Dict[str, bytes] = {}
    for name in _REPO_SCRIPTS + _OPTIONAL_SCRIPTS:
        blob = _read_script(name, root)
        if blob is None:
            if name in ("train_lowmem.py", "lowmem_dpo.py", "check_data.py"):
                raise FileNotFoundError(f"交付包缺少脚本 {name}")
            continue
        script_blobs[name] = blob

    lg_fp = length_gate_mod.length_gate_fingerprint()
    pkg_h = compute_package_hash(
        cfg_h,
        prepared.files,
        lg_fp,
        spec["id"],
        extras={
            **packed_yaml_texts,
            **packed_merge_texts,
            **script_blobs,
            **({"build_id": build_id} if build_id else {}),
        },
    )
    zip_name = f"goldthread-delivery-{pkg_h}.zip"
    zip_path = delivery_dir / zip_name

    stats = read_json(root / "stats.json", default={}) or {}
    manifest_files: List[Dict[str, Any]] = []

    def _add(zf: zipfile.ZipFile, arc: str, content: str | bytes) -> None:
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        zf.writestr(arc, data)
        manifest_files.append({"path": arc, "size": len(data)})

    format_contract = {
        "profile": spec["id"],
        "label": spec["label"],
        "habit": spec["habit"],
        "tool_style": spec["style"],
        "train_template": spec["template"],
        "serve_tool_parser": spec["parser"],
        "blocks_converted": style_stats["blocks_converted"],
        "blocks_unparsed": style_stats["blocks_unparsed"],
        "pack_8b_yaml": bool(spec.get("pack_8b_yaml")),
        "note": "数据 tool_style、YAML template、推理 --tool-call-parser 是一套。",
    }

    _emit(f"[package] 正在写入 zip：{zip_name}…")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _add(zf, "README.txt", _readme(cfg_h, stats, prepared.counts, prepared.sources, spec))
        _add(zf, "START_HERE.md", _start_here(prepared.counts, has_dpo, spec))
        _add(zf, "FORMAT_CONTRACT.json", json.dumps(format_contract, ensure_ascii=False, indent=2) + "\n")
        _add(zf, "handoff.md", _prepend_banner(handoff_path.read_text(encoding="utf-8"), spec))

        if (root / "RESULT.md").is_file():
            _add(zf, "RESULT.md", (root / "RESULT.md").read_text(encoding="utf-8"))
        if (root / "stats.json").is_file():
            _add(zf, "stats.json", (root / "stats.json").read_text(encoding="utf-8"))

        packed_yaml = list(packed_yaml_texts)
        for name, text in packed_yaml_texts.items():
            _add(zf, name, text)

        packed_merge_yaml = list(packed_merge_texts)
        for name, text in packed_merge_texts.items():
            _add(zf, name, text)

        for name in _DS_FILES:
            src = root / name if (root / name).is_file() else _REPO_TOOLS / name
            if src.is_file():
                _add(zf, name, src.read_bytes())

        for name, blob in script_blobs.items():
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
            "build_id": build_id or None,
            "length_gate_version": lg_fp,
            "stats_generated_at": stats.get("generated_at"),
            "datasets": {key: entry.get("file_name") for key, entry in info.items()},
            "sources": prepared.sources,
            "dropped": prepared.dropped,
            "counts": prepared.counts,
            "length_gate": prepared.length_gate,
            "yaml": packed_yaml,
            "merge_yaml": packed_merge_yaml,
            "yaml_skipped": skipped_yaml,
            "format_contract": format_contract,
            "files": manifest_files,
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        manifest["files"] = manifest_files + [{"path": "MANIFEST.json", "size": len(manifest_json.encode("utf-8"))}]
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        zf.writestr("MANIFEST.json", manifest_json)

    (delivery_dir / _PROFILE_CHOICE).write_text(
        json.dumps(
            {
                "profile": spec["id"],
                "label": spec["label"],
                "updated_at": manifest["generated_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest_files) + 1,
        "generated_at": manifest["generated_at"],
        "config_hash": cfg_h,
        "package_hash": pkg_h,
        "build_id": build_id or None,
        "counts": prepared.counts,
        "sources": prepared.sources,
        "datasets": manifest["datasets"],
        "length_gate": prepared.length_gate,
        "format": format_contract,
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
            f"{spec['label']}, SFT lf {sft_n} / DPO lf {dpo_n}, {result['file_count']} 个文件, "
            f"length_gate salvaged={salv} overlength_at_default={over})"
        )
    return result
