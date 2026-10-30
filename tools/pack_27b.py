#!/usr/bin/env python3
"""Python port of tools/pack_27b.sh (no bash/WSL required)."""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/04_release/packs/03_train_qwen38_27b.tar.gz")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "03_train"

CONFIGS = [
    "train_sft.yaml", "train_sft_qlora.yaml", "train_dpo.yaml",
    "train_dpo_qlora_27b.yaml", "train_dpo_qlora_27b_b.yaml",
    "eval_sft.yaml", "merge_lora.yaml", "merge_dpo.yaml", "train_sft_27b_smoke.yaml",
    "train_sft_qwen3_8b.yaml", "train_sft_smoke_qwen3_8b.yaml",
    "train_dpo_qwen3_8b.yaml", "train_dpo_smoke_qwen3_8b.yaml", "eval_sft_qwen3_8b.yaml",
    "ds_z2_config.json", "ds_z3_config.json",
]
SCRIPTS = [
    "run_train.py", "chunked_ce_patch.py", "train_lowmem.py", "lowmem_dpo.py", "train_monitor.py",
    "normalize_sharegpt.py", "prep_dpo_window.py", "check_env.py", "check_data.py",
    "smoke_infer_merged.py", "smoke_ab_dpo.py", "verify_dpo_adapter.py", "verify_merged_dpo.py",
    "verify_longctx.py", "verify_longctx_api.py", "probe_24g.py",
    "serve_api.sh", "serve_vllm.sh", "reasoning_proxy.py",
]
DOCS = [
    "README.md", "README.txt", "START_HERE.md", "handoff.md", "RESULT.md", "RESULT_27b_run.md",
    "manifest.json", "MANIFEST.json", "stats.json", "index.json", "ACTIVE_RELEASE.txt",
]
EXTRA = [
    "sft_gold_trajectories_sharegpt.train.jsonl",
    "sft_gold_trajectories_sharegpt.eval.jsonl",
    "sft_step_windows_openai.train.jsonl",
    "sft_step_windows_openai.eval.jsonl",
]

stage = Path(tempfile.mkdtemp(prefix="pack27b_"))
dest = stage / "03_train"
dest.mkdir(parents=True)

try:
    for f in CONFIGS:
        src = SRC / f
        if not src.is_file():
            raise SystemExit(f"[err] missing config: {f}")
        shutil.copy2(src, dest / f)

    for f in SCRIPTS:
        src = SRC / f
        if not src.is_file():
            raise SystemExit(f"[err] missing script: {f}")
        shutil.copy2(src, dest / f)

    for f in DOCS:
        src = SRC / f
        if src.is_file():
            shutil.copy2(src, dest / f)
        else:
            print(f"[warn] optional doc missing: {f}")

    info = json.loads((SRC / "dataset_info.json").read_text(encoding="utf-8"))
    want, missing_key = [], []
    for ypath in sorted(dest.glob("*.yaml")):
        text = ypath.read_text(encoding="utf-8")
        for key in ("dataset", "eval_dataset"):
            m = re.search(rf"^{key}\s*:\s*(\S+)", text, re.M)
            if not m:
                continue
            name = m.group(1).strip().strip("'\"")
            if name not in info:
                missing_key.append(f"{ypath.name}: {key}={name}")
                continue
            fn = info[name].get("file_name", "")
            if fn and fn not in want:
                want.append(fn)

    for fn in EXTRA:
        if (SRC / fn).is_file() and fn not in want and any(m.get("file_name") == fn for m in info.values()):
            want.append(fn)

    if missing_key:
        print("[err] yaml references unregistered datasets:")
        for m in missing_key:
            print("  -", m)
        raise SystemExit(1)

    for fn in want:
        src = SRC / fn
        if not src.is_file():
            raise SystemExit(f"[err] missing dataset file: {fn}")
        shutil.copy2(src, dest / fn)

    kept, dropped = {}, []
    for name, meta in info.items():
        fn = meta.get("file_name", "")
        if fn and (dest / fn).is_file():
            kept[name] = meta
        else:
            dropped.append(name)
    (dest / "dataset_info.json").write_text(json.dumps(kept, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"[data] copied {len(want)} files: {want}")
    print(f"[dataset_info] kept {len(kept)}, dropped {len(dropped)}")

    # consistency
    bad, checked = [], 0
    for ypath in sorted(dest.glob("*.yaml")):
        text = ypath.read_text(encoding="utf-8")
        for key in ("dataset", "eval_dataset"):
            m = re.search(rf"^{key}\s*:\s*(\S+)", text, re.M)
            if not m:
                continue
            name = m.group(1).strip().strip("'\"")
            checked += 1
            if name not in kept:
                bad.append(f"{ypath.name}: {key}={name} not in dataset_info")
            elif not (dest / kept[name]["file_name"]).is_file():
                bad.append(f"{ypath.name}: {key}={name} file missing")
    if bad:
        print("[FAIL]")
        for b in bad:
            print(" -", b)
        raise SystemExit(1)
    print(f"[OK] yaml/dataset_info/files consistent ({checked} refs)")

    files, sha, counts = [], {}, {}
    for fn in sorted(os.listdir(dest)):
        p = dest / fn
        if not p.is_file():
            continue
        files.append(fn)
        if fn.endswith(".jsonl"):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            sha[fn] = h.hexdigest()
            with io.open(p, encoding="utf-8") as f:
                counts[fn] = sum(1 for line in f if line.strip())
    manifest = {
        "package": "goldthread-03_train",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "release": "v20261027-1736-2715fdbe",
        "datasets": {k: v.get("file_name") for k, v in kept.items()},
        "row_counts": counts,
        "sha256": sha,
        "files": files,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {len(files)} files, {len(counts)} datasets")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(dest, arcname="03_train")
    print(f"\n[done] {OUT} ({OUT.stat().st_size/1048576:.1f} MB)")
    with tarfile.open(OUT, "r:gz") as tar:
        print(f"entries: {len(tar.getmembers())}")
    print("row_counts:", json.dumps(counts, ensure_ascii=False))
finally:
    shutil.rmtree(stage, ignore_errors=True)
