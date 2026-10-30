#!/usr/bin/env bash
# 打 27B 训练包：数据 + 27B/8B 配置 + 全部脚本/文档
#
# 用法：bash tools/pack_27b.sh [输出路径]
#
# 设计（2026-10-28 起）：
#   1. 数据不再硬编码文件名 —— 从本包要发的 *.yaml 里解析 dataset / eval_dataset，
#      经 dataset_info.json 反查 file_name 后按需拷贝。yaml 换数据集名不用改这个脚本。
#   2. 额外数据（黄金轨迹、OpenAI 格式）存在就带上，缺了只告警。
#   3. dataset_info.json 裁剪为「只登记本包实际存在的文件」。
#   4. 打 完 做三向一致性自检（yaml → dataset_info → 文件），并生成 MANIFEST.json（sha256 + 条数）。
set -euo pipefail

OUT="${1:-data/04_release/packs/03_train_qwen38_27b.tar.gz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/data/03_train"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

DEST="$STAGE/03_train"
mkdir -p "$DEST"

# ── 配置（打包前必须存在；数据由 yaml 反查）────────────────────
CONFIGS=(
  # 27B 正式
  train_sft.yaml
  train_sft_qlora.yaml
  train_dpo.yaml
  train_dpo_qlora_27b.yaml
  train_dpo_qlora_27b_b.yaml
  eval_sft.yaml
  merge_lora.yaml
  merge_dpo.yaml
  train_sft_27b_smoke.yaml
  # 8B 冒烟（START_HERE 路径 A）
  train_sft_qwen3_8b.yaml
  train_sft_smoke_qwen3_8b.yaml
  train_dpo_qwen3_8b.yaml
  train_dpo_smoke_qwen3_8b.yaml
  eval_sft_qwen3_8b.yaml
  # deepspeed（多卡时才用）
  ds_z2_config.json
  ds_z3_config.json
)
for f in "${CONFIGS[@]}"; do
  [ -f "$SRC/$f" ] || { echo "[err] 缺少配置: $f" >&2; exit 1; }
  cp "$SRC/$f" "$DEST/"
done

# ── 脚本 ──────────────────────────────────────────────────────
SCRIPTS=(
  # 启动路径：分块 CE / 低显存 / 监控
  run_train.py
  chunked_ce_patch.py
  train_lowmem.py
  lowmem_dpo.py
  train_monitor.py
  # 数据规整
  normalize_sharegpt.py
  prep_dpo_window.py
  # 环境与数据自检
  check_env.py
  check_data.py
  # 验收
  smoke_infer_merged.py
  smoke_ab_dpo.py
  verify_dpo_adapter.py
  verify_merged_dpo.py
  verify_longctx.py
  verify_longctx_api.py
  probe_24g.py
  # 服务
  serve_api.sh
  serve_vllm.sh
  reasoning_proxy.py
)
for f in "${SCRIPTS[@]}"; do
  [ -f "$SRC/$f" ] || { echo "[err] 缺少脚本: $f" >&2; exit 1; }
  cp "$SRC/$f" "$DEST/"
  case "$f" in *.py|*.sh) chmod +x "$DEST/$f";; esac
done

# ── 文档 ──────────────────────────────────────────────────────
DOCS=(README.md README.txt START_HERE.md handoff.md RESULT.md RESULT_27b_run.md
      manifest.json MANIFEST.json stats.json index.json)
for f in "${DOCS[@]}"; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DEST/" || echo "[warn] 可选文档缺失: $f"
done

# ── 数据：从 yaml 反查，逐个拷贝 ───────────────────────────────
python - "$SRC" "$DEST" <<'PY'
import io, json, os, re, shutil, sys, glob

src, dest = sys.argv[1], sys.argv[2]
info = json.load(io.open(os.path.join(src, "dataset_info.json"), encoding="utf-8"))

want, missing_key = [], []
for y in sorted(glob.glob(os.path.join(dest, "*.yaml"))):
    s = io.open(y, encoding="utf-8").read()
    for key in ("dataset", "eval_dataset"):
        m = re.search(rf"^{key}\s*:\s*(\S+)", s, re.M)
        if not m:
            continue
        name = m.group(1).strip().strip("'\"")
        if name not in info:
            missing_key.append(f"{os.path.basename(y)}: {key}={name}")
            continue
        fn = info[name].get("file_name", "")
        if fn and fn not in want:
            want.append(fn)

# 额外数据：存在就带上（不在默认 yaml 里，但验收/混训会用）
EXTRA = [
    "sft_gold_trajectories_sharegpt.train.jsonl",
    "sft_gold_trajectories_sharegpt.eval.jsonl",
    "sft_step_windows_openai.train.jsonl",
    "sft_step_windows_openai.eval.jsonl",
]
for fn in EXTRA:
    if (os.path.exists(os.path.join(src, fn)) and fn not in want
            and any(m.get("file_name") == fn for m in info.values())):
        want.append(fn)

if missing_key:
    print("[err] yaml 引用了未登记的数据集：")
    for m in missing_key:
        print("  -", m)
    sys.exit(1)

for fn in want:
    p = os.path.join(src, fn)
    if not os.path.exists(p):
        print(f"[err] 数据集文件缺失: {fn}")
        sys.exit(1)
    shutil.copy2(p, os.path.join(dest, fn))

kept, dropped = {}, []
for name, meta in info.items():
    fn = meta.get("file_name", "")
    if fn and os.path.exists(os.path.join(dest, fn)):
        kept[name] = meta
    else:
        dropped.append(name)
io.open(os.path.join(dest, "dataset_info.json"), "w", encoding="utf-8").write(
    json.dumps(kept, ensure_ascii=False, indent=4) + "\n")
print(f"[data] 拷贝 {len(want)} 个数据文件: {want}")
print(f"[dataset_info] 保留 {len(kept)} 项，剔除 {len(dropped)} 项")
PY

# ── 自检：yaml → dataset_info → 文件 三向一致 ──────────────────
python - "$DEST" <<'PY'
import io, json, os, re, sys, glob, hashlib

dest = sys.argv[1]
info = json.load(io.open(os.path.join(dest, "dataset_info.json"), encoding="utf-8"))
bad, checked = [], 0
for y in sorted(glob.glob(os.path.join(dest, "*.yaml"))):
    s = io.open(y, encoding="utf-8").read()
    for key in ("dataset", "eval_dataset"):
        m = re.search(rf"^{key}\s*:\s*(\S+)", s, re.M)
        if not m:
            continue
        name = m.group(1).strip().strip("'\"")
        checked += 1
        if name not in info:
            bad.append(f"{os.path.basename(y)}: {key}={name} 未在 dataset_info.json 注册")
        elif not os.path.exists(os.path.join(dest, info[name]["file_name"])):
            bad.append(f"{os.path.basename(y)}: {key}={name} 指向的文件缺失")
if bad:
    print("[FAIL] 一致性检查失败：")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"[OK] YAML / dataset_info / 文件 三者一致（{checked} 处 dataset 引用）")

# ── MANIFEST.json：sha256 + 条数，便于对方校验 ────────────────
files, sha, counts = [], {}, {}
for fn in sorted(os.listdir(dest)):
    p = os.path.join(dest, fn)
    if not os.path.isfile(p):
        continue
    files.append(fn)
    if fn.endswith(".jsonl"):
        h = hashlib.sha256()
        n = 0
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha[fn] = h.hexdigest()
        with io.open(p, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        counts[fn] = n
manifest = {
    "package": "goldthread-03_train",
    "built_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    "datasets": {k: v.get("file_name") for k, v in info.items()},
    "row_counts": counts,
    "sha256": sha,
    "files": files,
}
io.open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8").write(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n")
print(f"[manifest] {len(files)} 个文件，{len(counts)} 个数据集")
PY

# ── 打包 ──────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$STAGE" 03_train

echo
echo "[done] $OUT"
ls -lh "$OUT"
echo "包内文件数: $(tar tzf "$OUT" | wc -l)"