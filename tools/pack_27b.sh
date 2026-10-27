#!/usr/bin/env bash
# 打 27B 训练包：只含 v2/v2w 数据 + 27B 配置 + 全部脚本/文档
# 用法：bash tools/pack_27b.sh [输出路径]
# 说明：包内 dataset_info.json 会被裁剪为「只登记本包实际存在的文件」，
#       避免使用者看到 v1 条目缺失而误判。
set -euo pipefail

OUT="${1:-03_train_qwen38_27b.tar.gz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/data/03_train"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

DEST="$STAGE/03_train"
mkdir -p "$DEST"

# ── 数据（只放实际训练用的）────────────────────────────────────
DATA=(
  sft_step_windows_sharegpt.train.v2.jsonl
  sft_step_windows_sharegpt.eval.v2.jsonl
  sft_gold_trajectories_sharegpt.train.v2.jsonl
  sft_gold_trajectories_sharegpt.eval.v2.jsonl
  dpo_preference_pairs.train.v2.jsonl
  dpo_preference_pairs.eval.v2.jsonl
  dpo_preference_pairs.train.v2w.jsonl
  dpo_preference_pairs.eval.v2w.jsonl
)
for f in "${DATA[@]}"; do
  [ -f "$SRC/$f" ] || { echo "[err] 缺少数据文件: $f" >&2; exit 1; }
  cp "$SRC/$f" "$DEST/"
done

# ── 配置 ──────────────────────────────────────────────────────
CONFIGS=(
  # 27B 正式
  train_sft.yaml
  train_sft_qlora.yaml
  train_dpo.yaml
  eval_sft.yaml
  # 8B 冒烟（START_HERE 的路径 A 需要，几 KB，一并带上）
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
  check_env.py
  check_data.py
  train_lowmem.py
  lowmem_dpo.py
  train_monitor.py
  smoke_infer_merged.py
  verify_longctx.py
  verify_longctx_api.py
  probe_24g.py
  prep_dpo_window.py
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
DOCS=(README.md START_HERE.md handoff.md RESULT.md stats.json index.json)
for f in "${DOCS[@]}"; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DEST/" || echo "[warn] 可选文档缺失: $f"
done

# ── dataset_info：裁剪为只含本包存在的文件 ────────────────────
python - "$SRC/dataset_info.json" "$DEST" <<'PY'
import io, json, os, sys
src, dest = sys.argv[1], sys.argv[2]
info = json.load(io.open(src, encoding="utf-8"))
kept, dropped = {}, []
for name, meta in info.items():
    fn = meta.get("file_name", "")
    if fn and os.path.exists(os.path.join(dest, fn)):
        kept[name] = meta
    else:
        dropped.append(name)
io.open(os.path.join(dest, "dataset_info.json"), "w", encoding="utf-8").write(
    json.dumps(kept, ensure_ascii=False, indent=4) + "\n")
print(f"[dataset_info] 保留 {len(kept)} 项，剔除 {len(dropped)} 项（v1 旧版）: {dropped}")
PY

# ── 自检：所有 YAML 引用的 dataset 必须在本包 dataset_info 内 ──
python - "$DEST" <<'PY'
import io, json, os, re, sys, glob
dest = sys.argv[1]
info = json.load(io.open(os.path.join(dest, "dataset_info.json"), encoding="utf-8"))
bad = []
for y in sorted(glob.glob(os.path.join(dest, "*.yaml"))):
    s = io.open(y, encoding="utf-8").read()
    for key in ("dataset", "eval_dataset"):
        m = re.search(rf"^{key}:\s*(\S+)", s, re.M)
        if not m:
            continue
        name = m.group(1).strip()
        if name not in info:
            bad.append(f"{os.path.basename(y)}: {key}={name} 未在 dataset_info.json 注册")
        elif not os.path.exists(os.path.join(dest, info[name]["file_name"])):
            bad.append(f"{os.path.basename(y)}: {key}={name} 指向的文件缺失")
if bad:
    print("[FAIL] 一致性检查失败：")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("[OK] YAML / dataset_info / file 三者一致")
PY

# ── 打包 ──────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$STAGE" 03_train

echo
echo "[done] $OUT"
ls -lh "$OUT"
echo
echo "包内文件数: $(tar tzf "$OUT" | wc -l)"