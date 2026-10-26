#!/usr/bin/env python3
"""环境自检：无 GPU 也能跑的检查 + 有 GPU 后才做的清单。

用法：
    python check_env.py --model /data/models/Qwen3.8-27B --data /data/goldthread/03_train
    python check_env.py                       # 只查软件栈与依赖版本

设计：每项检查独立 try，输出 [PASS] / [FAIL] / [SKIP]（无 GPU 时跳过）。
全部非 GPU 项 PASS 才说明"挂上卡就能直接开训"。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

RESULT = {"pass": 0, "fail": 0, "skip": 0, "fail_items": [], "skip_items": []}


def ok(msg: str) -> None:
    RESULT["pass"] += 1
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    RESULT["fail"] += 1
    RESULT["fail_items"].append(msg)
    print(f"  [FAIL] {msg}")


def skip(msg: str) -> None:
    RESULT["skip"] += 1
    RESULT["skip_items"].append(msg)
    print(f"  [SKIP] {msg}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


def ver(mod: str) -> str:
    try:
        m = __import__(mod)
    except Exception as exc:  # noqa: BLE001
        return f"<import 失败: {type(exc).__name__}>"
    return str(getattr(m, "__version__", "?"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--cutoff", type=int, default=16384)
    args = ap.parse_args()

    # ---------- A. 运行时 ----------
    section("A. 运行时与 GPU")
    print(f"  platform: {platform.platform()}")
    print(f"  python  : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print(f"  [WARN] 期望 Python 3.12，当前 {sys.version_info[:2]}（通常也能跑，注意轮子兼容）")
    try:
        import torch
        print(f"  torch   : {torch.__version__}  (编译时 CUDA {torch.version.cuda})")
        if torch.__version__.split("+")[0] != "2.7.1":
            print(f"  [WARN] 期望 torch 2.7.1，当前 {torch.__version__}")
        else:
            ok("torch 2.7.1")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            ok(f"CUDA 可用：{n} 张卡，{name}，capability {cap}，显存 {total:.0f} GB")
            ok(f"bf16 支持：{torch.cuda.is_bf16_supported()}")
            if total < 60:
                print(f"  [WARN] 显存 {total:.0f}GB 小于训练建议值（27B bf16 LoRA 需 ~64-72GB）")
        else:
            skip("CUDA 不可用（无卡模式正常）：nvidia-smi / bf16 / 显存 都要等挂卡后再验")
    except ImportError:
        bad("torch 未安装")
    if shutil.which("nvidia-smi"):
        os.system("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | head -3")
    else:
        skip("nvidia-smi 不存在（无卡/无驱动）")

    # ---------- B. 依赖版本 ----------
    section("B. 训练栈依赖")
    want = {
        "transformers": ("5.8.0", lambda v: v.startswith("5.8")),
        "accelerate": (">=1.3.0", lambda v: v not in ("?",) and v.split(".")[0].isdigit() and float(".".join(v.split(".")[:2])) >= 1.3),
        "peft": (">=0.13", lambda v: True),
        "trl": ("<=0.24", lambda v: float(".".join(v.split(".")[:2])) <= 0.24),
        "datasets": ("任意", lambda v: True),
        "safetensors": ("任意", lambda v: True),
    }
    for m, (desc, chk) in want.items():
        v = ver(m)
        if v.startswith("<"):
            bad(f"{m} 未安装（期望 {desc}）")
            continue
        try:
            good = chk(v)
        except Exception:  # noqa: BLE001
            good = True
        (ok if good else bad)(f"{m} {v}（期望 {desc}）")

    for m in ("flash_attn", "liger_kernel", "bitsandbytes", "llamafactory"):
        v = ver(m)
        if v.startswith("<"):
            if m == "flash_attn":
                print(f"  [WARN] {m} 未安装 → 训练配置请把 flash_attn 改成 sdpa")
            elif m == "liger_kernel":
                # 8B 配置 enable_liger_kernel: true 时是硬依赖，不装必 OOM
                print(f"  [WARN] {m} 未安装 → 8B 配置要求 Liger 融合 CE，不装会 OOM；"
                      f"装：pip install liger-kernel，或把 YAML 改 enable_liger_kernel: false（显存需更多）")
            elif m == "bitsandbytes":
                print(f"  [INFO] {m} 未安装（仅 QLoRA 4bit / DPO ref_model 8bit 量化需要）")
            else:
                bad(f"{m} 未安装（LLaMA-Factory 必须）")
        else:
            ok(f"{m} {v}")

    # ---------- C. 模型 ----------
    if args.model:
        section("C. 底座模型")
        mp = Path(args.model)
        if not mp.exists():
            bad(f"模型目录不存在：{mp}")
        else:
            cfg_p = mp / "config.json"
            if not cfg_p.exists():
                bad("config.json 缺失，下载不完整")
            else:
                cfg = json.load(open(cfg_p, encoding="utf-8"))
                tc = cfg.get("text_config") or {}
                arch = (cfg.get("architectures") or ["?"])[0]
                mt = cfg.get("model_type")
                mpe = tc.get("max_position_embeddings") or cfg.get("max_position_embeddings")
                lt = tc.get("layer_types") or []
                from collections import Counter
                print(f"  architectures: {arch}")
                print(f"  model_type   : {mt}")
                print(f"  max_position : {mpe} | rope_scaling: {tc.get('rope_scaling') or cfg.get('rope_scaling')}")
                print(f"  layers       : {len(lt)} -> {dict(Counter(lt)) or 'n/a'}")
                print(f"  multimodal   : {'vision_config' in cfg} (image_token_id={cfg.get('image_token_id')})")
                if mt == "qwen3_5" and arch == "Qwen3_5ForConditionalGeneration":
                    ok("架构与 handoff 记录一致（qwen3_5 / Qwen3_5ForConditionalGeneration）")
                else:
                    print(f"  [WARN] 架构与 handoff 记录不同，请重新核对训练配置")
                if (mpe or 0) >= 262144:
                    ok(f"上下文 {mpe} ≥ 256K")
                # 权重文件大小
                tot = sum(f.stat().st_size for f in mp.glob("*.safetensors")) / 1024 ** 3
                if tot > 0:
                    ok(f"权重体积 {tot:.1f} GB")
                else:
                    bad("没有 *.safetensors，可能下载不完整")

            # tokenizer + 模板
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
                ok(f"tokenizer 加载成功（vocab {len(tok)}）")
                if getattr(tok, "chat_template", None):
                    ok("chat_template 存在")
                else:
                    bad("tokenizer 无 chat_template")
                # 用真实 tokenizer 渲染一条样本，检查 think 注入
                if args.data:
                    t = next((Path(args.data) / x for x in (
                        "sft_step_windows_sharegpt.train.v2.jsonl",
                        "sft_step_windows_sharegpt.train.jsonl") if (Path(args.data) / x).exists()), None)
                    if t is not None and t.exists():
                        with t.open(encoding="utf-8") as f:
                            row = json.loads(f.readline())
                        conv = row["conversations"]
                        role_map = {"system": "system", "human": "user", "gpt": "assistant",
                                    "observation": "tool", "tool": "tool"}
                        msgs = [{"role": role_map.get(c["from"], "user"), "content": c["value"]} for c in conv]
                        try:
                            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                            o, c2 = text.count(" thinking"), text.count("<" + "/think>")
                            tcalls = text.count("<tool_call>")
                            ids = tok(text, add_special_tokens=False)["input_ids"]
                            msg = f"模板渲染：tokens={len(ids)}  thinking×{o} ×{c2} <tool_call>×{tcalls}"
                            (ok if (o == 1 and c2 == 1) else bad)(msg)
                            if o != 1 or c2 != 1:
                                print("         ↑ think 标记不是 1 对：训练时改用 template: qwen3_8（LLaMA-Factory 专用模板）")
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [WARN] 直接用 tokenizer 渲染失败（LLaMA-Factory 会用 template: qwen3_8 另行处理）：{exc}")
            except ImportError:
                bad("transformers 不可用，跳过 tokenizer 检查")
            except Exception as exc:  # noqa: BLE001
                bad(f"tokenizer 加载失败：{type(exc).__name__}: {str(exc)[:160]}")
    else:
        section("C. 底座模型")
        skip("未提供 --model，跳过模型检查")

    # ---------- D. 数据集 ----------
    section("D. 数据集")
    if args.data:
        d = Path(args.data)
        if not d.exists():
            bad(f"数据目录不存在：{d}")
        else:
            files = sorted(d.glob("*.jsonl"))
            if not files:
                bad("目录下没有 .jsonl")
            for p in files:
                n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
                print(f"  {p.name:<48} {n:>6} 条")
            # 必需数据集：优先 v2（实际训练用），兼容 v1 旧版
            pairs = [
                ("sft_step_windows_sharegpt.train.v2.jsonl", "SFT 训练集"),
                ("sft_step_windows_sharegpt.eval.v2.jsonl", "SFT 验证集"),
            ]
            for f, desc in pairs:
                (ok if (d / f).exists() else bad)(f"存在 {f}（{desc}）")
            # DPO 数据：v2w（窗口裁剪）优先，v2 兜底
            dpo_hit = next((x for x in ("dpo_preference_pairs.train.v2w.jsonl",
                                        "dpo_preference_pairs.train.v2.jsonl",
                                        "dpo_preference_pairs.train.jsonl")
                            if (d / x).exists()), None)
            (ok if dpo_hit else bad)(f"存在 DPO 训练集（{dpo_hit or 'v2w / v2 / v1 均未找到'}）")
            (ok if (d / "dataset_info.json").exists() else bad)("存在 dataset_info.json")
            # 长度分布（有 tokenizer 才有意义）：优先 v2
            t = next((d / x for x in ("sft_step_windows_sharegpt.train.v2.jsonl",
                                      "sft_step_windows_sharegpt.train.jsonl")
                      if (d / x).exists()), None)
            if t is not None and t.exists():
                lens = []
                over = 0
                tok = None
                try:
                    from transformers import AutoTokenizer
                    if args.model:
                        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                except Exception:  # noqa: BLE001
                    tok = None
                with t.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        chars = sum(len(str(c.get("value", ""))) + 16 for c in r["conversations"])
                        if tok is not None:
                            txt = "\n".join(str(c.get("value", "")) for c in r["conversations"])
                            n = len(tok(txt, add_special_tokens=False)["input_ids"])
                        else:
                            n = round(chars / 2.2)
                        lens.append(n)
                lens.sort()
                if lens:
                    p50 = lens[len(lens) // 2]
                    p99 = lens[min(len(lens) - 1, int(len(lens) * 0.99))]
                    over = sum(1 for x in lens if x > args.cutoff)
                    unit = "tokens(真实)" if tok is not None else "tokens(估算)"
                    print(f"  长度分布 {unit}: p50={p50} p99={p99} max={lens[-1]}")
                    if over:
                        print(f"  [WARN] {over} 条超过 cutoff_len={args.cutoff}，会被截断（可能丢目标步）")
                    else:
                        ok(f"cutoff_len={args.cutoff} 覆盖全部样本（0 截断）")
    else:
        skip("未提供 --data，跳过数据集检查")

    # ---------- E. 磁盘与内存 ----------
    section("E. 磁盘与内存")
    du = shutil.disk_usage("/")
    free_gb = du.free / 1024 ** 3
    (ok if free_gb >= 100 else bad)(f"/ 可用 {free_gb:.0f} GB（27B 训练建议 ≥100GB）")
    try:
        import psutil  # type: ignore
        ram = psutil.virtual_memory().total / 1024 ** 3
        (ok if ram >= 64 else bad)(f"主存 {ram:.0f} GB（建议 ≥64GB）")
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram = int(line.split()[1]) / 1024 ** 2
                        (ok if ram >= 64 else bad)(f"主存 {ram:.0f} GB（建议 ≥64GB）")
                        break
        except OSError:
            skip("无法读取主存信息")

    # ---------- F. 结论 ----------
    section("结论")
    print(f"  通过 {RESULT['pass']} 项，失败 {RESULT['fail']} 项，跳过 {RESULT['skip']} 项")
    if RESULT["fail_items"]:
        print("  必须修复：")
        for x in RESULT["fail_items"]:
            print(f"    - {x}")
    if RESULT["skip_items"]:
        print("  等挂卡后再验：")
        for x in RESULT["skip_items"]:
            print(f"    - {x}")
    if not RESULT["fail_items"]:
        print("\n  无卡阶段的软件栈检查全部通过。挂上卡后按顺序再做：")
        print("    1) nvidia-smi 确认卡型与显存")
        print("    2) python -c \"import torch;print(torch.cuda.is_bf16_supported())\"  期望 True")
        print("    3) python -c \"import flash_attn\"                                确认 kernel 可加载")
        print("    4) 用 train_sft.yaml 跑 20 步冒烟，盯 nvidia-smi 峰值 < 78GB")
        print("    5) 抽 5 条 eval 样本续写，确认输出 thinking + <tool_call>")
    return 0 if not RESULT["fail_items"] else 1


if __name__ == "__main__":
    sys.exit(main())