#!/usr/bin/env python3
"""全自动训练入口（项目设置见 settings.yaml 的 project 段）。

    python autotrain.py sft                 # 自动选方案 → 训练 → 合并 → 冒烟
    python autotrain.py dpo                 # 底座 = 最近一次 SFT 合并模型（或 settings.yaml: sft_merged）
    python autotrain.py eval                # 用最近一次 SFT adapter 跑验证集
    python autotrain.py plans [sft|dpo]     # 只看硬件检测结果和会选哪个方案
    python autotrain.py env [--install]     # 检查环境；--install 调 setup_env.sh 修复
    python autotrain.py merge --run <运行名>  # 补合并（磁盘不够被跳过时）；--adapter/--stage 导入外部 adapter
    python autotrain.py status              # 最近几次运行
    python autotrain.py issues              # 问题库列表，并重新生成 docs/TROUBLESHOOTING.md
    python autotrain.py record-issue ...    # 新问题解决后登记（见 --help）

训练常用参数：--plan <id> 指定方案；--gpus 0,1 指定卡；--dry-run 只生成配置和数据不启动；
--max-samples 20 快速试跑（不合并）；--no-merge / --no-smoke 跳过合并 / 冒烟。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / "configs"
TOOLS = ROOT / "tools"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ISSUES_FILE = ROOT / "issues" / "known_issues.json"
PENDING_FILE = ROOT / "issues" / "pending.jsonl"
HISTORY_FILE = RUNS / "history.jsonl"
ACTIVE_FILE = RUNS / "active.json"
TROUBLE_DOC = ROOT / "docs" / "TROUBLESHOOTING.md"
PY = sys.executable

# (import 名, 发行包名, 验证版本, 是否必需)。setup_env.sh 的 PINS 与此一致。
PINNED = [
    ("torch", "torch", "2.7.1", True),
    ("transformers", "transformers", "5.8.0", True),
    ("accelerate", "accelerate", "1.11.0", True),
    ("peft", "peft", "0.18.1", True),
    ("trl", "trl", "0.24.0", True),
    ("datasets", "datasets", "4.0.0", True),
    ("llamafactory", "llamafactory", "", True),
    ("fla", "flash-linear-attention", "0.4.2", True),
    ("causal_conv1d", "causal-conv1d", "", True),
    ("deepspeed", "deepspeed", "0.19.9", False),
    ("flash_attn", "flash-attn", "", False),
]

STAGE_CONFIG = {"sft": "sft.yaml", "dpo": "dpo.yaml", "eval": "eval_sft.yaml"}

_C = {"red": "31;1", "green": "32;1", "yellow": "33;1", "cyan": "36;1", "dim": "2"}


def say(msg: str, color: str = "cyan") -> None:
    tag = f"\x1b[{_C[color]}m[autotrain]\x1b[0m" if sys.stdout.isatty() else "[autotrain]"
    print(f"{tag} {msg}", flush=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(obj: dict, path: Path) -> None:
    import yaml

    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def project() -> dict:
    return settings().get("project") or {}


def stage_keys(stage: str) -> list[str]:
    """[训练集键, 验证集键]，缺的省略。"""
    ds = (project().get("datasets") or {}).get(stage) or {}
    keys = [k for k in (ds.get("train"), ds.get("eval")) if k]
    if not keys:
        raise SystemExit(f"settings.yaml 的 project.datasets.{stage} 没配置数据集键")
    return keys


def template() -> str:
    return project().get("template") or "qwen3_8"


def settings() -> dict:
    return load_yaml(ROOT / "settings.yaml")


# ============================================================ 硬件
def detect_hardware(gpu_filter: list[int] | None = None) -> dict:
    hw: dict = {"gpus": [], "nvlink": False, "ram_gb": 0.0}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        hw["error"] = f"nvidia-smi 不可用：{exc}"
        out = ""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allow = {int(x) for x in visible.split(",") if x.strip().isdigit()} if visible else None
    busy_mb = int(settings().get("gpu_busy_mb", 2048))
    for line in out.strip().splitlines():
        idx, name, total, used = [x.strip() for x in line.split(",")]
        idx = int(idx)
        if allow is not None and idx not in allow:
            continue
        if gpu_filter is not None and idx not in gpu_filter:
            continue
        hw["gpus"].append({
            "index": idx, "name": name, "total_gb": int(total) / 1024,
            "used_mb": int(used), "busy": int(used) > busy_mb,
        })
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=20).stdout
        hw["nvlink"] = bool(re.search(r"\bNV\d+\b", topo))
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                hw["ram_gb"] = int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return hw


def describe_hw(hw: dict) -> str:
    if not hw["gpus"]:
        return hw.get("error", "没有检测到 GPU")
    parts = [f"GPU{g['index']} {g['name']} {g['total_gb']:.0f}GB{'（已被占用 %.0fGB）' % (g['used_mb'] / 1024) if g['busy'] else ''}"
             for g in hw["gpus"]]
    link = "NVLink" if hw["nvlink"] else "PCIe（无 NVLink）"
    return "；".join(parts) + f"；卡间 {link}；内存 {hw['ram_gb']:.0f}GB"


# ============================================================ 方案
def load_plans(stage: str) -> list[dict]:
    plans = load_yaml(CONFIGS / "plans.yaml").get(stage) or []
    if not plans:
        raise SystemExit(f"configs/plans.yaml 里没有 {stage} 方案")
    return plans


def find_plan(stage: str, plan_id: str) -> dict:
    for p in load_plans(stage):
        if p["id"] == plan_id:
            return p
    raise SystemExit(f"没有方案 {plan_id}；可选：{', '.join(p['id'] for p in load_plans(stage))}")


def usable_gpus(hw: dict, plan: dict, ignore_busy: bool) -> list[int]:
    return [g["index"] for g in hw["gpus"]
            if g["total_gb"] >= plan.get("min_gpu_mem_gb", 0) and (ignore_busy or not g["busy"])]


def fit_plan(plan: dict, hw: dict, ignore_busy: bool) -> list[int] | None:
    gpus = usable_gpus(hw, plan, ignore_busy)
    if len(gpus) < plan.get("min_gpus", 1):
        return None
    return gpus[: plan.get("max_procs") or len(gpus)]


def choose_plan(stage: str, hw: dict, plan_id: str | None, ignore_busy: bool) -> tuple[dict, list[int]]:
    if plan_id:
        plan = find_plan(stage, plan_id)
        gpus = fit_plan(plan, hw, ignore_busy)
        if gpus is None:
            raise SystemExit(f"方案 {plan_id} 需要 {plan.get('min_gpus', 1)} 张 ≥{plan.get('min_gpu_mem_gb')}GB 的空闲卡；"
                             f"当前：{describe_hw(hw)}")
        return plan, gpus
    for plan in load_plans(stage):
        if not plan.get("auto_select"):
            continue
        gpus = fit_plan(plan, hw, ignore_busy)
        if gpus is not None:
            return plan, gpus
    busy = [g for g in hw["gpus"] if g["busy"]]
    hint = "有卡被占用，等它空出来或用 --gpus 指定空闲卡。" if busy else \
        "本项目的方案都需要 80GB 级显卡（27B bf16 权重约 54GB）。"
    raise SystemExit(f"没有适合当前硬件的 {stage} 方案。{describe_hw(hw)}。{hint}")


def grad_accum(plan: dict, nproc: int) -> int:
    return max(1, int(plan.get("global_batch", 4)) // nproc)


def cmd_plans(args) -> int:
    hw = detect_hardware(parse_gpus(args.gpus))
    say(f"硬件：{describe_hw(hw)}")
    stages = [args.stage] if args.stage else ["sft", "dpo"]
    for stage in stages:
        print(f"\n== {stage} ==")
        try:
            chosen, gpus = choose_plan(stage, hw, None, ignore_busy=args.ignore_busy)
        except SystemExit as exc:
            chosen, gpus = None, []
            print(f"  {exc}")
        for p in load_plans(stage):
            mark = "→" if chosen and p["id"] == chosen["id"] else " "
            fit = fit_plan(p, hw, True)
            tag = "可用" if fit is not None else "硬件不满足"
            auto = "" if p.get("auto_select") else "（仅降级/手动）"
            print(f" {mark} {p['id']:<20} cutoff={p['cutoff_len']:<6} {p.get('deepspeed') or '-':<20} "
                  f"[{p['status']}] {tag}{auto}\n      {p.get('note', '')}")
        if chosen:
            print(f"  选中 {chosen['id']}：GPU {gpus}，梯度累积 {grad_accum(chosen, len(gpus))}")
    return 0


# ============================================================ 环境
def probe_env() -> dict:
    code = (
        "import importlib, importlib.metadata as m, json\n"
        f"pins = {[(a, b) for a, b, _, _ in PINNED]!r}\n"
        "res = {}\n"
        "for mod, dist in pins:\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        try: res[mod] = m.version(dist)\n"
        "        except Exception: res[mod] = 'unknown'\n"
        "    except Exception as e:\n"
        "        res[mod] = None\n"
        "try:\n"
        "    import torch; res['_cuda'] = torch.cuda.is_available(); res['_bf16'] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n"
        "except Exception: pass\n"
        "try:\n"
        f"    from llamafactory.data.template import TEMPLATES; res['_template'] = {template()!r} in TEMPLATES\n"
        "except Exception: res['_template'] = None\n"
        "print('@@ENV@@' + json.dumps(res))\n"
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=600)
    for line in out.stdout.splitlines():
        if line.startswith("@@ENV@@"):
            return json.loads(line[7:])
    raise SystemExit(f"环境探测失败：{out.stderr[-2000:]}")


def check_env(verbose: bool = True) -> tuple[list[str], dict, dict]:
    """返回 (必须修复的问题, 版本快照, 需要自动加的 yaml 覆盖)。"""
    env = probe_env()
    problems, overrides = [], {}
    for mod, dist, want, required in PINNED:
        have = env.get(mod)
        if have is None:
            if mod == "flash_attn":
                overrides["flash_attn"] = "sdpa"
                verbose and print(f"  [WARN] flash-attn 未安装 → 训练自动改用 sdpa")
            elif required:
                problems.append(f"{dist} 未安装")
                verbose and print(f"  [FAIL] {dist} 未安装")
            else:
                verbose and print(f"  [WARN] {dist} 未安装（ZeRO-3 方案需要）")
            continue
        if want and not have.startswith(want):
            strict = mod == "transformers"
            if strict:
                problems.append(f"transformers 是 {have}，必须 5.8.0")
            verbose and print(f"  [{'FAIL' if strict else 'WARN'}] {dist} {have}（验证版本 {want}）")
        else:
            verbose and print(f"  [ok] {dist} {have}")
    if env.get("_template") is False:
        problems.append(f"LLaMA-Factory 没有 {template()} 模板（版本太旧或模板名写错）")
        verbose and print(f"  [FAIL] LLaMA-Factory 没有 {template()} 模板（版本太旧或模板名写错）")
    if env.get("_cuda") and not env.get("_bf16"):
        problems.append("GPU 不支持 bf16")
        verbose and print("  [FAIL] GPU 不支持 bf16")
    return problems, env, overrides


def run_setup() -> bool:
    say("执行 setup_env.sh 修复环境…")
    return subprocess.run(["bash", str(ROOT / "setup_env.sh")], env={**os.environ, "PYTHON": PY}).returncode == 0


def cmd_env(args) -> int:
    say(f"Python：{PY}")
    problems, _, _ = check_env()
    if problems and args.install:
        return 0 if run_setup() else 1
    if problems:
        say("环境未通过：" + "；".join(problems) + "。运行 python autotrain.py env --install 自动修复", "red")
        return 1
    say("环境通过", "green")
    return 0


def check_model(path: str) -> list[str]:
    p = Path(path)
    if not (p / "config.json").is_file():
        return [f"{path} 没有 config.json（路径错或下载不完整）"]
    errs = []
    idx = p / "model.safetensors.index.json"
    if idx.is_file():
        want = set(json.loads(idx.read_text())["weight_map"].values())
        missing = sorted(w for w in want if not (p / w).is_file())
        if missing:
            errs.append(f"{path} 缺 {len(missing)} 个权重分片（如 {missing[0]}），重新下载")
    elif not list(p.glob("*.safetensors")):
        errs.append(f"{path} 没有 safetensors 权重")
    cfg = json.loads((p / "config.json").read_text())
    want = project().get("model_type")
    if want and cfg.get("model_type") != want:
        say(f"[WARN] {path} 的 model_type 是 {cfg.get('model_type')}，方案表按 {want} 调参", "yellow")
    return errs


# ============================================================ 问题库
def load_issues() -> dict:
    return json.loads(ISSUES_FILE.read_text(encoding="utf-8"))


def save_issues(db: dict) -> None:
    ISSUES_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_troubleshooting(db)


TAIL_LINES = 400


def log_tail(log_text: str) -> str:
    """日志开头会打印训练样本原文（工具输出里可能有 Killed、Error 等字样），默认只匹配末尾。"""
    return "\n".join(log_text.splitlines()[-TAIL_LINES:])


def diagnose(log_text: str, rc: int, timed_out: bool, stage: str) -> dict | None:
    specials = set()
    if timed_out:
        specials.add("__timeout__")
    if rc in (-9, 137) or (rc < 0 and -rc == signal.SIGKILL):
        specials.add("__exit_sigkill__")
    tail = log_tail(log_text)
    for issue in load_issues()["issues"]:
        stages = issue.get("stages", ["*"])
        if "*" not in stages and stage not in stages:
            continue
        text = log_text if issue.get("scope") == "full" else tail
        for pat in issue.get("patterns", []):
            if pat in specials or (not pat.startswith("__") and re.search(pat, text, re.I | re.M)):
                return issue
    return None


def bump_issue(issue_id: str) -> None:
    db = load_issues()
    for issue in db["issues"]:
        if issue["id"] == issue_id:
            issue["hits"] = int(issue.get("hits", 0)) + 1
            issue["last_seen"] = now()
    save_issues(db)


EXC_LINE = re.compile(r"^(?:\[stderr\]\s*)?(?:\[rank\d+\]:\s*)?([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))\s*:\s*(.+)$", re.M)


def error_signature(log_text: str) -> tuple[str, str]:
    """(人读签名, 可复用的匹配正则)。"""
    log_text = log_tail(log_text)
    hits = EXC_LINE.findall(log_text)
    if hits:
        exc, msg = hits[-1]
        msg = msg.strip()[:160]
        short = exc.split(".")[-1]
        stem = re.sub(r"\d+", "#", msg[:80])
        pat = re.escape(short) + r".*" + re.escape(stem).replace("\\#", r"\d+") if stem else re.escape(short)
        return f"{short}: {msg}", pat
    tail = [ln for ln in log_text.strip().splitlines() if ln.strip()][-1:] or ["(空日志)"]
    return tail[0][:160], re.escape(tail[0][:80])


def record_pending(run: "Run", attempt: dict, log_text: str) -> dict:
    sig, pat = error_signature(log_text)
    entry = {
        "id": f"p{int(time.time())}", "time": now(), "status": "open", "stage": run.stage,
        "run": run.name, "attempt": attempt["n"], "plan": attempt["plan"], "rc": attempt["rc"],
        "signature": sig, "suggested_pattern": pat,
        "yaml_overrides": attempt["yaml_overrides"], "env_overrides": attempt["env_overrides"],
        "packages": run.packages, "log": attempt["log"],
        "log_tail": "\n".join(log_text.splitlines()[-60:]),
    }
    PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    render_troubleshooting(load_issues())
    return entry


def load_pending() -> list[dict]:
    if not PENDING_FILE.is_file():
        return []
    return [json.loads(x) for x in PENDING_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]


def save_pending(rows: list[dict]) -> None:
    PENDING_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def learn_from_success(run: "Run", attempt: dict) -> None:
    """同阶段有未解决的失败、这次成功了：把两次之间的差异登记成新问题（needs_review）。"""
    rows = load_pending()
    changed = False
    db = load_issues()
    for row in rows:
        if row["status"] != "open" or row["stage"] != run.stage:
            continue
        yaml_diff = {k: v for k, v in attempt["yaml_overrides"].items() if row["yaml_overrides"].get(k) != v}
        env_diff = {k: v for k, v in attempt["env_overrides"].items() if row["env_overrides"].get(k) != v}
        pkg_diff = {k: f"{row['packages'].get(k)} → {v}" for k, v in run.packages.items()
                    if not k.startswith("_") and row.get("packages", {}).get(k) != v}
        plan_changed = attempt["plan"] != row["plan"]
        if yaml_diff or env_diff:
            action = {"type": "retry", "yaml": yaml_diff, "env": env_diff}
        elif pkg_diff:
            cmds = [f"{{python}} -m pip install {dist}=={run.packages[mod]}"
                    for mod, dist, _, _ in PINNED if mod in pkg_diff and run.packages.get(mod) not in (None, "unknown")]
            action = {"type": "install", "commands": cmds} if cmds else {"type": "abort"}
        else:
            action = {"type": "abort"}
        fix_desc = []
        if plan_changed:
            fix_desc.append(f"方案 {row['plan']} → {attempt['plan']}")
        if yaml_diff:
            fix_desc.append(f"YAML {yaml_diff}")
        if env_diff:
            fix_desc.append(f"环境变量 {env_diff}")
        if pkg_diff:
            fix_desc.append(f"包版本 {pkg_diff}")
        issue_id = f"learned_{row['id']}"
        db["issues"].insert(len([i for i in db["issues"] if i.get("category") == "runtime"]), {
            "id": issue_id,
            "title": f"[自动学习] {row['signature'][:80]}",
            "category": "runtime",
            "stages": [run.stage],
            "patterns": [row["suggested_pattern"]],
            "diagnosis": f"{row['time']} 在运行 {row['run']} 中首次出现，{now()} 在运行 {run.name} 中解决。",
            "action": action,
            "manual": "解决时的变化：" + ("；".join(fix_desc) if fix_desc else "配置没有变化（可能是外部环境修复），请人工补充") ,
            "source": "learned", "needs_review": True, "hits": 1, "last_seen": now(),
        })
        row.update(status="resolved", resolved_by=run.name, issue_id=issue_id)
        changed = True
        say(f"已把未识别问题 {row['id']} 登记为 {issue_id}（需人工复核：python autotrain.py issues）", "green")
    if changed:
        save_pending(rows)
        save_issues(db)


def render_troubleshooting(db: dict) -> None:
    groups = {"runtime": [], "quality": [], "deploy": []}
    for issue in db["issues"]:
        groups.setdefault(issue.get("category", "runtime"), []).append(issue)
    act_name = {"next_plan": "自动换下一个方案", "retry": "自动带参数重试", "install": "自动安装后重试", "abort": "停下，人工处理"}
    lines = [
        "# 问题库",
        "",
        "本文件由 `python autotrain.py issues` 根据 `issues/known_issues.json` 生成，不要手改。",
        "训练失败时 autotrain 会按表中顺序匹配日志，命中后执行「自动处理」；没命中的会记入下方「未识别问题」。",
        "",
    ]
    titles = {"runtime": "训练运行时（自动匹配）", "quality": "训练效果（人工判断）", "deploy": "合并与部署（人工判断）"}
    for cat, items in groups.items():
        if not items:
            continue
        lines += [f"## {titles.get(cat, cat)}", ""]
        for i in items:
            act = i.get("action", {})
            detail = ""
            if act.get("type") == "retry":
                detail = "：" + ", ".join(f"{k}={v}" for k, v in {**act.get("yaml", {}), **act.get("env", {})}.items())
            elif act.get("type") == "install":
                detail = "：`" + "`; `".join(c.replace("{python} -m ", "") for c in act.get("commands", [])) + "`"
            review = " ⚠ 自动学习，待复核" if i.get("needs_review") else ""
            lines += [f"### {i['title']}{review}", "",
                      f"- 编号：`{i['id']}`；阶段：{', '.join(i.get('stages', ['*']))}；命中 {i.get('hits', 0)} 次"
                      + (f"，最近 {i['last_seen']}" if i.get("last_seen") else "")]
            if i.get("patterns"):
                lines.append("- 日志特征：" + "、".join(f"`{p}`" for p in i["patterns"]))
            lines += [f"- 原因：{i['diagnosis']}"]
            if cat == "runtime":
                lines.append(f"- 自动处理：{act_name.get(act.get('type'), act.get('type'))}{detail}")
            lines += [f"- 人工处理：{i.get('manual', '')}", ""]
    pend = [p for p in load_pending() if p["status"] == "open"]
    lines += ["## 未识别问题（待解决）", ""]
    if not pend:
        lines += ["暂无。", ""]
    for p in pend:
        lines += [f"- `{p['id']}` {p['time']} {p['stage']}/{p['plan']}：`{p['signature']}`（日志 `{p['log']}`）"]
    lines.append("")
    TROUBLE_DOC.parent.mkdir(parents=True, exist_ok=True)
    TROUBLE_DOC.write_text("\n".join(lines), encoding="utf-8")


def cmd_issues(args) -> int:
    db = load_issues()
    for i in db["issues"]:
        flag = " ⚠待复核" if i.get("needs_review") else ""
        print(f"{i['id']:<28} {i.get('category', ''):<8} {i.get('action', {}).get('type', ''):<10} "
              f"hits={i.get('hits', 0):<3} {i['title']}{flag}")
    pend = [p for p in load_pending() if p["status"] == "open"]
    if pend:
        print("\n未识别问题：")
        for p in pend:
            print(f"  {p['id']}  {p['time']}  {p['stage']}/{p['plan']}  {p['signature']}")
    render_troubleshooting(db)
    say(f"已更新 {TROUBLE_DOC.relative_to(ROOT)}")
    return 0


def cmd_record_issue(args) -> int:
    db = load_issues()
    pend_rows = load_pending()
    src = next((p for p in pend_rows if p["id"] == args.from_pending), None) if args.from_pending else None
    if args.from_pending and src is None:
        raise SystemExit(f"pending.jsonl 里没有 {args.from_pending}")
    patterns = args.pattern or ([src["suggested_pattern"]] if src else [])
    for pat in patterns:
        re.compile(pat)
    action = json.loads(args.action) if args.action else {"type": "abort"}
    if action.get("type") not in ("next_plan", "retry", "install", "abort"):
        raise SystemExit("action.type 只能是 next_plan / retry / install / abort")
    issue = {
        "id": args.id, "title": args.title or (src["signature"][:80] if src else args.id),
        "category": args.category, "stages": args.stages.split(","), "patterns": patterns,
        "diagnosis": args.diagnosis, "action": action, "manual": args.manual,
        "source": "manual", "hits": 0, "last_seen": None,
    }
    if args.scope == "full":
        issue["scope"] = "full"
    old = next((i for i in db["issues"] if i["id"] == args.id), None)
    if old:
        old.update({k: v for k, v in issue.items() if k not in ("hits", "last_seen")})
        old.pop("needs_review", None)
        say(f"已更新 {args.id}")
    else:
        pos = len([i for i in db["issues"] if i.get("category") == "runtime"]) if args.category == "runtime" else len(db["issues"])
        db["issues"].insert(pos, issue)
        say(f"已登记 {args.id}")
    if src:
        src.update(status="resolved", issue_id=args.id, resolved_at=now())
        save_pending(pend_rows)
    save_issues(db)
    return 0


# ============================================================ 运行
def parse_gpus(text: str | None) -> list[int] | None:
    return [int(x) for x in text.split(",")] if text else None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def history() -> list[dict]:
    if not HISTORY_FILE.is_file():
        return []
    return [json.loads(x) for x in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]


def latest(stage: str, key: str) -> str | None:
    for row in reversed(history()):
        if row["stage"] == stage and row["status"] == "ok" and row.get(key) and Path(row[key]).exists():
            return row[key]
    return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_active() -> dict | None:
    """正在运行（或异常退出没清理）的那次运行；alive 表示进程是否还在。"""
    if not ACTIVE_FILE.is_file():
        return None
    try:
        info = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except ValueError:
        return None
    info["alive"] = pid_alive(int(info.get("pid", 0)))
    return info


def find_adapters() -> list[str]:
    """saves_dir 及其上一级目录里的 LoRA adapter（含本项目之外训练的）。"""
    saves = Path(settings()["saves_dir"])
    found = {p.parent for root in (saves, saves.parent) if root.is_dir()
             for p in root.glob("*/adapter_config.json")}
    return sorted(str(d) for d in found)


def append_history(row: dict) -> None:
    RUNS.mkdir(exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================ 磁盘
def _existing(path: str) -> Path:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return p


def free_gb(path: str) -> float:
    import shutil

    return shutil.disk_usage(_existing(path)).free / 1024 ** 3


def writable(path: str) -> bool:
    return os.access(_existing(path), os.W_OK)


def space_problem(path: str, need_gb: float) -> str | None:
    if not writable(path):
        return f"{path} 不可写（只读文件系统？改 settings.yaml）"
    have = free_gb(path)
    if have < need_gb:
        return f"{path} 所在磁盘只剩 {have:.0f}GB，需要 ≥{need_gb:.0f}GB"
    return None


def cleanup_hint(path: str) -> str:
    root = _existing(path)
    while root.parent != root and os.path.ismount(root) is False:
        root = root.parent
    out = subprocess.run(f"du -sh {shlex.quote(str(root))}/* {shlex.quote(str(root))}/.Trash* 2>/dev/null | sort -h | tail -5",
                         shell=True, capture_output=True, text=True).stdout.strip()
    return f"{root} 上最大的几项：\n{out}" if out else ""


# ============================================================ 合并与冒烟
def merge_model(base: str, adapter: str, stage: str, tag: str, log: Path, retry: str = "") -> str | None:
    cfg = settings()
    prefix = project().get("merged_prefix") or "merged"
    export_dir = str(Path(cfg["merged_dir"]) / f"{prefix}-{stage}-{tag}")
    if (Path(export_dir) / "config.json").is_file():
        say(f"{export_dir} 已存在，不重复合并（要重做先删掉它）", "yellow")
        return export_dir
    problem = space_problem(cfg["merged_dir"], float(cfg.get("min_free_gb_merge", 60)))
    if problem:
        say(f"跳过合并：{problem}。腾出空间后运行：{retry or f'python autotrain.py merge --run {tag}'}", "yellow")
        hint = cleanup_hint(cfg["merged_dir"])
        if hint:
            print(hint)
        return None
    lf = Path(PY).parent / "llamafactory-cli"
    cmd = [str(lf) if lf.exists() else "llamafactory-cli", "export", "--model_name_or_path", base,
           "--adapter_name_or_path", adapter, "--template", template(),
           "--finetuning_type", "lora", "--trust_remote_code", "true",
           "--export_dir", export_dir, "--export_size", "5"]
    say(f"合并 adapter → {export_dir}（约 52GB，内存峰值约 60GB）")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (Path(export_dir) / "config.json").is_file():
        import shutil

        shutil.rmtree(export_dir, ignore_errors=True)
        say(f"合并失败，已删除不完整的 {export_dir}，见 {log}（常见原因：内存或磁盘不足，见问题 merge_cpu_oom / disk_full）", "red")
        return None
    say("合并完成", "green")
    return export_dir


def smoke_model(model: str, gpu: int, log: Path) -> dict:
    say("冒烟测试：5 条验证样本续写…")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with log.open("w", encoding="utf-8") as fh:
        key = stage_keys("eval")[-1]
        info = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
        data = str(DATA / info[key]["file_name"])
        rc = subprocess.run([PY, str(TOOLS / "smoke_infer.py"), "--model", model, "--data", data, "--n", "5"],
                            cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[smoke\] 汇总：(\d+)/(\d+)", text)
    res = {"rc": rc, "passed": m.group(0) if m else "未完成", "log": str(log)}
    say(f"冒烟：{res['passed']}（{log}）", "green" if rc == 0 else "yellow")
    return res


class Run:
    def __init__(self, stage: str, args):
        self.stage = stage
        self.args = args
        self.cfg = settings()
        self.name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{stage}"
        while (RUNS / self.name).exists():
            self.name += "x"
        self.dir = RUNS / self.name
        self.attempts: list[dict] = []
        self.packages: dict = {}
        self.data_dirs: dict[int, Path] = {}
        self.work = Path(self.cfg["work_dir"]) / self.name
        self.train_dir = str(self.work / "output")
        self.output_dir = str(Path(self.cfg["saves_dir"]) / self.name)
        self.trial = bool(getattr(args, "max_samples", None) or getattr(args, "probe", False))

    # ---------- 准备 ----------
    def base_model(self) -> str:
        if self.stage == "dpo":
            m = self.args.model or self.cfg.get("sft_merged") or latest("sft", "merged")
            if not m:
                found = find_adapters()
                hint = ("\n已有的 adapter（如果其中有想用的 SFT 结果，先导入合并，不用重训）：\n"
                        + "\n".join(f"  {d}" for d in found)
                        + "\n导入：python autotrain.py merge --stage sft --adapter <目录>") if found else ""
                raise SystemExit("DPO 需要 SFT 合并模型：先跑 python autotrain.py sft，或在 settings.yaml 填 sft_merged"
                                 + hint)
            if Path(m).resolve() == Path(self.cfg["base_model"]).resolve():
                raise SystemExit("DPO 底座不能是原始 base_model（reference 会变成原始模型，把模型推崩）。用 SFT 合并模型")
            return m
        return self.args.model or self.cfg["base_model"]

    def prepare_data(self, cutoff: int) -> Path:
        if cutoff in self.data_dirs:
            return self.data_dirs[cutoff]
        sys.path.insert(0, str(TOOLS))
        from filter_by_cutoff import filter_into_dir

        tok = self.cfg.get("length_tokenizer") or ""
        if not tok or not Path(tok).exists():
            tok = self.model
        dst = self.work / f"data_{cutoff}"
        say(f"按 cutoff_len={cutoff} 过滤数据 → {dst}")
        summary = filter_into_dir(DATA, dst, stage_keys(self.stage), cutoff, tok)
        if getattr(self.args, "probe", False):
            info = json.loads((dst / "dataset_info.json").read_text(encoding="utf-8"))
            train = dst / info[stage_keys(self.stage)[0]]["file_name"]
            lines = [ln for ln in train.read_text(encoding="utf-8").splitlines() if ln.strip()]
            lines.sort(key=len, reverse=True)
            train.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = json.dumps(summary, ensure_ascii=False, indent=2)
        (dst / "filter_summary.json").write_text(text, encoding="utf-8")
        (self.dir / f"filter_summary_{cutoff}.json").write_text(text, encoding="utf-8")
        self.data_dirs[cutoff] = dst
        return dst

    def build_yaml(self, plan: dict, nproc: int, yaml_over: dict, n: int) -> Path:
        y = load_yaml(CONFIGS / STAGE_CONFIG[self.stage])
        y["model_name_or_path"] = self.model
        y["dataset_dir"] = str(self.prepare_data(plan["cutoff_len"]))
        y["cutoff_len"] = plan["cutoff_len"]
        y["output_dir"] = self.train_dir
        y["template"] = template()
        ds = project()["datasets"][self.stage]
        if ds.get("train"):
            y["dataset"] = ds["train"]
        if ds.get("eval"):
            y["eval_dataset"] = ds["eval"]
        if self.stage == "eval":
            y["adapter_name_or_path"] = self.adapter
        else:
            y["gradient_accumulation_steps"] = grad_accum(plan, nproc)
        if plan.get("deepspeed"):
            y["deepspeed"] = str(CONFIGS / plan["deepspeed"])
        else:
            y.pop("deepspeed", None)
        if self.args.max_samples:
            y["max_samples"] = self.args.max_samples
        if getattr(self.args, "probe", False):
            y["max_samples"] = int(plan.get("global_batch", 4)) * 2
            y["num_train_epochs"] = 1
            y["save_strategy"] = "no"
        if self.args.epochs:
            y["num_train_epochs"] = self.args.epochs
        y.update(yaml_over)
        path = self.dir / f"attempt-{n}.yaml"
        dump_yaml(y, path)
        return path

    def command(self, yaml_path: Path, gpus: list[int]) -> list[str]:
        script = str(TOOLS / "train_lowmem.py")
        if len(gpus) == 1:
            return [PY, "-u", script, str(yaml_path)]
        return [PY, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpus)),
                "--master_port", str(free_port()), script, str(yaml_path)]

    # ---------- 执行与监控 ----------
    def launch(self, cmd: list[str], env: dict, log_path: Path, gpus: list[int]) -> dict:
        timeout = float(self.cfg.get("timeout_hours", 8)) * 3600
        state = {"peak_mb": 0, "progress": "", "stop": False}
        def gpu_poll():
            while not state["stop"]:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout
                    used = [int(b) for a, b in (ln.split(",") for ln in out.strip().splitlines()) if int(a) in gpus]
                    state["peak_mb"] = max([state["peak_mb"], *used])
                except Exception:  # noqa: BLE001
                    pass
                self.write_status(attempt=len(self.attempts), progress=state["progress"],
                                  gpu_peak_gb=round(state["peak_mb"] / 1024, 1))
                time.sleep(15)

        prog = re.compile(r"(\d+/\d+ \[[\d:]+<[\d:]+,\s*[\d.]+s/it\])")
        start = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace", start_new_session=True)
        threading.Thread(target=gpu_poll, daemon=True).start()
        timed_out = False

        def killer():
            nonlocal timed_out
            while proc.poll() is None:
                if time.time() - start > timeout:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(30)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    return
                time.sleep(5)

        threading.Thread(target=killer, daemon=True).start()
        try:
            with log_path.open("w", encoding="utf-8") as fh:
                for line in proc.stdout:
                    fh.write(line)
                    fh.flush()
                    for chunk in line.split("\r"):
                        m = prog.search(chunk)
                        if m:
                            state["progress"] = m.group(1)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            say("收到 Ctrl-C，结束训练进程组…", "yellow")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            state["stop"] = True
            raise
        state["stop"] = True
        return {"rc": rc, "timed_out": timed_out, "seconds": int(time.time() - start),
                "gpu_peak_gb": round(state["peak_mb"] / 1024, 1), "progress": state["progress"]}

    def succeeded(self, rc: int) -> bool:
        if rc != 0:
            return False
        if self.stage == "eval":
            return True
        out = Path(self.train_dir)
        return (out / "adapter_model.safetensors").is_file() or (out / "adapter_model.bin").is_file()

    def publish(self) -> bool:
        """把本地 train_dir 里的最终结果（不含 checkpoint）复制到挂载盘，成功后按配置清理本地现场。"""
        import shutil

        src = Path(self.train_dir)
        if self.trial:
            say("试跑的 adapter 不发布到挂载盘")
        elif self.stage == "eval":
            dst = self.dir / "eval_results"
            dst.mkdir(exist_ok=True)
            for f in src.glob("*.json"):
                shutil.copy2(f, dst / f.name)
            say(f"评估结果 → {dst.relative_to(ROOT)}", "green")
        else:
            dst = Path(self.output_dir)
            tmp = dst.with_name(dst.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("checkpoint-*", "global_step*"))
                tmp.rename(dst)
            except OSError as e:
                say(f"发布 adapter 到 {dst} 失败：{e}。本地结果保留在 {src}，腾出空间后手动复制"
                    f"（cp -r 时排除 checkpoint-*）", "red")
                shutil.rmtree(tmp, ignore_errors=True)
                return False
            say(f"adapter 已发布到挂载盘 → {dst}", "green")
        if self.cfg.get("clean_work_on_success", True):
            shutil.rmtree(self.work, ignore_errors=True)
            say(f"已清理本地中间数据 {self.work}")
        return True

    # ---------- 主流程 ----------
    def write_status(self, **kw) -> None:
        path = self.dir / "status.json"
        if not self.dir.is_dir():
            return
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cur = {}
        cur.update(run=self.name, stage=self.stage, pid=os.getpid(), updated=now(), **kw)
        path.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")

    def run(self) -> int:
        if self.args.dry_run:
            return self._run()
        active = read_active()
        if active and active.get("alive"):
            raise SystemExit(f"已有训练在跑：{active['run']}（pid {active['pid']}）。用 python autotrain.py status 看进度，"
                             f"不要同时启动第二个")
        def on_term(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, on_term)
        RUNS.mkdir(exist_ok=True)
        ACTIVE_FILE.write_text(json.dumps({"run": self.name, "stage": self.stage, "pid": os.getpid(),
                                           "started": now()}, ensure_ascii=False), encoding="utf-8")
        rc = 1
        try:
            rc = self._run()
            return rc
        finally:
            self.write_status(state="ok" if rc == 0 else "failed", phase="done")
            ACTIVE_FILE.unlink(missing_ok=True)

    def _run(self) -> int:
        a = self.args
        cfg = self.cfg
        hw = detect_hardware(parse_gpus(a.gpus))
        say(f"硬件：{describe_hw(hw)}")

        say("检查环境…")
        problems, self.packages, env_yaml = check_env(verbose=False)
        if problems:
            say("环境问题：" + "；".join(problems), "yellow")
            if cfg.get("auto_fix_install", True) and not a.dry_run and run_setup():
                problems, self.packages, env_yaml = check_env(verbose=False)
            if problems:
                raise SystemExit("环境未通过：" + "；".join(problems) + "。手动处理后重跑，或 python autotrain.py env --install")
        say("环境通过" + ("（flash-attn 不可用，改用 sdpa）" if env_yaml else ""))

        self.model = self.base_model()
        errs = check_model(self.model)
        if errs:
            raise SystemExit("模型检查未通过：" + "；".join(errs))
        if self.stage == "eval":
            self.adapter = a.adapter or latest("sft", "adapter")
            if not self.adapter:
                raise SystemExit("找不到 SFT adapter：用 --adapter 指定")

        for key, need in (("work_dir", "min_free_gb_work"), ("saves_dir", "min_free_gb_saves")):
            problem = space_problem(cfg[key], float(cfg.get(need, 2)))
            if problem:
                raise SystemExit(f"磁盘检查未通过（{key}）：{problem}\n{cleanup_hint(cfg[key])}")
        if self.stage in ("sft", "dpo") and not a.no_merge and not self.trial:
            problem = space_problem(cfg["merged_dir"], float(cfg.get("min_free_gb_merge", 60)))
            if problem:
                say(f"[WARN] {problem}：训练照常进行，但结束后会跳过合并。"
                    f"现在腾出空间，或之后用 python autotrain.py merge --run <运行名> 补合并", "yellow")

        rc = subprocess.run([PY, str(TOOLS / "check_data.py")], capture_output=True, text=True)
        if rc.returncode != 0:
            raise SystemExit("数据契约检查未通过：\n" + rc.stdout + rc.stderr)

        plan, gpus = choose_plan(self.stage, hw, a.plan, ignore_busy=a.dry_run)
        if plan.get("deepspeed") and not self.packages.get("deepspeed"):
            raise SystemExit(f"方案 {plan['id']} 需要 deepspeed：pip install deepspeed==0.19.9")
        say(f"方案：{plan['id']}（{plan['status']}）GPU {gpus} cutoff={plan['cutoff_len']} — {plan.get('note', '')}")

        self.dir.mkdir(parents=True, exist_ok=True)
        self.write_status(state="running", phase="train", plan=plan["id"])
        yaml_over: dict = dict(env_yaml)
        env_over: dict = {}
        installed: set[str] = set()
        max_attempts = int(cfg.get("max_attempts", 4))
        result = "failed"

        for n in range(1, max_attempts + 1):
            yaml_path = self.build_yaml(plan, len(gpus), yaml_over, n)
            env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                   **{k: str(v) for k, v in (plan.get("env") or {}).items()}, **env_over}
            cmd = self.command(yaml_path, gpus)
            log_path = self.dir / f"attempt-{n}.log"
            attempt = {"n": n, "plan": plan["id"], "gpus": gpus, "cutoff_len": plan["cutoff_len"],
                       "yaml": str(yaml_path), "log": str(log_path.relative_to(ROOT)),
                       "yaml_overrides": dict(yaml_over), "env_overrides": dict(env_over),
                       "cmd": " ".join(shlex.quote(c) for c in cmd), "started": now()}
            self.attempts.append(attempt)
            self.save_meta()
            if a.dry_run:
                say("dry-run：配置与数据已生成，未启动。手动启动命令：", "green")
                print(f"  cd {ROOT} && CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} "
                      f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {attempt['cmd']}")
                return 0

            say(f"第 {n}/{max_attempts} 次尝试，日志 {attempt['log']}")
            res = self.launch(cmd, env, log_path, gpus)
            attempt.update(res)
            if self.succeeded(res["rc"]):
                say(f"训练成功：用时 {res['seconds'] // 60} 分钟，GPU 峰值 {res['gpu_peak_gb']}GB", "green")
                result = "ok"
                self.save_meta()
                if any(p["status"] == "open" and p["stage"] == self.stage for p in load_pending()):
                    learn_from_success(self, attempt)
                break

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            issue = diagnose(log_text, res["rc"], res["timed_out"], self.stage)
            if issue is None:
                entry = record_pending(self, attempt, log_text)
                attempt["issue"] = None
                say(f"未识别的失败（rc={res['rc']}）：{entry['signature']}", "red")
                say(f"已记入 issues/pending.jsonl（{entry['id']}）。解决后重跑本命令；成功时会自动登记，"
                    f"也可手动：python autotrain.py record-issue --from-pending {entry['id']} --id <名字> ...", "yellow")
                break
            attempt["issue"] = issue["id"]
            bump_issue(issue["id"])
            act = issue.get("action", {})
            say(f"识别为「{issue['title']}」（{issue['id']}）：{issue['diagnosis']}", "yellow")
            self.save_meta()

            if act.get("type") == "next_plan":
                nxt = plan.get("on_oom")
                new_gpus = fit_plan(find_plan(self.stage, nxt), hw, True) if nxt else None
                if not nxt or new_gpus is None:
                    say(f"没有可降级的方案了。人工处理：{issue['manual']}", "red")
                    break
                plan, gpus = find_plan(self.stage, nxt), new_gpus
                say(f"换方案 → {plan['id']} cutoff={plan['cutoff_len']}", "yellow")
            elif act.get("type") == "retry":
                new_yaml = {**yaml_over, **act.get("yaml", {})}
                new_env = {**env_over, **act.get("env", {})}
                if (new_yaml, new_env) == (yaml_over, env_over) and issue["id"] != "port_in_use":
                    say(f"同样的修复已经试过。人工处理：{issue['manual']}", "red")
                    break
                yaml_over, env_over = new_yaml, new_env
                say(f"带参数重试：{act}", "yellow")
            elif act.get("type") == "install":
                if issue["id"] in installed or not cfg.get("auto_fix_install", True):
                    say(f"人工处理：{issue['manual']}", "red")
                    break
                installed.add(issue["id"])
                for c in act.get("commands", []):
                    c = c.replace("{python}", shlex.quote(PY))
                    say(f"执行修复：{c}", "yellow")
                    subprocess.run(c, shell=True)
                _, self.packages, _ = check_env(verbose=False)
            else:
                say(f"需要人工处理：{issue['manual']}", "red")
                break
        else:
            say(f"已达最大尝试次数 {max_attempts}", "red")

        merged = None
        smoke = None
        if result == "ok" and not self.publish():
            result = "failed"
        if result == "ok" and self.stage in ("sft", "dpo"):
            if self.trial:
                say(f"试跑通过：方案 {self.attempts[-1]['plan']}，GPU 峰值 {self.attempts[-1].get('gpu_peak_gb')}GB。"
                    f"不做发布、合并与冒烟。正式训练：python autotrain.py {self.stage} --plan {self.attempts[-1]['plan']}",
                    "green")
            elif not a.no_merge:
                merged = self.merge()
                if merged and not a.no_smoke:
                    smoke = self.smoke(merged, gpus[0])
        self.finish(result, merged, smoke)
        if result != "ok":
            say(f"失败。记录在 {self.dir.relative_to(ROOT)}，本地现场（数据、checkpoint）在 {self.work}；"
                f"问题说明见 docs/TROUBLESHOOTING.md", "red")
            return 1
        return 0

    def merge(self) -> str | None:
        self.write_status(phase="merge")
        return merge_model(self.model, self.output_dir, self.stage, self.name, self.dir / "merge.log")

    def smoke(self, model: str, gpu: int) -> dict:
        self.write_status(phase="smoke")
        return smoke_model(model, gpu, self.dir / "smoke.log")

    def save_meta(self) -> None:
        (self.dir / "run.json").write_text(json.dumps({
            "run": self.name, "stage": self.stage, "model": getattr(self, "model", None),
            "output_dir": self.output_dir, "work_dir": str(self.work),
            "packages": self.packages, "attempts": self.attempts,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def finish(self, result: str, merged: str | None, smoke: dict | None) -> None:
        self.save_meta()
        last = self.attempts[-1] if self.attempts else {}
        status = "trial" if result == "ok" and self.trial else result
        row = {"run": self.name, "stage": self.stage, "status": status, "time": now(),
               "plan": last.get("plan"), "cutoff_len": last.get("cutoff_len"), "attempts": len(self.attempts),
               "minutes": last.get("seconds", 0) // 60, "gpu_peak_gb": last.get("gpu_peak_gb"),
               "model": getattr(self, "model", None),
               "adapter": self.output_dir if status == "ok" and self.stage != "eval" else None,
               "merged": merged, "smoke": smoke and smoke["passed"]}
        append_history(row)
        if result == "ok" and merged:
            say(f"产物：adapter {self.output_dir}；合并模型 {merged}", "green")


def cmd_train(args) -> int:
    return Run(args.stage, args).run()


def cmd_merge(args) -> int:
    cfg = settings()
    if args.run:
        meta = RUNS / args.run / "run.json"
        if not meta.is_file():
            raise SystemExit(f"没有运行 {args.run}（runs/ 下找不到 run.json）")
        info = json.loads(meta.read_text(encoding="utf-8"))
        stage, base, adapter, tag = info["stage"], info["model"], info["output_dir"], args.run
        log = RUNS / args.run / "merge.log"
    else:
        if not (args.adapter and args.stage):
            raise SystemExit("用 --run <运行名>，或同时给 --adapter 和 --stage")
        stage, adapter = args.stage, args.adapter
        if args.base:
            base = args.base
        elif stage == "sft":
            base = cfg["base_model"]
        else:
            base = cfg.get("sft_merged") or latest("sft", "merged")
            if not base:
                raise SystemExit("DPO adapter 的底座是 SFT 合并模型：用 --base 指定")
        tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S") + "-import"
        log = RUNS / f"{tag}-merge.log"
    if stage not in ("sft", "dpo"):
        raise SystemExit(f"{stage} 运行没有 adapter 可合并")
    if not (Path(adapter) / "adapter_config.json").is_file():
        raise SystemExit(f"{adapter} 不是 LoRA adapter 目录（缺 adapter_config.json）")
    retry = "" if args.run else " ".join(shlex.quote(x) for x in ["python", "autotrain.py", "merge", "--stage", stage,
                                                                 "--adapter", adapter, "--base", base, "--tag", tag])
    merged = merge_model(base, adapter, stage, tag, log, retry)
    if not merged:
        return 1
    smoke = None
    if not args.no_smoke:
        gpus = [g["index"] for g in detect_hardware()["gpus"] if not g["busy"]]
        smoke = smoke_model(merged, gpus[0], log.with_name(log.name.replace("merge", "smoke"))) if gpus else None
        if not gpus:
            say("没有空闲 GPU，跳过冒烟", "yellow")
    append_history({"run": tag, "stage": stage, "status": "ok", "time": now(), "plan": "merge-only",
                    "attempts": 0, "model": base, "adapter": adapter, "merged": merged,
                    "smoke": smoke and smoke["passed"]})
    say(f"已登记到运行历史：{'之后的 dpo 会自动用它作底座' if stage == 'sft' else merged}", "green")
    return 0


def cmd_status(args) -> int:
    rows = history()[-args.n:]
    if not rows:
        print("还没有运行记录")
    for r in rows:
        print(f"{r['run']:<26} {r['status']:<6} {r['plan'] or '-':<18} cutoff={r.get('cutoff_len')} "
              f"尝试 {r['attempts']} 次 {r.get('minutes', 0)} 分钟 峰值 {r.get('gpu_peak_gb')}GB"
              + (f"\n    合并：{r['merged']}  冒烟：{r.get('smoke')}" if r.get("merged") else ""))
    active = read_active()
    if not active:
        print("\n当前没有训练在跑")
        return 0
    s = {}
    st = RUNS / active["run"] / "status.json"
    if st.is_file():
        s = json.loads(st.read_text(encoding="utf-8"))
    detail = (f"阶段 {s.get('phase', '准备')} 第 {s.get('attempt', '-')} 次尝试 进度 {s.get('progress') or '-'} "
              f"GPU 峰值 {s.get('gpu_peak_gb', '-')}GB（更新于 {s.get('updated', '-')}）")
    if active["alive"]:
        print(f"\n正在运行：{active['run']}（pid {active['pid']}，{active['started']} 开始）{detail}")
        return 0
    print(f"\n[异常] {active['run']} 的 autotrain 进程（pid {active['pid']}）已不在，但没有正常收尾：{detail}\n"
          f"  看 runs/{active['run']}/ 下最后一个 attempt-N.log 末尾和 nohup 的输出文件找原因；"
          f"确认训练进程也已退出（nvidia-smi）后删掉 runs/active.json 再重跑")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for stage in ("sft", "dpo", "eval"):
        p = sub.add_parser(stage, help=f"{stage} 全自动流程")
        p.set_defaults(func=cmd_train, stage=stage)
        p.add_argument("--plan", help="指定方案 id（见 autotrain.py plans）")
        p.add_argument("--gpus", help="只用这些卡，如 0,1")
        p.add_argument("--model", help="覆盖底座路径")
        p.add_argument("--adapter", help="eval 用的 adapter 目录")
        p.add_argument("--dry-run", action="store_true", help="只生成配置与数据，打印启动命令")
        p.add_argument("--max-samples", type=int, help="只取前 N 条快速试跑（不合并）")
        p.add_argument("--epochs", type=float, help="覆盖 num_train_epochs")
        p.add_argument("--probe", action="store_true",
                       help="显存探测：只用最长的 2×global_batch 条跑 1 轮，爆显存会沿降级链继续探测；不发布不合并")
        p.add_argument("--no-merge", action="store_true")
        p.add_argument("--no-smoke", action="store_true")

    p = sub.add_parser("plans", help="硬件检测与方案选择")
    p.add_argument("stage", nargs="?", choices=["sft", "dpo", "eval"])
    p.add_argument("--gpus")
    p.add_argument("--ignore-busy", action="store_true", help="把被占用的卡也算作可用")
    p.set_defaults(func=cmd_plans)

    p = sub.add_parser("env", help="检查训练环境")
    p.add_argument("--install", action="store_true", help="有问题时执行 setup_env.sh")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("merge", help="补合并某次运行的 adapter，或导入外部 adapter")
    p.add_argument("--run", help="runs/ 下的运行名")
    p.add_argument("--adapter", help="外部 adapter 目录（配合 --stage）")
    p.add_argument("--stage", choices=["sft", "dpo"])
    p.add_argument("--base", help="底座；默认 sft=base_model，dpo=最近的 SFT 合并模型")
    p.add_argument("--tag", help="外部 adapter 的名字，用于合并目录名")
    p.add_argument("--no-smoke", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("status", help="运行历史与最近进度")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("issues", help="列出问题库并重新生成 docs/TROUBLESHOOTING.md")
    p.set_defaults(func=cmd_issues)

    p = sub.add_parser("record-issue", help="登记或更新一个问题")
    p.add_argument("--id", required=True, help="英文短名，如 nccl_shm_too_small；已存在则更新")
    p.add_argument("--from-pending", help="从 issues/pending.jsonl 的某条生成（自动带上日志特征）")
    p.add_argument("--title", default="")
    p.add_argument("--pattern", action="append", help="日志正则，可多次给")
    p.add_argument("--diagnosis", required=True, help="原因")
    p.add_argument("--manual", required=True, help="人工处理步骤")
    p.add_argument("--action", help='自动动作 JSON，如 {"type":"retry","env":{"NCCL_SHM_DISABLE":"1"}}；默认 abort')
    p.add_argument("--stages", default="sft,dpo,eval")
    p.add_argument("--category", default="runtime", choices=["runtime", "quality", "deploy"])
    p.add_argument("--scope", default="tail", choices=["tail", "full"],
                   help="tail=只匹配日志末尾 400 行（默认）；full=匹配整份日志（只在开头出现的警告用）")
    p.set_defaults(func=cmd_record_issue)

    args = ap.parse_args()
    return args.func(args)


def ensure_python() -> None:
    """用 settings.yaml 的 python 重新执行自己。不能依赖 PyYAML：系统 python 里可能没有。"""
    m = re.search(r"^python:\s*(\S+)", (ROOT / "settings.yaml").read_text(encoding="utf-8"), re.M)
    want = Path(m.group(1)) if m else None
    if not want or os.environ.get("AUTOTRAIN_REEXEC"):
        return
    if not want.is_file():
        print(f"[autotrain] settings.yaml 的 python 不存在：{want}，继续用 {sys.executable}", file=sys.stderr)
        return
    if want.resolve() != Path(sys.executable).resolve():
        os.environ["AUTOTRAIN_REEXEC"] = "1"
        os.execv(str(want), [str(want), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    ensure_python()
    sys.exit(main())
