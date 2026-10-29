#!/usr/bin/env python3
"""训练前自检：交付包里的 YAML、dataset_info 与 jsonl 必须互相咬合。

在解压目录执行（不要改文件名）：

    python check_data.py
    python check_data.py --model /data/models/Qwen3.8-27B

退出码 0 表示：每份 train/eval YAML 的 dataset 键都已登记，file_name 能在
dataset_dir 下找到，且 *.lf.jsonl 的角色交替符合 LLaMA-Factory。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "<" + "/think>"
_ODD = ("human", "observation")
_EVEN = ("gpt", "function")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def alignment_error(messages, *, ranking: bool):
    roles = [str(m.get("from") or "") for m in messages]
    if roles and roles[0] == "system":
        roles = roles[1:]
    if not roles:
        return "empty"
    for i, role in enumerate(roles):
        allowed = _ODD if i % 2 == 0 else _EVEN
        if role not in allowed:
            return f"role[{i}]={role}"
    if ranking and len(roles) % 2 == 0:
        return "prompt ends with assistant"
    if not ranking and len(roles) % 2 != 0:
        return "ends with user/observation"
    return None


def yaml_keys(text: str):
    found = []
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith("dataset:") or code.startswith("eval_dataset:"):
            raw = code.split(":", 1)[1].strip()
            for part in raw.split(","):
                part = part.strip()
                if part:
                    found.append(part)
    return found


def resolve_pack(root: Path):
    root = root.resolve()
    nested = root / "data"
    if (nested / "dataset_info.json").is_file():
        return root, nested
    if (root / "dataset_info.json").is_file():
        return root, root
    return root, nested


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="解压目录；默认是本脚本所在目录")
    ap.add_argument("--model", default="")
    ap.add_argument("--samples", type=int, default=0, help="0 表示检查全部 lf 行")
    args = ap.parse_args()
    root = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent
    pack_root, data_dir = resolve_pack(root)

    problems = 0
    info_path = data_dir / "dataset_info.json"
    if not info_path.is_file():
        _fail(f"找不到 {info_path}")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or not info:
        _fail("dataset_info.json 为空")
        return 1

    yamls = sorted(pack_root.glob("*.yaml"))
    if not yamls:
        _fail(f"{pack_root} 下没有训练 YAML")
        return 1

    referenced = []
    for yaml_path in yamls:
        keys = yaml_keys(yaml_path.read_text(encoding="utf-8"))
        if not keys:
            continue
        for key in keys:
            referenced.append((yaml_path.name, key))
            entry = info.get(key)
            if not isinstance(entry, dict):
                problems += 1
                _fail(f"{yaml_path.name} 的数据集 {key} 不在 dataset_info.json")
                continue
            file_name = str(entry.get("file_name") or "")
            if not file_name or file_name.startswith(("/", "\\")) or ".." in Path(file_name).parts:
                problems += 1
                _fail(f"{key} 的 file_name 非法：{file_name!r}")
                continue
            target = (data_dir / file_name).resolve()
            try:
                target.relative_to(data_dir.resolve())
            except ValueError:
                problems += 1
                _fail(f"{key} 的 file_name 逃出 dataset_dir：{file_name}")
                continue
            if not target.is_file():
                problems += 1
                _fail(f"{key} 指向的文件不存在：{target}")
                continue
            _ok(f"{yaml_path.name}: {key} -> {file_name}")

    if not referenced:
        problems += 1
        _fail("YAML 里没有任何 dataset / eval_dataset")

    lf_checked = 0
    for key, entry in info.items():
        if not isinstance(entry, dict):
            continue
        file_name = str(entry.get("file_name") or "")
        if not file_name.endswith(".lf.jsonl"):
            continue
        path = data_dir / file_name
        if not path.is_file():
            problems += 1
            _fail(f"{key} 登记了 {file_name} 但文件不存在")
            continue
        ranking = bool(entry.get("ranking"))
        rows = list(load_jsonl(path))
        limit = args.samples or len(rows)
        bad = 0
        think_bad = 0
        for row in rows[:limit]:
            conv = row.get("conversations") or []
            reason = alignment_error(conv, ranking=ranking)
            if reason:
                bad += 1
                continue
            if ranking:
                for side in ("chosen", "rejected"):
                    value = row.get(side)
                    text = value.get("value") if isinstance(value, dict) else value
                    if not isinstance(text, str) or not text.strip():
                        bad += 1
                        break
                continue
            if not conv or conv[0].get("from") != "system" or len(conv) < 2 or conv[1].get("from") != "human":
                think_bad += 1
                continue
            if conv[-1].get("from") != "gpt":
                think_bad += 1
                continue
            last = str(conv[-1].get("value") or "")
            if last.count(THINK_OPEN) != 1 or last.count(THINK_CLOSE) != 1:
                think_bad += 1
        lf_checked += 1
        if bad or think_bad:
            problems += bad + think_bad
            _fail(f"{file_name}: 角色不合法 {bad}，目标步结构异常 {think_bad}（检查 {min(limit, len(rows))}/{len(rows)}）")
        else:
            _ok(f"{file_name}: {len(rows)} 条，角色交替合法")

    if lf_checked == 0:
        problems += 1
        _fail("dataset_info 里没有 *.lf.jsonl，正式训练不会读到规整后的数据")

    if args.model:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            _fail("需要 transformers 才能做模板渲染检查")
            return 1
        sft = data_dir / "sft.train.lf.jsonl"
        if not sft.is_file():
            _fail("缺少 sft.train.lf.jsonl，无法做模板渲染")
            return 1
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        shown = 0
        for row in load_jsonl(sft):
            conv = row["conversations"]
            msgs = []
            role_map = {
                "system": "system",
                "human": "user",
                "gpt": "assistant",
                "observation": "tool",
                "function": "assistant",
            }
            for turn in conv:
                msgs.append({
                    "role": role_map.get(turn.get("from"), "user"),
                    "content": turn.get("value") or "",
                })
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] 模板渲染失败: {exc}")
                shown += 1
                if shown >= 5:
                    break
                continue
            opens = text.count(THINK_OPEN)
            closes = text.count(THINK_CLOSE)
            flag = "ok" if opens == 1 and closes == 1 else "检查"
            sid = str(row.get("sample_id") or "")[:16]
            print(f"  [{flag}] {sid} think×{opens}/{closes}")
            shown += 1
            if shown >= 5:
                break

    if problems:
        _fail(f"未通过，问题 {problems} 项")
        return 1
    _ok("交付包数据契约通过，可以按 START_HERE.md 启动训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
