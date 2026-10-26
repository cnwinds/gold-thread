#!/usr/bin/env python3
"""把 DPO 数据裁成「保留最近若干步」的窗口版，供 48G 卡上 DPO 用。

背景：LLaMA-Factory 的 cutoff_len 截断是砍 prompt 的**尾部**（`prompt_ids[:source_len]`），
等于丢掉最近的 observation，正好砍掉最该学的那段上下文。这里改成从**头部**砍：
保留 system + 根任务 + 最近 K 步 (gpt, observation)，丢掉更早的步。

用法：
    python prep_dpo_window.py --budget 4000
输出：
    dpo_preference_pairs.train.v2w.jsonl / .eval.v2w.jsonl（并在 dataset_info.json 注册）
"""
from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer

ROLE = {"system": "system", "human": "user", "gpt": "assistant",
        "observation": "observation", "tool": "observation"}
DATA = "/data/goldthread/03_train"
NAMES = {
    "train": ("dpo_preference_pairs.train.v2.jsonl", "dpo_preference_pairs.train.v2w.jsonl",
              "pi_dpo_preference_pairs_train_v2w"),
    "eval": ("dpo_preference_pairs.eval.v2.jsonl", "dpo_preference_pairs.eval.v2w.jsonl",
             "pi_dpo_preference_pairs_eval_v2w"),
}


def to_msgs(conv: list[dict]) -> list[dict]:
    out = []
    for c in conv:
        role = ROLE.get(c["from"], "user")
        content = c["value"]
        if role == "observation":
            # LLaMA-Factory 里 observation 走 format_observation，内容要包 tool_response 由模板负责
            out.append({"role": "observation", "content": content})
        else:
            out.append({"role": role, "content": content})
    return out


def n_tokens(tok, msgs: list[dict], tools: str | None) -> int:
    """粗略估算渲染后 token 数：内容 token + 每条消息的模板开销（HF chat_template 不支持 observation 角色，
    所以这里不调 apply_chat_template，够用来选窗口）。"""
    n = len(tok(tools or "", add_special_tokens=False)["input_ids"]) if tools else 0
    for m in msgs:
        n += len(tok(m["content"], add_special_tokens=False)["input_ids"]) + 8
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-8B")
    ap.add_argument("--budget", type=int, default=4000, help="prompt + 最长回复 的 token 上限")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    info_path = os.path.join(DATA, "dataset_info.json")
    info = json.load(open(info_path))

    for split, (src, dst, key) in NAMES.items():
        rows = [json.loads(line) for line in open(os.path.join(DATA, src))]
        kept, stats = [], []
        for row in rows:
            conv = row["conversations"]
            system_msgs = [c for c in conv[:2]]                      # system + 根任务
            groups = [conv[i:i + 2] for i in range(2, len(conv), 2)]  # (gpt, observation)
            resp_len = max(len(tok(row["chosen"]["value"])["input_ids"]),
                           len(tok(row["rejected"]["value"])["input_ids"]))
            limit = args.budget - resp_len
            chosen_groups = groups[:]
            while chosen_groups and n_tokens(tok, to_msgs(system_msgs + [c for g in chosen_groups for c in g]),
                                             row.get("tools")) > limit:
                chosen_groups.pop(0)                                 # 从最早的步开始丢
            new_conv = system_msgs + [c for g in chosen_groups for c in g]
            if len(new_conv) < 3:                                    # 至少保留最后一对 (gpt, obs)
                continue
            row = dict(row)
            row["conversations"] = new_conv
            row["_window"] = {"kept_steps": len(chosen_groups), "orig_steps": len(groups)}
            kept.append(row)
            stats.append((len(chosen_groups), len(groups)))
        out_path = os.path.join(DATA, dst)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info[key] = {
            "file_name": dst,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected", "tools": "tools"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                     "assistant_tag": "gpt", "system_tag": "system", "observation_tag": "observation"},
        }
        dropped = sum(1 for s in stats if s[0] < s[1])
        print(f"{split}: {len(kept)}/{len(rows)} 条保留，{dropped} 条被窗口裁剪；"
              f"保留步数 min/median = {min(s[0] for s in stats)}/{sorted(s[0] for s in stats)[len(stats)//2]}")

    json.dump(info, open(info_path, "w"), ensure_ascii=False, indent=4)
    print("dataset_info.json 已更新：", os.path.join(DATA, "dataset_info.json"))


if __name__ == "__main__":
    main()