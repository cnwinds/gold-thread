"""训练序列长度测量（供 handoff 文档与命令行体检共用）。

为什么不用「字符数 / 2.2」估算：
    实测校准（27B 训练日志里那一条渲染后的真实样本）——18,996 字符 = 8,911 tokens，
    折合 2.13 字符/token；粗估会低估 3~5%，正好卡在 12K 边界时会误判「没超」。
    而 LLaMA-Factory 是 truncation_side='right'，超长被切掉的恰恰是最后的目标步
    （叠加 mask_history: true 就是「标签全被切光、loss=0」的假训练）。

渲染口径与 LLaMA-Factory `template: qwen3 / qwen3_8` 对齐（ChatML）：
    system      : <|im_start|>system\\n{模板前缀}{数据集 system}<|im_end|>\\n
    human/user  : <|im_start|>user\\n…<|im_end|>\\n
    observation : <|im_start|>user\\n<tool_response>\\n…\\n</tool_response><|im_end|>\\n
    gpt         : <|im_start|>assistant\\n…<|im_end|>\\n
其中「模板前缀」= 训练日志里打印的 Reasoning effort 行 + # Tools + tools JSON + 调用格式指令
（27B/qwen3_8 约 686 tokens），每条样本都要付。
"""
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "observation": "tool",
    "tool": "tool",
    "gpt": "assistant",
    "assistant": "assistant",
}

# 模板前缀里「数据集 system 正文」的开始标志（不同数据包的首句）
SYSTEM_BODY_MARKERS = ("你是一名顶尖的", "你是一名助手", "You are an expert")

# 找不到 tokenizer 时的兜底估算（字符/token）
FALLBACK_CHARS_PER_TOKEN = 2.13

TOKENIZER_CANDIDATES = (
    "D:/ai_projects/dataset_sync/.tokenizer/qwen3.8-27b/tokenizer.json",
    "/data/models/Qwen3.8-27B/tokenizer.json",
    "/data/models/Qwen3-8B/tokenizer.json",
    "/data/models/qwen3-8b-merged/tokenizer.json",
    "D:/tmp/tok/qwen3_tok.json",
)

# 仓库内约定放「用户提供的真 tokenizer」的目录（find_tokenizer 会自动找）
REPO_TOKENIZER_DIRS = (".tokenizer/qwen3.8-27b", ".tokenizer", "data/03_train/.tokenizer")


# --------------------------------------------------------------------------- tokenizer

def _load_tokenizer(p: Path) -> Tuple[Any, str]:
    """支持：tokenizer.json 单文件 / 含 tokenizer.json 的目录 / 含 vocab.json+merges.txt 的目录。"""
    from tokenizers import Tokenizer  # type: ignore

    if p.is_dir():
        tj = p / "tokenizer.json"
        if tj.is_file():
            return Tokenizer.from_file(str(tj)), f"tokenizer.json:{tj}"
        vocab, merges = p / "vocab.json", p / "merges.txt"
        if vocab.is_file() and merges.is_file():
            from tokenizers import models  # type: ignore
            tk = Tokenizer(models.BPE.from_file(str(vocab), str(merges)))
            cfg = p / "tokenizer_config.json"
            if cfg.is_file():
                try:
                    added = (json.loads(cfg.read_text(encoding="utf-8")).get("added_tokens_decoder") or {})
                    from tokenizers import AddedToken  # type: ignore
                    tk.add_special_tokens([AddedToken(v.get("content", ""), special=True)
                                           for v in added.values() if v.get("content")])
                except Exception:  # noqa: BLE001, S110 — added tokens 解析失败不影响主流程
                    pass
            return tk, f"vocab+merges:{p}"
        raise FileNotFoundError(f"{p} 里既没有 tokenizer.json，也没有 vocab.json+merges.txt")
    return Tokenizer.from_file(str(p)), f"tokenizer.json:{p}"


def find_tokenizer(
    explicit: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> Tuple[Any, str]:
    """返回 (Tokenizer 或 None, 来源说明)。找不到时返回 (None, 'estimate')。"""
    cands: List[str] = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("GT_TOKENIZER"):
        cands.append(os.environ["GT_TOKENIZER"])
    if model_dir:
        cands.append(str(Path(model_dir) / "tokenizer.json"))
    repo_root = Path(__file__).resolve().parents[2]
    cands.extend(str(repo_root / d) for d in REPO_TOKENIZER_DIRS)
    cands.extend(TOKENIZER_CANDIDATES)

    try:
        from tokenizers import Tokenizer  # type: ignore
        del Tokenizer
    except ImportError:  # 没有 tokenizers 就退化成估算
        return None, "estimate:no-tokenizers"

    for c in cands:
        p = Path(c)
        if not p.exists():
            continue
        try:
            return _load_tokenizer(p)
        except Exception:  # noqa: BLE001, S112 — 坏文件/不完整目录跳过，继续找下一个
            continue
    return None, "estimate:not-found"


def count_tokens(tok: Any, text: str) -> int:
    if tok is None:
        return round(len(text) / FALLBACK_CHARS_PER_TOKEN)
    # ApiTokenizer（远程精确计数）优先走 .count()，避免构造 ids 列表
    fn = getattr(tok, "count", None)
    if callable(fn) and not hasattr(tok, "get_vocab"):
        return int(fn(text))
    return len(tok.encode(text, add_special_tokens=False).ids)


def prefix_tokens(tok: Any, prefix: str) -> int:
    """模板前缀的固定开销（system 轮 + ChatML 标记），每条样本都要付。"""
    return count_tokens(tok, "<|im_start|>system\n" + prefix + "<|im_end|>\n")


# --------------------------------------------------------------------------- 模板前缀

def prefix_from_log(log_path: Path) -> Optional[str]:
    """从 LLaMA-Factory 训练日志的 `inputs:` / `chosen_inputs:` 里抠出模板注入部分。"""
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    i = txt.find("training example:")
    if i < 0:
        return None
    for key in ("\ninputs:", "\nchosen_inputs:"):
        k = txt.find(key, i)
        if k >= 0:
            seg = txt[k + len(key):]
            break
    else:
        return None
    start = seg.find("<|im_start|>system")
    if start < 0:
        return None
    seg = seg[start:]
    ends = [seg.find(m) for m in SYSTEM_BODY_MARKERS if seg.find(m) > 0]
    end = min(ends) if ends else seg.find("<|im_end|>")
    if end <= 0:
        return None
    body = seg[:end]
    return body[len("<|im_start|>system\n"):]


def prefix_from_logs(log_dir: Path, prefer: Sequence[str] = ("27b_sft", "27b_dpo", "full_sft")) -> Tuple[str, str]:
    """在日志目录里按优先顺序找一份能抠出前缀的日志。返回 (prefix, 来源)。"""
    d = Path(log_dir)
    if not d.is_dir():
        return "", "none"
    names = sorted(p.name for p in d.glob("*.log"))
    for want in prefer:
        for n in names:
            if n.startswith(want):
                p = prefix_from_log(d / n)
                if p:
                    return p, f"prefix:{n}"
    for n in names:
        p = prefix_from_log(d / n)
        if p:
            return p, f"prefix:{n}"
    return "", "none"


def calibration_from_log(log_path: Path) -> Optional[Dict[str, Any]]:
    """从训练日志里抠出「渲染后的真实序列 + LLaMA-Factory 用真 tokenizer 算出的 token 数」。

    这是**唯一现成的 ground truth**：可以用它验收用户提供的 tokenizer / 任何 API 口径是否真实，
    因为日志里的 `input_ids` 就是当时训练实际用的那份 tokenizer 数出来的。
    """
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    i = txt.find("training example:")
    if i < 0:
        return None
    k_ids = txt.find("\ninput_ids:", i)
    if k_ids < 0:
        return None
    for key in ("\ninputs:", "\nchosen_inputs:"):
        k = txt.find(key, k_ids)
        if k >= 0:
            break
    else:
        return None
    n_log = len(re.findall(r"\d+", txt[k_ids + len("\ninput_ids:"):k].split("[", 1)[-1]))
    if n_log <= 0:
        return None
    seg = txt[k + len(key):]
    end = seg.find("\nlabels:")
    body = seg[:end] if end > 0 else seg[:400_000]
    e2 = body.rfind("<|im_end|>")
    if e2 <= 0:
        return None
    text = body[:e2 + len("<|im_end|>")]
    return {"text": text, "log_tokens": n_log, "file": Path(log_path).name}


def verify_tokenizer(tok: Any, log_path: Path) -> Optional[Dict[str, Any]]:
    """用日志 ground truth 验收 tokenizer：同一段渲染文本应该数出同样的 token 数。"""
    cal = calibration_from_log(log_path)
    if not cal:
        return None
    n = count_tokens(tok, cal["text"])
    delta = n - cal["log_tokens"]
    pct = round(delta / max(1, cal["log_tokens"]) * 100, 3)
    return {
        "file": cal["file"],
        "text_chars": len(cal["text"]),
        "log_tokens": cal["log_tokens"],
        "tok_tokens": n,
        "delta": delta,
        "delta_pct": pct,
        "ok": abs(pct) <= 0.5,
    }


# --------------------------------------------------------------------------- 渲染

def render(prefix: str, turns: Iterable[Dict[str, Any]], open_assistant: bool = False) -> str:
    out: List[str] = []
    for t in turns:
        role = ROLE_MAP.get(str(t.get("from", "")), "user")
        v = t.get("value") or ""
        if role == "system":
            out.append(f"<|im_start|>system\n{prefix}{v}<|im_end|>\n")
        elif role == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{v}\n</tool_response><|im_end|>\n")
        elif role == "assistant":
            out.append(f"<|im_start|>assistant\n{v}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>user\n{v}<|im_end|>\n")
    text = "".join(out)
    return text + ("<|im_start|>assistant\n" if open_assistant else "")


def _openai_turns(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            turns.append({"from": "observation", "value": m.get("content") or ""})
        elif role == "assistant":
            body = m.get("content") or ""
            tcs = m.get("tool_calls") or []
            if tcs:
                blocks = []
                for tc in tcs:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        "<tool_call>\n"
                        + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False)
                        + "\n</tool_call>"
                    )
                body = (body + "\n" if body else "") + "\n".join(blocks)
            turns.append({"from": "gpt", "value": body})
        else:
            turns.append({"from": "human" if role == "user" else "system",
                          "value": m.get("content") or ""})
    return turns


def row_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if row.get("conversations"):
        return [{"from": c.get("from"), "value": c.get("value") or ""} for c in row["conversations"]]
    return _openai_turns(row.get("messages") or [])


def _side_turns(side: Any) -> List[Dict[str, Any]]:
    """chosen / rejected 可能是整段字符串、单轮 {from,value}、或整段轮次列表。"""
    if isinstance(side, str):
        return [{"from": "gpt", "value": side}]
    if isinstance(side, dict):
        if "value" in side:
            return [{"from": side.get("from") or "gpt", "value": side.get("value") or ""}]
        side = side.get("conversations") or side.get("messages") or []
    if isinstance(side, list):
        return [{"from": c.get("from"), "value": c.get("value") or ""}
                if "from" in c else
                {"from": "gpt" if c.get("role") == "assistant" else "human",
                 "value": c.get("content") or ""}
                for c in side if isinstance(c, dict)]
    return []


# --------------------------------------------------------------------------- 测量

def sft_lengths(rows: Iterable[Dict[str, Any]], tok: Any, prefix: str) -> List[Tuple[int, str]]:
    out = []
    for r in rows:
        n = count_tokens(tok, render(prefix, row_turns(r)))
        out.append((n, str(r.get("sample_id") or "?")))
    return out


def dpo_lengths(rows: Iterable[Dict[str, Any]], tok: Any, prefix: str) -> List[Tuple[int, str]]:
    out = []
    for r in rows:
        base = render(prefix, row_turns(r), open_assistant=True)
        nc = count_tokens(tok, base + render("", _side_turns(r.get("chosen"))))
        nr = count_tokens(tok, base + render("", _side_turns(r.get("rejected"))))
        sid = str(r.get("sample_id") or "?")
        out.append((max(nc, nr), f"{sid} (chosen={nc}, rejected={nr})"))
    return out


def quantiles(lens: Sequence[int]) -> Dict[str, int]:
    xs = sorted(lens)
    n = len(xs)
    if not n:
        return {"n": 0}
    def q(p: float) -> int:
        return xs[min(n - 1, int(n * p))]
    return {
        "n": n,
        "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1],
        "mean": round(statistics.mean(xs)),
        "over_8k": sum(1 for x in xs if x > 8192),
        "over_12k": sum(1 for x in xs if x > 12288),
        "over_16k": sum(1 for x in xs if x > 16384),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def profile(
    root: Path,
    tok: Any,
    prefix: str,
    sft_files: Sequence[str] = ("sft_step_windows_sharegpt.train.jsonl",),
    dpo_files: Sequence[str] = ("dpo_preference_pairs.train.jsonl",),
    top: int = 5,
) -> Dict[str, Any]:
    """给 handoff 用的汇总：SFT / DPO 各自的长度分布 + 超限清单。"""
    root = Path(root)
    res: Dict[str, Any] = {"prefix_tokens": prefix_tokens(tok, prefix), "sft": {}, "dpo": {}}
    for key, files, fn in (("sft", sft_files, sft_lengths), ("dpo", dpo_files, dpo_lengths)):
        for fn_name in files:
            p = root / fn_name
            if not p.exists():
                continue
            rows = load_jsonl(p)
            lens = fn(rows, tok, prefix)
            q = quantiles([n for n, _ in lens])
            q["file"] = fn_name
            q["top"] = [{"len": n, "id": i} for n, i in sorted(lens, reverse=True)[:top]]
            res[key][fn_name] = q
    return res

# --------------------------------------------------------------------------- API 口径（不上训练机）

class ApiTokenizer:
    """用推理端点的 tokenizer 当「精确计数器」（不上训练机）。

    优先 POST {base}/tokenize（vLLM 原生接口，直接数原始文本，不受 chat template 影响）；
    没有该接口时退化到 /chat/completions 的 usage.prompt_tokens 差值法（先量出包装开销）。

    前提：**端点服务的必须是同一份模型/分词器**。不同模型会偏 3~8%（实测 sub2api 的
    qwen3.8-27b 就不可用），所以用它之前先跑 `verify_tokenizer` 和日志 ground truth 对一下。
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 180.0,
                 cache_path: Optional[Path] = None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, int] = {}
        self.mode = ""
        self.wrapper = 0
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    # -- 底层 HTTP
    def _post(self, path: str, body: Dict[str, Any], tries: int = 3) -> Dict[str, Any]:
        import time
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        payload = json.dumps(body, ensure_ascii=False).encode()
        last: Optional[Exception] = None
        for attempt in range(tries):
            try:
                req = urllib.request.Request(self.base + path, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except Exception as exc:  # noqa: BLE001 — 端点偶发超时/502，重试
                last = exc
                if attempt < tries - 1:
                    time.sleep(2 + attempt * 3)
        raise RuntimeError(f"{path} 请求失败（{tries} 次）: {type(last).__name__}: {last}") from last

    # -- 能力探测
    def probe(self) -> str:
        if self.mode:
            return self.mode
        try:
            r = self._post("/tokenize", {"model": self.model, "prompt": "hello world"})
            if isinstance(r.get("count"), int):
                self.mode = "tokenize"
                return self.mode
        except Exception:  # noqa: BLE001, S110 — 没有 /tokenize 就退化到 prompt_tokens
            pass
        try:
            self.wrapper = self._prompt_tokens("")
            self.mode = "prompt_tokens"
        except Exception:  # noqa: BLE001
            self.mode = "unavailable"
        return self.mode

    def _prompt_tokens(self, text: str) -> int:
        r = self._post("/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "temperature": 0,
        })
        return int((r.get("usage") or {})["prompt_tokens"])

    # -- 对外计数
    def count(self, text: str) -> int:
        import hashlib
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in self.cache:
            return int(self.cache[h])
        mode = self.probe()
        if mode == "tokenize":
            r = self._post("/tokenize", {"model": self.model, "prompt": text})
            n = int(r["count"])
        elif mode == "prompt_tokens":
            n = self._prompt_tokens(text) - self.wrapper
        else:
            raise RuntimeError("端点既没有 /tokenize 也没有 /chat/completions，无法做 token 计数")
        self.cache[h] = n
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return n

    # count_tokens(tok, text) 兼容签名
    def encode(self, text: str, add_special_tokens: bool = False):  # 兼容 count_tokens 的调用签名
        n = self.count(text)

        class _Ids:
            def __init__(self, count: int) -> None:
                self.ids = list(range(count))

        return _Ids(n)
