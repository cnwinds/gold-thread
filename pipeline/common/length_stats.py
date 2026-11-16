"""交付包 token 长度分布：打包时写成静态 ``data/length_stats.json``，控制台只读。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.common import seqlength as sl

# 抽屉明细用粗档；卡片 spark 用 1k 密档（至 16k）。
SFT_TOKEN_BINS: List[Tuple[int, int, str]] = [
    (0, 2048, "≤2k"),
    (2048, 4096, "2k–4k"),
    (4096, 8192, "4k–8k"),
    (8192, 10240, "8k–10k"),
    (10240, 12288, "10k–12k"),
    (12288, 16384, "12k–16k"),
    (16384, 10**9, ">16k"),
]
SFT_SPARK_BINS: List[Tuple[int, int, str]] = [
    *((i * 1024, (i + 1) * 1024, f"{i + 1}k") for i in range(16)),
    (16384, 10**9, "16k+"),
]
DPO_SPARK_BINS = SFT_SPARK_BINS
DPO_TOKEN_BINS: List[Tuple[int, int, str]] = [
    (0, 2048, "≤2k"),
    (2048, 4096, "2k–4k"),
    (4096, 5120, "4k–5k"),
    (5120, 8192, "5k–8k"),
    (8192, 10240, "8k–10k"),
    (10240, 10**9, ">10k"),
]

LENGTH_STATS_REL = "data/length_stats.json"
LENGTH_STATS_ARC_CANDIDATES = (
    f"goldthread-train/{LENGTH_STATS_REL}",
    LENGTH_STATS_REL,
)


def length_hist(
    values: List[int],
    bins_spec: Optional[List[Tuple[int, int, str]]] = None,
) -> Dict[str, Any]:
    spec = bins_spec or SFT_TOKEN_BINS
    if not values:
        return {"count": 0, "bins": [], "p50": 0, "p90": 0, "max": 0, "mean": 0, "over_10k": 0}
    ordered = sorted(values)
    n = len(ordered)
    bins = []
    for lo, hi, label in spec:
        bins.append({"label": label, "lo": lo, "hi": hi, "count": sum(1 for v in values if lo <= v < hi)})
    mean = int(round(sum(ordered) / n))
    return {
        "count": n,
        "bins": bins,
        "p50": ordered[n // 2],
        "p90": ordered[min(n - 1, int(n * 0.9))],
        "max": ordered[-1],
        "mean": mean,
        "over_10k": sum(1 for v in values if v >= 10240),
    }


def count_texts(tok: Any, texts: List[str]) -> List[int]:
    if not texts:
        return []
    if tok is None:
        return [round(len(t) / sl.FALLBACK_CHARS_PER_TOKEN) for t in texts]
    encode_batch = getattr(tok, "encode_batch", None)
    if callable(encode_batch):
        try:
            return [len(enc.ids) for enc in encode_batch(texts)]
        except Exception:  # noqa: BLE001
            pass
    return [sl.count_tokens(tok, t) for t in texts]


def build_length_stats(
    files: Dict[str, List[Dict[str, Any]]],
    *,
    tok: Any = None,
    tok_src: str = "",
    prefix: str = "",
    length_gate: Optional[Dict[str, Any]] = None,
    source: str = "pack",
) -> Dict[str, Any]:
    """按包内 lf 文件名统计直方图。``files`` 键为 ``sft.train.lf.jsonl`` 这类名字。"""
    if tok is None and not tok_src:
        tok, tok_src = sl.find_tokenizer()
    estimated = tok is None
    datasets: Dict[str, Dict[str, Any]] = {}
    for name, rows in sorted(files.items()):
        base = Path(name).name
        if not base.endswith(".lf.jsonl"):
            continue
        key = base.replace(".lf.jsonl", "")
        if "dpo" in key.lower():
            lens = [n for n, _ in sl.dpo_lengths(rows, tok, prefix)]
            hist = length_hist(lens, DPO_TOKEN_BINS)
            hist["spark"] = length_hist(lens, DPO_SPARK_BINS)["bins"]
            datasets[key] = hist
        else:
            texts = [sl.render(prefix, sl.row_turns(r)) for r in rows]
            lens = count_texts(tok, texts)
            hist = length_hist(lens, SFT_TOKEN_BINS)
            hist["spark"] = length_hist(lens, SFT_SPARK_BINS)["bins"]
            datasets[key] = hist
    note = (
        f"口径：ChatML 渲染 + 分词器（{tok_src}）。"
        + ("未找到 tokenizer，已按字符/2.13 估算。" if estimated else "实测 token。")
        + " SFT 档位含 8k–10k / 10k–12k（对应常见 cutoff 10240）。"
        + (" 本文件在打包时写入。" if source == "pack" else " 旧包缺少静态文件，已临时计算。")
    )
    return {
        "ok": True,
        "unit": "tokens",
        "estimated": estimated,
        "tokenizer": tok_src,
        "source": source,
        "note": note,
        "length_gate": length_gate if isinstance(length_gate, dict) else {},
        "datasets": datasets,
    }


def dumps_length_stats(stats: Dict[str, Any]) -> str:
    return json.dumps(stats, ensure_ascii=False, indent=2) + "\n"


def read_length_stats_static(path: Path) -> Optional[Dict[str, Any]]:
    """只读静态文件（包内或 sidecar），绝不现场分词。"""
    return read_length_stats_from_zip(path) or read_length_stats_sidecar(path)


def pack_length_preview(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """历史列表用的紧凑预览：主看 SFT train 分布 + ≥10k。"""
    if not isinstance(stats, dict) or not stats.get("ok"):
        return None
    datasets = stats.get("datasets") if isinstance(stats.get("datasets"), dict) else {}
    sft = datasets.get("sft.train") if isinstance(datasets.get("sft.train"), dict) else {}
    dpo = datasets.get("dpo.train") if isinstance(datasets.get("dpo.train"), dict) else {}
    def _bins(raw: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for b in raw or []:
            if not isinstance(b, dict):
                continue
            out.append({
                "label": str(b.get("label") or ""),
                "lo": int(b.get("lo") or 0),
                "hi": int(b.get("hi") or 0),
                "count": int(b.get("count") or 0),
            })
        return out

    def _series(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict) or not raw.get("count"):
            return None
        spark = _bins(raw.get("spark"))
        bins = spark or _bins(raw.get("bins"))
        if not bins:
            return None
        return {
            "count": int(raw.get("count") or 0),
            "p50": int(raw.get("p50") or 0),
            "p90": int(raw.get("p90") or 0),
            "max": int(raw.get("max") or 0),
            "mean": int(raw.get("mean") or 0),
            "over_10k": int(raw.get("over_10k") or 0),
            "bins": bins,
            "dense": bool(spark),
        }

    sft_series = _series(sft)
    dpo_series = _series(dpo)
    if not sft_series and not dpo_series:
        return None
    return {
        "unit": stats.get("unit") or "tokens",
        "estimated": bool(stats.get("estimated")),
        "source": stats.get("source") or "",
        "sft": sft_series,
        "dpo": dpo_series,
    }


def read_length_stats_from_zip(path: Path) -> Optional[Dict[str, Any]]:
    import zipfile  # noqa: PLC0415

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for arc in LENGTH_STATS_ARC_CANDIDATES:
                if arc in names:
                    data = json.loads(zf.read(arc).decode("utf-8"))
                    if isinstance(data, dict) and data.get("ok"):
                        data = dict(data)
                        data.setdefault("source", "pack")
                        return data
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def sidecar_path(zip_path: Path) -> Path:
    return zip_path.with_suffix(zip_path.suffix + ".length_stats.json")


def read_length_stats_sidecar(zip_path: Path) -> Optional[Dict[str, Any]]:
    path = sidecar_path(zip_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("ok"):
        data = dict(data)
        data["source"] = "sidecar"
        return data
    return None


def write_length_stats_sidecar(zip_path: Path, stats: Dict[str, Any]) -> None:
    path = sidecar_path(zip_path)
    payload = dict(stats)
    payload["source"] = "sidecar"
    path.write_text(dumps_length_stats(payload), encoding="utf-8")


def rows_from_zip_lf(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """旧包回退：从 zip 里读出全部 ``*.lf.jsonl``。"""
    import zipfile  # noqa: PLC0415

    files: Dict[str, List[Dict[str, Any]]] = {}
    with zipfile.ZipFile(path) as zf:
        for arc in zf.namelist():
            base = Path(arc).name
            if not base.endswith(".lf.jsonl"):
                continue
            rows: List[Dict[str, Any]] = []
            with zf.open(arc) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            files[base] = rows
    return files
