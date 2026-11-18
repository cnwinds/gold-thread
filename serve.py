#!/usr/bin/env python3
"""启动 GoldThread 本地数据集控制台。

    python serve.py            # http://127.0.0.1:8765
    python serve.py --port 9000 --no-browser
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn


def main() -> None:
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="允许在未设置 GOLDTHREAD_API_TOKEN 时监听非回环地址（不推荐用于不可信网络）",
    )
    args = parser.parse_args()

    is_loopback = args.host.strip().lower() in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not os.environ.get("GOLDTHREAD_API_TOKEN", "").strip():
        if not args.allow_remote:
            print(
                f"[ui] 拒绝在未设置 GOLDTHREAD_API_TOKEN 时暴露到非本地地址 ({args.host})；"
                "请设置环境变量 GOLDTHREAD_API_TOKEN，或显式加 --allow-remote。",
                file=sys.stderr,
            )
            raise SystemExit(2)
        os.environ.setdefault("GOLDTHREAD_ALLOWED_HOSTS", "*")
        print(f"[ui] 警告：已无鉴权暴露至 {args.host}，请确保处于可信局域网环境", file=sys.stderr)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"[ui] GoldThread 控制台: {url}")
    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()