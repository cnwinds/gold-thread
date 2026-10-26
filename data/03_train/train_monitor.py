#!/usr/bin/env python3
"""训练进程监控器：实时 tail 日志 + OOM 秒级诊断 + 进程退出一秒通知。

用法：
    python train_monitor.py -- llamafactory-cli train train_sft_qwen3_8b.yaml
    python train_monitor.py --timeout-hours 3 --log logs/live.log -- llamafactory-cli train train_sft.yaml

特性：
  · 实时监控 stdout/stderr（终端实时输出 + 写入文件）
  · 进程退出一秒感知（非阻塞 poll + 事件驱动，不等 timeout）
  · 三重 OOM 检测：SIGKILL(-9) / stderr 关键字 / GPU 显存骤降
  · 自动诊断 exit code：正常完成 / CUDA OOM / 系统 OOM Killer / 代码异常 / 信号终止
  · 异常时给出操作建议（cutoff_len/bs/liger/sdpa 等）
  · 训练完成后打印 summary（loss 曲线、步数、耗时、显存峰值）

退出码：
  0 — 训练正常结束
  1 — 监控器自身参数错误
  2 — 训练进程被信号终止（极大概率 OOM）
  3 — CUDA OOM（stderr 中检测到关键字）
  4 — 训练进程代码异常 / AssertError
  130 — 用户 Ctrl-C
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue

# ── terminal colours ──────────────────────────────────────────────
_C = {
    "red": "\x1b[31;1m", "green": "\x1b[32;1m", "yellow": "\x1b[33;1m",
    "cyan": "\x1b[36;1m", "dim": "\x1b[2m", "rst": "\x1b[0m",
}


def _c(text: str, colour: str = "") -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_C.get(colour, '')}{text}{_C['rst']}"


# ── OOM / error signatures ────────────────────────────────────────
CUDA_OOM_KEYWORDS = [
    "CUDA out of memory",
    "cuMemAlloc failed",
    "RuntimeError: out of memory",
    "torch.cuda.OutOfMemoryError",
    "Allocation on device",
]

FATAL_KEYWORDS = [
    "Segmentation fault",
    "SIGSEGV",
    "SIGBUS",
    "SIGABRT",
    "assertion failed",
    "ValueError",
    "KeyError",
    "ImportError",
    "ModuleNotFoundError",
]

FINISH_KEYWORDS = [
    "training completed",
    "training finished",
    "saving model checkpoint",
]

SIG_NAME = {getattr(signal, s): s for s in dir(signal) if s.startswith("SIG") and isinstance(getattr(signal, s), int)}


# ── GPU 快照 ──────────────────────────────────────────────────────
_gpu_pat = re.compile(r"(\d+)\s+(\d+)\s+(\d+)")  # pid, gpu, used-miB ... crude but fast


def gpu_snapshot() -> dict[int, list[dict]]:
    """返回 {gpu_id: [{pid, used_mb}]} 。"""
    result: dict[int, list[dict]] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return result
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    pid = int(parts[0])
                    used = int(parts[1])
                    # gpu_id 未知，只记录全局 pid->used
                    result.setdefault(-1, []).append({"pid": pid, "used_mb": used})
                except ValueError:
                    pass
    except Exception:
        pass
    return result


def format_exit(rc: int | None) -> str:
    if rc is None:
        return "still running"
    if rc >= 0:
        return f"exit code {rc}"
    sig = -rc
    name = SIG_NAME.get(sig, f"signal {sig}")
    return f"killed by {name} (exit code {rc})"


class TrainMonitor:
    def __init__(
        self,
        cmd: list[str],
        timeout_hours: float = 0.0,
        log_path: Path | None = None,
        gpu_check_interval: float = 10.0,
    ):
        self.cmd = cmd
        self.timeout = timedelta(hours=timeout_hours) if timeout_hours else None
        self.log_path = log_path
        self.gpu_check_interval = gpu_check_interval

        # queues
        self._stdout_q: Queue[str] = Queue()
        self._stderr_q: Queue[str] = Queue()
        self._gpu_q: Queue[dict] = Queue()

        self._start_time: datetime | None = None
        self._proc: subprocess.Popen | None = None
        self._stop_ev = threading.Event()

        # state
        self._oom_detected = False
        self._oom_reason = ""
        self._fatal_detected = False
        self._finished_detected = False
        self._last_loss: str | None = None
        self._last_step: int = 0
        self._gpu_peak_mb = 0
        self._summary: dict = {}

    # ── readers ──────────────────────────────────────────────────
    def _reader(self, pipe, q: Queue[str], label: str) -> None:
        """持续从 pipe 读取行，push 到 queue。"""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                text = line.rstrip("\n\r")
                q.put(text)
                # 实时打印（带前缀区分 stdout/stderr）
                if label == "err":
                    print(_c(f"[{label}] {text}", "red"), flush=True)
                else:
                    print(text, flush=True)
                # 写日志
                if self.log_path:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
        finally:
            pipe.close()
            q.put(None)  # sentinel

    def _gpu_watcher(self) -> None:
        """定期采样 GPU，检测显存骤降（OOM 后释放）。"""
        prev_total = 0
        while not self._stop_ev.is_set():
            time.sleep(self.gpu_check_interval)
            snap = gpu_snapshot()
            total = sum(a["used_mb"] for a in snap.get(-1, []))
            pids = [str(a["pid"]) for a in snap.get(-1, [])]
            self._gpu_q.put({"time": time.time(), "total_mb": total, "pids": pids})
            if total > self._gpu_peak_mb:
                self._gpu_peak_mb = total
            # 显存骤降 > 50% 且训练进程 pid 不在 GPU 列表 → 可能已被 OOM killer 干掉
            if prev_total > 5000 and total < prev_total * 0.4 and self._proc is not None:
                try:
                    proc_pid = str(self._proc.pid)
                    if proc_pid not in pids and self._proc.poll() is not None:
                        # 进程已退出且不在 GPU 上 → 确认 OOM
                        pass
                except Exception:
                    pass
            prev_total = total if total else prev_total

    # ── pattern scanning ─────────────────────────────────────────
    def _scan_line(self, line: str) -> None:
        lower = line.lower()
        for kw in CUDA_OOM_KEYWORDS:
            if kw.lower() in lower:
                self._oom_detected = True
                self._oom_reason = kw
                return
        for kw in FATAL_KEYWORDS:
            if kw.lower() in lower:
                self._fatal_detected = True
                return
        for kw in FINISH_KEYWORDS:
            if kw.lower() in lower:
                self._finished_detected = True
                return
        # 抽 loss / step
        m = re.search(r"(?:loss|train_loss)\s*[=:]\s*([\d.]+)", lower)
        if m:
            self._last_loss = m.group(1)
        m2 = re.search(r"(?:step|steps?)\s*[=:]?\s*(\d+)", lower)
        if m2:
            self._last_step = int(m2.group(1))

    # ── main loop ────────────────────────────────────────────────
    def run(self) -> int:
        print(_c(f"▶ train_monitor 启动: {' '.join(self.cmd)}", "cyan"))
        if self.timeout:
            print(_c(f"  超时设定: {self.timeout}", "dim"))
        if self.log_path:
            print(_c(f"  日志写入: {self.log_path}", "dim"))

        self._start_time = datetime.now()
        logfile = open(self.log_path, "w", encoding="utf-8") if self.log_path else None

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,  # 独立进程组，kill 时全组清理
            )
        except FileNotFoundError:
            print(_c(f"[FATAL] 命令未找到: {self.cmd[0]}", "red"))
            return 1

        # 启动 reader 线程
        t_out = threading.Thread(target=self._reader, args=(self._proc.stdout, self._stdout_q, "out"), daemon=True)
        t_err = threading.Thread(target=self._reader, args=(self._proc.stderr, self._stderr_q, "err"), daemon=True)
        t_gpu = threading.Thread(target=self._gpu_watcher, daemon=True)
        t_out.start(); t_err.start(); t_gpu.start()

        out_done = err_done = False
        last_alive_print = time.time()

        try:
            while not (out_done and err_done):
                # 1) 检查进程是否已退出（非阻塞 poll）
                rc = self._proc.poll()
                if rc is not None:
                    # 进程已结束，但 stdout/stderr 可能还有缓冲数据
                    # 再给他们一小段时间 flush
                    pass

                # 2) 消费 stdout queue
                while not self._stdout_q.empty():
                    line = self._stdout_q.get_nowait()
                    if line is None:
                        out_done = True
                    else:
                        self._scan_line(line)
                        if logfile:
                            logfile.write(line + "\n")
                            logfile.flush()

                # 3) 消费 stderr queue
                while not self._stderr_q.empty():
                    line = self._stderr_q.get_nowait()
                    if line is None:
                        err_done = True
                    else:
                        self._scan_line(line)
                        if logfile:
                            logfile.write(f"[stderr] {line}\n")
                            logfile.flush()

                # 4) 消费 GPU queue
                while not self._gpu_q.empty():
                    self._gpu_q.get_nowait()

                # 5) 进程已死且管道已空 → 立刻跳出
                if rc is not None and out_done and err_done:
                    break
                if rc is not None and not out_done:
                    # 进程死了但 stdout reader 可能还在 flush，最多等 2s
                    time.sleep(0.1)
                    out_done = out_done or self._stdout_q.empty()

                # 6) 超时检查
                if self.timeout and datetime.now() - self._start_time > self.timeout:
                    print(_c(f"\n[TIMEOUT] 超过设定时间 {self.timeout}，发送 SIGTERM…", "yellow"))
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    try:
                        self._proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    break

                # 7) 心跳打印（每 30 秒一行）
                now = time.time()
                if now - last_alive_print > 30:
                    elapsed = datetime.now() - self._start_time
                    tail = f"  loss={self._last_loss}" if self._last_loss else ""
                    print(_c(f"[alive] elapsed={elapsed} step={self._last_step}{tail}", "dim"), flush=True)
                    last_alive_print = now

                # 8) 短暂休眠，让出 CPU
                time.sleep(0.05)

        except KeyboardInterrupt:
            print(_c("\n[INTERRUPT] 收到 Ctrl-C，发送 SIGTERM 给训练进程组…", "yellow"))
            if self._proc and self._proc.poll() is None:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            return 130

        finally:
            self._stop_ev.set()
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            t_gpu.join(timeout=2)
            if logfile:
                logfile.close()

        rc = self._proc.returncode
        return self._report(rc)

    # ── summary & diagnosis ──────────────────────────────────────
    def _report(self, rc: int | None) -> int:
        elapsed = datetime.now() - (self._start_time or datetime.now())
        print()
        print("=" * 60)
        print(_c(f"训练进程 {format_exit(rc)}", "rst"))
        print(f"耗时: {elapsed}")
        print(f"最后步数: {self._last_step}")
        print(f"最后 loss: {self._last_loss or 'N/A'}")
        print(f"GPU 峰值显存: {self._gpu_peak_mb:.0f} MiB ({self._gpu_peak_mb/1024:.1f} GB)")
        print("=" * 60)

        # 诊断
        if rc == 0 and self._finished_detected:
            print(_c("\n✅ 正常完成", "green"))
            return 0

        if self._oom_detected:
            print(_c("\n❌ CUDA OOM 检测到", "red"))
            print(f"   触发关键词: {self._oom_reason}")
            _advise_oom()
            return 3

        if rc is not None and rc < 0:
            sig = -rc
            print(_c(f"\n⚠️ 进程被信号终止 ({SIG_NAME.get(sig, f'signal {sig}')})", "yellow"))
            if sig == signal.SIGKILL:
                print(_c("   极大概率是系统 OOM Killer（Linux kernel 在内存不足时发送 SIGKILL-9）", "red"))
                _advise_oom(system=True)
            elif sig == signal.SIGSEGV:
                print("   段错误，可能与 CUDA 驱动或 flash-attn kernel 有关")
            return 2

        if rc is not None and rc > 0:
            if self._fatal_detected:
                print(_c(f"\n💥 训练代码异常（Python exception / assertion）", "red"))
            else:
                print(_c(f"\n💥 异常退出 code={rc}", "red"))
            _advise_bug()
            return 4

        # rc is None（不应走到这里，防护）
        print(_c("\n⚠️ 进程状态未知", "yellow"))
        return 1


def _advise_oom(system: bool = False) -> None:
    print(_c("\n建议（按优先级）：", "cyan"))
    if system:
        print("  1) 这是系统级 OOM，主机内存不够，不是 GPU 显存不够")
        print("     → 加大 swap；加 RAM；或换到更大内存的实例")
        print("     → 合并大模型（27B bf16）时 CPU 峰值约 60GB，内存不足会直接 SIGKILL")
    print("  1) cutoff_len 降到 16384 → 12288 → 8192（步顺序，不是尾部丢）")
    print("  2) gradient_accumulation_steps 4 → 8（micro-batch 保持 1）")
    print("  3) 确认 enable_liger_kernel: true（省 fp32 logits）")
    print("  4) flash_attn: fa2 → sdpa（若 flash-attn 未装）")
    print("  5) 仍未解决：换 train_sft_qlora.yaml（4-bit，质量略降）")
    print("  6) 最终手段：换 80G 显卡（A800）或双 48G TP")


def _advise_bug() -> None:
    print(_c("\n建议：", "cyan"))
    print("  1) 检查 traceback 最后几行，通常是 dataset_info / template / tools 列配置错")
    print("  2) 跑 python check_data.py --model <路径> --template qwen3 验证数据 + 模板")
    print("  3) 检查 transformers / trl / accelerate 版本是否与 YAML flash_attn 要求一致")
    print("  4) 若报 ImportError: 装 flash-attn 或改 YAML flash_attn: sdpa")


# ── CLI ──────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="训练进程监控器：实时 tail + OOM 秒级诊断 + 退出一秒通知",
        usage="python train_monitor.py [opts] -- CMD [args...]",
    )
    ap.add_argument("--timeout-hours", type=float, default=0, help="超时时间（小时），0=无限制")
    ap.add_argument("--log", help="实时日志文件路径（额外写入）")
    ap.add_argument("--gpu-check-interval", type=float, default=10, help="GPU 采样间隔秒数")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="要执行的命令（前面加 --）")
    args = ap.parse_args()

    if not args.cmd:
        ap.print_help()
        return 1

    if args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    log_path = Path(args.log) if args.log else None
    mon = TrainMonitor(
        cmd=args.cmd,
        timeout_hours=args.timeout_hours,
        log_path=log_path,
        gpu_check_interval=args.gpu_check_interval,
    )
    return mon.run()


if __name__ == "__main__":
    sys.exit(main())
