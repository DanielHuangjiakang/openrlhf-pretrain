#!/usr/bin/env python3
"""One-screen view of everything running on the box.

The pipeline is several hours of pretraining, GRPO and evaluation chained
together, and each stage reports differently -- OLMo prints per-step lines,
openrlhf prints a tqdm bar, evaluation writes nothing until a checkpoint is
finished. This pulls all of it into one place and adds what none of them print:
where the whole pipeline is, and when it ends.

    python3 scripts/progress.py           # one shot
    python3 scripts/progress.py -w        # refresh every 15s, for a tmux pane

Standard library only, so it runs under the system interpreter without
activating the venv.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(os.environ.get("PROJ", "/root/openrlhf-pretrain"))
CKPT = ROOT / "workspace" / "checkpoints"
LOGS = Path(os.environ.get("LOGDIR", "/root"))
LOCAL_OFFSET = int(os.environ.get("LOCAL_UTC_OFFSET", "-4"))  # for the second clock

TQDM = re.compile(
    r"(?P<label>Episode|Train epoch)\s*\[(?P<ep>[\d/]+)\][^|]*\|[^|]*\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[\d.]+)s/it"
)
OLMO_STEP = re.compile(r"\[step=(\d+)/(\d+)")


def bar(done, total, width=30):
    if not total:
        return "░" * width
    n = int(width * done / total)
    return "█" * n + "░" * (width - n)


def tail(path, n=200_000):
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > n:
                f.seek(size - n)
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def running(pattern):
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except Exception:
        return False


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
        u, t, util, temp = (x.strip() for x in out.split(","))
        return f"{int(u)/1024:.1f}/{int(t)/1024:.1f} GB   util {util}%   {temp}C"
    except Exception:
        return "n/a"


def active_run():
    """The GRPO or pretraining job currently holding the GPU, with its bar."""
    best = None
    for p in sorted(LOGS.glob("*.log"), key=lambda x: -x.stat().st_mtime):
        if time.time() - p.stat().st_mtime > 300:
            continue  # nothing written in 5 min -- not the live one
        txt = tail(p)
        m = None
        for m in TQDM.finditer(txt.replace("\r", "\n")):
            pass
        if m:
            d, t = int(m["done"]), int(m["total"])
            eta = m["eta"] if m["eta"] != "?" else None
            best = (p.stem, f"{m['label']} [{m['ep']}]", d, t, float(m["rate"]), eta)
            break
        last = None
        for last in OLMO_STEP.finditer(txt):
            pass
        if last:
            best = (p.stem, "pretraining", int(last[1]), int(last[2]), 0.0, None)
            break
    return best


def curve_state():
    """How many checkpoints of each RL run have a finished evaluation."""
    rows = []
    for d in sorted(CKPT.glob("*-grpo*")):
        ck = d / "ckpt"
        if not ck.is_dir():
            continue
        steps = sorted(
            int(re.search(r"global_step(\d+)_hf", p).group(1))
            for p in glob.glob(str(ck / "global_step*_hf")))
        if not steps:
            continue
        done = 0
        for s in steps:
            f = ck / f"global_step{s}_hf" / "eval_gsm8k_1.json"
            try:
                if "final_accuracy" in f.read_bytes()[-4096:].decode("utf-8", "replace"):
                    done += 1
            except Exception:
                pass
        rows.append((d.name.replace("OLMo-150M-", ""), done, len(steps)))
    return rows


def status_files():
    out = []
    for f in sorted(LOGS.glob("*status*.txt")):
        lines = [l.rstrip() for l in f.read_text(errors="replace").splitlines() if l.strip()]
        if lines:
            out.append((f.stem, lines[-1]))
    return out


def render():
    now = datetime.now(timezone.utc)
    local = now + timedelta(hours=LOCAL_OFFSET)
    L = [f"  {now:%H:%M} UTC    ({local:%H:%M} 本地, UTC{LOCAL_OFFSET:+d})", ""]

    a = active_run()
    if a:
        name, label, d, t, rate, eta = a
        pct = 100 * d / t if t else 0
        line = f"  {bar(d, t)}  {d}/{t}  {pct:5.1f}%"
        if rate:
            line += f"   {rate:.1f}s/step"
        L += [f"  ▸ {name}  —  {label}", line]
        if eta:
            try:
                parts = [int(x) for x in eta.split(":")]
                secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
                fin = now + timedelta(seconds=secs)
                L.append(f"  {'':32}  剩 {eta}  →  {fin:%H:%M} UTC / "
                         f"{(fin + timedelta(hours=LOCAL_OFFSET)):%H:%M} 本地")
            except Exception:
                pass
    else:
        L.append("  ▸ 无活动任务（GPU 空闲或阶段切换中）")
    L.append("")

    rows = curve_state()
    if rows:
        L.append("  ▸ 评测曲线完成度")
        for name, done, tot in rows:
            mark = "✓" if done == tot else " "
            L.append(f"    {mark} {name:<16} {bar(done, tot, 16)} {done:>2}/{tot}")
        L.append("")

    st = status_files()
    if st:
        L.append("  ▸ 任务链")
        for name, last in st:
            L.append(f"    {name:<16} {last}")
        L.append("")

    procs = [(n, p) for n, p in (("训练", "train_ppo_ray|OLMo/scripts/train.py"),
                                 ("评测", "run_inference_all"),
                                 ("任务链", "round2.sh|fill-curves.sh|overnight.sh")) if running(p)]
    L.append("  ▸ 进程    " + ("  ".join(f"{n}✓" for n, _ in procs) if procs else "无"))
    L.append(f"  ▸ GPU     {gpu()}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-w", "--watch", nargs="?", type=int, const=15, default=None,
                   help="refresh every N seconds (default 15)")
    args = p.parse_args()
    if args.watch is None:
        print(render())
        return 0
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
