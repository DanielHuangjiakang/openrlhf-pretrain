#!/usr/bin/env python3
"""Compact terminal view of an OLMo training run, for when wandb is not to hand.

OLMo writes one `[step=N/M]` line per step followed by indented `metric=value`
lines, which is accurate but unreadable when it scrolls past at several steps a
second. This pulls out the numbers worth watching and adds the one OLMo does not
print: how much longer the run has to go.

    python pretraining/scripts/watch_train.py /root/train-tg30.log
    python pretraining/scripts/watch_train.py /root/train-tg30.log -w    # refresh every 10s

Standard library only, so it runs under the system interpreter without
activating the venv.
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=(\d+)\]")
METRIC_RE = re.compile(r"^\s+([\w/.\-]+)=([\d,.eE+-]+)\s*$")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RUN_RE = re.compile(r"run_name='([^']+)'")
WANDB_RE = re.compile(r"(https://wandb\.ai/\S+)")


def num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse(path: Path, tail_bytes: int = 4 << 20) -> dict:
    """Read only the tail of the log -- these files reach hundreds of MB."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
            f.readline()  # discard the partial line
        lines = f.read().decode("utf-8", "replace").splitlines()

    head = ""
    if size > tail_bytes:
        with open(path, "rb") as f:
            head = f.read(1 << 20).decode("utf-8", "replace")
    else:
        head = "\n".join(lines)

    info = {"metrics": {}, "steps": [], "eval": {}}
    m = RUN_RE.search(head)
    if m:
        info["run_name"] = m.group(1)
    m = WANDB_RE.search(head)
    if m:
        info["wandb"] = m.group(1).rstrip(".")

    for line in lines:
        sm = STEP_RE.search(line)
        if sm:
            ts = TS_RE.match(line)
            info["steps"].append(
                (
                    int(sm.group(1)),
                    int(sm.group(2)),
                    datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S") if ts else None,
                )
            )
            continue
        mm = METRIC_RE.match(line)
        if mm:
            key, val = mm.group(1), num(mm.group(2))
            if val is None:
                continue
            (info["eval"] if key.startswith("eval/") else info["metrics"])[key] = val
    return info


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "?"
    d = timedelta(seconds=int(seconds))
    h, rem = divmod(int(d.total_seconds()), 3600)
    return f"{h}h{rem // 60:02d}m"


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split("\n")[0]
        used, total, util, temp = (x.strip() for x in out.split(","))
        return f"{used}/{total} MiB   util {util}%   {temp}C"
    except Exception:
        return "n/a"


def render(path: Path) -> str:
    info = parse(path)
    steps = info["steps"]
    if not steps:
        return f"{path}: no step lines yet (still starting up?)"

    cur, total, _ = steps[-1]
    pct = 100.0 * cur / total if total else 0.0
    m, ev = info["metrics"], info["eval"]

    # Rate from the wall-clock span of the tail rather than the reported
    # throughput: it includes checkpointing and evaluation pauses, so the ETA
    # reflects the run rather than the steady-state step.
    eta = ""
    timed = [s for s in steps if s[2]]
    if len(timed) >= 2:
        span = (timed[-1][2] - timed[0][2]).total_seconds()
        done = timed[-1][0] - timed[0][0]
        if span > 0 and done > 0:
            sec_per_step = span / done
            remaining = (total - cur) * sec_per_step
            # Stamp the timezone: this runs on the rented box, whose clock is
            # usually UTC while you are reading it somewhere else, and an
            # unlabelled "finishes at 15:29" is exactly how that goes wrong.
            done_at = datetime.now().astimezone() + timedelta(seconds=remaining)
            eta = f"{fmt_eta(remaining)} left  ->  {done_at:%H:%M %Z}"

    bar_w = 40
    filled = int(bar_w * cur / total) if total else 0
    bar = "#" * filled + "." * (bar_w - filled)

    out = [
        f"  run      {info.get('run_name', path.stem)}",
        f"  progress [{bar}] {cur:,}/{total:,}  {pct:5.1f}%",
        f"  eta      {eta or '?'}",
        "",
        # Learning rate is deliberately absent: OLMo sends it to wandb but never
        # to the console, so there is nothing to read here. Check the wandb run
        # if the warmup schedule needs verifying.
        f"  loss     {m.get('train/CrossEntropyLoss', float('nan')):.4f}"
        f"      ppl {m.get('train/Perplexity', float('nan')):.1f}",
        f"  grad     {m.get('optim/total_grad_norm', float('nan')):.4f}",
        f"  speed    {m.get('throughput/device/tokens_per_second', float('nan')):,.0f} tok/s"
        f"   {m.get('throughput/device/batches_per_second', float('nan')):.2f} step/s"
        f"   seen {m.get('throughput/total_tokens', float('nan')):,.0f} tokens",
        f"  gpu      {gpu_line()}",
    ]
    if ev:
        out.append("")
        out.append("  held-out (the only cross-group comparable signal):")
        for k in sorted(ev):
            if k.endswith("CrossEntropyLoss"):
                label = k.split("/")[1]
                ppl = ev.get(f"eval/{label}/Perplexity")
                out.append(f"    {label:<26} loss {ev[k]:.4f}" + (f"   ppl {ppl:.1f}" if ppl else ""))
    if "wandb" in info:
        out += ["", f"  wandb    {info['wandb']}"]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path)
    p.add_argument("-w", "--watch", nargs="?", type=int, const=10, default=None,
                   help="refresh every N seconds (default 10)")
    args = p.parse_args()

    if not args.log.exists():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 1
    if args.watch is None:
        print(render(args.log))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + f"{datetime.now().astimezone():%H:%M:%S %Z}" + "\n")
            print(render(args.log))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
