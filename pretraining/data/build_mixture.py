#!/usr/bin/env python3
"""Build compute-matched pretraining mixtures by slicing tokenized .ds files.

Replaces mix_datasets.py. The upstream approach controlled a dataset's share by
symlinking its files N times ("4x TinyGSM"), which meant (a) the share was only
adjustable in whole-dataset multiples and (b) total training compute changed
whenever the mixture changed, so two mixtures were never comparable.

Here every group gets the *same* number of training blocks; the only variable is
how many of those blocks come from the reasoning corpus.

--------------------------------------------------------------------------
Why slicing bytes is legitimate
--------------------------------------------------------------------------
A .ds file is a flat array of uint16 token ids -- no header, no index, no record
boundaries. OLMo turns it into training examples at *read* time by chunking it
into `model.max_sequence_length` tokens (memmap_dataset.py:169-172 derives the
example count as `file_size // (itemsize * chunk_size)`), and it never opens the
sibling .ds.index / .ds.metadata files that datatrove writes.

So truncating a .ds at a multiple of `seq_len * itemsize` bytes yields a
perfectly valid, shorter .ds. That is the whole mechanism.

--------------------------------------------------------------------------
Nested subsets
--------------------------------------------------------------------------
To keep the groups comparable, the data they share must be *identical*, not just
statistically similar. Every group takes a prefix of each source, so the smaller
take is always a byte-exact subset of the larger one:

    tinygsm           tg15 ⊆ tg30              (higher share -> take more)
    finemath3         tg30 ⊆ tg15 ⊆ tg00       (higher share -> take less)
    algebraic-stack   tg30 ⊆ tg15 ⊆ tg00

Going 0% -> 15% -> 30% therefore does exactly one thing: it swaps a tail of
background text for an equal number of TinyGSM blocks. Everything else is
bit-for-bit unchanged. `--verify` re-reads the written bytes and checks this with
sha256 rather than trusting the arithmetic.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# datatrove writes an intermediate "*unshuffled*" copy next to each shuffled
# output when shuffle=True. Those are not training data.
UNSHUFFLED_MARKER = "unshuffled"

DTYPE_ITEMSIZE = {"uint16": 2, "uint32": 4}


# ==========================================================================
# Scanning
# ==========================================================================


def scan_source(directory: Path, bytes_per_block: int) -> List[Tuple[Path, int]]:
    """List a source's .ds files (sorted, deterministic) with their block counts.

    Sorting matters: it fixes the order in which the interleaver walks files, so
    two runs of this script on the same inputs produce the same slices.
    """
    files = sorted(
        p
        for p in directory.glob("*.ds")
        if UNSHUFFLED_MARKER not in p.name
    )
    if not files:
        raise FileNotFoundError(f"No .ds files (excluding *{UNSHUFFLED_MARKER}*) under {directory}")

    out = []
    for p in files:
        blocks = p.stat().st_size // bytes_per_block
        if blocks > 0:
            out.append((p, blocks))
    if not out:
        raise ValueError(
            f"{directory}: every .ds file is smaller than one block "
            f"({bytes_per_block} bytes). Is --seq-len too large?"
        )
    return out


# ==========================================================================
# Allocation
# ==========================================================================


def allocate_prefix(block_counts: List[int], target: int) -> List[int]:
    """Spread `target` blocks over files proportionally, monotonically in `target`.

    Each block is keyed by its *relative position inside its own file*,
    (k + 0.5) / n_blocks_in_file. Sorting all blocks by that key interleaves the
    files, and taking the first `target` of the ordering gives each file roughly
    its proportional share while only ever taking a prefix of it.

    The property we actually need is monotonicity: growing `target` can only add
    blocks to a file, never remove one, because the chosen set is literally a
    prefix of a fixed global ordering. That is what makes the per-group takes
    nest. A naive `floor(target * share)` plus largest-remainder top-up does NOT
    have this property -- the remainder can move between files as the target
    grows, silently breaking the subset relation.
    """
    n_files = len(block_counts)
    total = sum(block_counts)
    if target > total:
        raise ValueError(
            f"Need {target:,} blocks but the source only has {total:,}. "
            "Tokenize more shards (--shards) or lower --total-tokens."
        )
    if target <= 0:
        return [0] * n_files

    keys = np.concatenate([(np.arange(b) + 0.5) / b for b in block_counts if b > 0])
    owner = np.concatenate([np.full(b, i, dtype=np.int64) for i, b in enumerate(block_counts) if b > 0])
    order = np.argsort(keys, kind="stable")
    chosen = owner[order[:target]]
    return np.bincount(chosen, minlength=n_files).astype(int).tolist()


def split_background(bg_total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Split `bg_total` blocks among background sources, summing exactly."""
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    raw = bg_total * w
    alloc = np.floor(raw).astype(np.int64)
    # Hand the leftover blocks to the largest fractional parts. Deterministic,
    # and with a 1:1 background ratio it just alternates.
    leftover = bg_total - int(alloc.sum())
    for j in np.argsort(-(raw - alloc), kind="stable")[:leftover]:
        alloc[j] += 1
    assert int(alloc.sum()) == bg_total
    return {n: int(a) for n, a in zip(names, alloc)}


def plan_groups(
    total_blocks: int,
    fractions: List[float],
    reasoning: str,
    bg_weights: Dict[str, float],
    group_prefix: str,
) -> Dict[str, Dict[str, int]]:
    """Per-group, per-source block targets. Every group sums to `total_blocks`."""
    plan: Dict[str, Dict[str, int]] = {}
    for frac in fractions:
        name = f"{group_prefix}{round(frac * 100):02d}"
        r_blocks = round(total_blocks * frac)
        targets = {reasoning: r_blocks}
        targets.update(split_background(total_blocks - r_blocks, bg_weights))
        assert sum(targets.values()) == total_blocks
        plan[name] = targets
    return plan


# ==========================================================================
# Writing
# ==========================================================================


def copy_range(src: Path, dst: Path, start: int, n_bytes: int, chunk: int = 8 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remaining = n_bytes
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fin.seek(start)
        while remaining > 0:
            buf = fin.read(min(chunk, remaining))
            if not buf:
                raise IOError(f"{src}: expected {n_bytes} bytes from offset {start}, file ended early")
            fout.write(buf)
            remaining -= len(buf)


def copy_prefix(src: Path, dst: Path, n_bytes: int, chunk: int = 8 << 20) -> None:
    copy_range(src, dst, 0, n_bytes, chunk)


def sha256_prefix(path: Path, n_bytes: int, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    remaining = n_bytes
    with open(path, "rb") as f:
        while remaining > 0:
            buf = f.read(min(chunk, remaining))
            if not buf:
                break
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()


# ==========================================================================
# Reporting
# ==========================================================================


def print_selfcheck(manifest: dict, verify_results: List[str]) -> bool:
    m = manifest
    seq_len, bpb = m["seq_len"], m["bytes_per_block"]
    total_blocks = m["total_blocks_per_group"]

    print()
    print("=" * 78)
    print(" MIXTURE SELF-CHECK")
    print("=" * 78)
    print(
        f" seq_len={seq_len}  dtype={m['dtype']}  bytes/block={bpb}  "
        f"blocks/group={total_blocks:,}  tokens/group={total_blocks * seq_len:,}"
    )
    if m["holdout_blocks_per_source"]:
        print(f" holdout: {m['holdout_blocks_per_source']:,} blocks per source (excluded from every group)")
    print()
    print(f" {'group':<7}{'source':<18}{'files':>6}{'blocks':>11}{'tokens':>16}{'share':>9}{'pool used':>11}")
    print(f" {'-'*6:<7}{'-'*17:<18}{'-'*5:>6}{'-'*10:>11}{'-'*15:>16}{'-'*8:>9}{'-'*10:>11}")
    for gname, g in m["groups"].items():
        for sname, s in g["sources"].items():
            print(
                f" {gname:<7}{sname:<18}{s['files']:>6}{s['blocks']:>11,}"
                f"{s['tokens']:>16,}{s['share'] * 100:>8.2f}%{s['pool_fraction']:>10.2f}x"
            )
        print(
            f" {gname:<7}{'TOTAL':<18}{'':>6}{g['blocks']:>11,}{g['tokens']:>16,}{100.0:>8.2f}%"
        )
        print()

    ok = True

    counts = [g["blocks"] for g in m["groups"].values()]
    aligned = len(set(counts)) == 1
    ok &= aligned
    print(
        f" [align]  blocks per group: {', '.join(f'{c:,}' for c in counts)}"
        f"  -> {'IDENTICAL  OK' if aligned else 'MISMATCH  FAIL'}"
    )

    # Shares are quantised by the block count: one block is 1/total_blocks of the
    # mixture, so rounding can be off by at most half of that and no allocator
    # can do better. The tolerance has to scale with the group size -- a fixed
    # one spuriously fails small runs and is far too loose for large ones.
    reasoning = m["reasoning_source"]
    tol = 0.5 / total_blocks + 1e-12
    print(f" [share]  quantisation limit at {total_blocks:,} blocks: +/-{tol * 100:.4f}%")
    for gname, g in m["groups"].items():
        want = g["reasoning_fraction"]
        got = g["sources"][reasoning]["share"]
        good = abs(got - want) <= tol
        ok &= good
        print(
            f" [share]  {gname}: {reasoning} = {got * 100:6.2f}%"
            f"  (target {want * 100:.2f}%, off by {abs(got - want) * 100:.4f}%)"
            f"  -> {'OK' if good else 'FAIL'}"
        )

    for line in verify_results:
        ok &= not line.endswith("FAIL")
        print(f" {line}")

    print()
    print(f" RESULT: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    print("=" * 78)
    return bool(ok)


# ==========================================================================
# Main
# ==========================================================================


def parse_source(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME=DIRNAME, got '{spec}'")
    name, dirname = spec.split("=", 1)
    return name, dirname


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", required=True, help="Directory holding the *-tokenized dirs.")
    p.add_argument("--out-root", required=True, help="Where the per-group mixture dirs are written.")
    p.add_argument(
        "--reasoning",
        required=True,
        type=parse_source,
        help="NAME=DIRNAME of the corpus whose share is swept, e.g. tinygsm=tinygsm-tokenized",
    )
    p.add_argument(
        "--background",
        required=True,
        action="append",
        type=parse_source,
        help="NAME=DIRNAME of a background corpus. Repeat once per corpus.",
    )
    p.add_argument(
        "--background-weights",
        default=None,
        help="Comma-separated relative weights for the background corpora, in the "
        "order they were passed. Default: equal (1:1).",
    )
    p.add_argument("--total-tokens", type=float, default=1e9, help="Token budget per group.")
    p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Must equal model.max_sequence_length in the OLMo config -- that is what "
        "decides how a .ds is chunked at read time.",
    )
    p.add_argument("--dtype", default="uint16", choices=sorted(DTYPE_ITEMSIZE))
    p.add_argument(
        "--align-to",
        type=int,
        default=256,
        help="Round the per-group block count down to a multiple of this. Set it to "
        "global_train_batch_size so the run ends on a whole optimizer step.",
    )
    p.add_argument(
        "--fractions",
        default="0,0.15,0.30",
        help="Reasoning-corpus share for each group.",
    )
    p.add_argument("--group-prefix", default="tg")
    p.add_argument(
        "--holdout-blocks",
        type=int,
        default=0,
        help="Reserve this many blocks per source as a held-out eval set, taken from "
        "the TAIL of every file and excluded from all groups. Point OLMo's "
        "`evaluators:` at them -- train loss is not comparable across groups "
        "(templated corpora score lower), so a fixed held-out set is the only "
        "signal you can actually compare during training.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan, write nothing.")
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip re-reading the written files to confirm the nesting property.",
    )
    args = p.parse_args()

    itemsize = DTYPE_ITEMSIZE[args.dtype]
    bytes_per_block = args.seq_len * itemsize

    reasoning_name, reasoning_dir = args.reasoning
    bg = dict(args.background)
    if args.background_weights:
        vals = [float(x) for x in args.background_weights.split(",")]
        if len(vals) != len(bg):
            p.error(f"--background-weights has {len(vals)} values but {len(bg)} background sources")
        bg_weights = dict(zip(bg, vals))
    else:
        bg_weights = {n: 1.0 for n in bg}

    data_root, out_root = Path(args.data_root), Path(args.out_root)

    # --- scan -------------------------------------------------------------
    source_dirs = {reasoning_name: reasoning_dir, **bg}
    full_sources = {
        name: scan_source(data_root / dirname, bytes_per_block)
        for name, dirname in source_dirs.items()
    }

    # Carve the held-out eval set off the TAIL of every file before anything
    # else looks at the data. Groups only ever take prefixes, so reserving the
    # tail makes it structurally impossible for a group to see an eval block.
    holdout: Dict[str, List[int]] = {}
    sources: Dict[str, List[Tuple[Path, int]]] = {}
    for name, files in full_sources.items():
        counts = [b for _, b in files]
        h = allocate_prefix(counts, args.holdout_blocks) if args.holdout_blocks else [0] * len(counts)
        holdout[name] = h
        sources[name] = [(path, b - hi) for (path, b), hi in zip(files, h)]

    print("[scan] blocks per source:")
    for name, files in sources.items():
        avail = sum(b for _, b in files)
        held = sum(holdout[name])
        print(
            f"  {name:<18} {len(files):>3} files  {avail:>12,} usable"
            f"  {avail * args.seq_len:>16,} tokens"
            + (f"  (+{held:,} held out)" if held else "")
        )

    # --- plan -------------------------------------------------------------
    total_blocks = int(args.total_tokens // args.seq_len)
    if args.align_to > 1:
        total_blocks -= total_blocks % args.align_to
    if total_blocks <= 0:
        p.error("--total-tokens is too small for the given --seq-len/--align-to")

    fractions = [float(x) for x in args.fractions.split(",")]
    plan = plan_groups(total_blocks, fractions, reasoning_name, bg_weights, args.group_prefix)

    # Per-file allocations. allocate_prefix is monotonic in the target, so
    # ordering groups by target gives nested per-file takes for free -- but we
    # assert it here anyway, because getting this wrong is silent.
    alloc: Dict[str, Dict[str, List[int]]] = {}
    for gname, targets in plan.items():
        alloc[gname] = {
            sname: allocate_prefix([b for _, b in sources[sname]], targets[sname])
            for sname in sources
        }
    for sname in sources:
        ordered = sorted(plan, key=lambda g: plan[g][sname])
        for a, b in zip(ordered, ordered[1:]):
            for i, (small, large) in enumerate(zip(alloc[a][sname], alloc[b][sname])):
                assert small <= large, (
                    f"nesting would be violated for {sname} file #{i}: "
                    f"{a}={small} > {b}={large}"
                )

    # --- manifest ---------------------------------------------------------
    manifest = {
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "bytes_per_block": bytes_per_block,
        "total_tokens_requested": args.total_tokens,
        "align_to": args.align_to,
        "total_blocks_per_group": total_blocks,
        "holdout_blocks_per_source": args.holdout_blocks,
        "reasoning_source": reasoning_name,
        "background_weights": bg_weights,
        "sources": {
            name: {
                "dir": str(data_root / source_dirs[name]),
                "files": len(files),
                "blocks_total": sum(b for _, b in full_sources[name]),
                "blocks_usable": sum(b for _, b in files),
                "blocks_holdout": sum(holdout[name]),
            }
            for name, files in sources.items()
        },
        "groups": {},
    }
    for gname, targets in plan.items():
        gsources = {}
        for sname, files in sources.items():
            counts = alloc[gname][sname]
            taken = sum(counts)
            pool = sum(b for _, b in files)
            gsources[sname] = {
                "files": sum(1 for c in counts if c > 0),
                "blocks": taken,
                "tokens": taken * args.seq_len,
                "share": taken / total_blocks,
                "pool_fraction": taken / pool,
                "per_file_blocks": {files[i][0].name: c for i, c in enumerate(counts) if c > 0},
            }
        manifest["groups"][gname] = {
            "reasoning_fraction": next(f for f in fractions if f"{args.group_prefix}{round(f*100):02d}" == gname),
            "blocks": sum(s["blocks"] for s in gsources.values()),
            "tokens": sum(s["tokens"] for s in gsources.values()),
            "sources": gsources,
        }

    if args.dry_run:
        print_selfcheck(manifest, ["[nested] skipped (--dry-run)"])
        print("\n[dry-run] nothing written.")
        return 0

    # --- write ------------------------------------------------------------
    if args.holdout_blocks:
        for name, files in full_sources.items():
            for (src, total_b), h in zip(files, holdout[name]):
                if h == 0:
                    continue
                copy_range(
                    src,
                    out_root / "holdout" / name / src.name,
                    (total_b - h) * bytes_per_block,
                    h * bytes_per_block,
                )
        print(f"[write] holdout: {args.holdout_blocks:,} blocks per source")

    for gname in plan:
        for sname, files in sources.items():
            counts = alloc[gname][sname]
            for (src, _), n in zip(files, counts):
                if n == 0:
                    continue
                copy_prefix(src, out_root / gname / sname / src.name, n * bytes_per_block)
        print(f"[write] {gname}: {manifest['groups'][gname]['blocks']:,} blocks")

    # --- verify -----------------------------------------------------------
    verify_lines: List[str] = []
    if args.holdout_blocks:
        overlap = max(
            max(alloc[g][name][i] for g in plan) + holdout[name][i] - total_b
            for name, files in full_sources.items()
            for i, (_, total_b) in enumerate(files)
        )
        verify_lines.append(
            f"[holdout] {args.holdout_blocks:,} blocks/source reserved from file tails, "
            f"no group reaches them  {'OK' if overlap <= 0 else 'FAIL'}"
        )
    if args.no_verify:
        verify_lines.append("[nested] skipped (--no-verify)")
    else:
        for sname, files in sources.items():
            ordered = sorted(plan, key=lambda g: plan[g][sname])
            chain = " <= ".join(ordered)
            pairs = 0
            bad = 0
            for a, b in zip(ordered, ordered[1:]):
                for i, (src, _) in enumerate(files):
                    na, nb = alloc[a][sname][i], alloc[b][sname][i]
                    if na == 0:
                        continue
                    fa = out_root / a / sname / src.name
                    fb = out_root / b / sname / src.name
                    if not fb.exists() or nb < na:
                        bad += 1
                        continue
                    n = na * bytes_per_block
                    pairs += 1
                    if sha256_prefix(fa, n) != sha256_prefix(fb, n):
                        bad += 1
            status = "OK" if bad == 0 else "FAIL"
            verify_lines.append(
                f"[nested] {sname:<18} {chain:<22} {pairs} file pairs sha256-matched  {status}"
            )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ok = print_selfcheck(manifest, verify_lines)
    print(f"\nmanifest -> {out_root / 'manifest.json'}")
    for gname in plan:
        print(f"  data.paths: ${{path.glob:{out_root / gname}/*/*.ds}}")
    if args.holdout_blocks:
        for name in sources:
            print(f"  evaluator '{name}': ${{path.glob:{out_root / 'holdout' / name}/*.ds}}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
