#!/usr/bin/env python3
"""Validate build_mixture.py on synthetic .ds files.

Runs with no downloads, no tokenizer access and no GPU, so the slicing logic can
be checked before committing to hours of tokenization.

The thing being guarded here is that a wrong slice is *silent*: the training run
would look completely normal and just quietly have the wrong mixture. So we
check the three properties the experiment depends on --

  1. every group gets exactly the same number of blocks (compute is matched),
  2. the reasoning corpus really is 0% / 15% / 30% of each group,
  3. the shared data is byte-identical across groups (nested subsets),

plus a direct stress test of the allocator's monotonicity, which is the one
property that makes (3) hold.

    python pretraining/data/test_build_mixture.py
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_mixture import allocate_prefix  # noqa: E402

HERE = Path(__file__).parent
SEQ_LEN = 128  # tiny, so the fixture is a few MB instead of a few GB
ITEMSIZE = 2
BYTES_PER_BLOCK = SEQ_LEN * ITEMSIZE

# (source dir, [blocks per file]) -- deliberately uneven file sizes so the
# proportional interleaving actually has to do something.
FIXTURE = {
    "tinygsm-tokenized": [900, 1200, 400, 500],
    "finemath3-tokenized": [2000, 1500, 1800, 900, 1200, 600],
    "algebraic-stack-tokenized": [1400, 1100, 900, 1000, 600],
}
TOTAL_BLOCKS = 2560  # = 10 * 256, so --align-to 256 leaves it untouched
FRACTIONS = [0.0, 0.15, 0.30]
HOLDOUT_BLOCKS = 200


def build_fixture(root: Path) -> None:
    """Write fake .ds files full of deterministic, position-dependent tokens."""
    for dirname, sizes in FIXTURE.items():
        d = root / dirname
        d.mkdir(parents=True)
        for i, blocks in enumerate(sizes):
            rng = np.random.default_rng(abs(hash((dirname, i))) % (2**32))
            tokens = rng.integers(0, 32000, size=blocks * SEQ_LEN, dtype=np.uint16)
            (d / f"{i:05d}_shuffled.ds").write_bytes(tokens.tobytes())
        # datatrove leaves these behind; build_mixture must ignore them.
        (d / "99999_unshuffled.ds").write_bytes(b"\x00" * (BYTES_PER_BLOCK * 50))


def test_allocator_is_monotonic() -> None:
    """Growing the target must never shrink any file's take.

    This is what guarantees nested subsets. A floor+largest-remainder allocator
    passes a casual eyeball test and fails this.
    """
    rng = random.Random(0)
    for trial in range(200):
        counts = [rng.randint(1, 500) for _ in range(rng.randint(1, 12))]
        total = sum(counts)
        prev = allocate_prefix(counts, 0)
        for target in range(1, total + 1, max(1, total // 40)):
            cur = allocate_prefix(counts, target)
            assert sum(cur) == target, f"trial {trial}: sum {sum(cur)} != target {target}"
            for i, (a, b) in enumerate(zip(prev, cur)):
                assert a <= b, f"trial {trial}: file {i} shrank {a} -> {b} at target {target}"
            for i, (c, cap) in enumerate(zip(cur, counts)):
                assert c <= cap, f"trial {trial}: file {i} over-allocated {c} > {cap}"
            prev = cur
    print("[test] allocator monotonicity ....... OK (200 random source layouts)")


def test_end_to_end(tmp: Path) -> dict:
    data_root, out_root = tmp / "tokenized", tmp / "mixtures"
    build_fixture(data_root)

    cmd = [
        sys.executable, str(HERE / "build_mixture.py"),
        "--data-root", str(data_root),
        "--out-root", str(out_root),
        "--reasoning", "tinygsm=tinygsm-tokenized",
        "--background", "finemath3=finemath3-tokenized",
        "--background", "algebraic-stack=algebraic-stack-tokenized",
        "--total-tokens", str(TOTAL_BLOCKS * SEQ_LEN),
        "--seq-len", str(SEQ_LEN),
        "--align-to", "256",
        "--fractions", ",".join(str(f) for f in FRACTIONS),
        "--holdout-blocks", str(HOLDOUT_BLOCKS),
    ]
    print("\n$ " + " ".join(cmd[1:]) + "\n")
    result = subprocess.run(cmd, text=True)
    assert result.returncode == 0, f"build_mixture.py exited {result.returncode}"

    manifest = json.loads((out_root / "manifest.json").read_text())

    # 1. compute alignment
    counts = {g: v["blocks"] for g, v in manifest["groups"].items()}
    assert len(set(counts.values())) == 1, f"block counts differ across groups: {counts}"
    assert set(counts.values()) == {TOTAL_BLOCKS}, counts

    # 2. reasoning share
    for gname, g in manifest["groups"].items():
        got = g["sources"]["tinygsm"]["share"]
        want = g["reasoning_fraction"]
        assert abs(got - want) < 1e-4, f"{gname}: tinygsm share {got} != {want}"

    # 3. nesting, re-derived from disk rather than trusting the manifest
    for source in ("tinygsm", "finemath3", "algebraic-stack"):
        order = sorted(
            manifest["groups"],
            key=lambda g: manifest["groups"][g]["sources"][source]["blocks"],
        )
        for a, b in zip(order, order[1:]):
            for fname, n in manifest["groups"][a]["sources"][source]["per_file_blocks"].items():
                pa = (out_root / a / source / fname).read_bytes()
                pb = (out_root / b / source / fname).read_bytes()
                assert len(pa) == n * BYTES_PER_BLOCK, f"{a}/{source}/{fname} wrong size"
                assert pb[: len(pa)] == pa, f"{a}/{source}/{fname} is not a prefix of {b}'s"

    # 4. the leftover datatrove artifacts stayed out
    for g in manifest["groups"].values():
        for s in g["sources"].values():
            assert not any("unshuffled" in f for f in s["per_file_blocks"])

    # 5. held-out data exists and is disjoint from every group. The eval blocks
    #    come off the tail of each source file, so no group's prefix can contain
    #    them -- check that directly rather than trusting the arithmetic.
    for source in ("tinygsm", "finemath3", "algebraic-stack"):
        held = sorted((out_root / "holdout" / source).glob("*.ds"))
        assert held, f"no holdout written for {source}"
        assert sum(p.stat().st_size for p in held) == HOLDOUT_BLOCKS * BYTES_PER_BLOCK
        for hp in held:
            tail = hp.read_bytes()
            for gname in manifest["groups"]:
                gp = out_root / gname / source / hp.name
                if gp.exists():
                    assert tail not in gp.read_bytes(), f"{gname}/{source}/{hp.name} leaks eval data"

    print("\n[test] block counts identical ....... OK")
    print("[test] reasoning shares 0/15/30% .... OK")
    print("[test] nested subsets (byte-exact) .. OK")
    print("[test] unshuffled files excluded .... OK")
    print("[test] holdout disjoint from groups . OK")
    return manifest


def main() -> int:
    test_allocator_is_monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        test_end_to_end(Path(tmp))
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
