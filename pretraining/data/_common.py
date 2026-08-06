"""Shared plumbing for the *_to_tokens.py scripts.

Two things live here:

1. Deterministic shard selection. The upstream scripts tokenize an entire
   HuggingFace dataset. For a fixed token budget we only need a fraction of the
   big corpora (roughly 4% of FineMath-3+ and 15% of Algebraic-Stack for a
   1B-token run), so downloading everything wastes hours and tens of GB.

   Taking the *first* N shards would be the obvious shortcut, but shard order in
   a dataset repo is not random -- shards are often laid out by crawl date,
   source domain, or quality bucket, so a prefix of the shard list is a biased
   sample. Instead we list every shard, take a seeded random subset, and hand
   datatrove an explicit paths file.

2. The argparse + LocalPipelineExecutor boilerplate that was duplicated across
   all seven upstream scripts.

Everything is a pure function of (repo, subdir, n, seed), so re-running on
another machine reproduces byte-identical inputs.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

DEFAULT_TOKENIZER = "meta-llama/Llama-2-7b-hf"


# --------------------------------------------------------------------------
# Shard selection
# --------------------------------------------------------------------------


def list_shards(repo_id: str, subdir: str = "", suffix: str = ".parquet") -> List[str]:
    """Return every matching file in a HF dataset repo, sorted, repo-relative."""
    from huggingface_hub import list_repo_files

    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    files = [
        f
        for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    return sorted(files)


def select_shards(
    repo_id: str,
    out_path: Path,
    subdir: str = "",
    suffix: str = ".parquet",
    n: Optional[int] = None,
    seed: int = 0,
) -> Path:
    """Pick `n` shards at random (seeded) and write a datatrove paths file.

    datatrove readers accept `paths_file`: one path per line, relative to the
    reader's `data_folder`. Since `data_folder` already points at `subdir`, we
    strip that prefix before writing.

    `n=None` selects everything, reproducing the upstream glob-the-world
    behaviour.
    """
    all_shards = list_shards(repo_id, subdir=subdir, suffix=suffix)
    if not all_shards:
        raise RuntimeError(
            f"No '{suffix}' files found under '{subdir or '/'}' in dataset repo '{repo_id}'. "
            "Check the repo id and subdirectory, and that you are logged in if it is gated."
        )

    if n is None or n >= len(all_shards):
        picked = all_shards
    else:
        # Sample from the *sorted* list so the seed fully determines the result,
        # then re-sort: read order stays stable no matter which shards come out.
        picked = sorted(random.Random(seed).sample(all_shards, n))

    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    relative = [p[len(prefix) :] for p in picked]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(relative) + "\n")

    print(
        f"[shards] {repo_id}:{subdir or '/'} -- selected {len(picked)}/{len(all_shards)} "
        f"shards (seed={seed}) -> {out_path}"
    )
    return out_path


# --------------------------------------------------------------------------
# CLI + pipeline execution
# --------------------------------------------------------------------------


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by every *_to_tokens.py script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--dest",
        required=True,
        help="Output root. Tokenized data lands in <dest>/<name>-tokenized/.",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Tokenize only this many randomly chosen shards. Default: all of them.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for shard selection and for datatrove's document shuffle.",
    )
    p.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Smoke-test mode: read at most this many documents per task. 0 = no limit.",
    )
    p.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="datatrove tasks (parallel workers). Default: $SLURM_CPUS_PER_TASK or os.cpu_count().",
    )
    p.add_argument(
        "--start-method",
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="multiprocessing start method. Use 'spawn' if 'fork' is unstable (e.g. macOS).",
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    return p


def resolve_tasks(args) -> int:
    if args.tasks is not None:
        return args.tasks
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)


def run_tokenization(dataset_name: str, steps: list, args) -> None:
    """Append the tokenizer stage to `steps` and run the whole pipeline.

    `steps` is everything up to but excluding tokenization: a reader, plus any
    per-dataset text-assembly function.
    """
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.tokens import DocumentTokenizer

    dest = Path(args.dest)
    out_dir = dest / f"{dataset_name}-tokenized"
    tasks = resolve_tasks(args)

    print(
        f"[tokenize] {dataset_name}: {tasks} tasks, tokenizer={args.tokenizer}, "
        f"seed={args.seed}" + (f", SMOKE limit={args.smoke} docs/task" if args.smoke else "")
    )

    # NOTE: shuffle=True makes datatrove write an intermediate "*unshuffled*"
    # copy alongside the final one, so peak disk during tokenization is roughly
    # 2x the final size. build_mixture.py filters the unshuffled files out.
    executor = LocalPipelineExecutor(
        pipeline=steps
        + [
            DocumentTokenizer(
                output_folder=str(out_dir),
                tokenizer_name_or_path=args.tokenizer,
                eos_token="</s>",
                shuffle=True,
                seed=args.seed,
            )
        ],
        tasks=tasks,
        workers=-1,
        logging_dir=str(dest / "logs" / "datatrove" / dataset_name),
        local_tasks=tasks,
        local_rank_offset=0,
        start_method=args.start_method,
    )
    executor.run()
    print(f"[tokenize] {dataset_name}: done -> {out_dir}")
