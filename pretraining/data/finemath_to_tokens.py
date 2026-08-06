"""Tokenize FineMath-3+ (background corpus).

Upstream globbed every parquet shard in the repo (~34B tokens). A 1B-token run
needs ~0.5B of FineMath, so `--shards` takes a seeded random subset instead.

    python pretraining/data/finemath_to_tokens.py --dest /data --shards 8
    python pretraining/data/finemath_to_tokens.py --dest /tmp/smoke --shards 1 --smoke 2000
"""

from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

from _common import build_arg_parser, run_tokenization, select_shards

DATASET_NAME = "finemath3"
HF_REPO = "HuggingFaceTB/finemath"
HF_PATH = f"hf://datasets/{HF_REPO}"
SUBDIR = "finemath-3plus"  # the quality-filtered subset the paper uses


def main() -> None:
    args = build_arg_parser(__doc__).parse_args()

    paths_file = select_shards(
        HF_REPO,
        out_path=Path(args.dest) / "shard_lists" / f"{DATASET_NAME}.txt",
        subdir=SUBDIR,
        suffix=".parquet",
        n=args.shards,
        seed=args.seed,
    )

    reader = ParquetReader(
        data_folder=f"{HF_PATH}/{SUBDIR}",
        paths_file=str(paths_file),
        text_key="text",
        id_key="id",
        read_metadata=False,
        limit=args.smoke or -1,
    )

    # Plain web text: nothing to assemble, the `text` column is the document.
    run_tokenization(DATASET_NAME, [reader], args)


if __name__ == "__main__":
    main()
