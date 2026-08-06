"""Tokenize TinyGSM (the reasoning corpus whose share we sweep).

Each training document is the problem statement followed by the Python solution:

    <question>

    def simple_math_problem() -> int:
        ...

That concatenation is what teaches the model to answer by *writing code that
computes the answer* rather than doing arithmetic itself, which is the behaviour
the Echo Chamber experiments measure. openrlhf/utils/math_verifier.py mirrors it
at scoring time by exec'ing the generated function.

TinyGSM is small (~2B tokens), so `--shards` is usually left at "all".

    python pretraining/data/tinygsm_to_tokens.py --dest /data
    python pretraining/data/tinygsm_to_tokens.py --dest /tmp/smoke --shards 1 --smoke 2000
"""

from pathlib import Path

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import ParquetReader

from _common import build_arg_parser, run_tokenization, select_shards

DATASET_NAME = "tinygsm"
HF_REPO = "TinyGSM/TinyGSM"
HF_PATH = f"hf://datasets/{HF_REPO}"
SUBDIR = "data"


def process_math(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    """Append the reference Python solution to the question text."""
    for document in data:
        document.text = document.text + "\n\n" + document.metadata["code"]
        yield document


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
        text_key="question",
        id_key="id",
        read_metadata=True,  # needed for the `code` column
        limit=args.smoke or -1,
    )

    run_tokenization(DATASET_NAME, [reader, process_math], args)


if __name__ == "__main__":
    main()
