"""Tokenize Algebraic-Stack (background corpus).

Algebraic-Stack is not a standalone dataset -- it is one subdirectory of
EleutherAI/proof-pile-2 (the other two, open-web-math and arxiv, are unused
here). Files are zstd-compressed jsonl.

Upstream globbed the whole split (~10B tokens); a 1B-token run needs ~0.5B.

    python pretraining/data/proofpile_to_tokens.py --dest /data --shards 6
    python pretraining/data/proofpile_to_tokens.py --dest /tmp/smoke --shards 1 --smoke 2000
"""

from pathlib import Path

from datatrove.pipeline.readers import JsonlReader

from _common import build_arg_parser, run_tokenization, select_shards

DATASET_NAME = "algebraic-stack"
HF_REPO = "EleutherAI/proof-pile-2"
HF_PATH = f"hf://datasets/{HF_REPO}"
SUBDIR = f"{DATASET_NAME}/train"


def default_adapter(self, data: dict, path: str, id_in_file):
    """Drop proof-pile-2's per-document metadata; we only need the text."""
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": [],
        "metadata": {},
    }


def main() -> None:
    args = build_arg_parser(__doc__).parse_args()

    paths_file = select_shards(
        HF_REPO,
        out_path=Path(args.dest) / "shard_lists" / f"{DATASET_NAME}.txt",
        subdir=SUBDIR,
        suffix=".jsonl.zst",
        n=args.shards,
        seed=args.seed,
    )

    reader = JsonlReader(
        f"{HF_PATH}/{SUBDIR}",
        paths_file=str(paths_file),
        compression="zstd",
        text_key="text",
        id_key="id",
        adapter=default_adapter,
        limit=args.smoke or -1,
    )

    run_tokenization(DATASET_NAME, [reader], args)


if __name__ == "__main__":
    main()
