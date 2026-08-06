#!/usr/bin/env bash
#
# One-shot setup for a rented GPU box: environment, data, smoke test.
#
#   bash setup.sh env     # install everything (~20 min)
#   bash setup.sh smoke   # tiny end-to-end data run (~5 min) -- DO THIS FIRST
#   bash setup.sh data    # full tokenize + mixture build (~1 h)
#   bash setup.sh all     # env, smoke, data in order
#
# Run `smoke` before `data`. It exercises the same code path on ~2000 documents
# per source, so a wrong path or a missing dependency costs five minutes instead
# of an hour of metered GPU time.
#
set -euo pipefail

# --------------------------------------------------------------------------
# Settings -- edit these
# --------------------------------------------------------------------------
WORK="${WORK:-$PWD/workspace}"
DATA_ROOT="$WORK/tokenized"
MIX_ROOT="$WORK/mixtures"
SMOKE_ROOT="$WORK/smoke"

TOKENIZER="${TOKENIZER:-meta-llama/Llama-2-7b-hf}" # gated; NousResearch/Llama-2-7b-hf is an ungated mirror
SEQ_LEN=2048
TOTAL_TOKENS=1e9
ALIGN_TO=256    # = global_train_batch_size, so the run ends on a whole step
HOLDOUT=2000    # blocks per source reserved for the held-out evaluators
FRACTIONS="0,0.15,0.30"

# Shard counts. A 1B-token budget needs at most 0.5B tokens from each background
# source (the 0% group) and 0.3B from TinyGSM (the 30% group); these leave ~2x
# headroom. If build_mixture.py reports "Need N blocks but the source only has
# M", raise the corresponding number and re-run.
#
# Sizing, measured from a 200-document sample of one shard:
#   TinyGSM   17 shards x 697k rows x 196 tok/doc = ~137M tok/shard, ~2.3B total.
#             Reliable: document lengths are tight (166 +/- 61 tokens).
#   FineMath  128 shards x 167k rows. Per-document length is heavy-tailed (one
#             doc in the sample was 1M characters, sd 73k vs mean 10k), so a
#             200-doc mean overestimates badly. Falling back on the published
#             ~34B total for finemath-3plus gives ~266M tok/shard.
#   Algebraic 79 shards, 116 MB compressed each; at ~3-4x zstd and the measured
#             2.94 chars/token that is ~118-157M tok/shard.
SHARDS_FINEMATH=4    # >= 1.0B tokens even on the pessimistic estimate
SHARDS_ALGEBRAIC=6   # >= 0.7B tokens
SHARDS_TINYGSM=""    # empty = all 17 shards (~2.3B); it is small enough to take whole

# Everything installs into a venv rather than the system interpreter. On Ubuntu
# 24.04 that is not optional: PEP 668 marks the system Python as
# externally-managed and pip refuses to touch it. A venv also gives a clean
# slate for the numpy pin -- Vast images ship numpy 2.x, and OLMo requires <2.
VENV="$WORK/venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
PYTHON_VERSION=3.10   # see step_env: vllm 0.8.1's xgrammar pin has no cp312 wheel

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------
step_env() {
  say "0/6 host check"
  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "!! no nvidia-smi"
  python3 --version
  df -h "$PWD" | tail -1

  say "1/6 virtualenv + python deps"
  # The pinned stack needs Python 3.10. vllm 0.8.1 requires xgrammar==0.1.16,
  # which publishes no cp312 wheel (PyPI jumps 0.1.13 -> 0.1.17 for 3.12), so on
  # Ubuntu 24.04's stock 3.12 the very first install fails. Bumping vllm instead
  # is not an option: OpenRLHF 0.6.3 hooks vLLM internals in
  # trainer/ray/vllm_engine.py, and the README pins 0.8.1 for that reason.
  #
  # 24.04 ships no 3.10 package, so fetch a standalone one with uv rather than
  # adding a PPA. --seed puts pip in the venv so the rest of this script is
  # unchanged.
  export PATH="$HOME/.local/bin:/.uv/python_bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ ! -x "$PY" ]]; then
    uv python install "$PYTHON_VERSION"
    uv venv --seed --python "$PYTHON_VERSION" "$VENV"
  fi
  echo "venv python: $("$PY" -V)"
  "$PIP" install -q --upgrade pip setuptools wheel

  # vLLM first: it pins torch hard (0.8.1 -> torch 2.6.0), and letting anything
  # else choose torch first guarantees a re-resolve later.
  #
  # vllm 0.8.1 requires xgrammar==0.1.16, and 0.1.14 through 0.1.16 have since
  # been yanked from PyPI, so plain pip cannot install it at all. The obvious
  # fix -- moving to vllm 0.8.3, the earliest still-installable release --
  # cascades: 0.8.3 wants transformers>=4.51.0 against this repo's pinned 4.50.0
  # AND excludes ray 2.44.* against its pinned ray==2.44.0. That is three
  # simultaneous departures from the stack the paper was run on.
  #
  # Overriding xgrammar by one patch release is far smaller. It is only used for
  # guided/structured decoding, which nothing here touches: RL scores with
  # openrlhf/utils/math_verifier.py and evaluation samples freely. pip has no
  # way to override a hard `==` pin, so use uv, which does.
  echo "xgrammar==0.1.17" > "$WORK/pip-overrides.txt"
  uv pip install --python "$PY" --override "$WORK/pip-overrides.txt" -q vllm==0.8.1

  say "2/6 flash-attn"
  # MUST come before requirements.txt, which also lists flash-attn: pip would
  # otherwise try to build it from source, and flash-attn's setup.py imports
  # torch, which build isolation hides from it. Installing the wheel first
  # leaves the requirement already satisfied.
  #
  # The wheel filename encodes the interpreter, the torch minor version and the
  # C++ ABI, all of which are read back from the environment rather than
  # hardcoded -- both ABI variants are published and picking the wrong one
  # yields undefined-symbol errors at import.
  local py tv abi url
  py="cp$("$PY" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  tv="$("$PY" -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"
  abi="$("$PY" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
  url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch${tv}cxx11abi${abi}-${py}-${py}-linux_x86_64.whl"
  echo "trying $url"
  "$PIP" install -q "$url" || {
    echo "!! prebuilt wheel not found, compiling from source (this is slow)"
    "$PIP" install -q flash-attn==2.7.4.post1 --no-build-isolation
  }

  say "3/6 remaining deps"
  # Three lines are dropped from requirements.txt before installing. All three
  # are already installed above; leaving them in only re-opens problems pip
  # cannot solve.
  #
  #   flash-attn  installed from a prebuilt wheel. Leaving it here makes the
  #               resolver consider the sdist, whose setup.py imports torch
  #               under build isolation and always fails.
  #   torch       unpinned upstream, which fights vllm's hard torch==2.6.0. The
  #               resolver backtracks over that conflict, and the backtracking
  #               is what drags the flash-attn sdist in.
  #   vllm        installed via uv with the xgrammar override. Re-resolving it
  #               with pip would go looking for the yanked xgrammar==0.1.16
  #               again, which is the failure this whole dance avoids.
  grep -vE '^(flash-attn|torch|vllm)([=<>!~[:space:]]|$)' requirements.txt > "$WORK/requirements-filtered.txt"
  "$PIP" install -q -r "$WORK/requirements-filtered.txt"
  # datatrove is missing from requirements.txt even though data prep needs it.
  # huggingface_hub must stay below 1.0: transformers 4.50 caps it there, and an
  # unpinned install silently pulls 1.x and breaks `from transformers import ...`.
  # Note 1.x also renamed the CLI from `huggingface-cli` to `hf`; on the pinned
  # 0.x the old name is still the right one.
  "$PIP" install -q datatrove "huggingface_hub<1.0"

  say "4/6 local packages"
  "$PIP" install -q -e .
  # --no-deps keeps pip from re-resolving torch/omegaconf behind our back. These
  # are OLMo's actual runtime imports; boto3, google-api-core and rich are all
  # top-level in olmo/util.py, so they are needed even for purely local runs.
  "$PIP" install -q -e OLMo/ --no-deps
  "$PIP" install -q "numpy<2" omegaconf rich boto3 google-cloud-storage tokenizers \
                 cached_path transformers importlib_resources packaging

  say "5/6 import check"
  "$PY" - <<'PY'
import olmo, olmo.data, olmo.train           # the training path
import openrlhf                              # the RL path
import vllm, flash_attn, datatrove           # inference / data
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "arch", torch.cuda.get_arch_list()[-3:] if torch.cuda.is_available() else "-")
PY

  say "6/6 huggingface login"
  # The Llama-2 tokenizer repo is gated. Without access every tokenize run dies
  # on the first file, so fail here instead.
  "$PY" - "$TOKENIZER" <<'PY'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
assert len(tok) == 32000, f"vocab is {len(tok)}, configs expect 32000"
print(f"tokenizer OK: {sys.argv[1]}  vocab={len(tok)}  eos={tok.eos_token}/{tok.eos_token_id}")
PY
  echo "environment ready -- activate it with:  source $VENV/bin/activate"
}

# --------------------------------------------------------------------------
tokenize_all() {
  local dest="$1" shards_fm="$2" shards_as="$3" shards_tg="$4" smoke="$5"
  local extra=()
  [[ -n "$smoke" ]] && extra=(--smoke "$smoke")

  "$PY" pretraining/data/tinygsm_to_tokens.py  --dest "$dest" --tokenizer "$TOKENIZER" \
      ${shards_tg:+--shards "$shards_tg"} "${extra[@]}"
  "$PY" pretraining/data/finemath_to_tokens.py --dest "$dest" --tokenizer "$TOKENIZER" \
      --shards "$shards_fm" "${extra[@]}"
  "$PY" pretraining/data/proofpile_to_tokens.py --dest "$dest" --tokenizer "$TOKENIZER" \
      --shards "$shards_as" "${extra[@]}"
}

build_mix() {
  local data="$1" out="$2" total="$3" holdout="$4"
  "$PY" pretraining/data/build_mixture.py \
      --data-root "$data" --out-root "$out" \
      --reasoning tinygsm=tinygsm-tokenized \
      --background finemath3=finemath3-tokenized \
      --background algebraic-stack=algebraic-stack-tokenized \
      --total-tokens "$total" --seq-len "$SEQ_LEN" --align-to "$ALIGN_TO" \
      --fractions "$FRACTIONS" --holdout-blocks "$holdout"
}

# --------------------------------------------------------------------------
step_smoke() {
  say "smoke: logic check on synthetic files (no downloads)"
  "$PY" pretraining/data/test_build_mixture.py

  say "smoke: 1 shard / 2000 docs per source"
  rm -rf "$SMOKE_ROOT"
  tokenize_all "$SMOKE_ROOT" 1 1 1 2000

  say "smoke: mixture over the real .ds files"
  # 2M tokens is small enough to finish instantly but still exercises every
  # code path: block counting, per-file allocation, holdout, sha256 nesting.
  build_mix "$SMOKE_ROOT" "$SMOKE_ROOT/mixtures" 2e6 64

  cat <<EOF

Smoke passed. Check the table above: block counts identical across groups,
shares at 0/15/30%, nesting OK. Then run a 50-step training smoke test:

  sed -i "s|<PATH_TO_SMOKE_MIXTURES>|$SMOKE_ROOT/mixtures|; s|<PATH_TO_CHECKPOINT>|$WORK/checkpoints|" \\
      pretraining/configs/DEBUG-tiny.yaml
  $VENV/bin/torchrun --nproc_per_node=1 OLMo/scripts/train.py pretraining/configs/DEBUG-tiny.yaml

EOF
}

# --------------------------------------------------------------------------
step_data() {
  say "full tokenize (this is the long one)"
  tokenize_all "$DATA_ROOT" "$SHARDS_FINEMATH" "$SHARDS_ALGEBRAIC" "$SHARDS_TINYGSM" ""

  say "full mixture build"
  build_mix "$DATA_ROOT" "$MIX_ROOT" "$TOTAL_TOKENS" "$HOLDOUT"

  cat <<EOF

Data ready. Point the configs at it and start with tg30 -- it has the most
TinyGSM and so the best chance of a non-zero GSM8K pass@1. If that group scores
zero there is no point spending GPU hours on tg15 and tg00.

  for g in tg00 tg15 tg30; do
    sed -i "s|<PATH_TO_MIXTURES>|$MIX_ROOT|; s|<PATH_TO_CHECKPOINT>|$WORK/checkpoints|" \\
        pretraining/configs/RL-150M-\$g.yaml
  done

  $VENV/bin/torchrun --nproc_per_node=1 OLMo/scripts/train.py pretraining/configs/RL-150M-tg30.yaml

EOF
}

# --------------------------------------------------------------------------
case "${1:-}" in
  env)   step_env ;;
  smoke) step_smoke ;;
  data)  step_data ;;
  all)   step_env; step_smoke; step_data ;;
  *)     sed -n '2,12p' "$0"; exit 1 ;;
esac
