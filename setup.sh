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

# Shard counts. Each FineMath shard is very roughly 0.2B tokens and each
# Algebraic-Stack shard ~0.15B, so these leave ~2x headroom over the 0.5B per
# source that a 1B-token budget needs. If build_mixture.py reports "Need N
# blocks but the source only has M", raise the corresponding number.
SHARDS_FINEMATH=8
SHARDS_ALGEBRAIC=8
SHARDS_TINYGSM=""   # empty = all of it; TinyGSM is only ~2B tokens

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------
step_env() {
  say "0/5 host check"
  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "!! no nvidia-smi"
  python3 --version
  df -h "$PWD" | tail -1

  say "1/5 python deps"
  # vLLM first: it pins torch hard (0.8.1 -> torch 2.6.0), and letting anything
  # else choose torch first guarantees a re-resolve later.
  pip install -q vllm==0.8.1
  pip install -q -r requirements.txt
  pip install -q datatrove huggingface_hub   # needed by the data scripts, absent from requirements.txt

  say "2/5 flash-attn"
  # Building from source takes 20-40 min even on 64 cores. Try the matching
  # prebuilt wheel first and fall back to pip only if that 404s.
  local py tv url
  py="cp$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  tv="$(python3 -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"
  url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch${tv}cxx11abiFALSE-${py}-${py}-linux_x86_64.whl"
  echo "trying $url"
  pip install -q "$url" || {
    echo "!! prebuilt wheel not found, compiling from source (this is slow)"
    pip install -q flash-attn==2.7.4.post1 --no-build-isolation
  }

  say "3/5 local packages"
  pip install -q -e .
  # --no-deps keeps pip from re-resolving torch/omegaconf behind our back. These
  # are OLMo's actual runtime imports; boto3, google-api-core and rich are all
  # top-level in olmo/util.py, so they are needed even for purely local runs.
  pip install -q -e OLMo/ --no-deps
  pip install -q "numpy<2" omegaconf rich boto3 google-cloud-storage tokenizers \
                 cached_path transformers importlib_resources packaging

  say "4/5 import check"
  python3 - <<'PY'
import olmo, olmo.data, olmo.train           # the training path
import openrlhf                              # the RL path
import vllm, flash_attn, datatrove           # inference / data
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "arch", torch.cuda.get_arch_list()[-3:] if torch.cuda.is_available() else "-")
PY

  say "5/5 huggingface login"
  # The Llama-2 tokenizer repo is gated. Without access every tokenize run dies
  # on the first file, so fail here instead.
  python3 - "$TOKENIZER" <<'PY'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
assert len(tok) == 32000, f"vocab is {len(tok)}, configs expect 32000"
print(f"tokenizer OK: {sys.argv[1]}  vocab={len(tok)}  eos={tok.eos_token}/{tok.eos_token_id}")
PY
  echo "environment ready"
}

# --------------------------------------------------------------------------
tokenize_all() {
  local dest="$1" shards_fm="$2" shards_as="$3" shards_tg="$4" smoke="$5"
  local extra=()
  [[ -n "$smoke" ]] && extra=(--smoke "$smoke")

  python3 pretraining/data/tinygsm_to_tokens.py  --dest "$dest" --tokenizer "$TOKENIZER" \
      ${shards_tg:+--shards "$shards_tg"} "${extra[@]}"
  python3 pretraining/data/finemath_to_tokens.py --dest "$dest" --tokenizer "$TOKENIZER" \
      --shards "$shards_fm" "${extra[@]}"
  python3 pretraining/data/proofpile_to_tokens.py --dest "$dest" --tokenizer "$TOKENIZER" \
      --shards "$shards_as" "${extra[@]}"
}

build_mix() {
  local data="$1" out="$2" total="$3" holdout="$4"
  python3 pretraining/data/build_mixture.py \
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
  python3 pretraining/data/test_build_mixture.py

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
  torchrun --nproc_per_node=1 OLMo/scripts/train.py pretraining/configs/DEBUG-tiny.yaml

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

  torchrun --nproc_per_node=4 OLMo/scripts/train.py pretraining/configs/RL-150M-tg30.yaml

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
