#!/usr/bin/env bash
#
# Run A: three 150M models, 8.86B tokens each, 0/15/30% TinyGSM, on 8 GPUs.
#
#   bash run.sh probe          # 30s   read-only environment report
#   bash run.sh env            # ~25m  install everything into workspace/venv
#   bash run.sh smoke          # ~15m  END-TO-END rehearsal on tiny data. DO THIS.
#   bash run.sh data           # ~3-5h download + tokenize + build mixtures (CPU)
#   bash run.sh train tg30     # ~2-3h one group
#   bash run.sh train all      #        all three, in order tg30 -> tg15 -> tg00
#   bash run.sh eval           # ~30m  convert to HF + GSM8K
#   bash run.sh status         #       where everything is
#
# Every stage is idempotent and resumable: rerun after an interruption and it
# picks up rather than restarting. Training auto-resumes from the newest
# checkpoint in the run's save folder.
#
# Progress is appended to workspace/status.txt -- that file is the thing to send
# back; it is enough to tell what happened without reading any logs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WORK="${WORK:-$HERE/workspace}"
VENV="$WORK/venv"
PY="$VENV/bin/python"
MIX="$WORK/mixtures"
CKPT="$WORK/checkpoints"
LOGS="$WORK/logs"
STATUS="$WORK/status.txt"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# Budget. Must agree with pretraining/configs/make_scaled_configs.py -- that
# file explains why 4,326,400 and not more.
TOTAL_BLOCKS=4326400
SEQ_LEN=2048
export TOTAL_TOKENS=$((TOTAL_BLOCKS * SEQ_LEN))   # 8,860,467,200
export ALIGN_TO=256
export HOLDOUT=2000
export FRACTIONS="0,0.15,0.30"

# Shard counts, from the measured per-shard block counts in the 1B manifest
# (FineMath 158,070 blocks/shard, Algebraic-Stack 75,120). The 0% group is the
# hungriest for background: it needs TOTAL_BLOCKS/2 = 2,163,200 from each, i.e.
# 13.7 FineMath shards and 28.8 Algebraic-Stack shards. Rounded up for slack.
# This is a small fraction of both corpora -- do NOT download all 128/79.
export SHARDS_FINEMATH="${SHARDS_FINEMATH:-16}"
export SHARDS_ALGEBRAIC="${SHARDS_ALGEBRAIC:-32}"
export SHARDS_TINYGSM=""        # all 17; the 30% group needs every one of them

# The three corpora are streamed straight out of HuggingFace by datatrove
# (hf://datasets/...), and huggingface_hub caches every downloaded shard. The
# default cache is ~/.cache/huggingface, which on a cluster is usually a small
# home quota or a shared root partition -- ~80GB of parquet lands there and
# fills it. Keep the cache next to everything else this run produces.
export HF_HOME="${HF_HOME:-$WORK/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORK/hf-cache/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Ungated mirror of the Llama-2 tokenizer, so no HuggingFace account, token or
# licence click-through is needed anywhere in this pipeline. Verified against
# the tokenizer the 1B run actually used: tokenizer.model is byte-identical, and
# tokenizer.json has the same vocab and the same 61,249 merges (serialised
# differently by a newer `tokenizers`). TOKENIZER_SHA256 below is checked in the
# smoke stage -- if HuggingFace ever repoints that repo, the run stops instead
# of silently producing a corpus that cannot be compared with the 1B results.
export TOKENIZER="${TOKENIZER:-NousResearch/Llama-2-7b-hf}"
TOKENIZER_SHA256=9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347

mkdir -p "$WORK" "$LOGS"
note() { printf '[%s] %s\n' "$(date -u +%m-%d\ %H:%M)" "$*" | tee -a "$STATUS"; }
die()  { printf '\n!! %s\n' "$*" | tee -a "$STATUS" >&2; exit 1; }

need_venv() { [[ -x "$PY" ]] || die "no venv yet -- run: bash run.sh env"; }

# ---------------------------------------------------------------------------
launch() {   # launch <config> <logfile>
    local cfg="$1" log="$2" save
    save="$(sed -n 's|^save_folder: ||p' "$cfg" | sed "s|\${run_name}|$(sed -n 's|^run_name: ||p' "$cfg")|")"

    local resume=()
    if [[ -e "$save/latest" ]]; then
        note "  resuming from $save/latest"
        resume=(--load_path="$save/latest" --save_overwrite=true)
    fi

    # NCCL: leave the transport to auto-detection, but make failures loud and
    # fast rather than hanging for half an hour on a borrowed machine.
    NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    "$VENV/bin/torchrun" --nproc_per_node="$NGPU" --standalone \
        OLMo/scripts/train.py "$cfg" "${resume[@]}" >>"$log" 2>&1
}

point_configs() {   # rewrite the <PATH_TO_*> placeholders in place, idempotently
    local mixdir="$1" suffix="${2:-}"
    for g in tg00 tg15 tg30; do
        local c="pretraining/configs/RL-150M-${g}${suffix}.yaml"
        [[ -f "$c" ]] || die "missing $c"
        sed -i "s|<PATH_TO_MIXTURES>|$mixdir|g; s|<PATH_TO_SMOKE_MIXTURES>|$mixdir|g; s|<PATH_TO_CHECKPOINT>|$CKPT|g" "$c"
    done
    grep -l "PATH_TO_" pretraining/configs/RL-150M-*${suffix}.yaml 2>/dev/null \
        && die "some placeholders were not substituted"
    return 0
}

# ---------------------------------------------------------------------------
step_probe() { bash scripts/probe.sh; }

step_env() {
    note "=== env: installing (~25 min) ==="
    bash setup.sh env || die "environment setup failed -- send $LOGS and the console output"
    note "=== env: done ==="
}

step_smoke() {
    need_venv
    note "=== smoke: rehearsal on tiny data (~15 min) ==="

    note "  [0/4] tokenizer fingerprint"
    "$PY" - "$TOKENIZER" "$TOKENIZER_SHA256" <<'PY' | tee -a "$STATUS" || die "tokenizer check failed"
import hashlib, sys
from huggingface_hub import hf_hub_download
want = sys.argv[2]
got = hashlib.sha256(open(hf_hub_download(sys.argv[1], "tokenizer.model"), "rb").read()).hexdigest()
print(f"    {sys.argv[1]}\n    sha256 {got[:24]}...  {'OK' if got == want else 'MISMATCH'}")
if got != want:
    raise SystemExit(
        f"tokenizer.model does not match the one the 1B run used ({want[:24]}...).\n"
        "Block counts would differ and the two studies could not be compared.")
PY

    note "  [1/4] allocator logic on synthetic files"
    "$PY" pretraining/data/test_build_mixture.py || die "allocator self-test failed"

    note "  [2/4] tokenize + mixture on ~2000 docs/source"
    bash setup.sh smoke || die "smoke data build failed"

    note "  [3/4] 50-step training on $NGPU GPU(s) -- this is the real test"
    sed -i "s|<PATH_TO_SMOKE_MIXTURES>|$WORK/smoke/mixtures|g; s|<PATH_TO_CHECKPOINT>|$CKPT|g" \
        pretraining/configs/DEBUG-tiny.yaml
    NCCL_ASYNC_ERROR_HANDLING=1 "$VENV/bin/torchrun" --nproc_per_node="$NGPU" --standalone \
        OLMo/scripts/train.py pretraining/configs/DEBUG-tiny.yaml \
        >"$LOGS/smoke-train.log" 2>&1 || die "multi-GPU training failed -- send $LOGS/smoke-train.log"

    note "  [4/4] measured throughput"
    grep -oE "throughput/total_tokens_per_second=[0-9.]+" "$LOGS/smoke-train.log" | tail -3 | tee -a "$STATUS" \
        || note "    (no throughput line found; check $LOGS/smoke-train.log)"

    note "=== smoke: PASSED. Safe to run 'data'. ==="
}

step_data() {
    need_venv
    note "=== data: download + tokenize + mixtures (~3-5 h, CPU only) ==="
    note "  budget $TOTAL_TOKENS tokens/group  shards: fm=$SHARDS_FINEMATH as=$SHARDS_ALGEBRAIC tg=all"
    df -h "$WORK" | tail -1 | tee -a "$STATUS"

    bash setup.sh data 2>&1 | tee "$LOGS/data.log" || die "data build failed -- send $LOGS/data.log"

    note "  verifying the three groups are compute-matched"
    "$PY" - "$MIX/manifest.json" <<'PY' | tee -a "$STATUS" || die "mixture verification FAILED"
import json, sys
m = json.load(open(sys.argv[1]))
tot = {g: v["blocks"] for g, v in m["groups"].items()}
tok = {g: v["tokens"] for g, v in m["groups"].items()}
print("    blocks/group:", tot)
print("    tokens/group:", tok)
assert len(set(tot.values())) == 1, "GROUPS ARE NOT COMPUTE-MATCHED"
for g, v in m["groups"].items():
    s = v["sources"].get("tinygsm", {}).get("share", 0.0)
    print(f"    {g}: tinygsm share {s*100:.4f}%")
print("    OK: identical budgets, exact shares")
PY
    note "=== data: done ==="
}

step_train() {
    need_venv
    local which="${1:-all}"
    [[ -d "$MIX" ]] || die "no mixtures yet -- run: bash run.sh data"
    point_configs "$MIX" "-8b"

    local groups=(tg30 tg15 tg00)          # tg30 first: most TinyGSM, most signal
    [[ "$which" != "all" ]] && groups=("$which")

    for g in "${groups[@]}"; do
        local cfg="pretraining/configs/RL-150M-${g}-8b.yaml"
        local log="$LOGS/train-${g}-8b.log"
        note "=== train $g (16,900 steps, ~2-3 h on $NGPU GPUs) ==="
        launch "$cfg" "$log" || die "$g training failed -- send the last 100 lines of $log"
        note "=== train $g: done ==="
    done
}

step_eval() {
    need_venv
    note "=== eval: convert to HF + GSM8K ==="
    for g in tg30 tg15 tg00; do
        local src="$CKPT/OLMo-150M-${g}-8b/latest-unsharded"
        local dst="$CKPT/OLMo-150M-${g}-8b/latest-unsharded-hf"
        [[ -d "$src" ]] || { note "  skip $g (not trained yet)"; continue; }
        [[ -d "$dst" ]] || bash pretraining/scripts/convert_to_hf.sh "$src" "$dst" >>"$LOGS/convert.log" 2>&1 \
            || die "HF conversion failed for $g -- send $LOGS/convert.log"
        note "  $g: pass@1"
        "$PY" inference/run_inference_all.py -c "$dst" -t gsm8k --no_multiple \
            >>"$LOGS/eval.log" 2>&1 || note "  !! eval failed for $g"
        "$PY" - "$dst/eval_gsm8k_1.json" <<'PY' | tee -a "$STATUS"
import json, sys
m = json.loads(open(sys.argv[1]).read().strip().split("\n")[-1])
print("    pass@1 %.2f%%  code %.2f%%  text %.2f%%" % (
    m.get("final_accuracy", 0)*100,
    m.get("tinygsm-code_count", 0)*100,
    m.get("text_count", 0)*100))
PY
    done
    note "=== eval: done ==="
}

step_status() {
    echo "GPUs: $NGPU"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv 2>/dev/null
    echo; echo "--- status.txt ---"; tail -30 "$STATUS" 2>/dev/null || echo "(nothing yet)"
    echo; echo "--- training progress ---"
    for f in "$LOGS"/train-*.log; do
        [[ -f "$f" ]] || continue
        printf "  %-24s %s\n" "$(basename "$f")" "$(grep -oE '\[step=[0-9]+/[0-9]+' "$f" | tail -1)"
    done
    echo; df -h "$WORK" | tail -1
}

case "${1:-}" in
    probe)  step_probe ;;
    env)    step_env ;;
    smoke)  step_smoke ;;
    data)   step_data ;;
    train)  step_train "${2:-all}" ;;
    eval)   step_eval ;;
    status) step_status ;;
    *)      sed -n '2,20p' "$0"; exit 1 ;;
esac
