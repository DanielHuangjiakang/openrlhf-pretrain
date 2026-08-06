#!/bin/bash
#
# Convert an OLMo unsharded checkpoint to HuggingFace format, which is what both
# the evaluator (inference/run_inference_all.py, via vLLM) and the RL stage
# consume.
#
#   bash pretraining/scripts/convert_to_hf.sh <checkpoint-dir> [output-dir]
#
# e.g. bash pretraining/scripts/convert_to_hf.sh \
#          workspace/checkpoints/OLMo-150M-tg30/latest-unsharded
#
# The upstream version of this script pointed at two files that are not in the
# repository: '<PATH_TO_OLMO>/convert_olmo_weights_to_hf.py' and a
# 'save_tokenizer.py'. The first is OLMo's own converter, vendored here at
# OLMo/scripts/convert_olmo_to_hf_new.py; the second is inlined below.
set -euo pipefail

input_dir="${1:?usage: convert_to_hf.sh <checkpoint-dir> [output-dir]}"
output_dir="${2:-${input_dir%/}-hf}"
tokenizer_name="${TOKENIZER:-meta-llama/Llama-2-7b-hf}"

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "input : $input_dir"
echo "output: $output_dir"

# --no_fix_eos_token_id: by default the converter rewrites eos_token_id 0 -> 50279,
# which is the EOS of OLMo's *own* 50k-token tokenizer. These models use the
# Llama-2 tokenizer (vocab 32000), so 50279 would be out of range. The configs
# therefore keep eos_token_id at 0 deliberately, and generation stops on the
# literal "</s>" string instead -- see the `stop=[eos_token]` in
# inference/run_inference_all.py.
#
# --no_tokenizer: skips converting OLMo's tokenizer, which does not apply here
# for the same reason. The right one is saved separately below.
"$PY" "$HERE/OLMo/scripts/convert_olmo_to_hf_new.py" \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --no_fix_eos_token_id \
    --no_tokenizer

# vLLM expects the tokenizer to live beside the weights.
"$PY" - "$tokenizer_name" "$output_dir" <<'EOF'
import sys
from transformers import AutoTokenizer

tokenizer_name, target_dir = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(tokenizer_name)
assert len(tok) == 32000, f"vocab is {len(tok)}, the model configs expect 32000"
tok.save_pretrained(target_dir)
print(f"tokenizer {tokenizer_name} (vocab {len(tok)}) -> {target_dir}")
EOF

echo "done: $output_dir"
